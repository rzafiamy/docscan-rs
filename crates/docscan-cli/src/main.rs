//! `docscan`: convert a UVDoc checkpoint to GGUF, scan photos, or serve the
//! HTTP API.

mod server;

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use docscan::{Format, Geometry, Mode, Options, Paper, Scanner};
use std::path::{Path, PathBuf};
use std::time::Instant;

#[derive(Parser)]
#[command(
    name = "docscan",
    version,
    about = "Document photo to scan (UVDoc dewarping), GGUF weights"
)]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(clap::Args, Clone)]
struct RunOpts {
    /// docscan GGUF (from `docscan convert`)
    #[arg(short, long, env = "DOCSCAN_MODEL")]
    model: PathBuf,
    /// Run on the CPU even when a GPU build finds a GPU
    #[arg(long)]
    cpu: bool,
    /// CPU threads (default: all cores)
    #[arg(long)]
    threads: Option<usize>,
}

#[derive(clap::Args, Clone)]
struct ScanOpts {
    /// color (flattened light, white paper), gray, bw, original (geometry only)
    #[arg(long, default_value = "color")]
    mode: Mode,
    /// dewarp (perspective + curl), perspective (4 corners, straight lines), none
    #[arg(long, default_value = "dewarp")]
    geometry: Geometry,
    /// Four corners in photo pixels ("x,y,x,y,x,y,x,y"): skips detection
    #[arg(long)]
    corners: Option<String>,
    /// png, jpeg, pdf (default: from the output extension, else png)
    #[arg(long)]
    format: Option<Format>,
    /// Resolution written in JPEG / PDF
    #[arg(long, default_value_t = 300)]
    dpi: u16,
    /// PDF page: auto (image at its DPI), a4, letter
    #[arg(long, default_value = "auto")]
    paper: Paper,
    /// JPEG quality (also JPEG inside a PDF)
    #[arg(long, default_value_t = 90)]
    quality: u8,
    /// Longest output side in pixels
    #[arg(long, default_value_t = 8000)]
    max_side: usize,
}

impl ScanOpts {
    fn options(&self) -> Result<Options> {
        Ok(Options {
            geometry: self.geometry,
            mode: self.mode,
            corners: self
                .corners
                .as_deref()
                .map(docscan::scan::parse_corners)
                .transpose()?,
            max_side: self.max_side,
        })
    }
}

#[derive(Subcommand)]
enum Cmd {
    /// Convert UVDoc's checkpoint (best_model.pkl) to one GGUF
    Convert {
        checkpoint: PathBuf,
        #[arg(short, long)]
        output: PathBuf,
        /// Convolution weight dtype: f32 (default) or f16
        #[arg(long, default_value = "f32")]
        dtype: String,
        /// Model name stored in the file (default: output file stem)
        #[arg(long)]
        name: Option<String>,
    },
    /// Print what the model finds in photos (corners, score), as JSON lines
    Detect {
        #[command(flatten)]
        run: RunOpts,
        #[arg(required = true)]
        inputs: Vec<PathBuf>,
    },
    /// Turn photos into scans
    Scan {
        #[command(flatten)]
        run: RunOpts,
        #[command(flatten)]
        scan: ScanOpts,
        #[arg(required = true)]
        inputs: Vec<PathBuf>,
        /// Output file (one input) or directory (<name>_scan.<ext>)
        #[arg(short, long, default_value = ".")]
        output: PathBuf,
    },
    /// Serve the HTTP API (POST /v1/documents/scans, /v1/documents/detect)
    Serve {
        #[command(flatten)]
        run: RunOpts,
        #[arg(long, default_value = "127.0.0.1")]
        host: String,
        #[arg(long, default_value_t = 8791)]
        port: u16,
        /// Model id reported by /v1/models (default: the GGUF's name)
        #[arg(long)]
        model_id: Option<String>,
    },
}

fn open(run: &RunOpts) -> Result<(Scanner, String)> {
    if let Some(n) = run.threads {
        // candle's CPU kernels and ours use rayon's global pool, which reads
        // this once, at its first use (later than here)
        unsafe { std::env::set_var("RAYON_NUM_THREADS", n.to_string()) };
    }
    let device = docscan::pick_device(run.cpu)?;
    let t = Instant::now();
    let (tensors, name) = docscan::gguf::load(&run.model)?;
    let model = docscan::uvdoc::UvDoc::load(&tensors, &device, candle_core::DType::F32)?;
    let name = if name.is_empty() {
        run.model
            .file_stem()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_else(|| "docscan".into())
    } else {
        name
    };
    let dev = if device.is_cuda() {
        "cuda"
    } else if device.is_metal() {
        "metal"
    } else {
        "cpu"
    };
    eprintln!("loaded {name} ({dev}) in {:.2}s", t.elapsed().as_secs_f64());
    Ok((Scanner::new(model), name))
}

fn read_photo(p: &Path) -> Result<docscan::image::Image> {
    let bytes = std::fs::read(p).with_context(|| format!("reading {p:?}"))?;
    docscan::image::decode(&bytes).with_context(|| format!("decoding {p:?}"))
}

fn scan_files(run: &RunOpts, so: &ScanOpts, inputs: &[PathBuf], output: &Path) -> Result<()> {
    let (scanner, _) = open(run)?;
    let opts = so.options()?;
    let to_file = inputs.len() == 1 && output.extension().is_some() && !output.is_dir();
    let format = so.format.unwrap_or_else(|| {
        let ext = output.extension().map(|e| e.to_string_lossy().to_string());
        match (to_file, ext) {
            (true, Some(e)) => e.parse().unwrap_or_default(),
            _ => Format::Png,
        }
    });
    if !to_file {
        std::fs::create_dir_all(output)?;
    }
    for input in inputs {
        let t = Instant::now();
        let photo = read_photo(input)?;
        let s = scanner.scan(&photo, &opts)?;
        let bytes = docscan::encode(&s.image, format, so.quality, so.dpi, so.paper)?;
        let path = if to_file {
            output.to_path_buf()
        } else {
            let base = input
                .file_stem()
                .map(|s| s.to_string_lossy().into_owned())
                .unwrap_or_else(|| "scan".into());
            output.join(format!("{base}_scan.{}", format.ext()))
        };
        std::fs::write(&path, bytes)?;
        let det = s
            .detection
            .as_ref()
            .map(|d| format!(", score {:.2}", d.score))
            .unwrap_or_default();
        eprintln!(
            "{} -> {} ({}x{}, {:?}{det}) in {:.2}s",
            input.display(),
            path.display(),
            s.image.width,
            s.image.height,
            s.geometry,
            t.elapsed().as_secs_f64()
        );
        for w in s.detection.iter().flat_map(|d| &d.warnings) {
            eprintln!("  warning: {w}");
        }
    }
    Ok(())
}

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| "info".into()),
        )
        .with_writer(std::io::stderr)
        .init();
    match Cli::parse().cmd {
        Cmd::Convert {
            checkpoint,
            output,
            dtype,
            name,
        } => {
            let dt = docscan::gguf::parse_dtype(&dtype)?;
            let name = name.unwrap_or_else(|| {
                output
                    .file_stem()
                    .map(|s| s.to_string_lossy().into_owned())
                    .unwrap_or_default()
            });
            let counts = docscan::gguf::convert(&checkpoint, &output, dt, &name)?;
            let size = std::fs::metadata(&output)?.len() as f64 / 1e6;
            eprintln!(
                "wrote {} ({size:.1} MB): {}",
                output.display(),
                counts
                    .iter()
                    .map(|(d, n)| format!("{n} {d}"))
                    .collect::<Vec<_>>()
                    .join(", ")
            );
        }
        Cmd::Detect { run, inputs } => {
            let (scanner, _) = open(&run)?;
            for input in &inputs {
                let photo = read_photo(input)?;
                let t = Instant::now();
                let d = scanner.detect(&photo)?;
                let mut v = serde_json::to_value(&d)?;
                v["file"] = input.display().to_string().into();
                v["ms"] = (t.elapsed().as_secs_f64() * 1e3).round().into();
                println!("{v}");
            }
        }
        Cmd::Scan {
            run,
            scan,
            inputs,
            output,
        } => scan_files(&run, &scan, &inputs, &output)?,
        Cmd::Serve {
            run,
            host,
            port,
            model_id,
        } => {
            let (scanner, name) = open(&run)?;
            let rt = tokio::runtime::Runtime::new()?;
            rt.block_on(server::serve(
                scanner,
                model_id.unwrap_or(name),
                &host,
                port,
            ))?;
        }
    }
    Ok(())
}
