//! Document photo → scan, in Rust/Candle: UVDoc dewarping (one GGUF),
//! illumination flattening, gray / black-and-white, PNG / JPEG / PDF.
//!
//! Conservative by design: geometry is resampling, enhancement a per-pixel
//! tone curve; no generative model touches the pixels, so digits,
//! signatures and stamps are never redrawn.
//!
//! ```no_run
//! # fn main() -> anyhow::Result<()> {
//! let scanner = docscan::Scanner::open("models/uvdoc-f32.gguf".as_ref(), &candle_core::Device::Cpu)?;
//! let photo = docscan::image::decode(&std::fs::read("photo.jpg")?)?;
//! let scan = scanner.scan(&photo, &docscan::Options::default())?;
//! let png = docscan::encode(&scan.image, docscan::Format::Png, 90, 300, docscan::Paper::Auto)?;
//! std::fs::write("scan.png", png)?;
//! # Ok(()) }
//! ```

pub mod enhance;
pub mod gguf;
pub mod image;
pub mod pdf;
pub mod scan;
pub mod uvdoc;
pub mod warp;

pub use enhance::Mode;
pub use pdf::Paper;
pub use scan::{Detection, Format, Geometry, Options, Scan, Scanner, encode};

use anyhow::Result;
use candle_core::{DType, Device};
use std::path::Path;

impl Scanner {
    /// Opens a docscan GGUF (computes in F32: the model is small and the
    /// grid's precision is the output's geometry).
    pub fn open(path: &Path, device: &Device) -> Result<Self> {
        let (t, _) = gguf::load(path)?;
        Ok(Self::new(uvdoc::UvDoc::load(&t, device, DType::F32)?))
    }
}

/// The device for `cpu` = false: CUDA (or Metal) when built in, else the CPU.
pub fn pick_device(cpu: bool) -> Result<Device> {
    if cpu {
        return Ok(Device::Cpu);
    }
    #[cfg(feature = "cuda")]
    {
        return Ok(Device::new_cuda(0)?);
    }
    #[cfg(all(feature = "metal", not(feature = "cuda")))]
    {
        return Ok(Device::new_metal(0)?);
    }
    #[allow(unreachable_code)]
    Ok(Device::Cpu)
}
