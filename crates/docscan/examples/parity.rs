//! Dumps docscan's intermediates for `scripts/ref/ref.py`, which recomputes
//! them with UVDoc's PyTorch code and prints the differences.
//!
//!     cargo run --release --example parity -- models/uvdoc-f32.gguf photo.jpg tmp/rs.safetensors [--cpu]
//!     python scripts/ref/ref.py <UVDoc dir> best_model.pkl tmp/rs.safetensors

use anyhow::{Context, Result};
use candle_core::{Device, Tensor};
use docscan::uvdoc::{INPUT_H, INPUT_W};
use docscan::warp;
use std::collections::HashMap;

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let usage = "usage: parity <model.gguf> <photo> <out.safetensors> [--cpu]";
    let model = args.get(1).context(usage)?;
    let photo = args.get(2).context(usage)?;
    let out = args.get(3).context(usage)?;
    let device = docscan::pick_device(args.iter().any(|a| a == "--cpu"))?;
    let scanner = docscan::Scanner::open(model.as_ref(), &device)?;
    let img = docscan::image::decode(&std::fs::read(photo)?)?;
    let small = warp::resize_bilinear(&img, INPUT_W, INPUT_H);
    let grid = scanner.model.predict(&small)?;
    let unwarped = warp::unwarp(&img, &grid, img.width, img.height);
    let t = |d: &[f32], shape: &[usize]| Tensor::from_slice(d, shape, &Device::Cpu);
    let gp: Vec<f32> = grid.points.iter().flatten().copied().collect();
    let mut m = HashMap::new();
    m.insert(
        "image".to_string(),
        t(&img.data, &[img.height, img.width, 3])?,
    );
    m.insert("input".to_string(), t(&small.data, &[INPUT_H, INPUT_W, 3])?);
    m.insert("grid".to_string(), t(&gp, &[grid.gh, grid.gw, 2])?);
    m.insert(
        "unwarped".to_string(),
        t(&unwarped.data, &[img.height, img.width, 3])?,
    );
    candle_core::safetensors::save(&m, out)?;
    eprintln!("wrote {out}");
    Ok(())
}
