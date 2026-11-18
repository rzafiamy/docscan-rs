//! Synthetic "phone photo" of a flat scan, for trying the pipeline without
//! a camera: perspective, page curl, a shadow, uneven light, dark desk.
//!
//!     cargo run --release --example make_photo -- page.jpg tmp/photo.jpg

use anyhow::{Context, Result};
use docscan::image::{Image, decode};
use docscan::warp::square_to_quad;

fn invert(m: &[f64; 9]) -> [f64; 9] {
    let [a, b, c, d, e, f, g, h, i] = *m;
    let det = a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g);
    [
        (e * i - f * h) / det,
        (c * h - b * i) / det,
        (b * f - c * e) / det,
        (f * g - d * i) / det,
        (a * i - c * g) / det,
        (c * d - a * f) / det,
        (d * h - e * g) / det,
        (b * g - a * h) / det,
        (a * e - b * d) / det,
    ]
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let page = decode(&std::fs::read(
        args.get(1).context("usage: make_photo <page> <out.jpg>")?,
    )?)?;
    let out = args.get(2).context("usage: make_photo <page> <out.jpg>")?;
    let (w, h) = (2400usize, 3200usize);
    let quad = [
        [420.0, 330.0],
        [2010.0, 450.0],
        [2180.0, 2930.0],
        [260.0, 2830.0],
    ];
    let inv = invert(&square_to_quad(&quad));
    let mut img = Image::new(w, h, 3);
    let mut seed = 0x2545F491u32;
    for y in 0..h {
        for x in 0..w {
            let (xf, yf) = (x as f64, y as f64);
            let s = inv[6] * xf + inv[7] * yf + inv[8];
            let u = (inv[0] * xf + inv[1] * yf + inv[2]) / s;
            let mut v = (inv[3] * xf + inv[4] * yf + inv[5]) / s;
            // page curl: lines bow downwards in the middle, more at the bottom
            v -= 0.018 * (std::f64::consts::PI * u).sin() * (0.4 + v);
            seed ^= seed << 13;
            seed ^= seed >> 17;
            seed ^= seed << 5;
            let noise = (seed % 1000) as f32 / 1000.0 * 0.04;
            let px = &mut img.data[(y * w + x) * 3..(y * w + x) * 3 + 3];
            if (0.0..=1.0).contains(&u) && (0.0..=1.0).contains(&v) {
                let sx = (u * (page.width - 1) as f64).round() as usize;
                let sy = (v * (page.height - 1) as f64).round() as usize;
                // light falls off to the right; a hand's shadow at the bottom left
                let light = 1.0 - 0.35 * u as f32;
                let shadow = if ((u - 0.15).powi(2) + (v - 0.85).powi(2)).sqrt() < 0.22 {
                    0.55
                } else {
                    1.0
                };
                let tint = [1.0, 0.96, 0.85]; // warm lamp
                for (c, (o, t)) in px.iter_mut().zip(tint).enumerate() {
                    *o = (page.at(sx, sy, c) * light * shadow * t * 0.92 + noise).min(1.0);
                }
            } else {
                for (c, o) in px.iter_mut().enumerate() {
                    *o = 0.18 + 0.06 * c as f32 + noise * 2.0;
                }
            }
        }
    }
    std::fs::write(out, img.encode_jpeg(88, 72)?)?;
    eprintln!("wrote {out} ({w}x{h})");
    Ok(())
}
