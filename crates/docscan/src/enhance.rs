//! Scan look, without any learned model: illumination flattening (shadows,
//! uneven light, colour cast), levels, gray, black and white (Sauvola).
//!
//! Each output pixel is a monotonic function of the same input pixel and a
//! smooth background estimate: strokes, digits, stamps and signatures keep
//! their shape; nothing is drawn or inpainted.

use crate::image::Image;
use crate::warp::resize_bilinear;
use rayon::prelude::*;
use std::str::FromStr;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum Mode {
    /// Geometry only: the pixels as photographed.
    Original,
    /// Flattened illumination, white paper, colours kept.
    #[default]
    Color,
    Gray,
    /// Black and white (adaptive threshold).
    Bw,
}

impl FromStr for Mode {
    type Err = anyhow::Error;
    fn from_str(s: &str) -> anyhow::Result<Self> {
        Ok(match s.trim().to_ascii_lowercase().as_str() {
            "original" | "none" => Mode::Original,
            "color" | "colour" => Mode::Color,
            "gray" | "grey" | "grayscale" => Mode::Gray,
            "bw" | "binary" | "mono" => Mode::Bw,
            o => anyhow::bail!("unknown mode '{o}' (color, gray, bw, original)"),
        })
    }
}

/// Side of the working image the background is estimated on.
const BG_SIDE: usize = 512;
/// Normalised values at or below this become black, at or above `WHITE` white.
const BLACK_PERCENTILE: f32 = 0.005;
const WHITE: f32 = 0.94;

pub fn apply(img: &Image, mode: Mode) -> Image {
    match mode {
        Mode::Original => img.clone(),
        Mode::Color => levels(&flatten(img)),
        Mode::Gray => levels(&flatten(&img.to_gray())),
        Mode::Bw => sauvola(&levels(&flatten(&img.to_gray())), 0.2),
    }
}

/// Area-average downscale by an integer factor.
fn downscale(img: &Image, f: usize) -> Image {
    let (w, h, c) = (img.width.div_ceil(f), img.height.div_ceil(f), img.channels);
    let mut out = Image::new(w, h, c);
    out.data
        .par_chunks_mut(w * c)
        .enumerate()
        .for_each(|(y, row)| {
            for x in 0..w {
                let mut acc = [0.0f32; 3];
                let mut n = 0.0;
                for sy in y * f..((y + 1) * f).min(img.height) {
                    for sx in x * f..((x + 1) * f).min(img.width) {
                        for (ch, a) in acc.iter_mut().enumerate().take(c) {
                            *a += img.at(sx, sy, ch);
                        }
                        n += 1.0;
                    }
                }
                for ch in 0..c {
                    row[x * c + ch] = acc[ch] / n;
                }
            }
        });
    out
}

#[derive(Clone, Copy, PartialEq)]
enum Op {
    Max,
    Min,
    Mean,
}

/// Separable sliding filter of radius `r`, clamped at the edges.
fn filter(img: &Image, r: usize, op: Op) -> Image {
    let pass = |src: &Image, horizontal: bool| -> Image {
        let (w, h, c) = (src.width, src.height, src.channels);
        let mut out = Image::new(w, h, c);
        out.data
            .par_chunks_mut(w * c)
            .enumerate()
            .for_each(|(y, row)| {
                for x in 0..w {
                    for ch in 0..c {
                        let mut acc = match op {
                            Op::Max => f32::MIN,
                            Op::Min => f32::MAX,
                            Op::Mean => 0.0,
                        };
                        let mut n = 0.0;
                        for d in -(r as i64)..=(r as i64) {
                            let (sx, sy) = if horizontal {
                                ((x as i64 + d).clamp(0, w as i64 - 1) as usize, y)
                            } else {
                                (x, (y as i64 + d).clamp(0, h as i64 - 1) as usize)
                            };
                            let v = src.at(sx, sy, ch);
                            match op {
                                Op::Max => acc = acc.max(v),
                                Op::Min => acc = acc.min(v),
                                Op::Mean => {
                                    acc += v;
                                    n += 1.0;
                                }
                            }
                        }
                        row[x * c + ch] = if op == Op::Mean { acc / n } else { acc };
                    }
                }
            });
        out
    };
    pass(&pass(img, true), false)
}

/// Divides each channel by its paper background: shadows and light
/// gradients flatten to white, the paper's colour cast goes with them.
///
/// The background is a morphological closing (max then min filter, radius
/// wider than any stroke): ink disappears, while a shadow's edge stays where
/// it is. A blurred background would spread that edge into a band of gray
/// (black after thresholding) across the text next to it.
pub fn flatten(img: &Image) -> Image {
    let f = img.width.max(img.height).div_ceil(BG_SIDE).max(1);
    let small = downscale(img, f);
    let side = small.width.max(small.height);
    let r = (side / 80).max(3);
    let bg = filter(&filter(&small, r, Op::Max), r, Op::Min);
    // a light smoothing of the closing's flat blocks
    let bg = filter(&bg, 1, Op::Mean);
    let bg = resize_bilinear(&bg, img.width, img.height);
    let data = img
        .data
        .par_iter()
        .zip(bg.data.par_iter())
        .map(|(&v, &b)| (v / b.max(0.04)).min(1.0))
        .collect();
    Image {
        data,
        ..img.clone()
    }
}

/// Stretches luminance so the darkest ink is black and the paper white.
fn levels(img: &Image) -> Image {
    let gray = img.to_gray();
    let mut hist = [0usize; 256];
    for v in &gray.data {
        hist[(v.clamp(0.0, 1.0) * 255.0) as usize] += 1;
    }
    let target = (gray.data.len() as f32 * BLACK_PERCENTILE) as usize;
    let mut acc = 0;
    let mut black = 0.0;
    for (i, n) in hist.iter().enumerate() {
        acc += n;
        if acc > target {
            black = i as f32 / 255.0;
            break;
        }
    }
    // never stretch a light page into fake contrast
    let black = black.min(0.45);
    let span = (WHITE - black).max(0.2);
    let data = img
        .data
        .par_iter()
        .map(|&v| ((v - black) / span).clamp(0.0, 1.0))
        .collect();
    Image {
        data,
        ..img.clone()
    }
}

/// Sauvola threshold, `k` usually 0.2; window about 1/40 of the short side.
pub fn sauvola(gray: &Image, k: f32) -> Image {
    let (w, h) = (gray.width, gray.height);
    let win = (w.min(h) / 40).max(15) | 1;
    let r = win / 2;
    // integral images of v and v², (w+1)×(h+1)
    let stride = w + 1;
    let mut s = vec![0f64; stride * (h + 1)];
    let mut s2 = vec![0f64; stride * (h + 1)];
    for y in 0..h {
        let (mut row, mut row2) = (0f64, 0f64);
        for x in 0..w {
            let v = gray.data[y * w + x] as f64;
            row += v;
            row2 += v * v;
            s[(y + 1) * stride + x + 1] = s[y * stride + x + 1] + row;
            s2[(y + 1) * stride + x + 1] = s2[y * stride + x + 1] + row2;
        }
    }
    let mut out = Image::new(w, h, 1);
    out.data
        .par_chunks_mut(w)
        .enumerate()
        .for_each(|(y, line)| {
            let (y0, y1) = (y.saturating_sub(r), (y + r + 1).min(h));
            for (x, o) in line.iter_mut().enumerate() {
                let (x0, x1) = (x.saturating_sub(r), (x + r + 1).min(w));
                let n = ((y1 - y0) * (x1 - x0)) as f64;
                let rect = |t: &[f64]| {
                    t[y1 * stride + x1] - t[y0 * stride + x1] - t[y1 * stride + x0]
                        + t[y0 * stride + x0]
                };
                let mean = rect(&s) / n;
                let var = (rect(&s2) / n - mean * mean).max(0.0);
                let t = mean * (1.0 + k as f64 * (var.sqrt() / 0.5 - 1.0));
                *o = if (gray.data[y * w + x] as f64) > t {
                    1.0
                } else {
                    0.0
                };
            }
        });
    out
}
