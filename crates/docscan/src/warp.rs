//! Resampling: the network's input resize, the UVDoc unwarp (grid upsample +
//! `grid_sample`), and the 4-point perspective. Every output pixel is a
//! bilinear mix of input pixels: nothing is synthesised.

use crate::image::Image;
use rayon::prelude::*;

/// Resize with PyTorch `interpolate(mode="bilinear", align_corners=False)`
/// semantics (also OpenCV `INTER_LINEAR` on float images): half-pixel
/// centres, no antialiasing.
pub fn resize_bilinear(src: &Image, width: usize, height: usize) -> Image {
    let c = src.channels;
    let sx = src.width as f32 / width as f32;
    let sy = src.height as f32 / height as f32;
    let taps = |out: usize, scale: f32, n: usize| -> Vec<(usize, usize, f32)> {
        (0..out)
            .map(|i| {
                let r = ((i as f32 + 0.5) * scale - 0.5).max(0.0);
                let i0 = (r as usize).min(n - 1);
                let i1 = if i0 < n - 1 { i0 + 1 } else { i0 };
                (i0, i1, r - i0 as f32)
            })
            .collect()
    };
    let xs = taps(width, sx, src.width);
    let ys = taps(height, sy, src.height);
    let mut data = vec![0.0f32; width * height * c];
    data.par_chunks_mut(width * c)
        .zip(ys.par_iter())
        .for_each(|(row, &(y0, y1, ly))| {
            for (x, &(x0, x1, lx)) in xs.iter().enumerate() {
                for ch in 0..c {
                    let top = src.at(x0, y0, ch) * (1.0 - lx) + src.at(x1, y0, ch) * lx;
                    let bot = src.at(x0, y1, ch) * (1.0 - lx) + src.at(x1, y1, ch) * lx;
                    row[x * c + ch] = top * (1.0 - ly) + bot * ly;
                }
            }
        });
    Image {
        width,
        height,
        channels: c,
        data,
    }
}

/// A UVDoc grid: `gh` rows × `gw` columns of source positions in
/// `grid_sample` coordinates (-1..1, align_corners=True).
#[derive(Clone, Debug)]
pub struct Grid {
    pub gw: usize,
    pub gh: usize,
    /// x then y per point, row-major.
    pub points: Vec<[f32; 2]>,
}

impl Grid {
    pub fn at(&self, i: usize, j: usize) -> [f32; 2] {
        self.points[i * self.gw + j]
    }

    /// The grid in source pixels (align_corners=True).
    pub fn to_pixels(&self, width: usize, height: usize) -> Vec<[f32; 2]> {
        let (w, h) = ((width - 1) as f32, (height - 1) as f32);
        self.points
            .iter()
            .map(|p| [(p[0] + 1.0) * 0.5 * w, (p[1] + 1.0) * 0.5 * h])
            .collect()
    }
}

/// Bilinear sample with zeros outside, `grid_sample(align_corners=True)`.
#[inline]
fn sample(src: &Image, px: f32, py: f32, out: &mut [f32]) {
    let x0 = px.floor();
    let y0 = py.floor();
    let (lx, ly) = (px - x0, py - y0);
    let (x0, y0) = (x0 as i64, y0 as i64);
    let (w, h) = (src.width as i64, src.height as i64);
    out.fill(0.0);
    for (dy, wy) in [(0, 1.0 - ly), (1, ly)] {
        for (dx, wx) in [(0, 1.0 - lx), (1, lx)] {
            let (x, y) = (x0 + dx, y0 + dy);
            if x < 0 || y < 0 || x >= w || y >= h {
                continue;
            }
            let wgt = wx * wy;
            let base = (y as usize * src.width + x as usize) * src.channels;
            for (c, o) in out.iter_mut().enumerate() {
                *o += wgt * src.data[base + c];
            }
        }
    }
}

/// UVDoc's `bilinear_unwarping`: upsample the grid to `width`×`height`
/// (bilinear, align_corners=True), then `grid_sample` the source.
pub fn unwarp(src: &Image, grid: &Grid, width: usize, height: usize) -> Image {
    let c = src.channels;
    let (gw, gh) = (grid.gw, grid.gh);
    let axis = |out: usize, n: usize| -> Vec<(usize, usize, f32)> {
        (0..out)
            .map(|i| {
                let r = if out > 1 {
                    i as f32 * (n - 1) as f32 / (out - 1) as f32
                } else {
                    0.0
                };
                let i0 = (r as usize).min(n - 1);
                let i1 = (i0 + 1).min(n - 1);
                (i0, i1, r - i0 as f32)
            })
            .collect()
    };
    let xs = axis(width, gw);
    let ys = axis(height, gh);
    let sw = (src.width - 1) as f32 * 0.5;
    let sh = (src.height - 1) as f32 * 0.5;
    let mut data = vec![0.0f32; width * height * c];
    data.par_chunks_mut(width * c)
        .zip(ys.par_iter())
        .for_each(|(row, &(i0, i1, ly))| {
            for (x, &(j0, j1, lx)) in xs.iter().enumerate() {
                let mut g = [0.0f32; 2];
                for (k, gk) in g.iter_mut().enumerate() {
                    let top = grid.at(i0, j0)[k] * (1.0 - lx) + grid.at(i0, j1)[k] * lx;
                    let bot = grid.at(i1, j0)[k] * (1.0 - lx) + grid.at(i1, j1)[k] * lx;
                    *gk = top * (1.0 - ly) + bot * ly;
                }
                sample(
                    src,
                    (g[0] + 1.0) * sw,
                    (g[1] + 1.0) * sh,
                    &mut row[x * c..(x + 1) * c],
                );
            }
        });
    Image {
        width,
        height,
        channels: c,
        data,
    }
}

/// Corners in source pixels: top-left, top-right, bottom-right, bottom-left.
pub type Quad = [[f32; 2]; 4];

/// Homography mapping the unit square (0,0),(1,0),(1,1),(0,1) to `q`
/// (Heckbert's closed form), as a row-major 3×3.
pub fn square_to_quad(q: &Quad) -> [f64; 9] {
    let [[x0, y0], [x1, y1], [x2, y2], [x3, y3]] = q.map(|p| [p[0] as f64, p[1] as f64]);
    let (sx, sy) = (x0 - x1 + x2 - x3, y0 - y1 + y2 - y3);
    if sx.abs() < 1e-12 && sy.abs() < 1e-12 {
        // affine
        return [x1 - x0, x3 - x0, x0, y1 - y0, y3 - y0, y0, 0.0, 0.0, 1.0];
    }
    let (dx1, dx2, dy1, dy2) = (x1 - x2, x3 - x2, y1 - y2, y3 - y2);
    let den = dx1 * dy2 - dx2 * dy1;
    let g = (sx * dy2 - dx2 * sy) / den;
    let h = (dx1 * sy - sx * dy1) / den;
    [
        x1 - x0 + g * x1,
        x3 - x0 + h * x3,
        x0,
        y1 - y0 + g * y1,
        y3 - y0 + h * y3,
        y0,
        g,
        h,
        1.0,
    ]
}

/// Output size that keeps the source resolution of a quad: the longer of
/// each pair of opposite edges.
pub fn quad_size(q: &Quad) -> (usize, usize) {
    let d = |a: [f32; 2], b: [f32; 2]| ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2)).sqrt();
    let w = d(q[0], q[1]).max(d(q[3], q[2]));
    let h = d(q[0], q[3]).max(d(q[1], q[2]));
    ((w.round() as usize).max(1), (h.round() as usize).max(1))
}

/// Rectifies the quad `q` of `src` to a `width`×`height` image (pixel
/// centres: output (0,0) samples `q[0]`, the last pixel samples `q[2]`).
pub fn perspective(src: &Image, q: &Quad, width: usize, height: usize) -> Image {
    let m = square_to_quad(q);
    let c = src.channels;
    let fx = 1.0 / (width.max(2) - 1) as f64;
    let fy = 1.0 / (height.max(2) - 1) as f64;
    let mut data = vec![0.0f32; width * height * c];
    data.par_chunks_mut(width * c)
        .enumerate()
        .for_each(|(y, row)| {
            let v = y as f64 * fy;
            for x in 0..width {
                let u = x as f64 * fx;
                let w = m[6] * u + m[7] * v + m[8];
                let px = (m[0] * u + m[1] * v + m[2]) / w;
                let py = (m[3] * u + m[4] * v + m[5]) / w;
                sample(src, px as f32, py as f32, &mut row[x * c..(x + 1) * c]);
            }
        });
    Image {
        width,
        height,
        channels: c,
        data,
    }
}
