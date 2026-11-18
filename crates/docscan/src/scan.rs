//! Photo → scan: detection (UVDoc grid), geometry (dewarp, perspective or
//! none), enhancement, encoding.

use crate::enhance::{self, Mode};
use crate::image::Image;
use crate::pdf::{self, Paper};
use crate::uvdoc::{INPUT_H, INPUT_W, UvDoc};
use crate::warp::{self, Grid, Quad};
use anyhow::{Result, bail};
use serde::Serialize;
use std::str::FromStr;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum Geometry {
    /// UVDoc grid: perspective, curl and folds.
    #[default]
    Dewarp,
    /// The grid's four corners only, one homography: straight lines stay
    /// exactly straight (the most conservative correction).
    Perspective,
    /// No geometric change.
    None,
}

impl FromStr for Geometry {
    type Err = anyhow::Error;
    fn from_str(s: &str) -> Result<Self> {
        Ok(match s.trim().to_ascii_lowercase().as_str() {
            "dewarp" | "auto" | "" => Geometry::Dewarp,
            "perspective" | "flat" => Geometry::Perspective,
            "none" | "off" => Geometry::None,
            o => bail!("unknown geometry '{o}' (dewarp, perspective, none)"),
        })
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum Format {
    #[default]
    Png,
    Jpeg,
    Pdf,
}

impl FromStr for Format {
    type Err = anyhow::Error;
    fn from_str(s: &str) -> Result<Self> {
        Ok(match s.trim().to_ascii_lowercase().as_str() {
            "png" | "" => Format::Png,
            "jpeg" | "jpg" => Format::Jpeg,
            "pdf" => Format::Pdf,
            o => bail!("unknown format '{o}' (png, jpeg, pdf)"),
        })
    }
}

impl Format {
    pub fn mime(self) -> &'static str {
        match self {
            Format::Png => "image/png",
            Format::Jpeg => "image/jpeg",
            Format::Pdf => "application/pdf",
        }
    }
    pub fn ext(self) -> &'static str {
        match self {
            Format::Png => "png",
            Format::Jpeg => "jpg",
            Format::Pdf => "pdf",
        }
    }
}

#[derive(Clone, Debug)]
pub struct Options {
    pub geometry: Geometry,
    pub mode: Mode,
    /// Corners chosen by the user (source pixels, after EXIF rotation):
    /// perspective from these, no model call.
    pub corners: Option<Quad>,
    /// Longest output side, pixels (the source resolution is kept below it).
    pub max_side: usize,
}

impl Default for Options {
    fn default() -> Self {
        Self {
            geometry: Geometry::Dewarp,
            mode: Mode::Color,
            corners: None,
            max_side: 8000,
        }
    }
}

/// What the model found in a photo.
#[derive(Clone, Debug, Serialize)]
pub struct Detection {
    /// Top-left, top-right, bottom-right, bottom-left, source pixels.
    pub corners: Quad,
    /// 0..1: grid inside the photo and without folds; below 0.9 the
    /// geometry is doubtful (check the result against the photo).
    pub score: f32,
    /// Share of the photo the document covers.
    pub coverage: f32,
    /// Size that keeps the document's source resolution (width, height).
    pub size: (usize, usize),
    pub warnings: Vec<String>,
    #[serde(skip)]
    pub grid: Grid,
}

pub struct Scan {
    pub image: Image,
    pub detection: Option<Detection>,
    pub geometry: Geometry,
}

fn polyline(points: impl Iterator<Item = [f32; 2]>) -> f32 {
    let mut len = 0.0;
    let mut prev: Option<[f32; 2]> = None;
    for p in points {
        if let Some(q) = prev {
            len += ((p[0] - q[0]).powi(2) + (p[1] - q[1]).powi(2)).sqrt();
        }
        prev = Some(p);
    }
    len
}

fn quad_area(q: &Quad) -> f32 {
    let mut a = 0.0;
    for i in 0..4 {
        let (p, n) = (q[i], q[(i + 1) % 4]);
        a += p[0] * n[1] - n[0] * p[1];
    }
    (a * 0.5).abs()
}

/// Scores a grid and derives corners and output size (see [`Detection`]).
pub fn analyse(grid: Grid, width: usize, height: usize) -> Detection {
    let px = grid.to_pixels(width, height);
    let (gw, gh) = (grid.gw, grid.gh);
    let at = |i: usize, j: usize| px[i * gw + j];
    let inside = grid
        .points
        .iter()
        .filter(|p| p[0].abs() <= 1.02 && p[1].abs() <= 1.02)
        .count() as f32
        / grid.points.len() as f32;
    // cells keeping the orientation of the page (no fold-over, no collapse)
    let mut good = 0usize;
    let cells = (gh - 1) * (gw - 1);
    for i in 0..gh - 1 {
        for j in 0..gw - 1 {
            let (a, b, d) = (at(i, j), at(i, j + 1), at(i + 1, j));
            let cross = (b[0] - a[0]) * (d[1] - a[1]) - (b[1] - a[1]) * (d[0] - a[0]);
            if cross > 1e-3 {
                good += 1;
            }
        }
    }
    let folds = good as f32 / cells as f32;
    let corners: Quad = [at(0, 0), at(0, gw - 1), at(gh - 1, gw - 1), at(gh - 1, 0)];
    let coverage = (quad_area(&corners) / (width * height) as f32).min(1.0);
    let w = (0..gh)
        .map(|i| polyline((0..gw).map(|j| at(i, j))))
        .sum::<f32>()
        / gh as f32;
    let h = (0..gw)
        .map(|j| polyline((0..gh).map(|i| at(i, j))))
        .sum::<f32>()
        / gw as f32;
    let score = inside * folds;
    let mut warnings = Vec::new();
    if score < 0.9 {
        warnings.push(format!(
            "geometry uncertain (score {score:.2}): compare the result with the photo"
        ));
    }
    if coverage < 0.15 {
        warnings.push(format!(
            "document covers {:.0}% of the photo: low resolution, move closer",
            coverage * 100.0
        ));
    }
    Detection {
        corners,
        score,
        coverage,
        size: ((w.round() as usize).max(1), (h.round() as usize).max(1)),
        warnings,
        grid,
    }
}

fn cap(size: (usize, usize), max_side: usize) -> (usize, usize) {
    let m = size.0.max(size.1);
    if m <= max_side || max_side == 0 {
        return size;
    }
    let s = max_side as f64 / m as f64;
    (
        ((size.0 as f64 * s).round() as usize).max(1),
        ((size.1 as f64 * s).round() as usize).max(1),
    )
}

/// Puts four points in order: top-left, top-right, bottom-right, bottom-left.
pub fn order_corners(mut q: Quad) -> Quad {
    let cx = q.iter().map(|p| p[0]).sum::<f32>() / 4.0;
    let cy = q.iter().map(|p| p[1]).sum::<f32>() / 4.0;
    // clockwise on screen (y down) starting from the left
    q.sort_by(|a, b| {
        let ta = (a[1] - cy).atan2(a[0] - cx);
        let tb = (b[1] - cy).atan2(b[0] - cx);
        ta.total_cmp(&tb)
    });
    let tl = (0..4)
        .min_by(|&i, &j| (q[i][0] + q[i][1]).total_cmp(&(q[j][0] + q[j][1])))
        .unwrap();
    q.rotate_left(tl);
    q
}

/// Parses corners: JSON `[[x,y],[x,y],[x,y],[x,y]]` or eight numbers
/// separated by commas or spaces. Any order.
pub fn parse_corners(s: &str) -> Result<Quad> {
    let nums: Vec<f32> = if let Ok(v) = serde_json::from_str::<Vec<[f32; 2]>>(s) {
        v.into_iter().flatten().collect()
    } else {
        s.split(|c: char| c == ',' || c.is_whitespace() || c == ';')
            .filter(|t| !t.is_empty())
            .map(|t| t.parse::<f32>())
            .collect::<Result<_, _>>()
            .map_err(|_| anyhow::anyhow!("corners: expected 8 numbers or [[x,y],…]"))?
    };
    if nums.len() != 8 || nums.iter().any(|v| !v.is_finite()) {
        bail!("corners: expected 4 points (8 numbers), got {}", nums.len());
    }
    let q = [
        [nums[0], nums[1]],
        [nums[2], nums[3]],
        [nums[4], nums[5]],
        [nums[6], nums[7]],
    ];
    Ok(order_corners(q))
}

pub struct Scanner {
    pub model: UvDoc,
}

impl Scanner {
    pub fn new(model: UvDoc) -> Self {
        Self { model }
    }

    /// Runs the model on a photo.
    pub fn detect(&self, img: &Image) -> Result<Detection> {
        let small = warp::resize_bilinear(img, INPUT_W, INPUT_H);
        let grid = self.model.predict(&small)?;
        Ok(analyse(grid, img.width, img.height))
    }

    pub fn scan(&self, img: &Image, opts: &Options) -> Result<Scan> {
        if img.width < 16 || img.height < 16 {
            bail!("image too small ({}x{})", img.width, img.height);
        }
        let (geometry, detection, out) = if let Some(q) = opts.corners {
            let size = cap(warp::quad_size(&q), opts.max_side);
            (
                Geometry::Perspective,
                None,
                warp::perspective(img, &q, size.0, size.1),
            )
        } else {
            match opts.geometry {
                Geometry::None => (Geometry::None, None, img.clone()),
                g => {
                    let d = self.detect(img)?;
                    let out = if g == Geometry::Dewarp {
                        let size = cap(d.size, opts.max_side);
                        warp::unwarp(img, &d.grid, size.0, size.1)
                    } else {
                        let size = cap(warp::quad_size(&d.corners), opts.max_side);
                        warp::perspective(img, &d.corners, size.0, size.1)
                    };
                    (g, Some(d), out)
                }
            }
        };
        Ok(Scan {
            image: enhance::apply(&out, opts.mode),
            detection,
            geometry,
        })
    }
}

/// Encodes a scan. JPEG for a black-and-white scan is refused (artifacts
/// around every stroke): PNG or PDF.
pub fn encode(img: &Image, format: Format, quality: u8, dpi: u16, paper: Paper) -> Result<Vec<u8>> {
    let binary = img.channels == 1 && img.data.iter().all(|&v| v == 0.0 || v == 1.0);
    match format {
        Format::Png => img.encode_png(),
        Format::Jpeg if binary => {
            bail!("a black-and-white scan is not stored as JPEG: use png or pdf")
        }
        Format::Jpeg => img.encode_jpeg(quality, dpi),
        Format::Pdf => {
            let jpeg = if binary {
                None
            } else {
                Some(img.encode_jpeg(quality, dpi)?)
            };
            pdf::write(img, jpeg.as_deref(), dpi.max(1) as f64, paper)
        }
    }
}
