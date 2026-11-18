//! No download: a random-weight UVDoc for the plumbing, exact checks for
//! the resampling, enhancement and PDF code.

use candle_core::{DType, Device, Tensor};
use docscan::image::Image;
use docscan::scan::{order_corners, parse_corners};
use docscan::uvdoc::{GRID_H, GRID_W, INPUT_H, INPUT_W, UvDoc, convs};
use docscan::warp::{self, Grid};
use docscan::{Format, Geometry, Mode, Options, Paper, Scanner, encode};
use std::collections::HashMap;

fn random_weights() -> HashMap<String, Tensor> {
    let dev = Device::Cpu;
    let mut m = HashMap::new();
    for (name, s) in convs() {
        let fan_in = (s.in_ch * s.kernel * s.kernel) as f64;
        let w = Tensor::randn(
            0f32,
            (1.0 / fan_in).sqrt() as f32,
            (s.out_ch, s.in_ch, s.kernel, s.kernel),
            &dev,
        )
        .unwrap();
        m.insert(format!("{name}.weight"), w);
        m.insert(
            format!("{name}.bias"),
            Tensor::zeros(s.out_ch, DType::F32, &dev).unwrap(),
        );
    }
    m.insert("out2d.prelu".into(), Tensor::new(&[0.25f32], &dev).unwrap());
    m
}

/// A synthetic page: white paper, dark text bars, light falling off to the right.
fn page(w: usize, h: usize) -> Image {
    let mut img = Image::new(w, h, 3);
    for y in 0..h {
        for x in 0..w {
            let ink = (y / 6) % 4 == 0 && (x / 9) % 5 != 0 && x > 10 && x < w - 10;
            let light = 1.0 - 0.5 * x as f32 / w as f32;
            let v = if ink { 0.1 } else { 0.95 } * light;
            for c in 0..3 {
                img.data[(y * w + x) * 3 + c] = v;
            }
        }
    }
    img
}

fn identity_grid(gw: usize, gh: usize) -> Grid {
    let mut points = Vec::new();
    for i in 0..gh {
        for j in 0..gw {
            points.push([
                -1.0 + 2.0 * j as f32 / (gw - 1) as f32,
                -1.0 + 2.0 * i as f32 / (gh - 1) as f32,
            ]);
        }
    }
    Grid { gw, gh, points }
}

fn max_diff(a: &Image, b: &Image) -> f32 {
    assert_eq!(
        (a.width, a.height, a.channels),
        (b.width, b.height, b.channels)
    );
    a.data
        .iter()
        .zip(&b.data)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f32::max)
}

#[test]
fn identity_warps_return_the_source() {
    let src = page(160, 120);
    assert!(max_diff(&warp::resize_bilinear(&src, 160, 120), &src) < 1e-6);
    let g = identity_grid(GRID_W, GRID_H);
    assert!(max_diff(&warp::unwarp(&src, &g, 160, 120), &src) < 1e-4);
    let q = [[0.0, 0.0], [159.0, 0.0], [159.0, 119.0], [0.0, 119.0]];
    assert_eq!(warp::quad_size(&q), (159, 119));
    assert!(max_diff(&warp::perspective(&src, &q, 160, 120), &src) < 1e-4);
}

#[test]
fn perspective_maps_corners() {
    let mut src = Image::new(100, 80, 1);
    // one bright pixel at each corner of a tilted quad
    let q = [[10.0, 12.0], [85.0, 5.0], [92.0, 70.0], [6.0, 60.0]];
    for p in q {
        src.data[p[1] as usize * 100 + p[0] as usize] = 1.0;
    }
    let out = warp::perspective(&src, &q, 50, 40);
    for (x, y) in [(0, 0), (49, 0), (49, 39), (0, 39)] {
        assert!(out.data[y * 50 + x] > 0.99, "corner ({x},{y})");
    }
}

#[test]
fn corners_are_ordered_from_any_order() {
    let want = [[10.0, 10.0], [200.0, 15.0], [210.0, 300.0], [5.0, 290.0]];
    let shuffled = [want[2], want[0], want[3], want[1]];
    assert_eq!(order_corners(shuffled), want);
    assert_eq!(parse_corners("210,300 10,10 5,290 200,15").unwrap(), want);
    assert_eq!(
        parse_corners("[[5,290],[200,15],[10,10],[210,300]]").unwrap(),
        want
    );
    assert!(parse_corners("1,2,3").is_err());
}

#[test]
fn flattening_evens_the_paper_and_keeps_the_ink() {
    let src = page(400, 300);
    let out = docscan::enhance::apply(&src, Mode::Gray);
    assert_eq!(out.channels, 1);
    // paper left and right: both white now (were 0.95 and ~0.48)
    let (l, r) = (out.data[150 * 400 + 5], out.data[150 * 400 + 395]);
    assert!(l > 0.97 && r > 0.97, "paper {l} {r}");
    // ink stays dark, on both sides
    let ink = |x: usize| out.data[(24 + 2) * 400 + x];
    assert!(
        ink(30) < 0.2 && ink(370) < 0.2,
        "ink {} {}",
        ink(30),
        ink(370)
    );

    let bw = docscan::enhance::apply(&src, Mode::Bw);
    assert!(bw.data.iter().all(|&v| v == 0.0 || v == 1.0));
    assert_eq!(bw.data[150 * 400 + 395], 1.0);
    assert_eq!(bw.data[26 * 400 + 370], 0.0);
    assert_eq!(docscan::enhance::apply(&src, Mode::Original), src);
}

#[test]
fn gguf_round_trip_and_scan() {
    let dir = std::env::temp_dir().join(format!("docscan-test-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("tiny.gguf");
    let w = random_weights();
    docscan::gguf::write(
        &w,
        &path,
        docscan::gguf::parse_dtype("f16").unwrap(),
        "rand",
    )
    .unwrap();
    let (loaded, name) = docscan::gguf::load(&path).unwrap();
    assert_eq!(name, "rand");
    assert_eq!(loaded.len(), w.len());

    let model = UvDoc::load(&loaded, &Device::Cpu, DType::F32).unwrap();
    let x = Tensor::zeros((1, 3, INPUT_H, INPUT_W), DType::F32, &Device::Cpu).unwrap();
    assert_eq!(model.forward(&x).unwrap().dims(), [1, 2, GRID_H, GRID_W]);

    let scanner = Scanner::new(model);
    let photo = page(300, 400);
    let d = scanner.detect(&photo).unwrap();
    assert!((0.0..=1.0).contains(&d.score));
    for geometry in [Geometry::Dewarp, Geometry::Perspective, Geometry::None] {
        let opts = Options {
            geometry,
            max_side: 256,
            ..Options::default()
        };
        let s = scanner.scan(&photo, &opts).unwrap();
        assert!(s.image.width.max(s.image.height) <= 400);
        assert_eq!(s.geometry, geometry);
    }
    // user corners: no detection, exact size
    let opts = Options {
        corners: Some([[0.0, 0.0], [299.0, 0.0], [299.0, 399.0], [0.0, 399.0]]),
        mode: Mode::Original,
        ..Options::default()
    };
    let s = scanner.scan(&photo, &opts).unwrap();
    assert!(s.detection.is_none());
    assert_eq!((s.image.width, s.image.height), (299, 399));
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn encodings() {
    let color = docscan::enhance::apply(&page(120, 90), Mode::Color);
    let bw = docscan::enhance::apply(&page(120, 90), Mode::Bw);
    let png = encode(&color, Format::Png, 90, 300, Paper::Auto).unwrap();
    assert_eq!(&png[1..4], b"PNG");
    let back = docscan::image::decode(&png).unwrap();
    assert_eq!((back.width, back.height), (120, 90));
    let jpg = encode(&color, Format::Jpeg, 90, 300, Paper::Auto).unwrap();
    assert_eq!(&jpg[..2], &[0xFF, 0xD8]);
    assert!(encode(&bw, Format::Jpeg, 90, 300, Paper::Auto).is_err());

    for (img, paper, filter) in [
        (&color, Paper::Auto, "/DCTDecode"),
        (&bw, Paper::A4, "/FlateDecode"),
    ] {
        let pdf = encode(img, Format::Pdf, 90, 300, paper).unwrap();
        let text = String::from_utf8_lossy(&pdf);
        assert!(pdf.starts_with(b"%PDF-1.4"));
        assert!(text.contains(filter));
        // startxref points at the xref table
        let sx = text.rfind("startxref\n").unwrap() + 10;
        let off: usize = text[sx..].lines().next().unwrap().parse().unwrap();
        assert!(pdf[off..].starts_with(b"xref"));
        let mediabox = if paper == Paper::A4 {
            "[0 0 841.890 595.276]" // landscape image, landscape page
        } else {
            "[0 0 28.800 21.600]" // 120x90 px at 300 dpi
        };
        assert!(text.contains(mediabox), "{mediabox}");
    }
}
