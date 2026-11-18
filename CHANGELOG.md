# Changelog

All notable changes to **docscan-rs** are documented in this file. The format
is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

## [0.1.0] - 2026-11-18

### Added
- UVDoc inference in Rust/Candle (CUDA, Metal, CPU): dilated ResNet,
  reflect-padded 2-D head, 45×31 grid; batch norms folded at conversion.
- `docscan convert`: `best_model.pkl` → one GGUF (F32 or F16).
- Geometry: `dewarp` (grid upsample + `grid_sample`, as UVDoc), `perspective`
  (homography from the grid's corners), user `corners`, `none`; output size
  from the document's length in the photo.
- Detection score (grid inside the photo, no fold-over), coverage, warnings.
- Enhancement without a learned model: illumination flattening by
  morphological closing (shadow edges stay put), levels, gray, Sauvola black
  and white.
- PNG, JPEG (DPI in JFIF), one-page PDF (JPEG or exact Flate pixels; auto,
  A4 or Letter page). EXIF orientation applied on decode.
- `docscan scan`, `docscan detect`, `docscan serve`
  (`POST /v1/documents/scans`, `POST /v1/documents/detect`, `/v1/models`,
  `/health`).
- Parity tooling (`examples/parity.rs`, `scripts/ref/ref.py`), synthetic
  photo maker (`examples/make_photo.rs`), `tests/pipeline.rs` (no download).
