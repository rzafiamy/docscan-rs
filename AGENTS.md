# AGENTS.md

Guidance for AI agents working in this repository.

## Project

Document photo → scan in Rust/Candle: UVDoc dewarping (inference only), one
binary (`docscan`), weights in one GGUF written by `docscan convert` from
UVDoc's `best_model.pkl`. Engine of zallama's `docscan-server` backend
(`POST /v1/documents/scans`).

Pipeline: decode + EXIF orientation (`image.rs`) → resize to 488×712
(`warp::resize_bilinear`, torch bilinear / cv2 INTER_LINEAR semantics) →
network (`uvdoc.rs`) → 45×31 grid → `scan::analyse` (corners, score,
output size) → `warp::unwarp` (UVDoc `bilinear_unwarping`) or
`warp::perspective` → `enhance.rs` → `scan::encode` (`pdf.rs`).

## Where things are

- `crates/docscan/src/uvdoc.rs`: the network; `convs()` lists every folded
  convolution (name, shape, stride, dilation, padding).
- `crates/docscan/src/gguf.rs`: checkpoint names → folded names, BN folding,
  GGUF write / load.
- `crates/docscan/src/warp.rs`: resize, grid unwarp, homography.
- `crates/docscan/src/enhance.rs`: flattening (closing), levels, Sauvola.
- `crates/docscan/src/scan.rs`: options, detection analysis, `Scanner`.
- `crates/docscan-cli/src/`: CLI (`main.rs`), HTTP server (`server.rs`).
- `scripts/ref/ref.py`: PyTorch reference (imports UVDoc's checkout).

## Commands

```bash
cargo build --release -p docscan-cli
cargo test --release --workspace
cargo clippy --workspace --all-targets -- -D warnings
docscan convert best_model.pkl -o models/uvdoc-f32.gguf --name uvdoc

# synthetic photo, parity (Python with torch, safetensors; a UVDoc checkout)
cargo run --release --example make_photo -- page.jpg tmp/photo.jpg
cargo run --release --example parity -- models/uvdoc-f32.gguf tmp/photo.jpg tmp/rs.safetensors --cpu
python scripts/ref/ref.py <UVDoc dir> <UVDoc dir>/model/best_model.pkl tmp/rs.safetensors
```

CUDA: `--features cuda`, build into `--target-dir target-cuda`, needs
`/usr/local/cuda/bin` in PATH and `CUDA_COMPUTE_CAP`.

## Rules

- UVDoc's `model.py` and `utils.py` are the spec. F32 must match the
  reference: grid shift well under 0.01 px on a photo.
- Never add anything that synthesises pixels (generative restoration,
  inpainting, super-resolution) to the default path: the output must stay
  evidence of the photographed document. Anything of the kind would be an
  explicit opt-in mode, documented as such.
- Background estimation must not spread edges (closing, not blur): a gray
  band turns into a black bar over text in `bw`.
- Keep the CPU path dependency-free (no OpenCV, no system libs).
- Tests need no model download (`tests/pipeline.rs` builds a random one).
