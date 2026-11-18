# docscan-rs

Document photo → clean scan, in Rust: **UVDoc** dewarping (Verhoeven et
al., SIGGRAPH Asia 2023) on [Candle](https://github.com/huggingface/candle),
one 32 MB GGUF. A phone photo of an ID card, an invoice, a certificate or a
curled page comes out flat, cropped, with white paper and no shadows, as
PNG, JPEG or PDF.

- One binary, `docscan`: `convert`, `detect`, `scan`, `serve` (HTTP).
- **Conservative by design**, for official documents: the geometry is
  resampling (every output pixel is a bilinear mix of photo pixels) and the
  enhancement a per-pixel tone curve. No generative model, no inpainting,
  no super-resolution: a digit, a signature or a stamp is never redrawn.
- Detection score and warnings when the geometry is doubtful; `geometry =
  perspective` for a single homography (straight lines stay exactly
  straight); `corners` to force the user's own four points.
- EXIF orientation applied (phone photos are often stored sideways).
- CPU (about 0.5 s for a 8 MP photo), CUDA or Metal. Pure Rust: no OpenCV, no Python.
- The engine behind zallama's `docscan` modality
  ([rzafiamy/zallamars](https://github.com/rzafiamy/zallamars), backend
  `docscan-server`).

## Build

```bash
./build.sh            # CPU, or --cuda, --metal
```

## Model

Convert UVDoc's checkpoint (`model/best_model.pkl` in
[tanguymagne/UVDoc](https://github.com/tanguymagne/UVDoc), MIT). The file
is read as data: no code from it runs. Batch norms are folded into the
convolutions and the training-only 3-D head is dropped.

```bash
docscan convert best_model.pkl -o models/uvdoc-f32.gguf --name uvdoc   # 32 MB
docscan convert best_model.pkl -o models/uvdoc-f16.gguf --dtype f16     # 16 MB
```

| weights | grid vs PyTorch | note |
|---|---|---|
| `f32` (default) | 0.003 px | |
| `f16` | 0.1 px | half the size, visually identical |

## Scan

```bash
docscan scan -m models/uvdoc-f32.gguf photo.jpg -o scan.png
docscan scan -m … --mode bw photo.jpg -o scan.pdf --paper a4
docscan scan -m … --corners "120,80 1900,140 1980,2700 60,2650" photo.jpg -o id.jpg
docscan detect -m … *.jpg            # JSON lines: corners, score, coverage, size
```

| option | values |
|---|---|
| `--mode` | `color` (default: flattened light, white paper), `gray`, `bw` (Sauvola threshold), `original` (geometry only) |
| `--geometry` | `dewarp` (default: perspective + curl), `perspective` (4 corners), `none` |
| `--corners` | four photo points, any order, `x,y …` or `[[x,y],…]`: skips detection |
| `--format` | `png`, `jpeg`, `pdf` (default: from the output extension) |
| `--dpi`, `--paper`, `--quality`, `--max-side` | 300, `auto` / `a4` / `letter`, 90, 8000 |

The output keeps the document's resolution in the photo (the grid's
length), capped by `--max-side`. A black-and-white scan is never written as
JPEG (artifacts around every stroke).

## HTTP API

```bash
docscan serve -m models/uvdoc-f32.gguf --port 8791
curl -F file=@photo.jpg -F mode=gray -F format=pdf http://127.0.0.1:8791/v1/documents/scans -o scan.pdf
curl -F file=@photo.jpg http://127.0.0.1:8791/v1/documents/detect
```

- `POST /v1/documents/scans` (multipart): `file` and the options above
  (`mode`, `geometry`, `corners`, `format`, `dpi`, `paper`, `quality`,
  `max_side`). Returns the file with `X-Doc-Size` (`WxH`), `X-Doc-Corners`
  (JSON, photo pixels, TL TR BR BL), `X-Doc-Score` and `X-Doc-Warning` when
  the geometry is doubtful.
- `POST /v1/documents/detect` (multipart `file`): corners, score, coverage,
  size, warnings, photo width / height. For a live camera overlay.
- `GET /v1/models`, `GET /health`. Requests run one at a time. Errors:
  `{"error": {"message", "type"}}`.

## Accuracy

`examples/parity.rs` dumps the resized input, the grid and the unwarped
image; `scripts/ref/ref.py` recomputes them with UVDoc's own PyTorch code
and prints the differences (see `AGENTS.md`). `tests/pipeline.rs` runs
without any download (random-weight network, exact checks of the
resampling, enhancement and PDF code). `examples/make_photo.rs` turns a flat
scan into a synthetic phone photo (perspective, curl, shadow, warm light).

## Limits

- One page per photo; a page rotated 90° / 180° in the frame stays rotated
  (no orientation classifier yet).
- A shadow with a razor-sharp edge leaves a one-pixel line in `bw` mode.
- UVDoc was trained on A4-like pages; a small card in a large photo is
  found, but check `coverage` and the score.
- The score checks the grid (inside the photo, no fold-over), not what it
  frames: a photo without any document still gets a smooth grid and a high
  score. Deciding that there is a document is the caller's job.

## License

Apache-2.0 (code). UVDoc's model and weights: MIT (Tanguy Magne). See
`CREDITS.md`.
