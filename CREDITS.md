# Credits

- **UVDoc**: Floor Verhoeven, Tanguy Magne, Olga Sorkine-Hornung,
  "UVDoc: Neural Grid-based Document Unwarping" (SIGGRAPH Asia 2023).
  Network, input size, grid and unwarping follow `model.py` and `utils.py` of
  [tanguymagne/UVDoc](https://github.com/tanguymagne/UVDoc) (MIT,
  © 2023 Tanguy Magne), which `scripts/ref/ref.py` imports for the parity
  reference only.
- **Sauvola & Pietikäinen**, "Adaptive document image binarization"
  (Pattern Recognition, 2000): the `bw` threshold.
- **Candle** (Apache-2.0 / MIT, Hugging Face): tensors, CUDA/Metal kernels,
  GGUF and pickle readers.
- **image**, **flate2**, **rayon**: decoding / encoding, PDF compression,
  threads.

## Model weights

Not part of this repository. UVDoc's `best_model.pkl` is published with its
code under the MIT licence; the GGUF converted from it keeps that licence.
