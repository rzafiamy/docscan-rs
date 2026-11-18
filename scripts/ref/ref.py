"""PyTorch reference for docscan's parity check.

Loads the dump of `examples/parity.rs` (the decoded photo and docscan's
intermediates), recomputes them with UVDoc's own code (model.py, utils.py:
github.com/tanguymagne/UVDoc, MIT) and prints the relative errors.

    python scripts/ref/ref.py <UVDoc checkout> <best_model.pkl> tmp/rs.safetensors

The input resize uses torch bilinear (align_corners=False), which is what
OpenCV's INTER_LINEAR computes on float images (UVDoc's demo uses cv2).
Needs torch, numpy, safetensors.
"""
import sys

import torch
import torch.nn.functional as F
from safetensors.torch import load_file


def rel(a, b):
    return float((a - b).abs().max() / b.abs().max().clamp_min(1e-12))


def main():
    uvdoc_dir, ckpt, dump = sys.argv[1:4]
    sys.path.insert(0, uvdoc_dir)
    from model import UVDocnet
    from utils import IMG_SIZE, bilinear_unwarping

    rs = load_file(dump)
    img = rs["image"].permute(2, 0, 1).unsqueeze(0)  # 1,3,H,W
    h, w = img.shape[2:]
    inp = F.interpolate(img, size=(IMG_SIZE[1], IMG_SIZE[0]), mode="bilinear", align_corners=False)

    model = UVDocnet(num_filter=32, kernel_size=5)
    state = torch.load(ckpt, map_location="cpu", weights_only=True)["model_state"]
    model.load_state_dict(state)
    model.eval()
    with torch.no_grad():
        grid, _ = model(inp)  # 1,2,45,31
        # the network on docscan's own input, to separate resize from model error
        grid_on_rs, _ = model(rs["input"].permute(2, 0, 1).unsqueeze(0))
        unwarped = bilinear_unwarping(img, grid, (w, h))

    print(f"input     rel err {rel(rs['input'], inp[0].permute(1, 2, 0)):.2e}")
    print(f"grid      rel err {rel(rs['grid'], grid_on_rs[0].permute(1, 2, 0)):.2e}  (same input)")
    print(f"grid      rel err {rel(rs['grid'], grid[0].permute(1, 2, 0)):.2e}  (end to end)")
    px = (rs["grid"] - grid[0].permute(1, 2, 0)).abs().max() * max(w, h) / 2
    print(f"grid      max shift {px:.3f} px")
    d = (rs["unwarped"] - unwarped[0].permute(1, 2, 0)).abs()
    print(f"unwarped  max abs {float(d.max()):.2e}, mean abs {float(d.mean()):.2e} (0..1 scale)")


if __name__ == "__main__":
    main()
