"""Export the trained Tiny U-Net to ONNX and verify parity with PyTorch.

    ml/.venv/Scripts/python.exe scripts/export_onnx.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ml"))

import torch  # noqa: E402
from space_mapper.models import TinyUNet  # noqa: E402
from space_mapper.io import read_image  # noqa: E402
from space_mapper.datasets import resize_array  # noqa: E402

CKPT = ROOT / "ml/artifacts/tiny_unet_diverse.pt"
OUT = ROOT / "demo_static/model/tiny_unet_diverse.onnx"


def main() -> int:
    ckpt = torch.load(CKPT, map_location="cpu")
    model = TinyUNet(in_channels=ckpt["in_channels"], out_channels=1,
                     base_channels=ckpt["base_channels"])
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"loaded checkpoint: in_channels={ckpt['in_channels']} base={ckpt['base_channels']} resize={ckpt.get('resize')}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    dummy = torch.zeros(1, ckpt["in_channels"], 128, 128)
    torch.onnx.export(
        model, dummy, str(OUT),
        input_names=["input"], output_names=["logits"],
        opset_version=17, dynamo=False,
    )
    print(f"exported: {OUT} ({OUT.stat().st_size/1024:.0f} KB)")

    # --- parity check on a real chip, mirroring predict_with_checkpoint ---
    import onnxruntime as ort

    chip = ROOT / "data/raw/sen1floods11/images/Spain_5923267.tif"
    image = read_image(chip, normalize=True).astype(np.float32)  # HxWx13 in [0,1]
    small = resize_array(image, 128, is_mask=False)              # 128x128x13
    x = np.moveaxis(small, -1, 0)[None].astype(np.float32)       # 1x13x128x128

    with torch.no_grad():
        pt_logits = model(torch.from_numpy(x)).numpy()

    sess = ort.InferenceSession(str(OUT), providers=["CPUExecutionProvider"])
    ort_logits = sess.run(None, {"input": x})[0]

    diff = np.abs(pt_logits - ort_logits).max()
    pt_mask = (1 / (1 + np.exp(-pt_logits)) >= 0.5)
    ort_mask = (1 / (1 + np.exp(-ort_logits)) >= 0.5)
    agree = (pt_mask == ort_mask).mean()
    print(f"max |logit diff| = {diff:.2e}")
    print(f"mask agreement   = {agree*100:.4f}%")
    print(f"water fraction   = torch {pt_mask.mean()*100:.2f}% | onnx {ort_mask.mean()*100:.2f}%")
    if diff > 1e-3 or agree < 0.9999:
        raise SystemExit("PARITY CHECK FAILED")
    print("PARITY OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
