"""Smoke-test the actual published detector checkpoint on CPU.

The blank input is only a runtime smoke-test tensor; its output is not an
accuracy measurement and is not used by the Streamlit app.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from mmdet.apis import inference_detector, init_detector
from mmdet.utils import register_all_modules


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/roadguard_swin_infer.py")
    parser.add_argument("--checkpoint", required=True)
    args = parser.parse_args()

    config = Path(args.config)
    checkpoint = Path(args.checkpoint)
    if not config.is_file():
        raise FileNotFoundError(f"Missing inference config: {config}")
    if not checkpoint.is_file() or checkpoint.stat().st_size < 10_000_000:
        raise FileNotFoundError(f"Missing or invalid model checkpoint: {checkpoint}")

    torch.set_num_threads(min(4, torch.get_num_threads()))
    register_all_modules(init_default_scope=True)
    model = init_detector(str(config), str(checkpoint), device="cpu")
    model.eval()

    smoke_image = np.zeros((480, 640, 3), dtype=np.uint8)
    with torch.no_grad():
        result = inference_detector(model, smoke_image)
    if not hasattr(result, "pred_instances"):
        raise RuntimeError("MMDetection inference returned no pred_instances field")

    print("Real Swin checkpoint loaded and CPU inference completed.")
    print("Runtime smoke-test detections:", len(result.pred_instances))
    print("This synthetic smoke-test input is not an accuracy evaluation.")


if __name__ == "__main__":
    main()
