"""Deployment preflight: verify imports and compiled CPU MMCV operations."""
from __future__ import annotations

import cv2
import mmcv
import mmdet
import mmengine
import numpy
import streamlit
import torch
import torchvision
from mmcv.ops import RoIAlign, nms


def main() -> None:
    versions = {
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "mmcv": mmcv.__version__,
        "mmengine": mmengine.__version__,
        "mmdet": mmdet.__version__,
        "streamlit": streamlit.__version__,
        "numpy": numpy.__version__,
        "opencv": cv2.__version__,
    }
    for name, version in versions.items():
        print(f"{name}: {version}")

    assert torch.__version__.startswith("2.1.0"), versions
    assert torchvision.__version__.startswith("0.16.0"), versions
    assert mmcv.__version__ == "2.1.0", versions
    assert mmengine.__version__ == "0.10.7", versions
    assert mmdet.__version__ == "3.3.0", versions
    assert numpy.__version__ == "1.26.4", versions
    assert cv2.__version__ == "4.11.0", versions

    boxes = torch.tensor([[0., 0., 10., 10.], [1., 1., 9., 9.]])
    scores = torch.tensor([0.9, 0.8])
    kept, _ = nms(boxes, scores, 0.5)
    assert len(kept) == 1, f"Unexpected CPU NMS result: {kept}"

    features = torch.randn(1, 1, 8, 8)
    rois = torch.tensor([[0., 1., 1., 5., 5.]])
    pooled = RoIAlign(output_size=2, spatial_scale=1.0, sampling_ratio=0)(
        features, rois
    )
    assert tuple(pooled.shape) == (1, 1, 2, 2), tuple(pooled.shape)
    print("Imports, compiled MMCV CPU NMS and ROIAlign checks passed.")


if __name__ == "__main__":
    main()
