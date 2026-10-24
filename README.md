# RoadGuard Swin Transformer

Experimental road-damage detection with a **Swin Transformer Tiny backbone, FPN, and Faster R-CNN**, trained/evaluated on RDD2022.

> Research prototype. Predictions may be wrong or incomplete. Do not use as the sole basis for road-safety, infrastructure, or maintenance decisions.

## Current experiment snapshot
- 5 training epochs completed
- Validation COCO mAP@0.50:0.95: **9.00%**
- Validation COCO mAP@0.50: **24.70%**
- Validation COCO mAP@0.75: **4.00%**
- Test precision at confidence threshold 0.50: **30.98%**
- Test recall: **42.73%**
- Test F1: **35.92%**

Threshold 0.50 was selected using validation data from a sweep of 0.20, 0.30, 0.40, and 0.50. PR/F1 uses greedy one-to-one matching per image/class at IoU 0.50; these values do not replace COCO mAP.

## Project layout
- `configs/`: model and dataset configuration
- `scripts/`: training/evaluation utilities
- `results/`: measured evaluation metrics
- `docs/experiment-report.md`: full experiment report

Model checkpoint and road images are not stored in this repository. Keep checkpoints in Google Drive or a model-hosting service and document the exact checkpoint path/hash for reproducibility.

## Status
This is a preliminary five-epoch baseline. Transverse crack detections were not matched at the selected test threshold, and false positives remain substantial. Do not claim production readiness or improvement beyond the measured metrics.

See [the full experiment report](docs/experiment-report.md) for validation threshold comparisons, per-class metrics, and limitations.
