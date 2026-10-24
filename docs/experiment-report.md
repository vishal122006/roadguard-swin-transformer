# RoadGuard Swin Transformer — Experiment Report

## Overview
- **Task:** Multi-class road damage object detection
- **Dataset:** RDD2022
- **Architecture:** Swin Transformer Tiny backbone + Feature Pyramid Network (FPN) + Faster R-CNN
- **Training:** 5 epochs completed
- **Selected checkpoint:** `best_coco_bbox_mAP_epoch_5.pth` (kept outside GitHub due to size)
- **Confidence threshold:** 0.50, selected by validation F1 among tested thresholds
- **Matching IoU for PR/F1:** 0.50

## Validation COCO metrics at epoch 5
| Metric | Result |
|---|---:|
| COCO mAP@0.50:0.95 | 9.00% |
| COCO mAP@0.50 | 24.70% |
| COCO mAP@0.75 | 4.00% |

## Validation threshold comparison
| Threshold | Precision | Recall | F1 |
|---:|---:|---:|---:|
| 0.20 | 12.73% | 64.29% | 21.25% |
| 0.30 | 18.73% | 57.17% | 28.22% |
| 0.40 | 25.04% | 49.67% | 33.29% |
| 0.50 | 31.59% | 42.55% | 36.26% |

The 0.50 threshold achieved the highest micro-F1 among the four thresholds evaluated.

## Final test-split metrics at threshold 0.50
| Metric | Result |
|---|---:|
| Images processed | 1,156 |
| True positives | 426 |
| False positives | 949 |
| False negatives | 571 |
| Precision | 30.98% |
| Recall | 42.73% |
| F1 | 35.92% |

## Per-class test results
| Class | Precision | Recall | F1 |
|---|---:|---:|---:|
| D00_Longitudinal_Crack | 31.25% | 17.24% | 22.22% |
| D10_Transverse_Crack | 0.00% | 0.00% | 0.00% |
| D20_Alligator_Crack | 38.67% | 60.57% | 47.21% |
| D40_Pothole | 26.79% | 45.88% | 33.83% |

## Limitations and transparency
- Five training epochs completed.
- Validation threshold sweep tested 0.20, 0.30, 0.40, and 0.50. Threshold 0.50 produced the highest micro-F1 among tested thresholds.
- Precision/recall/F1 use greedy one-to-one matching per image and class at IoU 0.50; they are not COCO mAP.
- The test split was previously inspected in an exploratory epoch-1 evaluation at confidence 0.30. The test results should not be described as coming from a wholly untouched test set.
- No detections were matched for transverse cracks at the selected threshold.
- These results are a preliminary research baseline, not deployment-ready performance.
- Model checkpoint weights and dataset images are not included in the repository.

## Reproducibility artifacts
- `results/final_summary.json`: detailed measured metrics and experiment metadata
- `results/validation_threshold_sweep.csv`: validation threshold sweep
- `results/test_per_class.csv`: per-class test metrics
- `README.md`: project overview and quick start

The original YOLO Streamlit application is not included or replaced by this separate Swin experiment. Before further claims or deployment, inspect dataset label quality, split integrity, and sample predictions; investigate the transverse-crack class failure and false positives.
