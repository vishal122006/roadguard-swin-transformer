# RoadGuard Swin Transformer

A Streamlit road-damage inspection UI backed by **Swin Transformer Tiny + FPN + Faster R-CNN** trained on RDD2022.

> Research prototype: false positives and missed damage remain. Do not use predictions as the sole basis for road-safety or maintenance decisions.

## Features

- Dark RoadGuard inspection UI adapted from the original RoadGuard project.
- Original image vs. annotated prediction shown side by side on a consistent 1280 × 720 comparison canvas without stretching.
- Real Swin model inference with a selectable confidence threshold.
- Four trained classes: longitudinal crack, transverse crack, alligator crack, and pothole.
- Download annotated image and detection CSV.
- Recorded validation/test scorecard with a per-class breakdown.
- No YOLO fallback and no fabricated predictions. The app requires the actual trained Swin checkpoint.

## Run the Streamlit app

Use Python 3.11 in an environment with PyTorch, MMCV, MMEngine, and MMDetection versions compatible with the trained checkpoint. The training run used PyTorch 2.1.0+cu121, TorchVision 0.16.0+cu121, MMEngine 0.10.7, MMCV 2.1.0, and MMDetection 3.3.0 on an NVIDIA Tesla T4.

Install dependencies in a compatible environment. MMCV must use a wheel/build compatible with the installed PyTorch and CUDA runtime; if standard pip installation attempts a source build or fails, follow the OpenMMLab installation matrix for the matching environment before installing the remaining requirements.

\`\`\`bash
python -m venv .venv
# Activate the virtual environment first.
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
streamlit run app.py
\`\`\`

### Provide the trained checkpoint

Large weights are intentionally not included in GitHub. Either place the checkpoint at:

- \`weights/best_coco_bbox_mAP_epoch_5.pth\`

or set the environment variable \`ROADGUARD_SWIN_CHECKPOINT\` to the absolute path of the checkpoint. The default config is \`configs/roadguard_swin_infer.py\`; override it with \`ROADGUARD_SWIN_CONFIG\` if required.

Example on Linux/macOS:

\`\`\`bash
export ROADGUARD_SWIN_CHECKPOINT="/absolute/path/to/best_coco_bbox_mAP_epoch_5.pth"
streamlit run app.py
\`\`\`

Example in Windows PowerShell:

\`\`\`powershell
$env:ROADGUARD_SWIN_CHECKPOINT="C:\path\to\best_coco_bbox_mAP_epoch_5.pth"
streamlit run app.py
\`\`\`

If the checkpoint is missing, the app shows an error and does not silently load another model. The environment-specific MMCV ops are a key deployment constraint; first validate the app in the same kind of environment used for training (for example, Colab with the matching MMDetection stack).

## Recorded experiment snapshot

Five epochs were completed on RDD2022.

| Metric | Recorded result |
|---|---:|
| Validation COCO mAP@0.50:0.95 | 9.00% |
| Validation COCO mAP@0.50 | 24.70% |
| Validation COCO mAP@0.75 | 4.00% |
| Test precision at confidence 0.50 | 30.98% |
| Test recall | 42.73% |
| Test F1 | 35.92% |

The confidence threshold 0.50 was selected by comparing 0.20, 0.30, 0.40, and 0.50 on validation data. PR/F1 uses greedy one-to-one matching per image and class at IoU 0.50; it is not interchangeable with COCO mAP. The test split was inspected in an earlier exploratory evaluation at epoch 1, so it should not be presented as completely untouched. Transverse-crack detections were not matched at the selected test threshold.

See [the full experiment report](docs/experiment-report.md) and the JSON/CSV files in \`results/\` for details. The Streamlit scorecard shows these measured dataset-level results; it does not claim these values as the score for the current uploaded photo.

## Project layout

- \`app.py\`: Streamlit UI and Swin inference.
- \`configs/roadguard_swin_infer.py\`: inference model configuration.
- \`results/\`: measured summary and CSVs.
- \`docs/experiment-report.md\`: training and evaluation report.
- \`weights/\`: local-only trained checkpoint folder, ignored by Git.

The previous YOLO app remains in the separate repository [roadguard-ai](https://github.com/vishal122006/roadguard-ai). This repo uses the Swin model directly and does not copy the YOLO inference/fallback logic.


## Automatic checkpoint loading

If the epoch-5 checkpoint is absent from `weights/`, the app downloads it from the public `swin-epoch5` release asset on first detection. To publish that asset from Colab, see [checkpoint setup](docs/checkpoint-setup.md). Publishing the release makes the trained weights public; the uploader requires an explicit `PUBLISH` confirmation. The old YOLO repository is not modified.


## Test and launch in the existing Colab environment

A Colab smoke-test script runs the real Swin checkpoint on an annotated RDD2022 validation image and saves an annotated preview to Google Drive. See [the Colab test and launch instructions](docs/colab-smoke-test.md). The test image is selected from the validation split and is for a visual check, not a new accuracy metric.
