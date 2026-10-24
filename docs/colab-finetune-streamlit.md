# Colab fine-tuning and Streamlit workflow

This workflow keeps **training in Colab** (GPU) and **image upload/inference in Streamlit**. The Streamlit app is for inference; it should not run long GPU training jobs inside a web request.

## Current baseline

The saved five-epoch Swin-Tiny + FPN + Faster R-CNN checkpoint remains the baseline. Previously measured results were validation COCO mAP@0.50:0.95 = 9.0%, mAP@0.50 = 24.7%, and test precision/recall/F1 at confidence 0.50 = 30.98%/42.73%/35.92%. The test split was explored earlier, so use validation metrics for model selection and do not describe the test split as an untouched holdout.

The training audit found only 49 transverse-crack boxes in training and 6 in validation. This is severe class imbalance; fine-tuning alone cannot guarantee a useful transverse-crack detector.

## 1. Run fine-tuning in the existing Colab environment

Prerequisites:
- Google Drive mounted at `/content/drive`
- Existing Python environment at `/content/roadguard_py311/bin/python`
- Training repo and RDD2022 dataset at `/content/RoadGuard_SwinTransformer`
- Current checkpoint at `/content/drive/MyDrive/RoadGuard/work_dirs/roadguard_swin_tiny_faster_rcnn/best_coco_bbox_mAP_epoch_5.pth`
- This deployment repo cloned at `/content/roadguard-swin-transformer`

Run in a Colab cell:

```python
from pathlib import Path
import os
import subprocess

repo = Path("/content/roadguard-swin-transformer")
python = "/content/roadguard_py311/bin/python"
checkpoint = Path(
    "/content/drive/MyDrive/RoadGuard/work_dirs/"
    "roadguard_swin_tiny_faster_rcnn/best_coco_bbox_mAP_epoch_5.pth"
)
config = Path(
    "/content/RoadGuard_SwinTransformer/configs/"
    "roadguard_swin_tiny_faster_rcnn.py"
)
work_dir = Path(
    "/content/drive/MyDrive/RoadGuard/work_dirs/"
    "roadguard_swin_finetune_15ep"
)

assert checkpoint.is_file(), f"Missing checkpoint: {checkpoint}"
assert config.is_file(), f"Missing training config: {config}"
assert (repo / "scripts/fine_tune_colab.py").is_file(), (
    "Update the deployment repo to the branch/commit containing the fine-tune script."
)

env = os.environ.copy()
env["MPLBACKEND"] = "Agg"

cmd = [
    python, "-u", str(repo / "scripts/fine_tune_colab.py"),
    "--config", str(config),
    "--checkpoint", str(checkpoint),
    "--work-dir", str(work_dir),
    "--epochs", "15",
    "--lr", "2e-5",
    "--val-interval", "1",
    "--max-keep-ckpts", "3",
]
print("Starting controlled fine-tuning. This may take a while.")
subprocess.run(
    cmd,
    cwd="/content/RoadGuard_SwinTransformer",
    env=env,
    check=True,
)
print("Fine-tuning command completed. Inspect validation logs and best checkpoint.")
```

This starts a new run from the epoch-5 model weights with a lower learning rate and validates each epoch. It does not resume the old optimizer state, and it does not overwrite the original checkpoint. It saves outputs to a separate Google Drive directory.

If the run fails because the checkpoint metric key is not `coco/bbox_mAP`, inspect the validation metric names in the log and update `save_best` to the exact metric key. Do not assume the fine-tuned model is better just because training completes.

## 2. Compare checkpoints before deployment

1. Compare the best new checkpoint's validation COCO mAP@0.50:0.95 and mAP@0.50 with the baseline.
2. Run the same confidence-threshold evaluation on validation data for precision, recall, and F1.
3. Review per-class metrics, especially transverse cracks.
4. Keep the epoch-5 checkpoint if validation results do not improve. Do not select checkpoints using test-set results.

## 3. Test the new checkpoint with the real inference config

After confirming the new checkpoint exists, run the existing smoke test with these environment values in Colab:

```python
import os
import subprocess
from pathlib import Path

repo = Path("/content/roadguard-swin-transformer")
python = "/content/roadguard_py311/bin/python"
new_work_dir = Path(
    "/content/drive/MyDrive/RoadGuard/work_dirs/"
    "roadguard_swin_finetune_15ep"
)
candidates = sorted(new_work_dir.glob("best_coco_bbox_mAP_epoch_*.pth"))
if not candidates:
    raise FileNotFoundError("No best fine-tuned checkpoint found; check training logs.")
checkpoint = max(candidates, key=lambda p: p.stat().st_mtime)

env = os.environ.copy()
env.update({
    "MPLBACKEND": "Agg",
    "ROADGUARD_SWIN_CHECKPOINT": str(checkpoint),
    "ROADGUARD_SWIN_CONFIG": str(repo / "configs/roadguard_swin_infer.py"),
    "ROADGUARD_RDD_VAL_ANN":
        "/content/RoadGuard_SwinTransformer/data/rdd2022/annotations/instances_val.json",
    "ROADGUARD_RDD_IMAGES":
        "/content/RoadGuard_SwinTransformer/data/rdd2022/images",
    "ROADGUARD_SMOKE_THRESHOLD": "0.50",
    "ROADGUARD_SMOKE_OUTPUT_DIR":
        "/content/drive/MyDrive/RoadGuard/finetune_smoke_test",
})
subprocess.run(
    [python, "-u", str(repo / "scripts/smoke_test_checkpoint_colab.py")],
    cwd=str(repo), env=env, check=True,
)
```

A smoke test confirms inference works; it is not an accuracy evaluation.

## 4. Upload to the Streamlit app

The existing `app.py` already supports image uploads, real Swin inference, confidence filtering, side-by-side display, annotated-image download, and CSV export.

For the hosted app to load weights automatically, the checkpoint must be accessible to the deployment. The current app's default public GitHub release URL is `swin-epoch5/best_coco_bbox_mAP_epoch_5.pth`; that is the baseline asset, not the new fine-tuned checkpoint. Do not overwrite or publish weights without deciding that they should be public.

For a fine-tuned deployment:
- First evaluate and approve the new checkpoint.
- Publish the chosen checkpoint to a release or other authorized artifact storage.
- Update the app's `ROADGUARD_SWIN_CHECKPOINT_URL` setting to the new asset URL, or set `ROADGUARD_SWIN_CHECKPOINT` to a local checkpoint where the hosting environment supports it.
- Confirm that the host can install compatible PyTorch/MMCV/MMDetection dependencies. These OpenMMLab compiled dependencies can make a generic Streamlit Cloud deployment difficult; validate the exact hosting runtime before announcing it as deployed.

**Privacy note:** publishing a GitHub release asset makes the checkpoint publicly downloadable. Never upload private datasets, tokens, or credentials to the repository.
