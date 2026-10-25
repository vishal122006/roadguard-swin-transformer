#!/usr/bin/env bash
set -Eeuo pipefail

# One-command Crack/Pothole experiment runner.
# Run from the repository root after mounting Google Drive in a GPU-enabled Colab.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_ROOT="${ROADGUARD_DATA_ROOT:-/content/drive/MyDrive/RoadGuard/datasets/RDD2022_binary_coco}"
CHECKPOINT="${ROADGUARD_BASE_CHECKPOINT:-/content/drive/MyDrive/RoadGuard/work_dirs/roadguard_swin_tiny_faster_rcnn/best_coco_bbox_mAP_epoch_5.pth}"
WORK_DIR="${ROADGUARD_WORK_DIR:-/content/drive/MyDrive/RoadGuard/work_dirs/roadguard_swin_crack_binary_20ep}"
EPOCHS="${ROADGUARD_EPOCHS:-20}"
LR="${ROADGUARD_LR:-2e-5}"
ANNOTATION_DIR="$DATA_ROOT"

cd "$REPO_ROOT"

for f in "$CHECKPOINT" "$DATA_ROOT/instances_train.json" "$DATA_ROOT/instances_val.json" "$DATA_ROOT/instances_test.json"; do
  if [[ ! -f "$f" ]]; then
    echo "ERROR: Required file not found: $f" >&2
    exit 2
  fi
done

python - <<'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit(
        "No CUDA GPU is available. This runner intentionally stops before training "
        "rather than attempting expensive Swin training on CPU. In Colab, select a "
        "GPU runtime or run this script on a GPU machine."
    )
print("CUDA available:", torch.cuda.get_device_name(0))
PY

mkdir -p "$WORK_DIR"
python "$REPO_ROOT/scripts/fine_tune_crack_colab.py" \
  --config "$REPO_ROOT/configs/roadguard_swin_crack_train.py" \
  --checkpoint "$CHECKPOINT" \
  --data-root "$DATA_ROOT" \
  --annotation-dir "$ANNOTATION_DIR" \
  --work-dir "$WORK_DIR" \
  --epochs "$EPOCHS" --lr "$LR" --val-interval 1

BEST_CHECKPOINT="$(python - "$WORK_DIR" <<'PY'
import sys
from pathlib import Path
root = Path(sys.argv[1])
files = list(root.glob("best_coco_bbox_mAP_epoch_*.pth")) or list(root.glob("best_*.pth"))
if not files:
    raise SystemExit("No best validation checkpoint found in " + str(root))
print(max(files, key=lambda p: p.stat().st_mtime))
PY
)"

RESULTS="$WORK_DIR/results"
mkdir -p "$RESULTS"

# Select confidence threshold on validation only.
python "$REPO_ROOT/scripts/evaluate_crack_only.py" \
  --config "$REPO_ROOT/configs/roadguard_swin_crack_infer.py" \
  --checkpoint "$BEST_CHECKPOINT" \
  --ann "$DATA_ROOT/instances_val.json" \
  --images-dir "$DATA_ROOT" \
  --model-class-mode binary --device cuda:0 \
  --out "$RESULTS/validation.json"

VAL_THRESHOLD="$(python - "$RESULTS/validation.json" <<'PY'
import json, sys
data = json.load(open(sys.argv[1], encoding="utf-8"))
print(data["best_crack_f1_threshold_among_tested_thresholds"])
PY
)"

# Freeze validation-selected threshold for the held-out test report.
python "$REPO_ROOT/scripts/evaluate_crack_only.py" \
  --config "$REPO_ROOT/configs/roadguard_swin_crack_infer.py" \
  --checkpoint "$BEST_CHECKPOINT" \
  --ann "$DATA_ROOT/instances_test.json" \
  --images-dir "$DATA_ROOT" \
  --thresholds "$VAL_THRESHOLD" \
  --model-class-mode binary --device cuda:0 \
  --out "$RESULTS/test.json"

python - "$BEST_CHECKPOINT" "$WORK_DIR" "$VAL_THRESHOLD" <<'PY'
import json, sys
from pathlib import Path
checkpoint, work_dir, threshold = sys.argv[1], Path(sys.argv[2]), float(sys.argv[3])
summary = {
    "status": "completed",
    "best_checkpoint": checkpoint,
    "validation_selected_confidence_threshold": threshold,
    "validation_report": str(work_dir / "results/validation.json"),
    "test_report": str(work_dir / "results/test.json"),
    "test_csv": str(work_dir / "results/test.csv"),
    "note": "Test metrics use a threshold selected on validation. No metric is claimed unless these reports exist."
}
(work_dir / "experiment_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
print(json.dumps(summary, indent=2))
PY
