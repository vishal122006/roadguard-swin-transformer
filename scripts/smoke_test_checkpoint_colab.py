#!/usr/bin/env python3
"""Run a real Swin checkpoint inference on one RDD2022 validation image in Colab."""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

REPO_ROOT = Path(__file__).resolve().parents[1]
TRAIN_ROOT = Path("/content/RoadGuard_SwinTransformer")
ANN_PATH = Path(os.getenv(
    "ROADGUARD_RDD_VAL_ANN",
    str(TRAIN_ROOT / "data" / "rdd2022" / "annotations" / "instances_val.json"),
))
IMAGE_DIR = Path(os.getenv(
    "ROADGUARD_RDD_IMAGES",
    str(TRAIN_ROOT / "data" / "rdd2022" / "images"),
))
CHECKPOINT = Path(os.getenv(
    "ROADGUARD_SWIN_CHECKPOINT",
    "/content/drive/MyDrive/RoadGuard/work_dirs/roadguard_swin_tiny_faster_rcnn/best_coco_bbox_mAP_epoch_5.pth",
)).expanduser()
CONFIG = Path(os.getenv(
    "ROADGUARD_SWIN_CONFIG",
    str(REPO_ROOT / "configs" / "roadguard_swin_infer.py"),
)).expanduser()
OUTPUT_DIR = Path(os.getenv(
    "ROADGUARD_SMOKE_OUTPUT_DIR",
    "/content/drive/MyDrive/RoadGuard",
)).expanduser()
THRESHOLD = float(os.getenv("ROADGUARD_SMOKE_THRESHOLD", "0.50"))

CLASSES = [
    "D00_Longitudinal_Crack",
    "D10_Transverse_Crack",
    "D20_Alligator_Crack",
    "D40_Pothole",
]
DISPLAY = {
    CLASSES[0]: "Longitudinal crack",
    CLASSES[1]: "Transverse crack",
    CLASSES[2]: "Alligator crack",
    CLASSES[3]: "Pothole",
}
COLORS = [(255, 113, 136), (102, 217, 232), (199, 243, 107), (255, 184, 107)]


def pick_validation_image():
    if not ANN_PATH.is_file():
        raise FileNotFoundError(f"RDD2022 validation annotations not found: {ANN_PATH}")
    if not IMAGE_DIR.is_dir():
        raise FileNotFoundError(f"RDD2022 image folder not found: {IMAGE_DIR}")
    data = json.loads(ANN_PATH.read_text(encoding="utf-8"))
    images = {int(row["id"]): row for row in data.get("images", [])}
    counts = {}
    for ann in data.get("annotations", []):
        counts[int(ann["image_id"])] = counts.get(int(ann["image_id"]), 0) + 1

    candidates = sorted(
        data.get("images", []),
        key=lambda row: counts.get(int(row["id"]), 0),
        reverse=True,
    )
    for item in candidates:
        file_name = str(item["file_name"])
        candidate = IMAGE_DIR / file_name
        if candidate.is_file():
            return candidate, item, counts.get(int(item["id"]), 0)
        candidate = IMAGE_DIR / Path(file_name).name
        if candidate.is_file():
            return candidate, item, counts.get(int(item["id"]), 0)

    raise FileNotFoundError(
        "No annotated validation image could be found below IMAGE_DIR. "
        "Set ROADGUARD_RDD_IMAGES to the correct RDD2022 images directory."
    )


def font(size=22):
    for name in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf",
    ):
        try:
            return ImageFont.truetype(name, size=size)
        except OSError:
            pass
    return ImageFont.load_default()


def main():
    if not CHECKPOINT.is_file():
        raise FileNotFoundError(f"Epoch-5 checkpoint not found: {CHECKPOINT}")
    if not CONFIG.is_file():
        raise FileNotFoundError(f"Swin inference config not found: {CONFIG}")

    image_path, image_info, gt_count = pick_validation_image()
    print(f"Validation image: {image_path}")
    print(f"Image size in annotation: {image_info.get('width')} x {image_info.get('height')}")
    print(f"Ground-truth boxes in annotation: {gt_count}")
    print(f"Checkpoint: {CHECKPOINT} ({CHECKPOINT.stat().st_size / 1024**2:.1f} MB)")
    print(f"Config: {CONFIG}")
    print(f"Confidence threshold: {THRESHOLD:.2f}")

    import torch
    from mmdet.apis import init_detector, inference_detector
    from mmdet.utils import register_all_modules

    register_all_modules(init_default_scope=True)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print("Inference device:", device, flush=True)

    started = time.perf_counter()
    model = init_detector(str(CONFIG), str(CHECKPOINT), device=device)
    result = inference_detector(model, str(image_path))
    elapsed = time.perf_counter() - started

    instances = result.pred_instances
    boxes = instances.bboxes.detach().cpu().numpy()
    scores = instances.scores.detach().cpu().numpy()
    labels = instances.labels.detach().cpu().numpy()

    detections = []
    for box, score, label in zip(boxes, scores, labels):
        score = float(score)
        label = int(label)
        if score < THRESHOLD or not 0 <= label < len(CLASSES):
            continue
        detections.append({
            "class_id": label,
            "class_code": CLASSES[label],
            "class_name": DISPLAY[CLASSES[label]],
            "score": score,
            "bbox_xyxy": [round(float(v), 2) for v in box.tolist()],
        })

    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    draw_font = font()
    for det in detections:
        x1, y1, x2, y2 = det["bbox_xyxy"]
        color = COLORS[det["class_id"]]
        draw.rectangle((x1, y1, x2, y2), outline=color, width=max(3, image.width // 500))
        label_text = f'{det["class_name"]} · {det["score"]:.0%}'
        bounds = draw.textbbox((0, 0), label_text, font=draw_font)
        tw, th = bounds[2] - bounds[0], bounds[3] - bounds[1]
        tx = min(max(0, int(x1)), max(0, image.width - tw - 8))
        ty = max(0, int(y1) - th - 8)
        draw.rectangle((tx, ty, min(image.width, tx + tw + 8), min(image.height, ty + th + 8)), fill=(12, 18, 14))
        draw.text((tx + 4, ty + 3), label_text, font=draw_font, fill=color)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    annotated_path = OUTPUT_DIR / "roadguard_swin_smoke_test.jpg"
    report_path = OUTPUT_DIR / "roadguard_swin_smoke_test.json"
    image.save(annotated_path, quality=94)
    report = {
        "sample_image": str(image_path),
        "image_width": image.width,
        "image_height": image.height,
        "ground_truth_box_count_in_validation_annotation": gt_count,
        "checkpoint": str(CHECKPOINT),
        "config": str(CONFIG),
        "device": device,
        "confidence_threshold": THRESHOLD,
        "elapsed_seconds_including_model_load": round(elapsed, 3),
        "detection_count": len(detections),
        "detections": detections,
        "note": "Visual smoke test only; not an additional dataset-level accuracy metric."
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("\nSMOKE TEST COMPLETE")
    print("Detections:", len(detections))
    print("Time including model load:", f"{elapsed:.2f}s")
    print("Annotated image:", annotated_path)
    print("JSON details:", report_path)
    if detections:
        for item in detections[:10]:
            print(f" - {item['class_name']}: {item['score']:.1%} {item['bbox_xyxy']}")
    else:
        print("No boxes above the chosen threshold. This visual test does not mean the image is damage-free.")


if __name__ == "__main__":
    main()
