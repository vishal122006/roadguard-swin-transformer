#!/usr/bin/env python3
"""Validation-only confidence-threshold sweep; loads the detector once."""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

from mmdet.apis import inference_detector, init_detector
from mmdet.utils import register_all_modules


def box_iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    return inter / max(area_a + area_b - inter, 1e-12)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/roadguard_swin_tiny_faster_rcnn.py")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--ann", default="data/rdd2022/annotations/instances_val.json")
    parser.add_argument("--images-dir", default="data/rdd2022/images")
    parser.add_argument("--thresholds", nargs="+", type=float,
                        default=[0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80])
    parser.add_argument("--iou-thr", type=float, default=0.50)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--out", default="outputs/validation_threshold_sweep.json")
    args = parser.parse_args()

    import numpy as np

    ann_path, images_dir = Path(args.ann), Path(args.images_dir)
    payload = json.loads(ann_path.read_text(encoding="utf-8"))
    categories = sorted(payload["categories"], key=lambda item: item["id"])
    class_names = [item["name"] for item in categories]
    class_id_map = {item["id"]: idx for idx, item in enumerate(categories)}
    gt_by_image = {}
    for img in payload["images"]:
        gt_by_image[img["id"]] = defaultdict(list)
    for ann in payload["annotations"]:
        if ann.get("iscrowd", 0):
            continue
        class_idx = class_id_map.get(ann["category_id"])
        if class_idx is None:
            continue
        x, y, w, h = ann["bbox"]
        gt_by_image[ann["image_id"]][class_idx].append([x, y, x + w, y + h])

    register_all_modules(init_default_scope=True)
    model = init_detector(args.config, args.checkpoint, device=args.device)
    model.dataset_meta["classes"] = tuple(class_names)
    cached = {}
    missing = []
    min_threshold = min(args.thresholds)
    for position, img in enumerate(payload["images"], 1):
        path = images_dir / img["file_name"]
        if not path.is_file():
            missing.append(str(path))
            cached[img["id"]] = []
            continue
        pred = inference_detector(model, str(path)).pred_instances.cpu()
        boxes = pred.bboxes.numpy()
        scores = pred.scores.numpy()
        labels = pred.labels.numpy()
        cached[img["id"]] = [
            {"bbox": box.tolist(), "score": float(score), "label": int(label)}
            for box, score, label in zip(boxes, scores, labels)
            if float(score) >= min_threshold
        ]
        if position % 100 == 0:
            print(f"Cached validation predictions: {position}/{len(payload['images'])}")

    reports = []
    for threshold in sorted(set(args.thresholds)):
        totals = {name: {"TP": 0, "FP": 0, "FN": 0} for name in class_names}
        for img_id, gt_by_class in gt_by_image.items():
            for class_idx, class_name in enumerate(class_names):
                gt = gt_by_class.get(class_idx, [])
                candidates = [
                    det for det in cached[img_id]
                    if det["label"] == class_idx and det["score"] >= threshold
                ]
                candidates.sort(key=lambda item: item["score"], reverse=True)
                used = set()
                tp = fp = 0
                for det in candidates:
                    overlaps = [
                        (box_iou(det["bbox"], box), index)
                        for index, box in enumerate(gt) if index not in used
                    ]
                    best_iou, best_idx = max(overlaps, default=(0.0, -1), key=lambda item: item[0])
                    if best_idx >= 0 and best_iou >= args.iou_thr:
                        used.add(best_idx)
                        tp += 1
                    else:
                        fp += 1
                totals[class_name]["TP"] += tp
                totals[class_name]["FP"] += fp
                totals[class_name]["FN"] += len(gt) - len(used)
        overall = {key: sum(row[key] for row in totals.values()) for key in ("TP", "FP", "FN")}
        precision = overall["TP"] / max(overall["TP"] + overall["FP"], 1)
        recall = overall["TP"] / max(overall["TP"] + overall["FN"], 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-12)
        reports.append({
            "threshold": threshold, **overall,
            "precision": precision, "recall": recall, "f1": f1,
            "per_class": totals,
        })

    best = max(reports, key=lambda row: (row["f1"], row["precision"]))
    result = {
        "checkpoint": args.checkpoint,
        "annotation_file": str(ann_path),
        "images": len(payload["images"]),
        "iou_threshold": args.iou_thr,
        "method": "greedy per-image/per-class one-to-one matching",
        "thresholds": reports,
        "recommended_threshold_by_micro_f1": best["threshold"],
        "note": "Validation-only threshold selection; freeze this threshold before test evaluation.",
        "missing_images": missing,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    csv_out = out.with_suffix(".csv")
    with csv_out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["threshold", "TP", "FP", "FN", "precision", "recall", "f1"])
        writer.writeheader()
        for row in reports:
            writer.writerow({key: row[key] for key in writer.fieldnames})
    print(json.dumps({
        "recommended_threshold_by_micro_f1": result["recommended_threshold_by_micro_f1"],
        "thresholds": [{k: row[k] for k in ("threshold", "precision", "recall", "f1")} for row in reports],
        "output": str(out),
        "csv": str(csv_out),
    }, indent=2))


if __name__ == "__main__":
    main()
