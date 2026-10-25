#!/usr/bin/env python3
"""Evaluate four-class RoadGuard predictions as a unified Crack detector.

Crack combines D00 + D10 + D20. Pothole stays separate for reference.
Reports object-level precision/recall/F1, interpolated AP at IoU=0.50, and
image-level binary crack-present metrics. This script does not train a model.
"""
from __future__ import annotations
import argparse, csv, json
from collections import defaultdict
from pathlib import Path

MODEL_CLASSES = ("D00_Longitudinal_Crack", "D10_Transverse_Crack",
                 "D20_Alligator_Crack", "D40_Pothole")
CRACK_CODES = {"D00", "D10", "D20"}

def code_from_name(name: str) -> str:
    upper = name.upper()
    for code in ("D00", "D10", "D20", "D40"):
        if code in upper:
            return code
    for word, code in (("LONGITUDINAL","D00"), ("TRANSVERSE","D10"),
                        ("ALLIGATOR","D20"), ("POTHOLE","D40")):
        if word in upper:
            return code
    raise ValueError(f"Unrecognized RDD2022 category: {name!r}")

def as_target(name: str) -> str:
    return "Crack" if code_from_name(name) in CRACK_CODES else "Pothole"

def to_xyxy(box):
    x, y, w, h = map(float, box)
    return [x, y, x+w, y+h]

def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2-x1) * max(0.0, y2-y1)
    aa = max(0.0, a[2]-a[0]) * max(0.0, a[3]-a[1])
    ab = max(0.0, b[2]-b[0]) * max(0.0, b[3]-b[1])
    return inter / max(aa + ab - inter, 1e-12)

def match_counts(image_ids, gt, predictions, class_name, threshold, iou_thr):
    tp = fp = fn = 0
    for image_id in image_ids:
        targets = gt[image_id].get(class_name, [])
        candidates = [d for d in predictions.get(image_id, [])
                      if d["class"] == class_name and d["score"] >= threshold]
        candidates.sort(key=lambda d: d["score"], reverse=True)
        matched = set()
        for det in candidates:
            overlaps = [(iou(det["bbox"], box), idx)
                        for idx, box in enumerate(targets) if idx not in matched]
            ov, best = max(overlaps, default=(0.0, -1), key=lambda pair: pair[0])
            if best >= 0 and ov >= iou_thr:
                tp += 1
                matched.add(best)
            else:
                fp += 1
        fn += len(targets) - len(matched)
    return {"TP": tp, "FP": fp, "FN": fn}

def add_prf(counts):
    tp, fp, fn = counts["TP"], counts["FP"], counts["FN"]
    p = tp/(tp+fp) if tp+fp else 0.0
    r = tp/(tp+fn) if tp+fn else 0.0
    f1 = 2*p*r/(p+r) if p+r else 0.0
    return {**counts, "precision": p, "recall": r, "f1": f1}

def ap_at_iou(image_ids, gt, predictions, class_name, iou_thr):
    """101-point interpolated AP at one IoU threshold; not COCO mAP@[.50:.95]."""
    n_gt = sum(len(gt[i].get(class_name, [])) for i in image_ids)
    if n_gt == 0:
        return None
    ranked = [(d["score"], i, d["bbox"])
              for i in image_ids for d in predictions.get(i, [])
              if d["class"] == class_name]
    ranked.sort(key=lambda item: item[0], reverse=True)
    matched = {i: set() for i in image_ids}
    tp_flags, fp_flags = [], []
    for _, image_id, box in ranked:
        targets = gt[image_id].get(class_name, [])
        overlaps = [(iou(box, target), idx) for idx, target in enumerate(targets)
                    if idx not in matched[image_id]]
        ov, best = max(overlaps, default=(0.0, -1), key=lambda pair: pair[0])
        if best >= 0 and ov >= iou_thr:
            matched[image_id].add(best)
            tp_flags.append(1); fp_flags.append(0)
        else:
            tp_flags.append(0); fp_flags.append(1)
    if not ranked:
        return 0.0
    cumul_tp = cumul_fp = 0
    recalls, precisions = [], []
    for t, f in zip(tp_flags, fp_flags):
        cumul_tp += t; cumul_fp += f
        recalls.append(cumul_tp/n_gt)
        precisions.append(cumul_tp/max(cumul_tp+cumul_fp, 1))
    for idx in range(len(precisions)-2, -1, -1):
        precisions[idx] = max(precisions[idx], precisions[idx+1])
    return sum(max((p for r,p in zip(recalls, precisions) if r >= level), default=0.0)
               for level in (i/100 for i in range(101))) / 101.0

def image_level(image_ids, gt, predictions, threshold):
    tp = fp = tn = fn = 0
    for image_id in image_ids:
        actual = bool(gt[image_id].get("Crack", []))
        predicted = any(d["class"] == "Crack" and d["score"] >= threshold
                        for d in predictions.get(image_id, []))
        if actual and predicted: tp += 1
        elif not actual and predicted: fp += 1
        elif not actual and not predicted: tn += 1
        else: fn += 1
    accuracy = (tp+tn)/max(tp+fp+tn+fn, 1)
    precision = tp/(tp+fp) if tp+fp else 0.0
    recall = tp/(tp+fn) if tp+fn else 0.0
    specificity = tn/(tn+fp) if tn+fp else 0.0
    f1 = 2*precision*recall/(precision+recall) if precision+recall else 0.0
    return {"TP_images":tp, "FP_images":fp, "TN_images":tn, "FN_images":fn,
            "accuracy":accuracy, "precision":precision, "recall":recall,
            "specificity":specificity, "f1":f1,
            "definition":"Actual positive if any annotated crack exists; predicted positive if at least one predicted crack box meets threshold."}

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs/roadguard_swin_infer.py")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--ann", default="data/rdd2022/annotations/instances_val.json")
    p.add_argument("--images-dir", default="data/rdd2022/images")
    p.add_argument("--thresholds", nargs="+", type=float,
                   default=[0.10,0.20,0.30,0.40,0.50,0.60,0.70,0.80])
    p.add_argument("--iou-thr", type=float, default=0.50)
    p.add_argument("--min-score", type=float, default=0.001)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--out", default="results/crack_only_validation.json")
    args = p.parse_args()
    from mmengine import Config
    from mmdet.apis import inference_detector, init_detector
    from mmdet.utils import register_all_modules

    payload = json.loads(Path(args.ann).read_text(encoding="utf-8"))
    category_names = {int(c["id"]): str(c["name"]) for c in payload.get("categories", [])}
    info = {int(item["id"]): item for item in payload.get("images", [])}
    image_ids = list(info)
    gt = defaultdict(lambda: defaultdict(list))
    for ann in payload.get("annotations", []):
        if ann.get("iscrowd", 0):
            continue
        name = category_names.get(int(ann["category_id"]))
        if name:
            gt[int(ann["image_id"])][as_target(name)].append(to_xyxy(ann["bbox"]))

    cfg = Config.fromfile(args.config)
    try:
        cfg.model.test_cfg.rcnn.score_thr = min(args.min_score, min(args.thresholds))
    except Exception:
        pass
    register_all_modules(init_default_scope=True)
    model = init_detector(cfg, args.checkpoint, device=args.device)
    model.dataset_meta["classes"] = MODEL_CLASSES
    images_dir = Path(args.images_dir)
    predictions, missing = {}, []
    for pos, image_id in enumerate(image_ids, 1):
        filename = str(info[image_id]["file_name"])
        path = images_dir / filename
        if not path.is_file():
            path = images_dir / Path(filename).name
        if not path.is_file():
            missing.append(filename)
            predictions[image_id] = []
            continue
        result = inference_detector(model, str(path))
        instances = result.pred_instances.cpu()
        boxes, scores, labels = instances.bboxes.numpy(), instances.scores.numpy(), instances.labels.numpy()
        dets = []
        for box, score, label in zip(boxes, scores, labels):
            score, label = float(score), int(label)
            if score < args.min_score or not 0 <= label < len(MODEL_CLASSES):
                continue
            dets.append({"class":"Pothole" if label == 3 else "Crack",
                         "subtype":MODEL_CLASSES[label], "score":score,
                         "bbox":[float(v) for v in box.tolist()]})
        predictions[image_id] = dets
        if pos % 100 == 0 or pos == len(image_ids):
            print(f"Inferred {pos}/{len(image_ids)} images", flush=True)

    rows = []
    for threshold in sorted(set(args.thresholds)):
        rows.append({
            "threshold": threshold,
            "crack_object_level": add_prf(match_counts(image_ids,gt,predictions,"Crack",threshold,args.iou_thr)),
            "pothole_object_level": add_prf(match_counts(image_ids,gt,predictions,"Pothole",threshold,args.iou_thr)),
            "crack_image_level": image_level(image_ids,gt,predictions,threshold),
        })
    ap = {name: ap_at_iou(image_ids,gt,predictions,name,args.iou_thr)
          for name in ("Crack","Pothole")}
    best = max(rows, key=lambda r:(r["crack_object_level"]["f1"],r["crack_object_level"]["recall"]))
    report = {
        "task":"unified crack detection (D00/D10/D20 merged); pothole separate",
        "checkpoint":str(Path(args.checkpoint).resolve()),
        "annotation_file":str(Path(args.ann).resolve()),
        "images_in_annotations":len(image_ids),
        "images_inferred":len(image_ids)-len(missing),
        "missing_images":missing,
        "iou_threshold":args.iou_thr,
        "ap_at_iou_threshold":ap,
        "ap_note":"101-point interpolated AP at IoU=0.50; not COCO mAP@[.50:.95].",
        "threshold_sweep":rows,
        "best_crack_f1_threshold_among_tested_thresholds":best["threshold"],
        "selection_note":"Choose the threshold from validation; freeze it before testing.",
        "image_level_accuracy_note":"Image-level binary accuracy is separate from bounding-box accuracy and can be inflated by many crack-negative images."
    }
    out = Path(args.out); out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(report,indent=2),encoding="utf-8")
    csv_path = out.with_suffix(".csv")
    with csv_path.open("w",newline="",encoding="utf-8") as f:
        fields = ["threshold","crack_TP","crack_FP","crack_FN","crack_precision","crack_recall","crack_f1",
                  "crack_image_accuracy","crack_image_precision","crack_image_recall","crack_image_specificity","crack_image_f1",
                  "pothole_precision","pothole_recall","pothole_f1"]
        writer = csv.DictWriter(f,fieldnames=fields); writer.writeheader()
        for row in rows:
            c, im, pothole = row["crack_object_level"],row["crack_image_level"],row["pothole_object_level"]
            writer.writerow({"threshold":row["threshold"],"crack_TP":c["TP"],"crack_FP":c["FP"],"crack_FN":c["FN"],
                "crack_precision":c["precision"],"crack_recall":c["recall"],"crack_f1":c["f1"],
                "crack_image_accuracy":im["accuracy"],"crack_image_precision":im["precision"],
                "crack_image_recall":im["recall"],"crack_image_specificity":im["specificity"],"crack_image_f1":im["f1"],
                "pothole_precision":pothole["precision"],"pothole_recall":pothole["recall"],"pothole_f1":pothole["f1"]})
    print(json.dumps({"ap_at_iou_threshold":ap,"best_crack_f1_threshold":best["threshold"],
        "thresholds":[{"threshold":r["threshold"],"crack_object":r["crack_object_level"],"crack_image":r["crack_image_level"]} for r in rows],
        "json_output":str(out),"csv_output":str(csv_path),"missing_images":len(missing)},indent=2))

if __name__ == "__main__":
    main()
