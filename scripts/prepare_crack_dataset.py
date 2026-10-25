#!/usr/bin/env python3
"""Convert RDD2022 four-class COCO annotations to Crack + Pothole JSON."""
from __future__ import annotations
import argparse, json, re
from collections import Counter
from pathlib import Path

CRACK_CODES = {"D00", "D10", "D20"}

def rdd_code(name: str) -> str:
    upper = name.upper()
    match = re.search(r"D(?:00|10|20|40)", upper)
    if match:
        return match.group(0)
    for token, code in (("LONGITUDINAL","D00"),("TRANSVERSE","D10"),
                        ("ALLIGATOR","D20"),("POTHOLE","D40")):
        if token in upper:
            return code
    raise ValueError(f"Cannot map COCO category name to an RDD2022 class: {name!r}")

def convert(source: Path, destination: Path):
    data = json.loads(source.read_text(encoding="utf-8"))
    categories = {int(c["id"]): str(c["name"]) for c in data.get("categories", [])}
    old_to_new = {old_id: (1 if rdd_code(name) in CRACK_CODES else 2)
                  for old_id, name in categories.items()}
    converted = dict(data)
    converted["categories"] = [
        {"id": 1, "name": "Crack", "supercategory": "road_damage"},
        {"id": 2, "name": "Pothole", "supercategory": "road_damage"},
    ]
    annotations, counts, crack_images = [], Counter(), set()
    image_ids = {int(i["id"]) for i in data.get("images", [])}
    for ann in data.get("annotations", []):
        old_cat = int(ann["category_id"])
        if old_cat not in old_to_new:
            raise ValueError(f"Annotation references missing category_id={old_cat}")
        row = dict(ann)
        row["category_id"] = old_to_new[old_cat]
        annotations.append(row)
        label = "Crack" if row["category_id"] == 1 else "Pothole"
        counts[label] += 1
        if label == "Crack":
            crack_images.add(int(row["image_id"]))
    converted["annotations"] = annotations
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(converted, ensure_ascii=False), encoding="utf-8")
    return {"source": str(source), "output": str(destination),
            "images": len(data.get("images", [])), "annotations": len(annotations),
            "crack_boxes": counts["Crack"], "pothole_boxes": counts["Pothole"],
            "images_with_cracks": len(crack_images),
            "images_without_cracks": len(image_ids - crack_images)}

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train-ann", required=True)
    p.add_argument("--val-ann", required=True)
    p.add_argument("--test-ann", default="")
    p.add_argument("--out-dir", default="data/rdd2022_crack_binary/annotations")
    args = p.parse_args()
    out = Path(args.out_dir)
    sources = [("train", Path(args.train_ann)), ("val", Path(args.val_ann))]
    if args.test_ann:
        sources.append(("test", Path(args.test_ann)))
    results = []
    for split, source in sources:
        if not source.is_file():
            raise FileNotFoundError(source)
        results.append(convert(source, out / f"instances_{split}_crack_binary.json"))
    summary = out.parent / "conversion_summary.json"
    summary.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps({"splits": results, "summary": str(summary)}, indent=2))

if __name__ == "__main__":
    main()
