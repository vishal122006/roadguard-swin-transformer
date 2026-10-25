#!/usr/bin/env python3
"""Validation-driven accuracy fine-tuning for RoadGuard Swin in Colab.

Starts from an existing detector checkpoint, preserves the original weights,
uses class-balanced sampling plus label-safe photometric augmentation, and
selects the best checkpoint using validation COCO bbox mAP. It never selects
a checkpoint using test-set metrics.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True, help="MMDetection training config")
    p.add_argument("--checkpoint", required=True, help="Existing trained .pth checkpoint")
    p.add_argument("--work-dir", required=True, help="NEW output directory on Google Drive")
    p.add_argument("--epochs", type=int, default=24, help="Fine-tuning epochs (default: 24)")
    p.add_argument("--lr", type=float, default=2e-5, help="Fine-tuning learning rate")
    p.add_argument("--val-interval", type=int, default=1, help="Validate every N epochs")
    p.add_argument("--max-keep-ckpts", type=int, default=3)
    p.add_argument("--oversample-thr", type=float, default=0.03,
                   help="ClassBalancedDataset oversampling threshold; set 0 to disable")
    p.add_argument("--no-photometric-augmentation", action="store_true",
                   help="Disable PhotoMetricDistortion if it is unsuitable for the dataset")
    p.add_argument("--resume", action="store_true", help="Resume this same fine-tuning run from work-dir/latest.pth")
    return p.parse_args()


def _as_dict(config):
    return dict(config)


def audit_counts(annotation_file: Path):
    if not annotation_file.is_file():
        print(f"Dataset audit skipped; annotation file not found: {annotation_file}")
        return
    try:
        data = json.loads(annotation_file.read_text(encoding="utf-8"))
        categories = {c["id"]: c["name"] for c in data.get("categories", [])}
        counts = {name: 0 for name in categories.values()}
        ignored = 0
        for ann in data.get("annotations", []):
            if ann.get("iscrowd", 0):
                ignored += 1
            else:
                name = categories.get(ann.get("category_id"))
                if name is not None:
                    counts[name] += 1
        print(f"Training annotation audit ({annotation_file}):")
        print(f"  images={len(data.get('images', []))}, known_boxes={sum(counts.values())}, ignored_regions={ignored}")
        for name, count in counts.items():
            print(f"  {name}: {count}")
        rare = [name for name, count in counts.items() if count < 100]
        if rare:
            print("WARNING: very low training support for:", ", ".join(rare))
            print("Class balancing helps sampling but cannot replace missing/verified annotations.")
    except Exception as exc:
        print(f"WARNING: could not audit annotations: {type(exc).__name__}: {exc}")


def _apply_accuracy_recipe(cfg, args):
    # Disable external ImageNet initialization because the requested detector
    # checkpoint supplies the starting weights for the full network.
    if "model" in cfg and "backbone" in cfg.model:
        cfg.model.backbone.init_cfg = None

    dataset_cfg = cfg.train_dataloader.dataset
    if "pipeline" not in dataset_cfg:
        raise KeyError("Expected train_dataloader.dataset.pipeline in the supplied config.")

    pipeline = [dict(step) for step in dataset_cfg.pipeline]
    types = [step.get("type") for step in pipeline]
    if not args.no_photometric_augmentation and "PhotoMetricDistortion" not in types:
        load_ann_idx = next(
            (i for i, step in enumerate(pipeline) if step.get("type") == "LoadAnnotations"),
            None,
        )
        if load_ann_idx is None:
            raise RuntimeError("Cannot add PhotoMetricDistortion: LoadAnnotations was not found.")
        pipeline.insert(load_ann_idx + 1, dict(type="PhotoMetricDistortion"))
        print("Accuracy recipe: enabled PhotoMetricDistortion (bbox-safe color/brightness changes).")
    dataset_cfg.pipeline = pipeline

    if args.oversample_thr > 0:
        if dataset_cfg.get("type") == "ClassBalancedDataset":
            print("Dataset is already class-balanced; leaving its wrapper unchanged.")
        else:
            base_dataset = _as_dict(dataset_cfg)
            cfg.train_dataloader.dataset = dict(
                type="ClassBalancedDataset",
                oversample_thr=float(args.oversample_thr),
                dataset=base_dataset,
            )
            print(f"Accuracy recipe: ClassBalancedDataset oversample_thr={args.oversample_thr:g}.")
    else:
        print("Class-balanced sampling disabled by request.")

    # Keep the model's original warm-up, but schedule the two LR drops relative
    # to this fine-tuning run. This avoids the old bug where 15 epochs ended
    # before the original epoch-16/22 milestones ever ran.
    schedulers = list(cfg.get("param_scheduler", []))
    for scheduler in schedulers:
        if scheduler.get("type") == "MultiStepLR" and scheduler.get("by_epoch", False):
            first = max(1, int(round(args.epochs * 0.67)))
            second = max(first + 1, int(round(args.epochs * 0.87)))
            milestones = sorted(set(m for m in (first, second) if m < args.epochs))
            scheduler.begin = 0
            scheduler.end = args.epochs
            scheduler.milestones = milestones
            scheduler.gamma = float(scheduler.get("gamma", 0.1))
            print(f"Accuracy recipe: MultiStepLR milestones={milestones}, epochs={args.epochs}.")
    cfg.param_scheduler = schedulers

    cfg.train_cfg.max_epochs = int(args.epochs)
    cfg.train_cfg.val_interval = int(args.val_interval)

    if "optim_wrapper" not in cfg or "optimizer" not in cfg.optim_wrapper:
        raise KeyError("Expected cfg.optim_wrapper.optimizer in training config.")
    cfg.optim_wrapper.optimizer.lr = float(args.lr)


def main():
    args = parse_args()
    config_path = Path(args.config).expanduser().resolve()
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    work_dir = Path(args.work_dir).expanduser().resolve()

    if not config_path.is_file():
        raise FileNotFoundError(f"Training config not found: {config_path}")
    if not checkpoint_path.is_file() or checkpoint_path.stat().st_size < 10_000_000:
        raise FileNotFoundError(f"Starting checkpoint missing or too small: {checkpoint_path}")
    if args.epochs < 5 or args.val_interval < 1 or args.lr <= 0:
        raise ValueError("For a meaningful run, epochs must be >= 5; val-interval and lr must be positive.")
    if args.oversample_thr < 0:
        raise ValueError("--oversample-thr cannot be negative.")
    latest_checkpoint = work_dir / "latest.pth"
    if args.resume:
        if (work_dir / ".training_complete").is_file():
            raise RuntimeError(f"This run is already marked complete: {work_dir}. Choose a new work-dir for a new experiment.")
        if not latest_checkpoint.is_file():
            raise FileNotFoundError(f"Cannot resume: latest checkpoint missing: {latest_checkpoint}")
    elif work_dir.exists() and any(work_dir.iterdir()):
        raise FileExistsError(
            f"Output directory is not empty: {work_dir}. Choose a NEW work-dir "
            "for a new experiment, or pass --resume to continue this exact run."
        )

    os.environ.setdefault("MPLBACKEND", "Agg")
    from mmengine import Config
    from mmengine.runner import Runner
    from mmdet.utils import register_all_modules

    register_all_modules(init_default_scope=True)
    cfg = Config.fromfile(str(config_path))
    cfg.work_dir = str(work_dir)
    if args.resume:
        cfg.resume = True
        cfg.load_from = None
        print(f"Resuming existing fine-tuning run from: {latest_checkpoint}")
    else:
        cfg.load_from = str(checkpoint_path)
        cfg.resume = False

    # Print the current dataset statistics before making any training changes.
    audit_counts(config_path.parent.parent / "data/rdd2022/annotations/instances_train.json")
    _apply_accuracy_recipe(cfg, args)

    if "default_hooks" not in cfg:
        cfg.default_hooks = {}
    if "checkpoint" not in cfg.default_hooks:
        cfg.default_hooks.checkpoint = dict(type="CheckpointHook")
    cfg.default_hooks.checkpoint.interval = 1
    cfg.default_hooks.checkpoint.max_keep_ckpts = int(args.max_keep_ckpts)
    cfg.default_hooks.checkpoint.save_best = "coco/bbox_mAP"
    cfg.default_hooks.checkpoint.rule = "greater"
    cfg.default_hooks.checkpoint.save_last = True
    work_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 76)
    print("ROADGUARD SWIN — ACCURACY FINE-TUNING")
    print(f"Config:              {config_path}")
    print(f"Starting checkpoint: {checkpoint_path}")
    print(f"Output directory:    {work_dir}")
    print(f"Epochs:              {args.epochs}")
    print(f"Learning rate:       {args.lr:g}")
    print("Selection metric:    validation COCO bbox mAP")
    print("Original checkpoint: unchanged")
    print("Resume mode:         " + str(args.resume))
    print("Test split:          not used for checkpoint selection")
    print("=" * 76)

    runner = Runner.from_cfg(cfg)
    runner.train()

    candidates = sorted(work_dir.glob("best_coco_bbox_mAP_epoch_*.pth"))
    if not candidates:
        raise RuntimeError(
            "Training ended but no best_coco_bbox_mAP checkpoint was written. "
            "Inspect validation logs for the exact metric key and checkpoint-hook errors."
        )
    best = max(candidates, key=lambda p: p.stat().st_mtime)
    print(f"Best validation-selected checkpoint: {best}")
    print(f"Checkpoint size: {best.stat().st_size / 1024**2:.1f} MB")
    (work_dir / ".training_complete").write_text(
        f"completed; best_checkpoint={best.name}\\n", encoding="utf-8"
    )
    print(f"Training run finished. All artifacts: {work_dir}")


if __name__ == "__main__":
    main()
