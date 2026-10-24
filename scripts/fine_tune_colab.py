#!/usr/bin/env python3
"""Fine-tune the existing RoadGuard MMDetection checkpoint in Colab.

This script starts a NEW fine-tuning run from the trained epoch-5 weights.
It does not resume the old optimizer/scheduler state and does not modify the
original checkpoint. Validation mAP is used to select the best new checkpoint.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True, help="Original MMDetection training config")
    p.add_argument("--checkpoint", required=True, help="Existing trained .pth checkpoint")
    p.add_argument("--work-dir", required=True, help="NEW output directory; use Google Drive")
    p.add_argument("--epochs", type=int, default=15, help="Fine-tuning epochs (default: 15)")
    p.add_argument("--lr", type=float, default=2e-5, help="Fine-tuning learning rate (default: 2e-5)")
    p.add_argument("--val-interval", type=int, default=1, help="Validate every N epochs")
    p.add_argument("--max-keep-ckpts", type=int, default=3, help="Keep this many recent checkpoints")
    return p.parse_args()


def main():
    args = parse_args()
    config_path = Path(args.config).expanduser().resolve()
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    work_dir = Path(args.work_dir).expanduser().resolve()

    if not config_path.is_file():
        raise FileNotFoundError(f"Training config not found: {config_path}")
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Starting checkpoint not found: {checkpoint_path}")
    if checkpoint_path.stat().st_size < 10 * 1024 * 1024:
        raise ValueError(f"Checkpoint is unexpectedly small: {checkpoint_path}")
    if args.epochs < 1 or args.val_interval < 1 or args.lr <= 0:
        raise ValueError("epochs, val-interval, and lr must be positive")

    # Avoid the Colab inline matplotlib backend inside the training subprocess.
    os.environ.setdefault("MPLBACKEND", "Agg")

    from mmengine import Config
    from mmengine.runner import Runner
    from mmdet.utils import register_all_modules

    register_all_modules(init_default_scope=True)
    cfg = Config.fromfile(str(config_path))

    # Start a new fine-tuning run from model weights, not the old optimizer state.
    cfg.load_from = str(checkpoint_path)
    cfg.resume = False
    cfg.work_dir = str(work_dir)

    if "train_cfg" not in cfg:
        raise KeyError("Training config has no train_cfg.")
    cfg.train_cfg.max_epochs = int(args.epochs)
    cfg.train_cfg.val_interval = int(args.val_interval)

    # Lower LR for fine-tuning the already-trained detector.
    if "optim_wrapper" not in cfg or "optimizer" not in cfg.optim_wrapper:
        raise KeyError("Expected cfg.optim_wrapper.optimizer in training config.")
    cfg.optim_wrapper.optimizer.lr = float(args.lr)

    # Select the best checkpoint using validation COCO bbox mAP.
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

    print("=" * 72)
    print("ROADGUARD SWIN — FINE-TUNING RUN")
    print("=" * 72)
    print(f"Config:             {config_path}")
    print(f"Starting checkpoint:{checkpoint_path}")
    print(f"Output directory:   {work_dir}")
    print(f"Fine-tune epochs:   {args.epochs}")
    print(f"Learning rate:      {args.lr:g}")
    print(f"Validation interval:{args.val_interval}")
    print("Best checkpoint:    selected by validation coco/bbox_mAP")
    print("Original checkpoint will not be modified.")
    print("=" * 72)

    runner = Runner.from_cfg(cfg)
    runner.train()

    best = work_dir / "best_coco_bbox_mAP_epoch_1.pth"
    best_candidates = sorted(work_dir.glob("best_coco_bbox_mAP_epoch_*.pth"))
    if best_candidates:
        best = max(best_candidates, key=lambda p: p.stat().st_mtime)
        print(f"Best-checkpoint file detected: {best}")
        print(f"Size: {best.stat().st_size / 1024**2:.1f} MB")
    else:
        print(
            "WARNING: No best_coco_bbox_mAP_epoch_*.pth found. "
            "Inspect the training log and checkpoint hook metric name."
        )
    print(f"Training run finished. All artifacts are in: {work_dir}")


if __name__ == "__main__":
    main()
