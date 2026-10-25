#!/usr/bin/env python3
"""Fine-tune a two-class Crack/Pothole Swin detector from the four-class checkpoint.

The classifier/regressor outputs are initialized by merging D00/D10/D20 weights
for Crack, copying D40 outputs to Pothole, and preserving background class logits.
The original checkpoint is never overwritten.
"""
from __future__ import annotations
import argparse, json, os
from pathlib import Path

def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True, help="Original four-class MMDetection training config")
    p.add_argument("--checkpoint", required=True, help="Original four-class checkpoint")
    p.add_argument("--data-root", required=True, help="RDD2022 root containing images/")
    p.add_argument("--annotation-dir", required=True, help="Folder with binary train/validation JSON files")
    p.add_argument("--work-dir", required=True, help="New output directory, preferably on Google Drive")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--val-interval", type=int, default=1)
    p.add_argument("--max-keep-ckpts", type=int, default=3)
    p.add_argument("--no-photometric-augmentation", action="store_true")
    p.add_argument("--resume", action="store_true")
    return p.parse_args()

def walk_dicts(node):
    if isinstance(node, dict):
        yield node
        for value in list(node.values()):
            if isinstance(value, (dict,list)):
                yield from walk_dicts(value)
    elif isinstance(node, list):
        for value in node:
            if isinstance(value,(dict,list)):
                yield from walk_dicts(value)

def patch_dataset(dataset_cfg, annotation_file, data_root):
    # Our generated COCO file_name values already include images/<split>/..., so
    # use an empty img prefix to avoid images/images/... path duplication.
    ann_data = json.loads(Path(annotation_file).read_text(encoding="utf-8"))
    first_name = next((str(row.get("file_name", "")) for row in ann_data.get("images", [])), "")
    img_prefix = "" if first_name.startswith("images/") else "images/"
    leaves = 0
    for node in walk_dicts(dataset_cfg):
        typ = str(node.get("type",""))
        if typ == "CocoDataset":
            node["ann_file"] = str(annotation_file)
            node["data_root"] = str(data_root) + os.sep
            prefix = dict(node.get("data_prefix",{}))
            prefix["img"] = img_prefix
            node["data_prefix"] = prefix
            node["metainfo"] = dict(classes=("Crack","Pothole"))
            leaves += 1
    if leaves == 0:
        raise RuntimeError("No CocoDataset leaf was found in the supplied training config. Inspect dataset wrappers.")
    return leaves

def add_photometric_augmentation(dataset_cfg):
    for node in walk_dicts(dataset_cfg):
        if str(node.get("type","")) != "CocoDataset" or not isinstance(node.get("pipeline"),list):
            continue
        pipeline = [dict(step) for step in node["pipeline"]]
        types = [step.get("type") for step in pipeline]
        if "PhotoMetricDistortion" not in types and "LoadAnnotations" in types:
            pipeline.insert(types.index("LoadAnnotations")+1,dict(type="PhotoMetricDistortion"))
            node["pipeline"] = pipeline

def merge_checkpoint(source, destination):
    import torch
    checkpoint = torch.load(str(source),map_location="cpu")
    state = checkpoint.get("state_dict",checkpoint)
    keys = {
        "cw": next((k for k in state if k.endswith("roi_head.bbox_head.fc_cls.weight")),None),
        "cb": next((k for k in state if k.endswith("roi_head.bbox_head.fc_cls.bias")),None),
        "rw": next((k for k in state if k.endswith("roi_head.bbox_head.fc_reg.weight")),None),
        "rb": next((k for k in state if k.endswith("roi_head.bbox_head.fc_reg.bias")),None),
    }
    if not all(keys.values()):
        raise KeyError("Checkpoint is missing expected Faster R-CNN bbox_head fc_cls/fc_reg parameters.")
    cw,cb,rw,rb = (state[keys[k]] for k in ("cw","cb","rw","rb"))
    if cw.shape[0] != 5 or cb.shape[0] != 5:
        raise ValueError(f"Expected 4-class classifier output with background (5 rows); got {cw.shape} and {cb.shape}.")
    if rw.shape[0] != 16 or rb.shape[0] != 16:
        raise ValueError(f"Expected class-specific 4-class box regression (16 rows); got {rw.shape} and {rb.shape}.")
    # Original outputs: D00, D10, D20, D40, background.
    # New outputs: Crack, Pothole, background.
    state[keys["cw"]] = torch.stack([cw[0:3].mean(0),cw[3],cw[4]],dim=0)
    state[keys["cb"]] = torch.stack([cb[0:3].mean(0),cb[3],cb[4]],dim=0)
    reg_w = rw.reshape(4,4,-1)
    reg_b = rb.reshape(4,4)
    state[keys["rw"]] = torch.stack([reg_w[0:3].mean(0),reg_w[3]],dim=0).reshape(8,-1)
    state[keys["rb"]] = torch.stack([reg_b[0:3].mean(0),reg_b[3]],dim=0).reshape(8)
    if "state_dict" in checkpoint:
        checkpoint["state_dict"] = state
        if not isinstance(checkpoint.get("meta"),dict):
            checkpoint["meta"] = {}
        checkpoint["meta"]["roadguard_class_merge"] = {
            "old":["D00_Longitudinal_Crack","D10_Transverse_Crack","D20_Alligator_Crack","D40_Pothole"],
            "new":["Crack","Pothole"],
            "method":"mean of three crack outputs; copied pothole and retained background classifier output"
        }
    else:
        checkpoint = state
    destination.parent.mkdir(parents=True,exist_ok=True)
    torch.save(checkpoint,str(destination))
    print(f"Saved merged two-class initialization: {destination}",flush=True)

def main():
    args = parse_args()
    config_path = Path(args.config).expanduser().resolve()
    source_checkpoint = Path(args.checkpoint).expanduser().resolve()
    data_root = Path(args.data_root).expanduser().resolve()
    ann_dir = Path(args.annotation_dir).expanduser().resolve()
    work_dir = Path(args.work_dir).expanduser().resolve()
    # Accept the filenames produced by the Colab COCO preparation cell and the
    # repository conversion script, without requiring the user to rename files.
    def annotation_path(split):
        candidates = [ann_dir / f"instances_{split}_crack_binary.json", ann_dir / f"instances_{split}.json"]
        return next((p for p in candidates if p.is_file()), candidates[0])
    train_ann = annotation_path("train")
    val_ann = annotation_path("val")
    test_ann = annotation_path("test")
    for path in (config_path,source_checkpoint,train_ann,val_ann):
        if not path.is_file():
            raise FileNotFoundError(path)
    if not (data_root/"images").is_dir():
        raise FileNotFoundError(f"Expected image directory {data_root/'images'}")
    if args.epochs < 5 or args.lr <= 0 or args.val_interval < 1:
        raise ValueError("epochs must be >= 5 and learning rate/validation interval must be positive")
    latest = work_dir/"latest.pth"
    if args.resume:
        if not latest.is_file():
            raise FileNotFoundError(f"Resume was requested but {latest} is missing.")
    elif work_dir.exists() and any(work_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty; use a new path or pass --resume: {work_dir}")

    os.environ.setdefault("MPLBACKEND","Agg")
    from mmengine import Config
    from mmengine.runner import Runner
    from mmdet.utils import register_all_modules
    register_all_modules(init_default_scope=True)
    cfg = Config.fromfile(str(config_path))
    cfg.work_dir = str(work_dir)
    cfg.model.roi_head.bbox_head.num_classes = 2
    # The full detector checkpoint is the initialization source; avoid unnecessary external backbone downloads.
    if cfg.get("model") and cfg.model.get("backbone"):
        cfg.model.backbone.init_cfg = None
    cfg.metainfo = dict(classes=("Crack","Pothole"))
    patch_dataset(cfg.train_dataloader.dataset,train_ann,data_root)
    if not args.no_photometric_augmentation:
        add_photometric_augmentation(cfg.train_dataloader.dataset)
    patch_dataset(cfg.val_dataloader.dataset,val_ann,data_root)
    if cfg.get("test_dataloader") and test_ann.is_file():
        patch_dataset(cfg.test_dataloader.dataset,test_ann,data_root)

    for key in ("val_evaluator","test_evaluator"):
        evaluator = cfg.get(key)
        if evaluator:
            for node in walk_dicts(evaluator):
                if "ann_file" in node:
                    if key == "val_evaluator":
                        node["ann_file"] = str(val_ann)
                    elif test_ann.is_file():
                        node["ann_file"] = str(test_ann)

    cfg.train_cfg.max_epochs = args.epochs
    cfg.train_cfg.val_interval = args.val_interval
    cfg.optim_wrapper.optimizer.lr = args.lr
    schedulers = list(cfg.get("param_scheduler",[]))
    for scheduler in schedulers:
        if scheduler.get("type") == "MultiStepLR" and scheduler.get("by_epoch",False):
            milestones = sorted(set(m for m in
                (max(1,int(args.epochs*0.67)),max(2,int(args.epochs*0.87))) if m < args.epochs))
            scheduler.begin, scheduler.end, scheduler.milestones = 0,args.epochs,milestones
    cfg.param_scheduler = schedulers
    if "default_hooks" not in cfg:
        cfg.default_hooks = {}
    if "checkpoint" not in cfg.default_hooks:
        cfg.default_hooks.checkpoint = dict(type="CheckpointHook")
    cfg.default_hooks.checkpoint.interval = 1
    cfg.default_hooks.checkpoint.max_keep_ckpts = args.max_keep_ckpts
    cfg.default_hooks.checkpoint.save_best = "coco/bbox_mAP"
    cfg.default_hooks.checkpoint.rule = "greater"
    cfg.default_hooks.checkpoint.save_last = True

    adapted = work_dir/"initial_crack_merged.pth"
    if args.resume:
        cfg.resume = True
        cfg.load_from = None
    else:
        merge_checkpoint(source_checkpoint,adapted)
        cfg.resume = False
        cfg.load_from = str(adapted)
    work_dir.mkdir(parents=True,exist_ok=True)
    print("="*70)
    print("ROADGUARD SWIN — CRACK/POTHOLE FINE-TUNING")
    print(f"Config: {config_path}")
    print(f"Original checkpoint (unchanged): {source_checkpoint}")
    print(f"Data root: {data_root}")
    print(f"Annotations: {ann_dir}")
    print(f"Work directory: {work_dir}")
    print(f"Epochs: {args.epochs}; LR: {args.lr}; validation every epoch: {args.val_interval}")
    print("Model selection: validation COCO bbox mAP. Test split does not select checkpoints.")
    print("="*70,flush=True)
    runner = Runner.from_cfg(cfg)
    runner.train()
    candidates = list(work_dir.glob("best_coco_bbox_mAP_epoch_*.pth")) or list(work_dir.glob("best_*.pth"))
    if not candidates:
        raise RuntimeError("No best checkpoint written. Inspect validation metric key and training log.")
    best = max(candidates,key=lambda x:x.stat().st_mtime)
    (work_dir/".training_complete").write_text(json.dumps({
        "best_checkpoint":best.name,"epochs_requested":args.epochs
    },indent=2),encoding="utf-8")
    print(f"Training finished; best validation checkpoint: {best}",flush=True)

if __name__ == "__main__":
    main()
