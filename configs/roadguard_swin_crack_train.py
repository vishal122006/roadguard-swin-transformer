"""Training config for RoadGuard's two-class Swin-T/FPN/Faster R-CNN experiment.

Dataset paths and class metadata are patched by scripts/fine_tune_crack_colab.py.
The base file supplies the checkpoint-compatible detector architecture.
"""
_base_ = "./roadguard_swin_infer.py"

default_scope = "mmdet"
dataset_type = "CocoDataset"
classes = ("Crack", "Pothole")

train_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="LoadAnnotations", with_bbox=True),
    dict(type="Resize", scale=(1333, 800), keep_ratio=True),
    dict(type="RandomFlip", prob=0.5),
    dict(type="PackDetInputs"),
]

test_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="Resize", scale=(1333, 800), keep_ratio=True),
    dict(type="LoadAnnotations", with_bbox=True),
    dict(type="PackDetInputs"),
]

train_dataloader = dict(
    batch_size=2,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type="DefaultSampler", shuffle=True),
    batch_sampler=dict(type="AspectRatioBatchSampler"),
    dataset=dict(
        type=dataset_type,
        data_root="",
        ann_file="",
        data_prefix=dict(img=""),
        metainfo=dict(classes=classes),
        filter_cfg=dict(filter_empty_gt=False, min_size=32),
        pipeline=train_pipeline,
    ),
)

val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type="DefaultSampler", shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root="",
        ann_file="",
        data_prefix=dict(img=""),
        metainfo=dict(classes=classes),
        test_mode=True,
        pipeline=test_pipeline,
    ),
)
test_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type="DefaultSampler", shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root="",
        ann_file="",
        data_prefix=dict(img=""),
        metainfo=dict(classes=classes),
        test_mode=True,
        pipeline=test_pipeline,
    ),
)

val_evaluator = dict(
    type="CocoMetric",
    ann_file="",
    metric="bbox",
    classwise=True,
)
test_evaluator = dict(
    type="CocoMetric",
    ann_file="",
    metric="bbox",
    classwise=True,
)

train_cfg = dict(type="EpochBasedTrainLoop", max_epochs=20, val_interval=1)
val_cfg = dict(type="ValLoop")
test_cfg = dict(type="TestLoop")

optim_wrapper = dict(
    type="OptimWrapper",
    optimizer=dict(type="AdamW", lr=2e-5, betas=(0.9, 0.999), weight_decay=0.05),
    clip_grad=dict(max_norm=1.0, norm_type=2),
)

param_scheduler = [
    dict(type="LinearLR", start_factor=0.001, by_epoch=False, begin=0, end=500),
    dict(type="MultiStepLR", begin=0, end=20, by_epoch=True, milestones=[14, 18], gamma=0.1),
]

default_hooks = dict(
    timer=dict(type="IterTimerHook"),
    logger=dict(type="LoggerHook", interval=50),
    param_scheduler=dict(type="ParamSchedulerHook"),
    checkpoint=dict(
        type="CheckpointHook",
        interval=1,
        max_keep_ckpts=3,
        save_best="coco/bbox_mAP",
        rule="greater",
        save_last=True,
    ),
    sampler_seed=dict(type="DistSamplerSeedHook"),
)
log_processor = dict(type="LogProcessor", window_size=50, by_epoch=True)
visualizer = dict(type="DetLocalVisualizer", vis_backends=[], name="visualizer")
env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method="fork", opencv_num_threads=0),
    dist_cfg=dict(backend="nccl"),
)
launcher = "none"
work_dir = "./work_dirs/roadguard_swin_crack_binary"
