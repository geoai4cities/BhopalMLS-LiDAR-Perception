"""
Sonata Configuration for Kerala MLS - INSTANCE SEGMENTATION Full Fine-Tuning
PG-v1m2 + FT-9ch: 9-Channel Input (XYZ + RGB + normals)

FULL FINE-TUNING: enc_mode=False, freeze_encoder=False, entire model trains
Based on:
- Kerala semantic FT: sonata_140gb_12class_finetune_9ch.py
- Kerala Concerto instance FT: concerto_140gb_12class_insseg_ft_9ch.py

Instance Segmentation uses PG-v1m2 (PointGroup) which adds:
- bias_head: Predicts offset from each point to its instance centroid
- Ball query clustering at inference to group points into instances

Architecture (enc_mode=False, freeze_encoder=False):
  Encoder: 48->96->192->384->512 (TRAINABLE, pre-trained init)
  Decoder: 384->192->96->64 (TRAINABLE, random init)
  seg_head: 64 -> 12 classes (TRAINABLE)
  bias_head: 64 -> 3 offset (TRAINABLE)

Stuff classes (excluded from instance clustering):
  -1: padding, 0: Ground, 1: Vegetation, 3: Road, 4: Wire

GPU: H200 (143GB VRAM)
Estimated Memory: ~80-100GB (full model trainable)
"""

_base_ = ["../_base_/default_runtime.py"]

# Misc settings
batch_size = 16
batch_size_val = 1
batch_size_test = 1
num_worker = 16
num_worker_test = 1
mix_prob = 0.8
empty_cache = False
enable_amp = True
clip_grad = 3.0
evaluate = True  # REQUIRED for InsSegEvaluator

# HDF5 file path
H5_PATH = "data/Kerala/Kerala_scene_v1_corridor.h5"

# Pre-trained checkpoint
SONATA_CHECKPOINT = "sonata_pretrained/pretrain-sonata-v1m1-0-base.pth"

# Class configuration
num_classes = 12
segment_ignore_index = (-1, 0, 1, 3, 4)  # padding + Ground + Vegetation + Road + Wire (stuff classes)

class_names = [
    "Ground",                    # 0: seq 0-3 (4 subclasses)
    "Vegetation",                # 1: seq 4-7 (4 subclasses)
    "Building",                  # 2: seq 8-11 (4 subclasses)
    "Road",                      # 3: seq 12 (1 subclass)
    "Wire",                      # 4: seq 13 (1 subclass)
    "Pole and Tower",            # 5: seq 14-16 (3 subclasses)
    "Wire-Structure Connector",  # 6: seq 17-18 (2 subclasses)
    "Vehicle",                   # 7: seq 19-28 (10 subclasses)
    "Board",                     # 8: seq 29-31 (3 subclasses)
    "Fencing",                   # 9: seq 32-36 (5 subclasses)
    "Human and Animal",          # 10: seq 37-43 (7 subclasses)
    "Road Side Object",          # 11: seq 44-47 (4 subclasses)
]

class_groups = [
    list(range(0, 4)),      # 0: Ground
    list(range(4, 8)),      # 1: Vegetation
    list(range(8, 12)),     # 2: Building
    [12],                   # 3: Road
    [13],                   # 4: Wire
    list(range(14, 17)),    # 5: Pole and Tower
    list(range(17, 19)),    # 6: Wire-Structure Connector
    list(range(19, 29)),    # 7: Vehicle
    list(range(29, 32)),    # 8: Board
    list(range(32, 37)),    # 9: Fencing
    list(range(37, 44)),    # 10: Human and Animal
    list(range(44, 48)),    # 11: Road Side Object
]

class_weight = [
    2.03,     # 0: Ground (13.81%)
    1.00,     # 1: Vegetation (56.74%) — dominant class
    3.20,     # 2: Building (5.56%)
    1.71,     # 3: Road (19.41%)
    10.97,    # 4: Wire (0.47%)
    11.34,    # 5: Pole and Tower (0.44%)
    100.44,   # 6: Wire-Structure Connector (0.01%) — rarest class
    10.41,    # 7: Vehicle (0.52%)
    12.46,    # 8: Board (0.37%)
    4.74,     # 9: Fencing (2.52%)
    20.02,    # 10: Human and Animal (0.14%)
    76.34,    # 11: Road Side Object (0.01%)
]

# =============================================================================
# MODEL CONFIGURATION - PG-v1m2 (PointGroup) FULL FINE-TUNING
# =============================================================================
model = dict(
    type="PG-v1m2",
    backbone=dict(
        type="PT-v3m2",
        in_channels=9,
        order=("z", "z-trans", "hilbert", "hilbert-trans"),
        stride=(2, 2, 2, 2),
        # SONATA ENCODER (Official) - TRAINABLE
        enc_depths=(3, 3, 3, 12, 3),
        enc_channels=(48, 96, 192, 384, 512),
        enc_num_head=(3, 6, 12, 24, 32),
        enc_patch_size=(1024, 1024, 1024, 1024, 1024),
        # SONATA DECODER (Official) - TRAINABLE
        dec_depths=(2, 2, 2, 2),
        dec_channels=(64, 96, 192, 384),
        dec_num_head=(4, 6, 12, 24),
        dec_patch_size=(1024, 1024, 1024, 1024),
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        drop_path=0.3,
        shuffle_orders=True,
        pre_norm=True,
        enable_rpe=False,
        enable_flash=True,
        upcast_attention=False,
        upcast_softmax=False,
        traceable=False,
        mask_token=False,
        enc_mode=False,          # FULL FT: Use decoder
        freeze_encoder=False,    # FULL FT: Train encoder too
    ),
    backbone_out_channels=64,
    semantic_num_classes=num_classes,
    semantic_ignore_index=-1,
    segment_ignore_index=segment_ignore_index,
    instance_ignore_index=-1,
    cluster_thresh=1.5,
    cluster_closed_points=600,
    cluster_propose_points=200,
    cluster_min_points=50,
    voxel_size=0.01,  # MUST match grid_size
    criteria=[
        dict(type="CrossEntropyLoss", weight=class_weight, loss_weight=1.0, ignore_index=-1),
        dict(type="LovaszLoss", mode="multiclass", loss_weight=1.0, ignore_index=-1),
    ],
    freeze_backbone=False,  # FULL FT: Train everything
)

# =============================================================================
# SCHEDULER SETTINGS
# =============================================================================
epoch = 100
eval_epoch = 100
optimizer = dict(type="AdamW", lr=0.002, weight_decay=0.005)
scheduler = dict(
    type="OneCycleLR",
    max_lr=[0.002, 0.0002],
    pct_start=0.04,
    anneal_strategy="cos",
    div_factor=10.0,
    final_div_factor=100.0,
)
param_dicts = [dict(keyword="block", lr=0.0002)]

# =============================================================================
# DATASET CONFIGURATION
# =============================================================================
# PointClip — computed from Kerala_scene_v1_corridor.h5 bounds + 10% margin
# Raw bounds: X[-78.27, 116.93], Y[-143.83, 90.79], Z[-27.94, 46.75]  ->  ceil(|raw|*1.1) -> *0.2
_point_clip_range = (-87*0.2, -159*0.2, -31*0.2, 129*0.2, 100*0.2, 52*0.2)

dataset_type = "KeralaSceneHDF5Dataset"

data = dict(
    num_classes=num_classes,
    ignore_index=-1,
    names=class_names,

    train=dict(
        type=dataset_type,
        h5_path=H5_PATH,
        split="train",
        num_points=None,
        use_intensity=False,
        use_gps_time=False,
        use_normals=True,
        use_rgb=True,
        remap_classes=True,
        class_groups=class_groups,
        class_names=class_names,
        load_instance=True,
        transform=[
            dict(type="RandomScale", scale=[0.2, 0.2]),
            dict(type="RandomRotate", angle=[-1, 1], axis="z", center=[0, 0, 0], p=0.5),
            dict(type="RandomScale", scale=[0.9, 1.1]),
            dict(type="RandomFlip", p=0.5),
            dict(type="RandomJitter", sigma=0.001, clip=0.004),
            dict(type="GridSample", grid_size=0.01, hash_type="fnv", mode="train", return_grid_coord=True),
            dict(type="PointClip", point_cloud_range=_point_clip_range),
            dict(type="NormalizeColor"),
            dict(type="InstanceParser", segment_ignore_index=segment_ignore_index, instance_ignore_index=-1),
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=("coord", "grid_coord", "segment", "instance", "instance_centroid", "bbox"),
                feat_keys=("coord", "color", "normal"),
            ),
        ],
        test_mode=False,
        ignore_index=-1,
    ),

    val=dict(
        type=dataset_type,
        h5_path=H5_PATH,
        split="val",
        num_points=None,
        use_intensity=False,
        use_gps_time=False,
        use_normals=True,
        use_rgb=True,
        remap_classes=True,
        class_groups=class_groups,
        class_names=class_names,
        load_instance=True,
        transform=[
            dict(type="RandomScale", scale=[0.2, 0.2]),
            dict(
                type="Copy",
                keys_dict={
                    "coord": "origin_coord",
                    "segment": "origin_segment",
                    "instance": "origin_instance",
                },
            ),
            dict(type="GridSample", grid_size=0.01, hash_type="fnv", mode="train", return_grid_coord=True),
            dict(type="PointClip", point_cloud_range=_point_clip_range),
            dict(type="NormalizeColor"),
            dict(type="InstanceParser", segment_ignore_index=segment_ignore_index, instance_ignore_index=-1),
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=("coord", "grid_coord", "segment", "instance",
                      "origin_coord", "origin_segment", "origin_instance",
                      "instance_centroid", "bbox", "name"),
                feat_keys=("coord", "color", "normal"),
                offset_keys_dict=dict(offset="coord", origin_offset="origin_coord"),
            ),
        ],
        test_mode=False,
        ignore_index=-1,
    ),

    test=dict(
        type=dataset_type,
        h5_path=H5_PATH,
        split="test",
        num_points=None,
        use_intensity=False,
        use_gps_time=False,
        use_normals=True,
        use_rgb=True,
        remap_classes=True,
        class_groups=class_groups,
        class_names=class_names,
        load_instance=True,
        transform=[
            dict(type="RandomScale", scale=[0.2, 0.2]),
            dict(
                type="Copy",
                keys_dict={
                    "coord": "origin_coord",
                    "segment": "origin_segment",
                    "instance": "origin_instance",
                },
            ),
            dict(type="GridSample", grid_size=0.01, hash_type="fnv", mode="train", return_grid_coord=True),
            dict(type="PointClip", point_cloud_range=_point_clip_range),
            dict(type="NormalizeColor"),
            dict(type="InstanceParser", segment_ignore_index=segment_ignore_index, instance_ignore_index=-1),
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=("coord", "grid_coord", "segment", "instance",
                      "origin_coord", "origin_segment", "origin_instance",
                      "instance_centroid", "bbox", "name"),
                feat_keys=("coord", "color", "normal"),
                offset_keys_dict=dict(offset="coord", origin_offset="origin_coord"),
            ),
        ],
        test_mode=False,
        ignore_index=-1,
    ),
)

# =============================================================================
# HOOKS (Sonata pre-training checkpoint mapping)
# =============================================================================
hooks = [
    dict(type="CheckpointLoader", keywords="module.student.backbone", replacement="module.backbone"),
    dict(type="IterationTimer", warmup_iter=2),
    dict(type="InformationWriter"),
    dict(type="InsSegEvaluator", segment_ignore_index=segment_ignore_index, instance_ignore_index=-1),
    dict(type="CheckpointSaver", save_freq=None),
    dict(type="PreciseEvaluator", test_last=False),
]

# Tester
test = dict(
    type="InsSegTester",
    segment_ignore_index=segment_ignore_index,
    instance_ignore_index=-1,
    verbose=False,
)

# Pre-trained weights
weight = SONATA_CHECKPOINT
