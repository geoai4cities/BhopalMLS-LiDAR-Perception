"""
Concerto Configuration for Kerala MLS - FULL FINE-TUNING (Official Style)
FT-9ch: 9-Channel Input (XYZ + zero-RGB + normals) - BHOPAL CHECKPOINT COMPATIBLE

FULL FINE-TUNING: enc_mode=False, freeze_encoder=False, entire model trains
Date: 2026-04-11

Dataset: Kerala V1 corridor (48 -> 12 parent classes)
Input Channels: 9 = XYZ(3) + RGB(3) + normal(3) - OFFICIAL CONCERTO INPUT
Note: Kerala RGB is all zeros (no color data available); loader returns
      a zero-filled 'color' tensor so input shape stays checkpoint-compatible.

Kerala: 71 paths, 48 training classes -> 12 parent classes
Train: paths 01-38, Val: paths 39-51, Test: paths 52-71

FROZEN:     None
TRAINABLE:  ~240M params (entire model — encoder + decoder + seg_head)
"""

_base_ = ["../_base_/default_runtime.py"]

# =============================================================================
# MISC SETTINGS (OFFICIAL)
# =============================================================================
batch_size = 16           # Will be overridden by script for GPU-specific settings
batch_size_val = 1
batch_size_test = 1
num_worker = 16
num_worker_test = 1
mix_prob = 0.8
empty_cache = False
enable_amp = True
clip_grad = 3.0

# HDF5 file path (relative to the repository root working directory)
H5_PATH = "data/Kerala/Kerala_scene_v1_corridor.h5"

# Pre-trained checkpoint path
CONCERTO_CHECKPOINT = "Concerto_pretrained/concerto_large_outdoor.pth"

# =============================================================================
# CLASS CONFIGURATION — Kerala (48 -> 12 parent classes)
# =============================================================================
num_classes = 12
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

# Class weights — sqrt-scaled inverse frequency (from Kerala_corridor_class_distribution.h5)
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
# MODEL CONFIGURATION — OFFICIAL FULL FINE-TUNING
# =============================================================================
model = dict(
    type="DefaultSegmentorV2",
    num_classes=num_classes,
    backbone_out_channels=64,  # DECODER OUTPUT
    backbone=dict(
        type="PT-v3m2",
        in_channels=9,  # XYZ(3) + RGB(3) + normal(3)
        order=("z", "z-trans", "hilbert", "hilbert-trans"),
        stride=(2, 2, 2, 2),
        # CONCERTO LARGE ENCODER (Official) - TRAINABLE
        enc_depths=(3, 3, 3, 12, 3),
        enc_channels=(64, 128, 256, 512, 768),
        enc_num_head=(4, 8, 16, 32, 48),
        enc_patch_size=(1024, 1024, 1024, 1024, 1024),
        # CONCERTO LARGE DECODER (Official) - TRAINABLE
        dec_depths=(2, 2, 2, 2),
        dec_channels=(64, 96, 192, 384),
        dec_num_head=(4, 6, 12, 24),
        dec_patch_size=(1024, 1024, 1024, 1024),
        # Shared params
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
        enc_mode=False,
        freeze_encoder=False,
    ),
    freeze_backbone=False,
    criteria=[
        dict(type="CrossEntropyLoss", weight=class_weight, loss_weight=1.0, ignore_index=-1),
        dict(type="LovaszLoss", mode="multiclass", loss_weight=1.0, ignore_index=-1),
    ],
)

# =============================================================================
# SCHEDULER SETTINGS (OFFICIAL - Differential Learning Rates)
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
dataset_type = "KeralaSceneHDF5Dataset"

# PointClip — Bhopal-wide PCR for clean BM<->KM transfer (matches TR/PL).
# KM-native PCR (commented below) clipped here at Y_min and Z edges.
# Kerala_scene_v1_corridor.h5 bounds: X[-78.27, 116.93], Y[-143.83, 90.79],
# Z[-27.94, 46.75]  ->  ceil(|raw|*1.1) -> *0.2
# _point_clip_range = (-87*0.2, -159*0.2, -31*0.2, 129*0.2, 100*0.2, 52*0.2)
_point_clip_range = (-29.8, -21.8, -4.4, 42.6, 31.8, 8.6)

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
        transform=[
            dict(type="RandomScale", scale=[0.2, 0.2]),
            dict(type="RandomRotate", angle=[-1, 1], axis="z", center=[0, 0, 0], p=0.5),
            dict(type="RandomScale", scale=[0.9, 1.1]),
            dict(type="RandomFlip", p=0.5),
            dict(type="RandomJitter", sigma=0.001, clip=0.004),
            dict(
                type="GridSample",
                grid_size=0.01,
                hash_type="fnv",
                mode="train",
                return_grid_coord=True,
            ),
            dict(type="PointClip", point_cloud_range=_point_clip_range),
            dict(type="NormalizeColor"),
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=("coord", "grid_coord", "segment"),
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
        transform=[
            dict(type="RandomScale", scale=[0.2, 0.2]),
            dict(type="Copy", keys_dict={"segment": "origin_segment"}),
            dict(
                type="GridSample",
                grid_size=0.01,
                hash_type="fnv",
                mode="train",
                return_grid_coord=True,
                return_inverse=True,
            ),
            dict(type="PointClip", point_cloud_range=_point_clip_range),
            dict(type="NormalizeColor"),
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=("coord", "grid_coord", "segment", "origin_segment", "inverse"),
                feat_keys=("coord", "color", "normal"),
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
        transform=[
            dict(type="RandomScale", scale=[0.2, 0.2]),
            dict(type="PointClip", point_cloud_range=_point_clip_range),
            dict(type="Copy", keys_dict={"segment": "origin_segment"}),
            dict(
                type="GridSample",
                grid_size=0.005,
                hash_type="fnv",
                mode="train",
                return_inverse=True,
            ),
            dict(type="NormalizeColor"),
        ],
        test_mode=True,
        test_cfg=dict(
            voxelize=dict(
                type="GridSample",
                grid_size=0.01,
                hash_type="fnv",
                mode="test",
                return_grid_coord=True,
            ),
            crop=None,
            post_transform=[
                dict(type="ToTensor"),
                dict(
                    type="Collect",
                    keys=("coord", "grid_coord", "index"),
                    feat_keys=("coord", "color", "normal"),
                ),
            ],
            aug_transform=[
                [dict(type="RandomScale", scale=[0.9, 0.9])],
                [dict(type="RandomScale", scale=[0.95, 0.95])],
                [dict(type="RandomScale", scale=[1, 1])],
                [dict(type="RandomScale", scale=[1.05, 1.05])],
                [dict(type="RandomScale", scale=[1.1, 1.1])],
                [dict(type="RandomScale", scale=[0.9, 0.9]), dict(type="RandomFlip", p=1)],
                [dict(type="RandomScale", scale=[0.95, 0.95]), dict(type="RandomFlip", p=1)],
                [dict(type="RandomScale", scale=[1, 1]), dict(type="RandomFlip", p=1)],
                [dict(type="RandomScale", scale=[1.05, 1.05]), dict(type="RandomFlip", p=1)],
                [dict(type="RandomScale", scale=[1.1, 1.1]), dict(type="RandomFlip", p=1)],
            ],
        ),
        ignore_index=-1,
    ),
)

# =============================================================================
# HOOKS (For Released Weights: concerto_large_outdoor.pth)
# =============================================================================
hooks = [
    dict(
        type="CheckpointLoader",
        keywords="module.",
        replacement="module.backbone.",
    ),
    dict(type="IterationTimer", warmup_iter=2),
    dict(type="InformationWriter"),
    dict(type="SemSegEvaluator"),
    dict(type="CheckpointSaver", save_freq=None),
    dict(type="PreciseEvaluator", test_last=False),
]

# Pre-trained weights path
weight = CONCERTO_CHECKPOINT
