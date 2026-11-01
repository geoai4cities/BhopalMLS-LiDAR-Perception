"""
Utonia Configuration for Kerala MLS - FULL FINE-TUNING (4-Channel Input)
FT-4ch: 9-channel input with RGB/normals zeroed (geometry-only, pretrained-compatible)
FULL FINE-TUNING: enc_mode=False, freeze_encoder=False, entire model trains
Date: 2026-04-13

Dataset: Kerala V1 corridor (48 -> 12 parent classes)
Input Channels: 9 = XYZ(3) + zeros(3) + zeros(3) [RGB/normals zeroed at runtime]

Input: 9ch = XYZ(3) + RGB(3) + normals(3). RGB and normals are loaded from H5 but zeroed at runtime via ZeroColor + ZeroNormal transforms, so only XYZ geometry reaches the model. Because in_channels=9, the pretrained checkpoint's input stem is shape-compatible and loads cleanly.

TRAINABLE: ~115M params (entire model — encoder + decoder + seg_head), all random init
"""

_base_ = ["../_base_/default_runtime.py"]

# =============================================================================
# MISC SETTINGS (OFFICIAL)
# =============================================================================
batch_size = 16
batch_size_val = 1
batch_size_test = 1
num_worker = 16
num_worker_test = 1
mix_prob = 0.8
empty_cache = False
enable_amp = True

# HDF5 file path
H5_PATH = "data/Kerala/Kerala_scene_v1_corridor.h5"

# Pre-trained checkpoint path - Sonata (not loaded for 4ch; see header note)
UTONIA_CHECKPOINT = "Utonia_pretrained/utonia.pth"

# =============================================================================
# CLASS CONFIGURATION
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

# Class weights
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
# MODEL CONFIGURATION - FULL FINE-TUNING (4-channel input, from scratch)
# =============================================================================
model = dict(
    type="DefaultSegmentorV2",
    num_classes=num_classes,
    backbone_out_channels=96,  # Decoder output channels (first dec_channel)
    backbone=dict(
        type="PT-v3m3",
        in_channels=9,  # OFFICIAL: Always 9 channels (coord + color + normal)
        order=("z", "z-trans", "hilbert", "hilbert-trans"),
        stride=(2, 2, 2, 2),
        # UTONIA ENCODER (Official)
        enc_depths=(3, 3, 3, 12, 3),
        enc_channels=(54, 108, 216, 432, 576),
        enc_num_head=(3, 6, 12, 24, 32),
        enc_patch_size=(1024, 1024, 1024, 1024, 1024),
        # UTONIA DECODER (Official)
        dec_depths=(3, 3, 3, 3),
        dec_channels=(96, 96, 192, 384),
        dec_num_head=(8, 8, 16, 32),  # head_dim must be divisible by 3 for RoPE: 96/8=12, 192/16=12, 384/32=12
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
        enc_mode=False,        # FULL FT: Use decoder
        freeze_encoder=False,  # FULL FT: Train encoder too
        rope_base=10,
        shift_coords=None,
        jitter_coords=None,
        rescale_coords=None,
    ),
    freeze_backbone=False,  # FULL FT: Train everything
    criteria=[
        dict(type="CrossEntropyLoss", weight=class_weight, loss_weight=1.0, ignore_index=-1),
        dict(type="LovaszLoss", mode="multiclass", loss_weight=1.0, ignore_index=-1),
    ],
)

# =============================================================================
# SCHEDULER SETTINGS (OFFICIAL - with differential LR)
# =============================================================================
epoch = 100
eval_epoch = 100

optimizer = dict(type="AdamW", lr=0.002, weight_decay=0.005)

scheduler = dict(
    type="OneCycleLR",
    max_lr=[0.002, 0.0002],  # Head LR, Backbone LR (10x lower)
    pct_start=0.04,
    anneal_strategy="cos",
    div_factor=10.0,
    final_div_factor=100.0,
)

# OFFICIAL: Backbone gets 10x lower learning rate
param_dicts = [dict(keyword="block", lr=0.0002)]

# =============================================================================
# DATASET CONFIGURATION — 9ch with zeroed RGB/normals (geometry only)
# =============================================================================
# PointClip — Bhopal-wide PCR for clean BM<->KM transfer (matches TR/PL).
# KM-native PCR (commented below) clipped here at Y_min and Z edges.
# Kerala_scene_v1_corridor.h5 bounds: X[-78.27, 116.93], Y[-143.83, 90.79],
# Z[-27.94, 46.75]  ->  ceil(|raw|*1.1) -> *0.2
# _point_clip_range = (-87*0.2, -159*0.2, -31*0.2, 129*0.2, 100*0.2, 52*0.2)
_point_clip_range = (-29.8, -21.8, -4.4, 42.6, 31.8, 8.6)

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
        use_normals=True,     # Loaded then zeroed via ZeroNormal transform
        use_rgb=True,         # Loaded then zeroed via ZeroColor transform
        remap_classes=True,
        class_groups=class_groups,
        class_names=class_names,
        transform=[
            dict(type="RandomScale", scale=[0.2, 0.2]),
            dict(type="RandomRotate", angle=[-1, 1], axis="z", center=[0, 0, 0], p=0.5),
            dict(type="RandomScale", scale=[0.9, 1.1]),
            dict(type="RandomFlip", p=0.5),
            dict(type="RandomJitter", sigma=0.0025, clip=0.01),
            dict(
                type="GridSample",
                grid_size=0.01,
                hash_type="fnv",
                mode="train",
                return_grid_coord=True,
            ),
            dict(type="PointClip", point_cloud_range=_point_clip_range),
            dict(type="ZeroColor"),   # Zero RGB
            dict(type="ZeroNormal"),  # Zero normals -> geometry only
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
        use_normals=True,     # Loaded then zeroed via ZeroNormal transform
        use_rgb=True,         # Loaded then zeroed via ZeroColor transform
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
            dict(type="ZeroColor"),   # Zero RGB
            dict(type="ZeroNormal"),  # Zero normals -> geometry only
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
        use_normals=True,     # Loaded then zeroed via ZeroNormal transform
        use_rgb=True,         # Loaded then zeroed via ZeroColor transform
        remap_classes=True,
        class_groups=class_groups,
        class_names=class_names,
        transform=[
            dict(type="RandomScale", scale=[0.2, 0.2]),
            dict(type="PointClip", point_cloud_range=_point_clip_range),
            dict(type="ZeroColor"),   # Zero RGB
            dict(type="ZeroNormal"),  # Zero normals -> geometry only
            dict(type="Copy", keys_dict={"segment": "origin_segment"}),
            dict(
                type="GridSample",
                grid_size=0.005,
                hash_type="fnv",
                mode="train",
                return_inverse=True,
            ),
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
# HOOKS
# =============================================================================
hooks = [
    dict(type="CheckpointLoader", keywords="module.", replacement="module.backbone."),
    dict(type="IterationTimer", warmup_iter=2),
    dict(type="InformationWriter"),
    dict(type="SemSegEvaluator"),
    dict(type="CheckpointSaver", save_freq=None),
    dict(type="PreciseEvaluator", test_last=False),
]

weight = UTONIA_CHECKPOINT