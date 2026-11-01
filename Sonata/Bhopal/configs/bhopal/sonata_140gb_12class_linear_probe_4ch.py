"""
Sonata Configuration for Bhopal MLS - LINEAR PROBING (Official Style)
LP-3ch: XYZ Only - Geometry Only (zeros for color/normal)

TRUE LINEAR PROBING: enc_mode=True, only seg_head trains
Based on: pointcept/configs/sonata/semseg-sonata-v1m1-0a-scannet-lin.py

Date: 2026-02-05
Status: ALIGNED WITH OFFICIAL SONATA CONFIG

Dataset: Bhopal V9 (with normals + RGB, but zeros used for color/normal)
Input Channels: 9 = XYZ(3) + zeros(3) + zeros(3)
Meaningful Features: 3ch (XYZ only), color/normal zeroed via transforms

=============================================================================
OFFICIAL LINEAR PROBING ARCHITECTURE (enc_mode=True)
=============================================================================
| Encoder Stage | Channels | Output                           |
|---------------|----------|----------------------------------|
| Stage 0       | 48       | Concatenated to final output     |
| Stage 1       | 96       | Concatenated to final output     |
| Stage 2       | 192      | Concatenated to final output     |
| Stage 3       | 384      | Concatenated to final output     |
| Stage 4       | 512      | Concatenated to final output     |
| TOTAL         | 1232     | backbone_out_channels            |

FROZEN:     ~85M params (entire Sonata backbone including embedding)
TRAINABLE:  ~14.8K params (seg_head linear: 1232 -> 12 classes)

Note: Uses Concerto-style approach (ZeroColor + ZeroNormal transforms)
=============================================================================

GPU: H200 (143GB VRAM)
Estimated Memory: ~30-40GB (encoder-only, no decoder)
"""

_base_ = ["../_base_/default_runtime.py"]

# =============================================================================
# MISC SETTINGS (OFFICIAL)
# =============================================================================
batch_size = 16  # Will be overridden by script for GPU-specific settings
batch_size_val = 1  # Matching official (None = auto 1/gpu from default_runtime.py)
batch_size_test = 1
num_worker = 16
num_worker_test = 1
mix_prob = 0.8  # OFFICIAL: MixUp augmentation enabled
empty_cache = False
enable_amp = True

# HDF5 file path - V9 Dataset (with normals and RGB)
# Paths relative to the repository root (the working directory)
H5_PATH = "data/Bhopal/bhopal_scene_v9_normals_rgb.h5"

# Pre-trained checkpoint path - Sonata (indoor pre-trained, no outdoor-specific available)
SONATA_CHECKPOINT = "sonata_pretrained/pretrain-sonata-v1m1-0-base.pth"

# =============================================================================
# CLASS CONFIGURATION
# =============================================================================
num_classes = 12
class_names = [
    "Ground",              # 0: Original classes 0-3
    "Vegetation",          # 1: Original classes 4-7
    "Building",            # 2: Original classes 8-12
    "Road",                # 3: Original class 13
    "Wire",                # 4: Original class 14
    "Pole and Tower",      # 5: Original classes 15-18
    "Wire-Structure Connector",  # 6: Original class 19
    "Vehicle",             # 7: Original classes 20-34
    "Board",               # 8: Original classes 35-37
    "Fencing",             # 9: Original classes 38-42
    "Human and Animal",    # 10: Original classes 43-49
    "Road Side Object",    # 11: Original classes 50-61
]

class_groups = [
    list(range(0, 4)),     # 0: Ground (0-3)
    list(range(4, 8)),     # 1: Vegetation (4-7)
    list(range(8, 13)),    # 2: Building (8-12)
    [13],                  # 3: Road (13)
    [14],                  # 4: Wire (14)
    list(range(15, 19)),   # 5: Pole and Tower (15-18)
    [19],                  # 6: Wire-Structure Connector (19)
    list(range(20, 35)),   # 7: Vehicle (20-34)
    list(range(35, 38)),   # 8: Board (35-37)
    list(range(38, 43)),   # 9: Fencing (38-42)
    list(range(43, 50)),   # 10: Human and Animal (43-49)
    list(range(50, 62)),   # 11: Road Side Object (50-61)
]

# Class weights - Sqrt-scaled inverse frequency
class_weight = [
    4.9549,   # 0: Ground (2.65%)
    4.6483,   # 1: Vegetation (3.01%)
    1.0000,   # 2: Building (65.12%) - dominant class
    2.1667,   # 3: Road (13.87%)
    4.3169,   # 4: Wire (3.49%)
    8.0928,   # 5: Pole and Tower (0.99%)
    10.5119,  # 6: Wire-Structure Connector (0.59%)
    3.3875,   # 7: Vehicle (5.67%)
    9.6172,   # 8: Board (0.70%)
    15.8514,  # 9: Fencing (0.26%) - rarest class
    8.9056,   # 10: Human and Animal (0.82%)
    4.8175,   # 11: Road Side Object (2.81%)
]

# =============================================================================
# MODEL CONFIGURATION - OFFICIAL LINEAR PROBING (enc_mode=True)
# =============================================================================
model = dict(
    type="DefaultSegmentorV2",
    num_classes=num_classes,
    backbone_out_channels=1232,  # SONATA: Sum of all encoder channels (48+96+192+384+512)
    backbone=dict(
        type="PT-v3m2",
        in_channels=9,  # OFFICIAL: Always 9 channels (coord + color + normal)
        order=("z", "z-trans", "hilbert", "hilbert-trans"),
        stride=(2, 2, 2, 2),
        # SONATA ENCODER (Official) - Smaller than Concerto
        enc_depths=(3, 3, 3, 12, 3),
        enc_channels=(48, 96, 192, 384, 512),  # SONATA channels (not Concerto 64-128-256-512-768)
        enc_num_head=(3, 6, 12, 24, 32),       # SONATA heads (not Concerto 4-8-16-32-48)
        enc_patch_size=(1024, 1024, 1024, 1024, 1024),
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
        enc_mode=True,  # OFFICIAL: Encoder-only mode (no decoder)
        freeze_encoder=False,  # Handled by freeze_backbone
    ),
    freeze_backbone=True,  # OFFICIAL: Freeze entire backbone
    criteria=[
        dict(type="CrossEntropyLoss", weight=class_weight, loss_weight=1.0, ignore_index=-1),
        dict(type="LovaszLoss", mode="multiclass", loss_weight=1.0, ignore_index=-1),
    ],
)

# =============================================================================
# SCHEDULER SETTINGS (OFFICIAL)
# =============================================================================
epoch = 100
eval_epoch = 100  # Must equal epoch for proper display

optimizer = dict(type="AdamW", lr=0.002, weight_decay=0.005)  # OFFICIAL: weight_decay=0.005

scheduler = dict(
    type="OneCycleLR",
    max_lr=[0.002, 0.0002],  # OFFICIAL: [base_lr, block_lr]
    pct_start=0.04,  # OFFICIAL
    anneal_strategy="cos",
    div_factor=10.0,
    final_div_factor=100.0,  # OFFICIAL: 100 (not 1000)
)

# OFFICIAL: Blocks get reduced learning rate (not completely frozen)
param_dicts = [dict(keyword="block", lr=0.0002)]

# =============================================================================
# DATASET CONFIGURATION - 3ch (XYZ only, zeros for color/normal)
# =============================================================================
dataset_type = "BhopalSceneHDF5Dataset"

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
            # OFFICIAL transform order (matching SemanticKITTI/Sonata)
            dict(type="RandomScale", scale=[0.2, 0.2]),
            dict(type="RandomRotate", angle=[-1, 1], axis="z", center=[0, 0, 0], p=0.5),
            dict(type="RandomScale", scale=[0.9, 1.1]),
            dict(type="RandomFlip", p=0.5),
            # OFFICIAL 3ch jitter (larger than 9ch for geometry-only)
            dict(type="RandomJitter", sigma=0.0025, clip=0.01),
            dict(
                type="GridSample",
                grid_size=0.01,
                hash_type="fnv",
                mode="train",
                return_grid_coord=True,
            ),
            # BHOPAL PointClip: Based on analyze_bhopal_pointcloud_range.py (10% margin)
            dict(type="PointClip", point_cloud_range=(-149*0.2, -109*0.2, -22*0.2, 213*0.2, 159*0.2, 43*0.2)),
            # SphereCrop REMOVED: Official Concerto nuScenes/Waymo configs comment it out.
            # Removing preserves full spatial context for better segmentation.
            dict(type="ZeroColor"),   # Zero RGB → geometry only
            dict(type="ZeroNormal"),  # Zero normals → geometry only
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
            dict(type="PointClip", point_cloud_range=(-149*0.2, -109*0.2, -22*0.2, 213*0.2, 159*0.2, 43*0.2)),
            dict(type="ZeroColor"),
            dict(type="ZeroNormal"),
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
            dict(type="PointClip", point_cloud_range=(-149*0.2, -109*0.2, -22*0.2, 213*0.2, 159*0.2, 43*0.2)),
            dict(type="Copy", keys_dict={"segment": "origin_segment"}),
            dict(
                type="GridSample",
                grid_size=0.005,
                hash_type="fnv",
                mode="train",
                return_inverse=True,
            ),
            dict(type="ZeroColor"),
            dict(type="ZeroNormal"),
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
                [
                    dict(type="RandomScale", scale=[0.9, 0.9]),
                    dict(type="RandomFlip", p=1),
                ],
                [
                    dict(type="RandomScale", scale=[0.95, 0.95]),
                    dict(type="RandomFlip", p=1),
                ],
                [dict(type="RandomScale", scale=[1, 1]), dict(type="RandomFlip", p=1)],
                [
                    dict(type="RandomScale", scale=[1.05, 1.05]),
                    dict(type="RandomFlip", p=1),
                ],
                [
                    dict(type="RandomScale", scale=[1.1, 1.1]),
                    dict(type="RandomFlip", p=1),
                ],
            ],
        ),
        ignore_index=-1,
    ),
)

# =============================================================================
# HOOKS (For Pre-training Weights: pretrain-sonata-v1m1-0-base.pth)
# =============================================================================
hooks = [
    dict(
        type="CheckpointLoader",
        # Pre-training checkpoint has keys: module.student.backbone.*
        # Map to: module.backbone.*
        keywords="module.student.backbone",
        replacement="module.backbone",
    ),
    dict(type="IterationTimer", warmup_iter=2),
    dict(type="InformationWriter"),
    dict(type="SemSegEvaluator"),
    dict(type="CheckpointSaver", save_freq=None),
    dict(type="PreciseEvaluator", test_last=False),
]

# Pre-trained weights path
weight = SONATA_CHECKPOINT
