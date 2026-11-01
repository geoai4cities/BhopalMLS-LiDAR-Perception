"""
KpConvX Configuration for Bhopal MLS - TRAIN FROM SCRATCH
9ch: Full Features (XYZ + RGB + normals)

TRAIN FROM SCRATCH: No pre-trained weights, entire model trains from random init
Architecture: KpConvX-L (Apple KPConvX, CVPR 2024)

Date: 2026-02-12
Status: NEW

Dataset: Bhopal V9 (with normals + RGB)
Input Channels: 9 = XYZ(3) + RGB(3) + normal(3)

=============================================================================
KpConvX-L ARCHITECTURE (Train from Scratch, init_channels=64)
=============================================================================
| Component       | Channels              | Status                           |
|-----------------|----------------------|----------------------------------|
| Stem Block      | 64                   | TRAINABLE (random init)          |
| Encoder Layer 0 | 64                   | TRAINABLE (3 blocks)             |
| Encoder Layer 1 | 90                   | TRAINABLE (3 blocks)             |
| Encoder Layer 2 | 128                  | TRAINABLE (9 blocks)             |
| Encoder Layer 3 | 181                  | TRAINABLE (12 blocks)            |
| Encoder Layer 4 | 256                  | TRAINABLE (3 blocks)             |
| Decoder Layer 0 | 181                  | TRAINABLE                        |
| Decoder Layer 1 | 128                  | TRAINABLE                        |
| Decoder Layer 2 | 90                   | TRAINABLE                        |
| Decoder Layer 3 | 64                   | TRAINABLE                        |
| Head            | -> 12 classes        | TRAINABLE                        |

TRAINABLE: ~40M params (entire model from scratch)
=============================================================================

GPU: H200 (143GB VRAM)
Estimated Memory: ~30-50GB
"""

_base_ = ["../_base_/default_runtime.py"]

# =============================================================================
# MISC SETTINGS
# =============================================================================
batch_size = 12  # Matching official KpConvX (8x80GB: batch_size=12)
batch_size_val = 1  # Matching official (None = auto 1/gpu from default_runtime.py)
batch_size_test = 1
num_worker = 16
num_worker_test = 1
mix_prob = 0.8
empty_cache = False
enable_amp = True
clip_grad = 3.0

# HDF5 file path - V9 Dataset (with normals and RGB)
H5_PATH = "data/Bhopal/bhopal_scene_v9_normals_rgb.h5"

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
    1.0000,   # 2: Building (65.12%)
    2.1667,   # 3: Road (13.87%)
    4.3169,   # 4: Wire (3.49%)
    8.0928,   # 5: Pole and Tower (0.99%)
    10.5119,  # 6: Wire-Structure Connector (0.59%)
    3.3875,   # 7: Vehicle (5.67%)
    9.6172,   # 8: Board (0.70%)
    15.8514,  # 9: Fencing (0.26%)
    8.9056,   # 10: Human and Animal (0.82%)
    4.8175,   # 11: Road Side Object (2.81%)
]

# =============================================================================
# MODEL CONFIGURATION - KpConvX-L (Train from Scratch)
# =============================================================================
model = dict(
    type="DefaultSegmentor",
    backbone=dict(
        type="kpconvx_base",
        input_channels=9,          # coord(3) + RGB(3) + normal(3)
        num_classes=num_classes,
        dim=3,
        task='cloud_segmentation',
        kp_mode='kpconvx',
        shell_sizes=(1, 14, 28),
        kp_radius=2.1,
        kp_aggregation='nearest',
        kp_influence='constant',
        kp_sigma=2.1,
        share_kp=False,
        conv_groups=-1,
        inv_groups=8,
        inv_act='sigmoid',
        inv_grp_norm=True,
        kpx_upcut=False,
        subsample_size=0.04,
        neighbor_limits=(12, 16, 20, 20, 20),  # Official fixed value (paper)
        layer_blocks=(3, 3, 9, 12, 3),         # KpConvX-L
        init_channels=64,                       # Official value (all configs)
        channel_scaling=1.414,
        radius_scaling=2.2,
        decoder_layer=True,
        grid_pool=True,
        upsample_n=3,
        first_inv_layer=1,
        drop_path_rate=0.3,
        norm='batch',
        bn_momentum=0.1,
        smooth_labels=False,
        class_w=(),
    ),
    criteria=[
        dict(type="CrossEntropyLoss", weight=class_weight, loss_weight=1.0, ignore_index=-1),
        dict(type="LovaszLoss", mode="multiclass", loss_weight=1.0, ignore_index=-1),
    ],
)

# =============================================================================
# SCHEDULER SETTINGS
# =============================================================================
epoch = 450
eval_epoch = 50  # Validate every 50 epochs (official: eval_epoch=200 with epoch=1000)

optimizer = dict(type="AdamW", lr=0.005, weight_decay=0.05)

scheduler = dict(
    type="OneCycleLR",
    max_lr=0.005,
    pct_start=0.05,
    anneal_strategy="cos",
    div_factor=100.0,
    final_div_factor=1000.0,
)

# =============================================================================
# DATASET CONFIGURATION - 9ch (coord + color + normal)
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
        use_normals=True,   # 9ch: normals enabled
        use_rgb=True,       # 9ch: RGB enabled
        remap_classes=True,
        class_groups=class_groups,
        class_names=class_names,
        transform=[
            dict(type="RandomScale", scale=[0.2, 0.2]),
            dict(type="CenterShift", apply_z=True),
            dict(type="RandomDropout", dropout_ratio=0.2, dropout_application_ratio=0.2),
            dict(type="RandomRotate", angle=[-1, 1], axis="z", center=[0, 0, 0], p=0.5),
            dict(type="RandomScale", scale=[0.9, 1.1]),
            dict(type="RandomFlip", p=0.5),
            dict(type="RandomJitter", sigma=0.001, clip=0.004),
            dict(type="ElasticDistortion", distortion_params=[[0.2, 0.4], [0.8, 1.6]]),
            dict(
                type="GridSample",
                grid_size=0.01,
                hash_type="fnv",
                mode="train",
                return_grid_coord=True,
                return_min_coord=True,
            ),
            dict(type="PointClip", point_cloud_range=(-149*0.2, -109*0.2, -22*0.2, 213*0.2, 159*0.2, 43*0.2)),
            dict(type="SphereCrop", sample_rate=0.8, mode="random"),
            dict(type="SphereCrop", point_max=120000, mode="random"),
            dict(type="CenterShift", apply_z=False),
            dict(type="NormalizeColor"),
            dict(type="ShufflePoint"),
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
            dict(type="CenterShift", apply_z=True),
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
            dict(type="CenterShift", apply_z=False),
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
            dict(type="PointClip", point_cloud_range=(-149*0.2, -109*0.2, -22*0.2, 213*0.2, 159*0.2, 43*0.2)),
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
            # Official KpConvX TTA: 4 rotations + 4 rot*scale(0.95) + 4 rot*scale(1.05) + 1 flip = 13
            aug_transform=[
                [dict(type="RandomRotateTargetAngle", angle=[0], axis="z", center=[0, 0, 0], p=1)],
                [dict(type="RandomRotateTargetAngle", angle=[1/2], axis="z", center=[0, 0, 0], p=1)],
                [dict(type="RandomRotateTargetAngle", angle=[1], axis="z", center=[0, 0, 0], p=1)],
                [dict(type="RandomRotateTargetAngle", angle=[3/2], axis="z", center=[0, 0, 0], p=1)],
                [dict(type="RandomRotateTargetAngle", angle=[0], axis="z", center=[0, 0, 0], p=1),
                 dict(type="RandomScale", scale=[0.95, 0.95])],
                [dict(type="RandomRotateTargetAngle", angle=[1/2], axis="z", center=[0, 0, 0], p=1),
                 dict(type="RandomScale", scale=[0.95, 0.95])],
                [dict(type="RandomRotateTargetAngle", angle=[1], axis="z", center=[0, 0, 0], p=1),
                 dict(type="RandomScale", scale=[0.95, 0.95])],
                [dict(type="RandomRotateTargetAngle", angle=[3/2], axis="z", center=[0, 0, 0], p=1),
                 dict(type="RandomScale", scale=[0.95, 0.95])],
                [dict(type="RandomRotateTargetAngle", angle=[0], axis="z", center=[0, 0, 0], p=1),
                 dict(type="RandomScale", scale=[1.05, 1.05])],
                [dict(type="RandomRotateTargetAngle", angle=[1/2], axis="z", center=[0, 0, 0], p=1),
                 dict(type="RandomScale", scale=[1.05, 1.05])],
                [dict(type="RandomRotateTargetAngle", angle=[1], axis="z", center=[0, 0, 0], p=1),
                 dict(type="RandomScale", scale=[1.05, 1.05])],
                [dict(type="RandomRotateTargetAngle", angle=[3/2], axis="z", center=[0, 0, 0], p=1),
                 dict(type="RandomScale", scale=[1.05, 1.05])],
                [dict(type="RandomFlip", p=1)],
            ],
        ),
        ignore_index=-1,
    ),
)

# =============================================================================
# HOOKS (No pre-trained weights - train from scratch)
# =============================================================================
hooks = [
    dict(type="CheckpointLoader"),
    dict(type="IterationTimer", warmup_iter=2),
    dict(type="InformationWriter"),
    dict(type="SemSegEvaluator"),
    dict(type="CheckpointSaver", save_freq=None),
    dict(type="PreciseEvaluator", test_last=False),
]
