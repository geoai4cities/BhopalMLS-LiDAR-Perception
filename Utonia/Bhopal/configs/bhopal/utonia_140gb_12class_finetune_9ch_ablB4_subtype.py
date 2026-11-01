"""Factor-attribution ablation B4_subtype: subtype-swapped control: as B2, but Vehicle keeps Car (the subtype the target benchmarks are made of) instead of the dominant two-wheeler; separates variety from subtype match.
Generated from utonia_140gb_12class_finetune_9ch.py; only H5_PATH and class_groups differ.
Subset definition: data/Bhopal/ablation_subsets.json
"""
"""
Utonia Configuration for Bhopal MLS - FULL FINE-TUNING (Official Style)
FT-9ch: 9-Channel Input (XYZ + RGB + normals) - MATCHES OFFICIAL UTONIA

FULL FINE-TUNING: enc_mode=False, freeze_encoder=False, entire model trains
Date: 2026-03-11
Status: ALIGNED WITH OFFICIAL UTONIA CONFIG

Dataset: Bhopal V9 (with normals + RGB)
Input Channels: 9 = XYZ(3) + RGB(3) + normal(3) - OFFICIAL UTONIA INPUT

=============================================================================
OFFICIAL FULL FINE-TUNING ARCHITECTURE (enc_mode=False, freeze_encoder=False)
=============================================================================
The entire model is trainable with differential learning rates:
- Backbone (encoder + decoder) gets 10x lower learning rate
- Segmentation head gets full learning rate

| Encoder Stage | Channels | Heads | head_dim | Status                     |
|---------------|----------|-------|----------|----------------------------|
| Stage 0       | 54       | 3     | 18       | TRAINABLE (pre-trained)    |
| Stage 1       | 108      | 6     | 18       | TRAINABLE (pre-trained)    |
| Stage 2       | 216      | 12    | 18       | TRAINABLE (pre-trained)    |
| Stage 3       | 432      | 24    | 18       | TRAINABLE (pre-trained)    |
| Stage 4       | 576      | 32    | 18       | TRAINABLE (pre-trained)    |

| Decoder Stage | Channels | Heads | head_dim | Status                     |
|---------------|----------|-------|----------|----------------------------|
| Stage 3       | 384      | 32    | 12       | TRAINABLE (random init)    |
| Stage 2       | 192      | 16    | 12       | TRAINABLE (random init)    |
| Stage 1       | 96       | 8     | 12       | TRAINABLE (random init)    |
| Stage 0       | 96       | 8     | 12       | TRAINABLE (random init)    |
| Seg Head      | 96 -> 12 |       |          | TRAINABLE (random init)    |

NOTE: Decoder dec_num_head=(8, 8, 16, 32) instead of PTv3 default (6, 6, 12, 32)
because Utonia's Point3DRoPE requires head_dim % 3 == 0. The decoder is randomly
initialized (not in pre-trained checkpoint), so different head counts are fine.

TRAINABLE: ~210M params (encoder ~176M + decoder ~30M + seg_head 1152)

This is FULL FINE-TUNING - entire model adapts to the target dataset!
=============================================================================

GPU: H200 (143GB VRAM)
Estimated Memory: ~80-90GB (full encoder+decoder training)
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
clip_grad = 3.0  # OFFICIAL: Gradient clipping for stability

# HDF5 file path - V9 Dataset (with normals and RGB)
H5_PATH = "data/Bhopal/bhopal_scene_v9_ablB4_subtype.h5"

# Pre-trained checkpoint path
UTONIA_CHECKPOINT = "Utonia_pretrained/utonia.pth"

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

class_groups_eval = [  # full grouping (val/test)
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
class_groups = [
    [1],  # Ground: single fine subclass only (train)
    [7],  # Vegetation: single fine subclass only (train)
    [10],  # Building: single fine subclass only (train)
    [13],  # Road: single fine subclass only (train)
    [14],  # Wire: single fine subclass only (train)
    [16],  # Pole and Tower: single fine subclass only (train)
    [19],  # Wire-Structure Connector: single fine subclass only (train)
    [21],  # Vehicle: single fine subclass only (train)
    [37],  # Board: single fine subclass only (train)
    [39],  # Fencing: single fine subclass only (train)
    [43],  # Human and Animal: single fine subclass only (train)
    [53],  # Road Side Object: single fine subclass only (train)
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
# MODEL CONFIGURATION - OFFICIAL FULL FINE-TUNING (enc_mode=False, freeze_encoder=False)
# =============================================================================
model = dict(
    type="DefaultSegmentorV2",
    num_classes=num_classes,
    backbone_out_channels=96,  # DECODER OUTPUT (not 1386 like linear probing)
    backbone=dict(
        type="PT-v3m3",
        in_channels=9,  # XYZ(3) + RGB(3) + normal(3)
        order=("z", "z-trans", "hilbert", "hilbert-trans"),
        stride=(2, 2, 2, 2),
        # UTONIA ENCODER (Official) - TRAINABLE
        enc_depths=(3, 3, 3, 12, 3),
        enc_channels=(54, 108, 216, 432, 576),
        enc_num_head=(3, 6, 12, 24, 32),
        enc_patch_size=(1024, 1024, 1024, 1024, 1024),
        # UTONIA DECODER (Official) - TRAINABLE
        dec_depths=(3, 3, 3, 3),
        dec_channels=(96, 96, 192, 384),
        dec_num_head=(8, 8, 16, 32),  # head_dim must be divisible by 3 for RoPE: 96/8=12, 192/16=12, 384/32=12
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
        # KEY SETTINGS FOR FULL FINE-TUNING
        enc_mode=False,        # Use decoder
        freeze_encoder=False,  # Train encoder (with lower LR)
        rope_base=10,
        shift_coords=None,
        jitter_coords=None,
        rescale_coords=None,
    ),
    freeze_backbone=False,  # Train everything
    criteria=[
        dict(type="CrossEntropyLoss", weight=class_weight, loss_weight=1.0, ignore_index=-1),
        dict(type="LovaszLoss", mode="multiclass", loss_weight=1.0, ignore_index=-1),
    ],
)

# =============================================================================
# SCHEDULER SETTINGS (OFFICIAL - Differential Learning Rates)
# =============================================================================
epoch = 100
eval_epoch = 100  # Must equal epoch for proper display

optimizer = dict(type="AdamW", lr=0.002, weight_decay=0.005)  # OFFICIAL: weight_decay=0.005

scheduler = dict(
    type="OneCycleLR",
    max_lr=[0.002, 0.0002],  # OFFICIAL: [base_lr, block_lr] - backbone gets 10x lower
    pct_start=0.04,  # OFFICIAL
    anneal_strategy="cos",
    div_factor=10.0,
    final_div_factor=100.0,  # OFFICIAL: 100 (not 1000)
)

# OFFICIAL: Backbone blocks get reduced learning rate (10x lower)
param_dicts = [dict(keyword="block", lr=0.0002)]

# =============================================================================
# DATASET CONFIGURATION
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
        use_normals=True,
        use_rgb=True,
        remap_classes=True,
        class_groups=class_groups,
        class_names=class_names,
        transform=[
            # OFFICIAL transform order (matching SemanticKITTI/Utonia)
            dict(type="RandomScale", scale=[0.2, 0.2]),
            dict(type="RandomRotate", angle=[-1, 1], axis="z", center=[0, 0, 0], p=0.5),
            dict(type="RandomScale", scale=[0.9, 1.1]),
            dict(type="RandomFlip", p=0.5),
            # OFFICIAL 9ch jitter (smaller than 4ch for color data sensitivity)
            dict(type="RandomJitter", sigma=0.001, clip=0.004),
            dict(
                type="GridSample",
                grid_size=0.01,
                hash_type="fnv",
                mode="train",
                return_grid_coord=True,
            ),
            # BHOPAL PointClip: Based on analyze_bhopal_pointcloud_range.py (10% margin)
            # Bhopal data: X=[-135,193], Y=[-99,144], Z=[-20,39] -> with margin: X=[-149,213], Y=[-109,159], Z=[-22,43]
            # SemanticKITTI/Utonia official (too small for Bhopal): (-35.2*0.2, -35.2*0.2, -4*0.2, 35.2*0.2, 35.2*0.2, 2*0.2)
            dict(type="PointClip", point_cloud_range=(-149*0.2, -109*0.2, -22*0.2, 213*0.2, 159*0.2, 43*0.2)),
            # SphereCrop REMOVED: Official nuScenes/Waymo configs comment it out.
            # Removing preserves full spatial context (critical for dense Bhopal MLS data).
            dict(type="NormalizeColor"),
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=("coord", "grid_coord", "segment"),
                feat_keys=("coord", "color", "normal"),  # OFFICIAL ORDER
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
        class_groups=class_groups_eval,
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
            # BHOPAL PointClip: Based on analyze_bhopal_pointcloud_range.py (10% margin)
            # Bhopal data: X=[-135,193], Y=[-99,144], Z=[-20,39] -> with margin: X=[-149,213], Y=[-109,159], Z=[-22,43]
            # SemanticKITTI/Utonia official (too small for Bhopal): (-35.2*0.2, -35.2*0.2, -4*0.2, 35.2*0.2, 35.2*0.2, 2*0.2)
            dict(type="PointClip", point_cloud_range=(-149*0.2, -109*0.2, -22*0.2, 213*0.2, 159*0.2, 43*0.2)),
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
        class_groups=class_groups_eval,
        class_names=class_names,
        transform=[
            dict(type="RandomScale", scale=[0.2, 0.2]),
            # BHOPAL PointClip: Based on analyze_bhopal_pointcloud_range.py (10% margin)
            # Bhopal data: X=[-135,193], Y=[-99,144], Z=[-20,39] -> with margin: X=[-149,213], Y=[-109,159], Z=[-22,43]
            # SemanticKITTI/Utonia official (too small for Bhopal): (-35.2*0.2, -35.2*0.2, -4*0.2, 35.2*0.2, 35.2*0.2, 2*0.2)
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
# HOOKS (For Released Weights: utonia.pth)
# =============================================================================
hooks = [
    dict(
        type="CheckpointLoader",
        keywords="module.",           # Released weights have direct keys (embedding.*, enc.*)
        replacement="module.backbone.",  # Map to backbone.embedding.*, backbone.enc.*
    ),
    dict(type="IterationTimer", warmup_iter=2),
    dict(type="InformationWriter"),
    dict(type="SemSegEvaluator"),
    dict(type="CheckpointSaver", save_freq=None),
    dict(type="PreciseEvaluator", test_last=False),
]

# Pre-trained weights path
weight = UTONIA_CHECKPOINT
