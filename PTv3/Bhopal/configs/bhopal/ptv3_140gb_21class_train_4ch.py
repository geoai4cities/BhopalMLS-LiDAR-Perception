"""
PTv3 Configuration for Bhopal MLS - TRAIN FROM SCRATCH
4ch: XYZ + Intensity
21-Class Fine-Grained Taxonomy

TRAIN FROM SCRATCH: No pre-trained weights, entire model trains from random init
Architecture: PT-v3m1 (original Point Transformer V3)

Date: 2026-03-14
Status: NEW

Dataset: Bhopal V9
Input Channels: 4 = XYZ(3) + intensity(1) - Matching SemanticKITTI 4ch convention

=============================================================================
PT-v3m1 ARCHITECTURE (Train from Scratch)
=============================================================================
| Component       | Channels              | Status                           |
|-----------------|----------------------|----------------------------------|
| Encoder Stage 0 | 32                   | TRAINABLE (random init)          |
| Encoder Stage 1 | 64                   | TRAINABLE (random init)          |
| Encoder Stage 2 | 128                  | TRAINABLE (random init)          |
| Encoder Stage 3 | 256                  | TRAINABLE (random init)          |
| Encoder Stage 4 | 512                  | TRAINABLE (random init)          |
| Decoder Stage 0 | 256                  | TRAINABLE (random init)          |
| Decoder Stage 1 | 128                  | TRAINABLE (random init)          |
| Decoder Stage 2 | 64                   | TRAINABLE (random init)          |
| Decoder Stage 3 | 64                   | TRAINABLE (random init)          |
| Seg Head        | 64 -> 21 classes     | TRAINABLE (random init)          |

TRAINABLE: ~22M params (entire model from scratch)
=============================================================================

GPU: H200 (143GB VRAM)
Estimated Memory: ~30-45GB
"""

_base_ = ["../_base_/default_runtime.py"]

# =============================================================================
# MISC SETTINGS
# =============================================================================
batch_size = 12  # Matching official PTv3 NuScenes/Waymo
batch_size_val = 1  # Matching official (None = auto 1/gpu)
batch_size_test = 1
num_worker = 16
num_worker_test = 1
mix_prob = 0.8  # CopyPaste enabled (matching official PTv3 configs). SphereCrop removed to prevent OOB errors.
empty_cache = False
enable_amp = True
clip_grad = 1.0  # Prevent gradient explosion / NaN propagation

# HDF5 file path - V9 Dataset (with normals and RGB)
H5_PATH = "data/Bhopal/bhopal_scene_v9_normals_rgb.h5"

# =============================================================================
# CLASS CONFIGURATION - 21-Class Fine-Grained Taxonomy
# =============================================================================
num_classes = 21
class_names = [
    "Ground",                # 0  - STUFF
    "Stairs",                # 1
    "Vegetation",            # 2  - STUFF
    "Building (other)",      # 3
    "Commercial",            # 4
    "Road",                  # 5  - STUFF
    "Wire",                  # 6  - STUFF
    "Pole (other)",          # 7
    "Electric Pole",         # 8
    "Street Light",          # 9
    "2-Wheeler",             # 10
    "3-Wheeler",             # 11
    "4-Wheeler",             # 12
    "Hand-Cart",             # 13
    "Board",                 # 14
    "Fencing",               # 15
    "Human",                 # 16
    "Road Side Furniture",   # 17
    "Shop Shade",            # 18
    "Mannequin",             # 19
    "Road Side Obj (other)", # 20
]

class_groups = [
    [0, 1, 2],                          # 0:  Ground (Flat + Gently Sloping + Ground gen)
    [3],                                 # 1:  Stairs (Steeply Sloping Ground)
    [4, 5, 6, 7],                        # 2:  Vegetation (all subtypes)
    [8, 9, 11, 12],                      # 3:  Building (other) (Gen+Residential+Industrial+Special)
    [10],                                # 4:  Commercial
    [13],                                # 5:  Road
    [14],                                # 6:  Wire
    [15, 17, 19],                        # 7:  Pole (other) (Gen+MobileTower+WireStruct)
    [16],                                # 8:  Electric Pole
    [18],                                # 9:  Street Light
    [22, 23, 32, 33, 34],               # 10: 2-Wheeler
    [25, 27, 28, 30],                   # 11: 3-Wheeler
    [20, 21, 26, 29, 31],               # 12: 4-Wheeler
    [24],                                # 13: Hand-Cart
    [35, 36, 37],                        # 14: Board
    [38, 39, 40, 41, 42],               # 15: Fencing
    [43, 44, 45, 46],                   # 16: Human
    [53],                                # 17: Road Side Furniture
    [52],                                # 18: Shop Shade
    [57],                                # 19: Mannequin
    [50, 51, 54, 55, 56, 58, 59, 60, 61],  # 20: Road Side Obj (other)
]
# NOTE: Original IDs 47 (Animal), 48 (Dog), 49 (Cow) are NOT in any group

# Class weights - Sqrt-scaled inverse frequency
class_weight = [
    4.8805, 7.7263, 3.6122, 1.8059, 1.0000, 1.7738, 3.6638, 7.1549, 7.1867,
    20.0000, 3.5210, 8.8729, 5.1955, 11.8747, 7.5868, 16.5094, 7.4137, 7.1182,
    7.2990, 16.3963, 13.9078,
]

# =============================================================================
# MODEL CONFIGURATION - PT-v3m1 (Train from Scratch)
# =============================================================================
model = dict(
    type="DefaultSegmentorV2",
    num_classes=num_classes,
    backbone_out_channels=64,  # dec_channels[0]
    backbone=dict(
        type="PT-v3m1",
        in_channels=4,  # XYZ(3) + intensity(1)
        order=["z", "z-trans", "hilbert", "hilbert-trans"],
        stride=(2, 2, 2, 2),
        # PT-v3m1 ENCODER
        enc_depths=(2, 2, 2, 6, 2),
        enc_channels=(32, 64, 128, 256, 512),
        enc_num_head=(2, 4, 8, 16, 32),
        enc_patch_size=(1024, 1024, 1024, 1024, 1024),
        # PT-v3m1 DECODER
        dec_depths=(2, 2, 2, 2),
        dec_channels=(64, 64, 128, 256),
        dec_num_head=(4, 4, 8, 16),
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
        enc_mode=False,  # Full encoder+decoder
        # PDNorm settings (PT-v3m1 specific)
        pdnorm_bn=False,
        pdnorm_ln=False,
        pdnorm_decouple=True,
        pdnorm_adaptive=False,
        pdnorm_affine=True,
        pdnorm_conditions=("Bhopal",),
    ),
    criteria=[
        dict(type="CrossEntropyLoss", weight=class_weight, loss_weight=1.0, ignore_index=-1),
        dict(type="LovaszLoss", mode="multiclass", loss_weight=1.0, ignore_index=-1),
    ],
)

# =============================================================================
# SCHEDULER SETTINGS
# =============================================================================
epoch = 100
eval_epoch = 100

optimizer = dict(type="AdamW", lr=0.002, weight_decay=0.005)

scheduler = dict(
    type="OneCycleLR",
    max_lr=[0.002, 0.0002],  # [base_lr, block_lr] — matching official PTv3 configs
    pct_start=0.04,
    anneal_strategy="cos",
    div_factor=10.0,
    final_div_factor=100.0,
)

# Differential LR: transformer blocks get 10x lower LR (matching official PTv3 NuScenes/Waymo)
param_dicts = [dict(keyword="block", lr=0.0002)]

# =============================================================================
# DATASET CONFIGURATION - 4ch (XYZ + intensity)
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
        use_intensity=True,    # 4ch: XYZ + intensity
        use_gps_time=False,
        use_normals=False,
        use_rgb=False,
        remap_classes=True,
        class_groups=class_groups,
        class_names=class_names,
        transform=[
            dict(type="RandomScale", scale=[0.2, 0.2]),
            dict(type="RandomRotate", angle=[-1, 1], axis="z", center=[0, 0, 0], p=0.5),
            dict(type="RandomScale", scale=[0.9, 1.1]),
            dict(type="RandomFlip", p=0.5),
            dict(type="RandomJitter", sigma=0.001, clip=0.004),  # Scaled for 0.2x coords (official: 0.005/0.02 at 1x)
            dict(
                type="GridSample",
                grid_size=0.01,
                hash_type="fnv",
                mode="train",
                return_grid_coord=True,
            ),
            dict(type="PointClip", point_cloud_range=(-149*0.2, -109*0.2, -22*0.2, 213*0.2, 159*0.2, 43*0.2)),
            # SphereCrop removed: not used in official PTv3 configs, and caused OOB errors with CopyPaste
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=("coord", "grid_coord", "segment"),
                feat_keys=("coord", "strength"),
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
        use_intensity=True,
        use_gps_time=False,
        use_normals=False,
        use_rgb=False,
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
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=("coord", "grid_coord", "segment", "origin_segment", "inverse"),
                feat_keys=("coord", "strength"),
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
        use_intensity=True,
        use_gps_time=False,
        use_normals=False,
        use_rgb=False,
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
                    feat_keys=("coord", "strength"),
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
