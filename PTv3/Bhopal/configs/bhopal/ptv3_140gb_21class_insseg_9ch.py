"""
PTv3 Configuration for Bhopal MLS - INSTANCE SEGMENTATION (PointGroup)
PG-v1m2 + PT-v3m1: 9-Channel Input (XYZ + RGB + normals)
21-Class Fine-Grained Taxonomy

Based on:
- Bhopal semantic seg: ptv3_140gb_21class_train_9ch.py
- LitePT insseg: litept_L_21class_insseg_9ch.py (PointGroup model pattern)

Instance Segmentation uses PG-v1m2 (PointGroup) which adds:
- bias_head: Predicts offset from each point to its instance centroid
- Ball query clustering at inference to group points into instances

Stuff classes (excluded from instance clustering):
  -1: padding, 0: Ground, 2: Vegetation, 5: Road, 6: Wire

Thing classes evaluated for mAP (17 total):
  1: Stairs, 3: Building(other), 4: Commercial,
  7: Pole(other), 8: Electric Pole, 9: Street Light,
  10: 2-Wheeler, 11: 3-Wheeler, 12: 4-Wheeler, 13: Hand-Cart,
  14: Board, 15: Fencing, 16: Human,
  17: Road Side Furniture, 18: Shop Shade, 19: Mannequin, 20: RSO(other)

Architecture: PT-v3m1 (original Point Transformer V3)
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
| InsSeg Head     | 64 -> 21 classes     | TRAINABLE (random init)          |

TRAINABLE: ~22M params (entire model from scratch)
=============================================================================

Date: 2026-03-18
Status: NEW

Dataset: Bhopal V9 (with normals + RGB)
Input Channels: 9 = XYZ(3) + RGB(3) + normal(3)

GPU: H200 (143GB VRAM)
Estimated Memory: ~40-55GB (PointGroup uses more than semantic seg)
"""

_base_ = ["../_base_/default_runtime.py"]

# =============================================================================
# MISC SETTINGS
# =============================================================================
batch_size = 12  # Matching official PTv3 NuScenes/Waymo
batch_size_val = 1
batch_size_test = 1
num_worker = 16
num_worker_test = 1
mix_prob = 0.8  # CopyPaste enabled (matching official PTv3 configs)
empty_cache = False
enable_amp = True
clip_grad = 1.0  # Prevent gradient explosion / NaN propagation
evaluate = True  # REQUIRED for InsSegEvaluator

# HDF5 file path - V9 Dataset (with normals and RGB)
H5_PATH = "data/Bhopal/bhopal_scene_v9_normals_rgb.h5"

save_path = "exp/bhopal/ptv3_21class_insseg_9ch"

# =============================================================================
# 21-CLASS CONFIGURATION
# =============================================================================
num_classes = 21
segment_ignore_index = (-1, 0, 2, 5, 6)  # padding + Ground + Vegetation + Road + Wire

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
    4.8805,   # 0:  Ground (2.00%)
    7.7263,   # 1:  Stairs (0.80%)
    3.6122,   # 2:  Vegetation (3.65%)
    1.8059,   # 3:  Building (other) (14.62%)
    1.0000,   # 4:  Commercial (47.66%)
    1.7738,   # 5:  Road (15.15%)
    3.6638,   # 6:  Wire (3.55%)
    7.1549,   # 7:  Pole (other) (0.93%)
    7.1867,   # 8:  Electric Pole (0.92%)
    20.0000,  # 9:  Street Light (0.03%) - capped
    3.5210,   # 10: 2-Wheeler (3.84%)
    8.8729,   # 11: 3-Wheeler (0.61%)
    5.1955,   # 12: 4-Wheeler (1.77%)
    11.8747,  # 13: Hand-Cart (0.34%)
    7.5868,   # 14: Board (0.83%)
    16.5094,  # 15: Fencing (0.17%)
    7.4137,   # 16: Human (0.87%)
    7.1182,   # 17: Road Side Furniture (0.94%)
    7.2990,   # 18: Shop Shade (0.89%)
    16.3963,  # 19: Mannequin (0.18%)
    13.9078,  # 20: Road Side Obj (other) (0.25%)
]

# =============================================================================
# MODEL CONFIGURATION - PG-v1m2 (PointGroup) + PT-v3m1
# =============================================================================
model = dict(
    type="PG-v1m2",
    backbone=dict(
        type="PT-v3m1",
        in_channels=9,  # XYZ(3) + RGB(3) + normal(3)
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
    backbone_out_channels=64,  # dec_channels[0]
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
)

# =============================================================================
# SCHEDULER SETTINGS
# =============================================================================
epoch = 150
eval_epoch = 25

optimizer = dict(type="AdamW", lr=0.002, weight_decay=0.005)

scheduler = dict(
    type="OneCycleLR",
    max_lr=[0.002, 0.0002],  # [base_lr, block_lr] -- matching official PTv3 configs
    pct_start=0.04,
    anneal_strategy="cos",
    div_factor=10.0,
    final_div_factor=100.0,
)

# Differential LR: transformer blocks get 10x lower LR (matching official PTv3 NuScenes/Waymo)
param_dicts = [dict(keyword="block", lr=0.0002)]

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
        load_instance=True,
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
            dict(type="PointClip", point_cloud_range=(-149*0.2, -109*0.2, -22*0.2, 213*0.2, 159*0.2, 43*0.2)),
            dict(type="SphereCrop", sample_rate=0.8, mode="random"),
            dict(type="NormalizeColor"),
            dict(type="InstanceParser", segment_ignore_index=segment_ignore_index, instance_ignore_index=-1),
            dict(type="ToTensor"),
            dict(type="Update", keys_dict={"grid_size": 0.01}),
            dict(
                type="Collect",
                keys=("coord", "grid_coord", "segment", "instance", "instance_centroid", "bbox", "grid_size"),
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
            dict(
                type="GridSample",
                grid_size=0.01,
                hash_type="fnv",
                mode="train",
                return_grid_coord=True,
            ),
            dict(type="PointClip", point_cloud_range=(-149*0.2, -109*0.2, -22*0.2, 213*0.2, 159*0.2, 43*0.2)),
            dict(type="NormalizeColor"),
            dict(type="InstanceParser", segment_ignore_index=segment_ignore_index, instance_ignore_index=-1),
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=(
                    "coord", "grid_coord", "segment", "instance",
                    "origin_coord", "origin_segment", "origin_instance",
                    "instance_centroid", "bbox", "name",
                ),
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
            dict(
                type="GridSample",
                grid_size=0.01,
                hash_type="fnv",
                mode="train",
                return_grid_coord=True,
            ),
            dict(type="PointClip", point_cloud_range=(-149*0.2, -109*0.2, -22*0.2, 213*0.2, 159*0.2, 43*0.2)),
            dict(type="NormalizeColor"),
            dict(type="InstanceParser", segment_ignore_index=segment_ignore_index, instance_ignore_index=-1),
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=(
                    "coord", "grid_coord", "segment", "instance",
                    "origin_coord", "origin_segment", "origin_instance",
                    "instance_centroid", "bbox", "name",
                ),
                feat_keys=("coord", "color", "normal"),
                offset_keys_dict=dict(offset="coord", origin_offset="origin_coord"),
            ),
        ],
        test_mode=False,
        ignore_index=-1,
    ),
)

# =============================================================================
# HOOKS (from-scratch training, no pretrained checkpoint)
# =============================================================================
hooks = [
    dict(type="CheckpointLoader"),
    dict(type="IterationTimer", warmup_iter=2),
    dict(type="InformationWriter"),
    dict(
        type="InsSegEvaluator",
        segment_ignore_index=segment_ignore_index,
        instance_ignore_index=-1,
    ),
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
