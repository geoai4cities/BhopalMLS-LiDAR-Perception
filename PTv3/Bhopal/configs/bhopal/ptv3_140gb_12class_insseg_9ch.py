"""
PTv3 Configuration for Bhopal MLS - INSTANCE SEGMENTATION (PointGroup)
PG-v1m2 + PT-v3m1: 9-Channel Input (XYZ + RGB + normals)

Based on:
- Bhopal semantic seg: ptv3_140gb_12class_train_9ch.py
- LitePT insseg: litept_L_12class_insseg_9ch.py (PointGroup model pattern)

Instance Segmentation uses PG-v1m2 (PointGroup) which adds:
- bias_head: Predicts offset from each point to its instance centroid
- Ball query clustering at inference to group points into instances

Stuff classes (excluded from instance clustering):
  -1: padding, 0: Ground, 1: Vegetation, 3: Road, 4: Wire

Thing classes evaluated for mAP (8 total):
  2: Building, 5: Pole and Tower, 6: Wire-Structure Connector,
  7: Vehicle, 8: Board, 9: Fencing, 10: Human and Animal,
  11: Road Side Object

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
| InsSeg Head     | 64 -> 12 classes     | TRAINABLE (random init)          |

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

save_path = "exp/bhopal/ptv3_12class_insseg_9ch"

# =============================================================================
# CLASS CONFIGURATION
# =============================================================================
num_classes = 12
segment_ignore_index = (-1, 0, 1, 3, 4)  # padding + Ground + Vegetation + Road + Wire (stuff classes)

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
