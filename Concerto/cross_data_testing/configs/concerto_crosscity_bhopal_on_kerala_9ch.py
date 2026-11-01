"""
Cross-City Test Config: Bhopal Model -> Kerala Data (with Coordinate Alignment)

Tests a BhopalMLS-trained Concerto FT-9ch model on Kerala corridor data.
Uses Bhopal's PointClip range so Kerala data is constrained to the
Bhopal model's learned coordinate distribution. Some Kerala points
outside Bhopal's range will be clipped.

=============================================================================
COORDINATE ALIGNMENT (Why this config exists)
=============================================================================
Bhopal model trained PointClip (after 0.2x scale):
  X=[-29.8, 42.6], Y=[-21.8, 31.8], Z=[-4.4, 8.6]
Kerala data after 0.2x scale (training PointClip):
  X=[-17.4, 25.8], Y=[-31.8, 20.0], Z=[-6.2, 10.4]

Kerala exceeds Bhopal's range on Y_min (-31.8 < -21.8), Z_min (-6.2 < -4.4),
and Z_max (10.4 > 8.6). PointClip will discard out-of-range points. The
mismatch is mild (~1.5x on Y_min), much less severe than the BM<->SK 5x
case — no extra rescaling is required.
=============================================================================

Dataset: KeralaSceneHDF5Dataset (target data)
Model: Concerto Large FT (same architecture as training)
Input: 9ch (XYZ + RGB + normals)
Classes: 12 (unified mapping)
"""

_base_ = ["../../Bhopal/configs/_base_/default_runtime.py"]

# =============================================================================
# MISC SETTINGS (test-only, these are not critical)
# =============================================================================
batch_size = 1
batch_size_val = 1
batch_size_test = 1
num_worker = 16
num_worker_test = 1
mix_prob = 0.0
empty_cache = False
enable_amp = True

# HDF5 file path - Kerala corridor v1
H5_PATH = "data/Kerala/Kerala_scene_v1_corridor.h5"

# =============================================================================
# CLASS CONFIGURATION (12 unified classes — same schema as Bhopal)
# =============================================================================
num_classes = 12
class_names = [
    "Ground",                    # 0
    "Vegetation",                # 1
    "Building",                  # 2
    "Road",                      # 3
    "Wire",                      # 4
    "Pole and Tower",            # 5
    "Wire-Structure Connector",  # 6
    "Vehicle",                   # 7
    "Board",                     # 8
    "Fencing",                   # 9
    "Human and Animal",          # 10
    "Road Side Object",          # 11
]

# Kerala raw -> unified 12 classes (48 raw classes)
class_groups = [
    [0, 1, 2, 3],                                       # 0: Ground
    [4, 5, 6, 7],                                       # 1: Vegetation
    [8, 9, 10, 11],                                     # 2: Building
    [12],                                               # 3: Road
    [13],                                               # 4: Wire
    [14, 15, 16],                                       # 5: Pole and Tower
    [17, 18],                                           # 6: Wire-Structure Connector
    [19, 20, 21, 22, 23, 24, 25, 26, 27, 28],           # 7: Vehicle
    [29, 30, 31],                                       # 8: Board
    [32, 33, 34, 35, 36],                               # 9: Fencing
    [37, 38, 39, 40, 41, 42, 43],                       # 10: Human and Animal
    [44, 45, 46, 47],                                   # 11: Road Side Object
]

# =============================================================================
# MODEL CONFIGURATION (same architecture, loaded from Bhopal checkpoint)
# =============================================================================
model = dict(
    type="DefaultSegmentorV2",
    num_classes=num_classes,
    backbone_out_channels=64,
    backbone=dict(
        type="PT-v3m2",
        in_channels=9,
        order=("z", "z-trans", "hilbert", "hilbert-trans"),
        stride=(2, 2, 2, 2),
        enc_depths=(3, 3, 3, 12, 3),
        enc_channels=(64, 128, 256, 512, 768),
        enc_num_head=(4, 8, 16, 32, 48),
        enc_patch_size=(1024, 1024, 1024, 1024, 1024),
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
        enc_mode=False,
        freeze_encoder=False,
    ),
    freeze_backbone=False,
    criteria=[
        dict(type="CrossEntropyLoss", loss_weight=1.0, ignore_index=-1),
        dict(type="LovaszLoss", mode="multiclass", loss_weight=1.0, ignore_index=-1),
    ],
)

# Scheduler (not used for testing, but required by config system)
epoch = 1
eval_epoch = 1
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
# COORDINATE ALIGNMENT PARAMETERS
# =============================================================================
# Bhopal model's training PointClip (after 0.2x scale)
BHOPAL_POINT_CLIP = (-29.8, -21.8, -4.4, 42.6, 31.8, 8.6)

# =============================================================================
# DATASET CONFIGURATION
# =============================================================================
dataset_type = "KeralaSceneHDF5Dataset"
ignore_index = -1

data = dict(
    num_classes=num_classes,
    ignore_index=ignore_index,
    names=class_names,

    # Train/Val: minimal stubs (not used during testing)
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
            dict(type="GridSample", grid_size=0.01, hash_type="fnv", mode="train", return_grid_coord=True),
            dict(type="PointClip", point_cloud_range=BHOPAL_POINT_CLIP),
            dict(type="NormalizeColor"),
            dict(type="ToTensor"),
            dict(type="Collect", keys=("coord", "grid_coord", "segment"), feat_keys=("coord", "color", "normal")),
        ],
        test_mode=False,
        ignore_index=ignore_index,
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
            dict(type="GridSample", grid_size=0.01, hash_type="fnv", mode="train", return_grid_coord=True, return_inverse=True),
            dict(type="PointClip", point_cloud_range=BHOPAL_POINT_CLIP),
            dict(type="NormalizeColor"),
            dict(type="ToTensor"),
            dict(type="Collect", keys=("coord", "grid_coord", "segment", "origin_segment", "inverse"), feat_keys=("coord", "color", "normal")),
        ],
        test_mode=False,
        ignore_index=ignore_index,
    ),

    # =========================================================================
    # TEST: Kerala data with Bhopal-aligned coordinate transforms
    # =========================================================================
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
            # 1. Standard 0.2x coordinate scaling (same as all Concerto configs)
            dict(type="RandomScale", scale=[0.2, 0.2]),
            # 2. Bhopal model's PointClip — clips KL points outside BM range
            dict(type="PointClip", point_cloud_range=BHOPAL_POINT_CLIP),
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
        ignore_index=ignore_index,
    ),
)

# Hooks (not used for testing, but required by config system)
hooks = [
    dict(type="IterationTimer", warmup_iter=2),
    dict(type="InformationWriter"),
    dict(type="SemSegEvaluator"),
    dict(type="CheckpointSaver", save_freq=None),
]
