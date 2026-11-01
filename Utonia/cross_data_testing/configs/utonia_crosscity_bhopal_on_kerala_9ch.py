"""
Cross-City Test Config: Utonia BM Model -> Kerala Corridor Data

Tests a Utonia-Bhopal-trained FT-9ch model on Kerala corridor data
for the BM <-> KL cross-region cell.

=============================================================================
COORDINATE ALIGNMENT
=============================================================================
BM Utonia training PointClip (after 0.2x scale):
  X=[-29.8, 42.6], Y=[-21.8, 31.8], Z=[-4.4, 8.6]
KL data after 0.2x scale (KL Utonia training PointClip):
  X=[-17.4, 25.8], Y=[-31.8, 20.0], Z=[-6.2, 10.4]

KL exceeds BM's range on Y_min (-31.8 < -21.8), Z_min (-6.2 < -4.4),
and Z_max (10.4 > 8.6). PointClip discards out-of-range points. Mismatch
is mild (~1.5x on Y_min) — much less severe than the BM<->SK 5x case, no
extra rescaling required.

Source-PCR convention (matches Concerto BM<->KL cross-configs): BM Utonia
was trained on the Bhopal-wide PCR only, so KL data must be clipped into
that distribution at eval time. (Unlike BM<->PL where PL Utonia was
deliberately trained with the Bhopal-wide PCR — that shortcut is not
available here without retraining KL.)
=============================================================================

Model:    Utonia FT-9ch (PT-v3m3, BM-12 schema) — same arch as training
Source:   Utonia/Bhopal/outputs/utonia_bhopal_finetune_9ch/model/model_best.pth
Target:   data/Kerala/Kerala_scene_v1_corridor.h5
Input:    9ch (XYZ + RGB + normal)
Classes:  12 unified BM-12 schema. KL fine 48 raw classes mapped via class_groups.
"""

_base_ = ["../../Bhopal/configs/_base_/default_runtime.py"]

# =============================================================================
# MISC SETTINGS (test-only stubs — required by the config system)
# =============================================================================
batch_size = 1
batch_size_val = 1
batch_size_test = 1
num_worker = 16
num_worker_test = 1
mix_prob = 0.0
empty_cache = False
enable_amp = True

H5_PATH = "data/Kerala/Kerala_scene_v1_corridor.h5"

# =============================================================================
# CLASS CONFIGURATION (12 unified classes)
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

# Kerala raw (48) -> unified BM-12 (same mapping used at KL Utonia training).
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
# MODEL CONFIGURATION (must match Utonia FT-9ch training arch exactly)
# =============================================================================
model = dict(
    type="DefaultSegmentorV2",
    num_classes=num_classes,
    backbone_out_channels=96,
    backbone=dict(
        type="PT-v3m3",
        in_channels=9,
        order=("z", "z-trans", "hilbert", "hilbert-trans"),
        stride=(2, 2, 2, 2),
        enc_depths=(3, 3, 3, 12, 3),
        enc_channels=(54, 108, 216, 432, 576),
        enc_num_head=(3, 6, 12, 24, 32),
        enc_patch_size=(1024, 1024, 1024, 1024, 1024),
        dec_depths=(3, 3, 3, 3),
        dec_channels=(96, 96, 192, 384),
        dec_num_head=(8, 8, 16, 32),  # Point3DRoPE: head_dim div-by-3
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
        rope_base=10,
        shift_coords=None,
        jitter_coords=None,
        rescale_coords=None,
    ),
    freeze_backbone=False,
    criteria=[
        dict(type="CrossEntropyLoss", loss_weight=1.0, ignore_index=-1),
        dict(type="LovaszLoss", mode="multiclass", loss_weight=1.0, ignore_index=-1),
    ],
)

# Scheduler (not used for testing — required by the config system)
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
# COORDINATE ALIGNMENT — BM training PCR (source model)
# =============================================================================
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

    # train/val: minimal stubs (not used at test time)
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
    # TEST: KL data with BM-aligned coords (clips KL points outside BM PCR)
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
            dict(type="RandomScale", scale=[0.2, 0.2]),
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
                [dict(type="RandomScale", scale=[0.9, 0.9]), dict(type="RandomFlip", p=1)],
                [dict(type="RandomScale", scale=[0.95, 0.95]), dict(type="RandomFlip", p=1)],
                [dict(type="RandomScale", scale=[1, 1]), dict(type="RandomFlip", p=1)],
                [dict(type="RandomScale", scale=[1.05, 1.05]), dict(type="RandomFlip", p=1)],
                [dict(type="RandomScale", scale=[1.1, 1.1]), dict(type="RandomFlip", p=1)],
            ],
        ),
        ignore_index=ignore_index,
    ),
)

# Hooks (not used by test.py, but required by the config system)
hooks = [
    dict(type="IterationTimer", warmup_iter=2),
    dict(type="InformationWriter"),
    dict(type="SemSegEvaluator"),
    dict(type="CheckpointSaver", save_freq=None),
]
