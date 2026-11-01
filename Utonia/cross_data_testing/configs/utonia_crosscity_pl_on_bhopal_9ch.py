"""
Cross-City Test Config: Utonia PL Model -> Bhopal Data

Tests a Utonia-Paris-Lille-3D-trained FT-9ch model on Bhopal MLS data
for the the paper's cross-region table PL <-> BM transfer cell.

=============================================================================
COORDINATE ALIGNMENT
=============================================================================
PL Utonia training PointClip (after 0.2x scale) — Bhopal-wide:
  X=[-29.8, 42.6], Y=[-21.8, 31.8], Z=[-4.4, 8.6]

This is the SAME range Bhopal Utonia training uses (we deliberately picked
Bhopal-wide PCR for the PL configs to make cross-domain a checkpoint swap).
So a BM scene loaded under PL training transforms is NOT clipped — every
BM point that the BM-trained model would have seen is also visible to the
PL-trained model.

The convenience comes for free: the cross-region table BM<->PL is the cleanest cross-region
cell in the paper because PCR alignment is exact (unlike BM<->KL which
clips ~1.5x and BM<->SK which needs 5x rescaling).
=============================================================================

Model:    Utonia FT-9ch (PT-v3m3, BM-12 schema) — same arch as training
Source:   Utonia/Paris-Lille-3D/outputs/utonia_paris_lille_finetune_9ch/model/model_best.pth
Target:   data/Bhopal/bhopal_scene_v9_normals_rgb.h5
Input:    9ch (XYZ + RGB + normal); BM has ~40% real RGB vs PL's all-zero
          padding — PL-trained weights ignore color signal so BM RGB is
          unused at eval (architectural caveat, not a config bug).
Classes:  12 unified BM-12 schema
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

H5_PATH = "data/Bhopal/bhopal_scene_v9_normals_rgb.h5"

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

# Bhopal raw (62) -> unified BM-12 (same mapping used by all BM cross-data
# configs in Concerto/cross_data_testing/configs/).
class_groups = [
    [0, 1, 2, 3],                                       # 0: Ground
    [4, 5, 6, 7],                                       # 1: Vegetation
    [8, 9, 10, 11, 12],                                 # 2: Building
    [13],                                               # 3: Road
    [14],                                               # 4: Wire
    [15, 16, 17, 18],                                   # 5: Pole and Tower
    [19],                                               # 6: Wire-Structure Connector
    [20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34],  # 7: Vehicle
    [35, 36, 37],                                       # 8: Board
    [38, 39, 40, 41, 42],                               # 9: Fencing
    [43, 44, 45, 46, 47, 48, 49],                       # 10: Human and Animal
    [50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61],   # 11: Road Side Object
]

# =============================================================================
# MODEL CONFIGURATION (must match Utonia FT-9ch training arch exactly so the
# saved state_dict loads with strict=True via test.py)
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
# COORDINATE ALIGNMENT — PL training PCR (Bhopal-wide; matches BM training too)
# =============================================================================
BHOPAL_POINT_CLIP = (-29.8, -21.8, -4.4, 42.6, 31.8, 8.6)

# =============================================================================
# DATASET CONFIGURATION
# =============================================================================
dataset_type = "BhopalSceneHDF5Dataset"
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
    # TEST: BM data with PL-aligned coords (PCR is identical, so no clipping)
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
