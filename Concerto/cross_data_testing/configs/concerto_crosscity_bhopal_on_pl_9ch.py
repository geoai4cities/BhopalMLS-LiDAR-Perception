"""
Cross-City Test Config: Concerto BM Model -> Paris-Lille-3D Data

Tests a Concerto/Bhopal-trained FT-9ch model on Paris-Lille-3D data
for the BM <-> PL cross-region cell (Concerto-L row).

=============================================================================
COORDINATE ALIGNMENT
=============================================================================
BM Concerto training PointClip (after 0.2x scale):
  X=[-29.8, 42.6], Y=[-21.8, 31.8], Z=[-4.4, 8.6]

This is the SAME range used by PL Concerto training (we deliberately picked
Bhopal-wide PCR for PL configs to match BM). Therefore PL scenes loaded
under BM training transforms are NOT clipped — coord distribution is
identical at training and cross-eval time.

The convenience comes for free: the cross-region table BM<->PL is the cleanest cross-region
cell because PCR alignment is exact (unlike BM<->KL which clips ~1.5x and
BM<->SK which needs 5x rescaling).
=============================================================================

Dataset: ParisLilleSceneHDF5Dataset (target data)
Model:   Concerto Large FT (PT-v3m2; same architecture as training)
Source:  Concerto/Bhopal/outputs/concerto_bhopal_finetune_9ch[_indoor]/model/model_best.pth
Target:  data/Paris-Lille-3D/paris_lille_scene_v1_normals.h5
Input:   9ch (XYZ + RGB + normal); PL has all-zero RGB padding so BM-trained
         weights see no color signal at eval (architectural caveat, not a
         config bug).
Classes: 12 unified BM-12 schema. PL fine 9-digit IDs are mapped to BM-12
         via PL_FINE_TO_BM12 (imported from pointcept.datasets.paris_lille).
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

H5_PATH = "data/Paris-Lille-3D/paris_lille_scene_v1_normals.h5"

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
    "Wire-Structure Connector",  # 6: ABSENT in PL — predictions for this
                                 #    class will be reported as ~0 IoU
    "Vehicle",                   # 7
    "Board",                     # 8
    "Fencing",                   # 9
    "Human and Animal",          # 10
    "Road Side Object",          # 11
]

# PL fine 9-digit IDs -> unified BM-12. Same mapping used at PL Concerto
# training; imported from the dataset class for explicit visibility.
from pointcept.datasets.paris_lille import PL_FINE_TO_BM12 as learning_map

# =============================================================================
# MODEL CONFIGURATION (must match Concerto FT-9ch training arch exactly)
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
# COORDINATE ALIGNMENT — BM training PCR (Bhopal-wide; same as PL training)
# =============================================================================
BHOPAL_POINT_CLIP = (-29.8, -21.8, -4.4, 42.6, 31.8, 8.6)

# =============================================================================
# DATASET CONFIGURATION
# =============================================================================
dataset_type = "ParisLilleSceneHDF5Dataset"
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
        learning_map=learning_map,
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
        learning_map=learning_map,
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
    # TEST: PL data with BM-aligned coords (PCR is identical, so no clipping)
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
        learning_map=learning_map,
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
