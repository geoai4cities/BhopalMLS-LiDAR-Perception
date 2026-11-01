"""
Concerto Configuration for Paris-Lille-3D - FULL FINE-TUNING (BM-12 schema)
FT-9ch: 9-Channel Input (XYZ + RGB + normals); RGB is zero-padded for PL.

Cloned from:
  Concerto/Bhopal/configs/bhopal/concerto_140gb_12class_finetune_9ch.py
with PL deltas: H5_PATH, dataset_type, learning_map, class_weight.

Source: Paris-Lille-3D 50-class fine taxonomy (Roynard 2017). The dataset
class applies a learning_map (PL_FINE_TO_BM12 in
pointcept/datasets/paris_lille.py) that remaps the 9-digit fine class IDs to
the BhopalMLS unified 12-class taxonomy at load time, so that the cross-region table
self-trained rows {BM->BM, KM->KM, PL->PL} are directly comparable on a
single schema. Same pattern as Concerto/SemanticKITTI/configs/semantickitti/
concerto_12class_finetune_9ch.py.

Class coverage (BM-12) on PL training data:
  BM 0 Ground            <- PL sidewalk(202030000) + curb(202040000) +
                            island(202050000) + grass/lawn(202060000) + ...
  BM 1 Vegetation        <- PL natural(304*) (tree, bush, hedge)
  BM 2 Building          <- PL building(203000000)
  BM 3 Road              <- PL road(202020000)            (~44% of points)
  BM 4 Wire              <- PL wire(302030600)            (rare but present)
  BM 5 Pole and Tower    <- PL post + bollard + lamp + traffic light + sign
  BM 6 Wire-Str-Conn     <- ABSENT in PL (no PL equivalent for insulators)
  BM 7 Vehicle           <- PL all 2-/4-wheelers (303030*, 303040*)
  BM 8 Board             <- PL signboard(302020700)
  BM 9 Fencing           <- PL barrier + grid + chain + low wall + wired fence
  BM 10 Human and Animal <- PL pedestrian variants(303020*)
  BM 11 Road Side Object <- PL trash can + meter + bench + shelter + ...

Only BM 6 (Wire-Structure Connector) carries the 1.0 placeholder weight
since PL has no labelled equivalent. All other BM-12 classes have at least
one PL fine ID feeding into them.

GPU target: single 40GB A100 (batch_size=4, grad_accum=16, eff. batch 64).
Mirrors the SemanticKITTI 12c FT-9ch memory profile.

Date: 2026-05-04
"""

_base_ = ["../_base_/default_runtime.py"]

# =============================================================================
# MISC SETTINGS (single 40GB GPU)
# =============================================================================
batch_size = 4
batch_size_val = 1
batch_size_test = 1
num_worker = 16
num_worker_test = 1
mix_prob = 0.8
empty_cache = True
enable_amp = True
clip_grad = 3.0
gradient_accumulation_steps = 16  # effective batch 4 * 16 = 64

# Paris-Lille-3D HDF5 (V9-compatible schema; produced by
# data/Paris-Lille-3D/preprocess_pl3d_to_hdf5.py)
H5_PATH = "data/Paris-Lille-3D/paris_lille_scene_v1_normals.h5"

# Pretrained Concerto checkpoint (released outdoor; matches SK/BM precedent)
CONCERTO_CHECKPOINT = "Concerto_pretrained/concerto_large_outdoor.pth"

# =============================================================================
# UNIFIED 12-CLASS CONFIGURATION (matches BhopalMLS / SemanticKITTI 12c)
# =============================================================================
num_classes = 12
class_names = [
    "Ground",                    # 0: PL sidewalk + curb + island + grass + ...
    "Vegetation",                # 1: PL tree + bush + hedge + natural
    "Building",                  # 2: PL building
    "Road",                      # 3: PL road (202020000)
    "Wire",                      # 4: PL wire (302030600)
    "Pole and Tower",            # 5: PL post + bollard + lamp + traffic light/sign
    "Wire-Structure Connector",  # 6: ABSENT in PL — placeholder
    "Vehicle",                   # 7: PL all 2-/4-wheelers (cars, vans, trucks, ...)
    "Board",                     # 8: PL signboard (302020700)
    "Fencing",                   # 9: PL barrier + grid + chain + low wall + ...
    "Human and Animal",          # 10: PL pedestrian variants
    "Road Side Object",          # 11: PL trash can + meter + bench + shelter + ...
]

# Learning map: PL3D fine 9-digit class IDs -> BM-12 unified.
# The dataset class (ParisLilleSceneHDF5Dataset) provides this as its default
# learning_map (PL_FINE_TO_BM12 in pointcept/datasets/paris_lille.py), built
# from the official Roynard 2017 50-class taxonomy. We import & reuse it
# here for explicit visibility in the config; passing it via `learning_map=`
# is identical to omitting it (dataset class falls back to its default).
from pointcept.datasets.paris_lille import PL_FINE_TO_BM12 as learning_map

# Classes absent from PL training signal: only Wire-Structure Connector
# (BM 6) — there is no PL fine class for insulators / wire-structure
# connectors. All other BM-12 classes have at least one PL fine class
# mapped via the learning_map above.
missing_classes = [6]  # Wire-Structure Connector

# Sqrt-inverse-frequency class weights computed from the actual PL3D
# 50-class fine taxonomy training-split histogram (116.4M supervised points
# across 11 of 12 BM-12 classes; only BM 6 Wire-Str-Conn is absent in PL).
# Source: paris_lille_scene_v1_normals.h5 splits/train_indices.
# Formula: w_c = sqrt(N_supervised / (n_present * count_c)).
class_weight = [
     0.8778,   # 0: Ground            (11.797%)
     1.0364,   # 1: Vegetation        ( 8.464%)
     0.6239,   # 2: Building          (23.354%)
     0.4407,   # 3: Road              (46.801%) — dominant
    27.4756,   # 4: Wire              ( 0.012%) — rarest present class
     3.9779,   # 5: Pole and Tower    ( 0.574%)
     1.0000,   # 6: Wire-Str-Conn     (ABSENT — placeholder)
     1.4060,   # 7: Vehicle           ( 4.599%)
    11.5161,   # 8: Board             ( 0.069%)
     1.5448,   # 9: Fencing           ( 3.809%)
     7.7743,   # 10: Human and Animal ( 0.150%)
     4.9578,   # 11: Road Side Object ( 0.370%)
]

# =============================================================================
# MODEL CONFIGURATION - FULL FINE-TUNING (enc_mode=False, freeze_encoder=False)
# =============================================================================
model = dict(
    type="DefaultSegmentorV2",
    num_classes=num_classes,
    backbone_out_channels=64,
    backbone=dict(
        type="PT-v3m2",
        in_channels=9,  # XYZ(3) + RGB(3, zero-padded) + normal(3)
        order=("z", "z-trans", "hilbert", "hilbert-trans"),
        stride=(2, 2, 2, 2),
        # Concerto LARGE encoder (trainable)
        enc_depths=(3, 3, 3, 12, 3),
        enc_channels=(64, 128, 256, 512, 768),
        enc_num_head=(4, 8, 16, 32, 48),
        enc_patch_size=(1024, 1024, 1024, 1024, 1024),
        # Concerto LARGE decoder (trainable)
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
        dict(type="CrossEntropyLoss", weight=class_weight, loss_weight=1.0, ignore_index=-1),
        dict(type="LovaszLoss", mode="multiclass", loss_weight=1.0, ignore_index=-1),
    ],
)

# =============================================================================
# SCHEDULER (differential LR, official Concerto)
# =============================================================================
epoch = 100
eval_epoch = 100

optimizer = dict(type="AdamW", lr=0.002, weight_decay=0.005)

scheduler = dict(
    type="OneCycleLR",
    max_lr=[0.002, 0.0002],  # head full lr, backbone block 10x lower
    pct_start=0.04,
    anneal_strategy="cos",
    div_factor=10.0,
    final_div_factor=100.0,
)

param_dicts = [dict(keyword="block", lr=0.0002)]

# =============================================================================
# DATASET CONFIGURATION
# =============================================================================
dataset_type = "ParisLilleSceneHDF5Dataset"
ignore_index = -1

# PointClip: PL tiles are pre-centered at HDF5 write time (5m XY tile size,
# Z-range ~30m max). The Bhopal range (in 0.2x-scaled coords) is wide enough
# to never clip a centered PL tile and matches the cross-region inference
# config so the same range is used for PL training and BM<->PL evaluation.
# Bhopal raw bounds + 10% margin: X=[-149,213], Y=[-109,159], Z=[-22,43].
PCR = (-149*0.2, -109*0.2, -22*0.2, 213*0.2, 159*0.2, 43*0.2)

data = dict(
    num_classes=num_classes,
    ignore_index=ignore_index,
    names=class_names,

    train=dict(
        type=dataset_type,
        h5_path=H5_PATH,
        split="train",
        num_points=None,
        use_intensity=False,
        use_gps_time=False,
        use_normals=True,
        use_rgb=True,           # zero-padded; keeps 9ch input compatibility
        remap_classes=True,
        learning_map=learning_map,
        class_names=class_names,
        transform=[
            dict(type="RandomScale", scale=[0.2, 0.2]),
            dict(type="RandomRotate", angle=[-1, 1], axis="z", center=[0, 0, 0], p=0.5),
            dict(type="RandomScale", scale=[0.9, 1.1]),
            dict(type="RandomFlip", p=0.5),
            dict(type="RandomJitter", sigma=0.001, clip=0.004),
            dict(
                type="GridSample",
                grid_size=0.01,
                hash_type="fnv",
                mode="train",
                return_grid_coord=True,
            ),
            dict(type="PointClip", point_cloud_range=PCR),
            dict(type="NormalizeColor"),
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=("coord", "grid_coord", "segment"),
                feat_keys=("coord", "color", "normal"),
            ),
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
            dict(
                type="GridSample",
                grid_size=0.01,
                hash_type="fnv",
                mode="train",
                return_grid_coord=True,
                return_inverse=True,
            ),
            dict(type="PointClip", point_cloud_range=PCR),
            dict(type="NormalizeColor"),
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=("coord", "grid_coord", "segment", "origin_segment", "inverse"),
                feat_keys=("coord", "color", "normal"),
            ),
        ],
        test_mode=False,
        ignore_index=ignore_index,
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
        learning_map=learning_map,
        class_names=class_names,
        transform=[
            dict(type="RandomScale", scale=[0.2, 0.2]),
            dict(type="PointClip", point_cloud_range=PCR),
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

# =============================================================================
# HOOKS - FINE-TUNING (strict=False for partial loading; decoder is random init)
# =============================================================================
hooks = [
    dict(
        type="CheckpointLoader",
        keywords="module.",
        replacement="module.backbone.",
        strict=False,
    ),
    dict(type="IterationTimer", warmup_iter=2),
    dict(type="InformationWriter"),
    dict(type="SemSegEvaluator"),
    dict(type="CheckpointSaver", save_freq=None),
    dict(type="PreciseEvaluator", test_last=False),
]

weight = CONCERTO_CHECKPOINT
