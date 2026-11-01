"""
Utonia Configuration for Paris-Lille-3D - FULL FINE-TUNING (BM-12 schema)
FT-9ch: 9-Channel Input (XYZ + RGB + normals); RGB is zero-padded for PL.

Cloned from:
  Concerto/Paris-Lille-3D/configs/paris_lille/concerto_140gb_12class_finetune_9ch.py
with the Utonia backbone deltas (PT-v3m3, narrower channels, RoPE-aware decoder
heads). The dataset / HDF5 / learning_map / class_weight are reused verbatim
from the Concerto/PL pipeline; only the model and the pretrained checkpoint
change.

Architecture deltas vs Concerto/PL FT (cross-checked against
Utonia/Bhopal/configs/bhopal/utonia_140gb_12class_finetune_9ch.py):
  - backbone.type: PT-v3m2 -> PT-v3m3
  - enc_channels: (64,128,256,512,768) -> (54,108,216,432,576)
  - enc_num_head: (4,8,16,32,48) -> (3,6,12,24,32)
  - dec_depths: (2,2,2,2) -> (3,3,3,3)
  - dec_channels: (64,96,192,384) -> (96,96,192,384)
  - dec_num_head: (4,6,12,24) -> (8,8,16,32) (Point3DRoPE: head_dim div-by-3
    -> 96/8=12, 192/16=12, 384/32=12)
  - backbone_out_channels: 64 -> 96
  - extra m3 fields: rope_base, shift_coords, jitter_coords, rescale_coords
  - weight: concerto_large_outdoor.pth -> utonia.pth

GPU target: single 40GB A100 (batch_size=4, grad_accum=16, eff. batch 64).

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
empty_cache = True       # 40GB GPU; helps fragmentation. Bhopal Utonia uses
                         # False on H200 — that knob is memory/throughput only.
enable_amp = True
clip_grad = 3.0
gradient_accumulation_steps = 16  # effective batch 4 * 16 = 64

# Paris-Lille-3D HDF5 (V9-compatible schema; produced by
# data/Paris-Lille-3D/preprocess_pl3d_to_hdf5.py)
H5_PATH = "data/Paris-Lille-3D/paris_lille_scene_v1_normals.h5"

# Pretrained Utonia checkpoint (single released variant; matches BM/SK precedent)
UTONIA_CHECKPOINT = "Utonia_pretrained/utonia.pth"

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
# Dataset class default; imported here for explicit visibility.
from pointcept.datasets.paris_lille import PL_FINE_TO_BM12 as learning_map

# Only Wire-Structure Connector (BM 6) absent from PL training signal.
missing_classes = [6]

# Sqrt-inverse-frequency class weights from the PL3D 50-class fine taxonomy
# training-split histogram (116.4M supervised points; BM 6 absent).
# Source: paris_lille_scene_v1_normals.h5 splits/train_indices.
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
    backbone_out_channels=96,        # Utonia decoder output (vs Concerto's 64)
    backbone=dict(
        type="PT-v3m3",              # Utonia backbone (vs Concerto's PT-v3m2)
        in_channels=9,               # XYZ(3) + RGB(3, zero-padded) + normal(3)
        order=("z", "z-trans", "hilbert", "hilbert-trans"),
        stride=(2, 2, 2, 2),
        # Utonia encoder (trainable)
        enc_depths=(3, 3, 3, 12, 3),
        enc_channels=(54, 108, 216, 432, 576),
        enc_num_head=(3, 6, 12, 24, 32),
        enc_patch_size=(1024, 1024, 1024, 1024, 1024),
        # Utonia decoder (trainable)
        dec_depths=(3, 3, 3, 3),
        dec_channels=(96, 96, 192, 384),
        dec_num_head=(8, 8, 16, 32),  # Point3DRoPE: head_dim div-by-3 (96/8=12, 192/16=12, 384/32=12)
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
        # m3 extras (absent in m2)
        rope_base=10,
        shift_coords=None,
        jitter_coords=None,
        rescale_coords=None,
    ),
    freeze_backbone=False,
    criteria=[
        dict(type="CrossEntropyLoss", weight=class_weight, loss_weight=1.0, ignore_index=-1),
        dict(type="LovaszLoss", mode="multiclass", loss_weight=1.0, ignore_index=-1),
    ],
)

# =============================================================================
# SCHEDULER (differential LR; matches Bhopal Utonia FT)
# =============================================================================
epoch = 100
eval_epoch = 100

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
# DATASET CONFIGURATION
# =============================================================================
dataset_type = "ParisLilleSceneHDF5Dataset"
ignore_index = -1

# PointClip: PL tiles are pre-centered at HDF5 write time (5m XY tile size,
# Z-range ~30m max). Bhopal-wide bounds are wide enough to never clip a
# centered PL tile; matches the cross-region inference convention so the same
# range is used for PL training and BM<->PL evaluation.
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
        use_rgb=True,
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
# HOOKS - Released utonia.pth carries embedding.* + enc.* keys only; decoder
# and seg head are random-init. CheckpointLoader is non-strict by default
# (matches Bhopal Utonia FT signature exactly — no `strict=` field).
# =============================================================================
hooks = [
    dict(
        type="CheckpointLoader",
        keywords="module.",
        replacement="module.backbone.",
    ),
    dict(type="IterationTimer", warmup_iter=2),
    dict(type="InformationWriter"),
    dict(type="SemSegEvaluator"),
    dict(type="CheckpointSaver", save_freq=None),
    dict(type="PreciseEvaluator", test_last=False),
]

weight = UTONIA_CHECKPOINT
