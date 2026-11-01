"""
Concerto Configuration for Toronto-3D - FULL FINE-TUNING (BM-12 schema)
FT-9ch: 9-Channel Input (XYZ + REAL RGB + normals).

Cloned from Concerto/Paris-Lille-3D/configs/paris_lille/
concerto_140gb_12class_finetune_9ch.py with TR deltas:
  - dataset_type:  ParisLilleSceneHDF5Dataset -> Toronto3DSceneHDF5Dataset
  - H5_PATH:       data/Toronto_3D/toronto_scene_v1_normals_rgb.h5
  - remap form:    learning_map(PL_FINE_TO_BM12) -> class_groups(TR9_TO_BM12_GROUPS)
                   (TR's 9 dense small labels are cleaner expressed as
                   class_groups; mirrors the Bhopal convention.)
  - RGB:           PL3D zero-padded -> Toronto real RGB
  - class_weight:  recomputed from TR train histogram (L001+L003+L004)

Source: Toronto-3D (Tan et al. 2020) 9-class taxonomy. The dataset class
applies a class_groups remap (TR9_TO_BM12_GROUPS in
pointcept/datasets/toronto.py) that lifts the 9 native classes to the
unified BhopalMLS-12 schema at load time, so that the paper the cross-region table
self-trained rows {BM->BM, KM->KM, PL->PL, TR->TR} are directly
comparable on a single schema.

Class coverage (BM-12) on Toronto training data:
  BM 0  Ground            <- ABSENT (TR doesn't separate sidewalk/curb;
                              all goes to BM-3 via TR-1)
  BM 1  Vegetation        <- TR-3 Natural
  BM 2  Building          <- TR-4 Building
  BM 3  Road              <- TR-1 Road + TR-2 Road_markings (~63% of pts)
  BM 4  Wire              <- TR-5 Utility_line              (~0.5%)
  BM 5  Pole and Tower    <- TR-6 Pole                      (~1%)
  BM 6  Wire-Str-Conn     <- ABSENT
  BM 7  Vehicle           <- TR-7 Car                        (~3%)
  BM 8  Board             <- ABSENT
  BM 9  Fencing           <- TR-8 Fence                      (~0.02%, very rare)
  BM 10 Human and Animal  <- ABSENT
  BM 11 Road Side Object  <- ABSENT

Five BM-12 classes are absent in TR training (0, 6, 8, 10, 11) and
carry the 1.0 placeholder weight. mIoU should be reported over the 7
present classes (1, 2, 3, 4, 5, 7, 9) as in the paper.

GPU target: single 40GB A100 (batch_size=4, grad_accum=16, eff. batch 64).
Mirrors the Bhopal/PL3D 12c FT-9ch memory profile.

Date: 2026-05-05
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

# Toronto-3D HDF5 (V9-compatible schema; produced by
# data/Toronto_3D/preprocess_toronto_to_hdf5.py)
H5_PATH = "data/Toronto_3D/toronto_scene_v1_normals_rgb.h5"

# Pretrained Concerto checkpoint (released outdoor; matches BM/PL precedent)
CONCERTO_CHECKPOINT = "Concerto_pretrained/concerto_large_outdoor.pth"

# =============================================================================
# UNIFIED 12-CLASS CONFIGURATION (matches BhopalMLS / Paris-Lille-3D 12c)
# =============================================================================
num_classes = 12
class_names = [
    "Ground",                    # 0: ABSENT in TR
    "Vegetation",                # 1: TR-3 Natural
    "Building",                  # 2: TR-4 Building
    "Road",                      # 3: TR-1 Road + TR-2 Road_markings
    "Wire",                      # 4: TR-5 Utility_line
    "Pole and Tower",            # 5: TR-6 Pole
    "Wire-Structure Connector",  # 6: ABSENT in TR
    "Vehicle",                   # 7: TR-7 Car
    "Board",                     # 8: ABSENT in TR
    "Fencing",                   # 9: TR-8 Fence
    "Human and Animal",          # 10: ABSENT in TR
    "Road Side Object",          # 11: ABSENT in TR
]

# TR-9 -> BM-12 unified, expressed as Bhopal-style class_groups. The
# dataset class (Toronto3DSceneHDF5Dataset) provides this as its default,
# but we import & reuse it here for explicit visibility.
from pointcept.datasets.toronto import TR9_TO_BM12_GROUPS as class_groups

# Classes absent from TR training signal: BM 0, 6, 8, 10, 11. These
# carry the 1.0 placeholder weight since TR has no labelled equivalents.
missing_classes = [0, 6, 8, 10, 11]

# Sqrt-inverse-frequency class weights from the Toronto-3D 9-class
# training-split histogram (L001+L003+L004; 66.5M points after BM-12
# remap, 64.4M supervised). Formula:
#   w_c = sqrt(N_supervised / (n_present * count_c))
# where n_present = 7 (BM 1, 2, 3, 4, 5, 7, 9). Source:
# data/Toronto_3D/compute_class_weights.py against
# toronto_scene_v1_normals_rgb.h5 splits/train_indices.
class_weight = [
     1.0000,    # 0: Ground                       (ABSENT)
     1.4764,    # 1: Vegetation                   ( 6.554% from TR-3 Natural)
     0.7193,    # 2: Building                     (27.608% from TR-4 Building)
     0.5019,    # 3: Road                         (56.714% from TR-1+2)  — dominant
     3.9972,    # 4: Wire                         ( 0.894% from TR-5 Utility_line)
     3.5699,    # 5: Pole and Tower               ( 1.121% from TR-6 Pole)
     1.0000,    # 6: Wire-Structure Connector     (ABSENT)
     1.4701,    # 7: Vehicle                      ( 6.610% from TR-7 Car)
     1.0000,    # 8: Board                        (ABSENT)
     5.3513,    # 9: Fencing                      ( 0.499% from TR-8 Fence)  — rarest present
     1.0000,    # 10: Human and Animal            (ABSENT)
     1.0000,    # 11: Road Side Object            (ABSENT)
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
        in_channels=9,  # XYZ(3) + RGB(3, real Toronto) + normal(3)
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
dataset_type = "Toronto3DSceneHDF5Dataset"
ignore_index = -1

# PointClip: TR tiles are pre-centered at HDF5 write time (5m XY tile size,
# Z-range ~30m max for L001-L004). Use the SAME PCR as Bhopal/PL3D training
# (post-0.2x scale, Bhopal-wide) so cross-region inference is a drop-in
# checkpoint swap with no clipping mismatch.
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
        use_rgb=True,           # REAL RGB (Toronto-3D has uchar 0-255 R,G,B)
        remap_classes=True,
        class_groups=class_groups,
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
        class_groups=class_groups,
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
