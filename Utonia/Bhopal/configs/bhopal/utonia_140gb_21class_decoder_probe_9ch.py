"""
Utonia Configuration for Bhopal MLS - DECODER PROBING (Official Style)
DP-9ch: 9-Channel Input (XYZ + RGB + normals) - 21-Class Fine-Grained Taxonomy

DECODER PROBING: enc_mode=False, freeze_encoder=True, only decoder + seg_head train
Status: ALIGNED WITH OFFICIAL UTONIA CONFIG

Dataset: Bhopal V9 (with normals + RGB)
Input Channels: 9 = XYZ(3) + RGB(3) + normal(3) - OFFICIAL UTONIA INPUT

21-Class Taxonomy (from 62 original subclasses):
  - Stairs separated from Ground (mobility-relevant)
  - 2/3/4-Wheeler + Hand-Cart (India-specific vehicle taxonomy)
  - Road Side Object split into Furniture, Shop Shade, Mannequin, Other
  - Animals (Cow/Dog/Animal) mapped to ignore (too few instances)

=============================================================================
OFFICIAL DECODER PROBING ARCHITECTURE (enc_mode=False, freeze_encoder=True)
=============================================================================
| Encoder Stage | Channels | Heads | head_dim | Status                  |
|---------------|----------|-------|----------|-------------------------|
| Stage 0       | 54       | 3     | 18       | FROZEN (pre-trained)    |
| Stage 1       | 108      | 6     | 18       | FROZEN (pre-trained)    |
| Stage 2       | 216      | 12    | 18       | FROZEN (pre-trained)    |
| Stage 3       | 432      | 24    | 18       | FROZEN (pre-trained)    |
| Stage 4       | 576      | 32    | 18       | FROZEN (pre-trained)    |

| Decoder Stage | Channels | Heads | head_dim | Status                  |
|---------------|----------|-------|----------|-------------------------|
| Stage 3       | 384      | 32    | 12       | TRAINABLE (random init) |
| Stage 2       | 192      | 16    | 12       | TRAINABLE (random init) |
| Stage 1       | 96       | 8     | 12       | TRAINABLE (random init) |
| Stage 0       | 96       | 8     | 12       | TRAINABLE (random init) |
| TOTAL OUTPUT  | 96       |       |          | backbone_out_channels   |

NOTE: Decoder dec_num_head=(8, 8, 16, 32) instead of PTv3 default (6, 6, 12, 32)
because Utonia's Point3DRoPE requires head_dim % 3 == 0. The decoder is randomly
initialized (not in pre-trained checkpoint), so different head counts are fine.

FROZEN:     ~176M params (encoder including embedding)
TRAINABLE:  ~30-40M params (decoder) + ~2016 params (seg_head: 96 -> 21 classes)

This is DECODER PROBING - encoder frozen, decoder trainable!
=============================================================================
"""

_base_ = ["../_base_/default_runtime.py"]

# =============================================================================
# MISC SETTINGS (OFFICIAL)
# =============================================================================
batch_size = 16
batch_size_val = 1
batch_size_test = 1
num_worker = 16
num_worker_test = 1
mix_prob = 0.8
empty_cache = False
enable_amp = True

H5_PATH = "data/Bhopal/bhopal_scene_v9_normals_rgb.h5"
UTONIA_CHECKPOINT = "Utonia_pretrained/utonia.pth"

# =============================================================================
# 21-CLASS CONFIGURATION
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
    [22, 23, 32, 33, 34],               # 10: 2-Wheeler (Mobike+Cycle+RiderOnVeh+MobikeRider+CycleRider)
    [25, 27, 28, 30],                   # 11: 3-Wheeler (Tuktuk+Rickshaw+E-Rickshaw+DynTuktuk)
    [20, 21, 26, 29, 31],               # 12: 4-Wheeler (VehicleGen+Car+Truck+DynCar+DynTruck)
    [24],                                # 13: Hand-Cart
    [35, 36, 37],                        # 14: Board (Board+SignBoard+AdvtBoard)
    [38, 39, 40, 41, 42],               # 15: Fencing (all subtypes)
    [43, 44, 45, 46],                   # 16: Human (HumanAnimal gen+Male+Female+Child)
    [53],                                # 17: Road Side Furniture
    [52],                                # 18: Shop Shade
    [57],                                # 19: Mannequin
    [50, 51, 54, 55, 56, 58, 59, 60, 61],  # 20: Road Side Obj (other)
]
# NOTE: Original IDs 47 (Animal), 48 (Dog), 49 (Cow) are NOT in any group

class_weight = [
    4.8805,   # 0:  Ground (2.00%)
    7.7263,   # 1:  Stairs (0.80%)
    3.6122,   # 2:  Vegetation (3.65%)
    1.8059,   # 3:  Building (other) (14.62%)
    1.0000,   # 4:  Commercial (47.66%) - dominant class
    1.7738,   # 5:  Road (15.15%)
    3.6638,   # 6:  Wire (3.55%)
    7.1549,   # 7:  Pole (other) (0.93%)
    7.1867,   # 8:  Electric Pole (0.92%)
    20.0000,  # 9:  Street Light (0.03%) - capped from 38.96
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
# MODEL CONFIGURATION - OFFICIAL DECODER PROBING (enc_mode=False, freeze_encoder=True)
# =============================================================================
model = dict(
    type="DefaultSegmentorV2",
    num_classes=num_classes,
    backbone_out_channels=96,  # DECODER OUTPUT (not 1386 like linear probing)
    backbone=dict(
        type="PT-v3m3",
        in_channels=9,  # XYZ(3) + RGB(3) + normal(3)
        order=("z", "z-trans", "hilbert", "hilbert-trans"),
        stride=(2, 2, 2, 2),
        # UTONIA ENCODER (Official) - FROZEN
        enc_depths=(3, 3, 3, 12, 3),
        enc_channels=(54, 108, 216, 432, 576),
        enc_num_head=(3, 6, 12, 24, 32),
        enc_patch_size=(1024, 1024, 1024, 1024, 1024),
        # UTONIA DECODER (Official) - TRAINABLE
        dec_depths=(3, 3, 3, 3),
        dec_channels=(96, 96, 192, 384),
        dec_num_head=(8, 8, 16, 32),  # head_dim must be divisible by 3 for RoPE: 96/8=12, 192/16=12, 384/32=12
        dec_patch_size=(1024, 1024, 1024, 1024),
        # Shared params
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
        # KEY CHANGES FOR DECODER PROBING
        enc_mode=False,       # Use decoder (not encoder-only)
        freeze_encoder=True,  # Freeze encoder, train decoder
        rope_base=10,
        shift_coords=None,
        jitter_coords=None,
        rescale_coords=None,
    ),
    freeze_backbone=False,  # Allow decoder training (encoder frozen via freeze_encoder)
    criteria=[
        dict(type="CrossEntropyLoss", weight=class_weight, loss_weight=1.0, ignore_index=-1),
        dict(type="LovaszLoss", mode="multiclass", loss_weight=1.0, ignore_index=-1),
    ],
)

# =============================================================================
# SCHEDULER SETTINGS (OFFICIAL)
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
        use_normals=True,
        use_rgb=True,
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
            dict(type="PointClip", point_cloud_range=(-149*0.2, -109*0.2, -22*0.2, 213*0.2, 159*0.2, 43*0.2)),
            dict(type="NormalizeColor"),
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=("coord", "grid_coord", "segment"),
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
            dict(type="NormalizeColor"),
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=("coord", "grid_coord", "segment", "origin_segment", "inverse"),
                feat_keys=("coord", "color", "normal"),
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
        ignore_index=-1,
    ),
)

# =============================================================================
# HOOKS (For Released Weights: utonia.pth)
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
