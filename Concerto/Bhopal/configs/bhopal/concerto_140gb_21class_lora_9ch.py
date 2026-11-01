"""
Concerto Configuration for Bhopal MLS - LoRA FINE-TUNING (Official Style)
LoRA-9ch: 9-Channel Input (XYZ + RGB + normals) - 21-Class Fine-Grained Taxonomy

LoRA FINE-TUNING: enc_mode=True, freeze_encoder=True, only LoRA adapters train

21-Class Taxonomy (from 62 original subclasses):
  - Stairs separated from Ground (mobility-relevant)
  - 2/3/4-Wheeler + Hand-Cart (India-specific vehicle taxonomy)
  - Road Side Object split into Furniture, Shop Shade, Mannequin, Other
  - Animals (Cow/Dog/Animal) mapped to ignore (too few instances)

=============================================================================
OFFICIAL LoRA FINE-TUNING ARCHITECTURE (enc_mode=True, freeze_encoder=True)
=============================================================================
Low-Rank Adaptation (LoRA) adds small trainable matrices to frozen layers:
- Base weights are frozen
- LoRA adapters (rank=8) add ~0.1% trainable params to attention layers

| Component       | Channels              | Status                           |
|-----------------|----------------------|----------------------------------|
| Encoder Base    | 64->128->256->512->768   | FROZEN (pre-trained weights)     |
| LoRA Adapters   | rank=8 per layer     | TRAINABLE (~0.2M params)         |
| Decoder         | Not used             | enc_mode=True                    |
| Seg Head        | 1728 -> 21 classes    | TRAINABLE (~36.3K params)        |

FROZEN:     ~207.67M params (encoder base weights)
TRAINABLE:  ~0.2M params (LoRA adapters) + ~36.3K params (seg_head)

This is LoRA FINE-TUNING - parameter-efficient adaptation!
=============================================================================

LoRA Parameters (Official):
- lora_r = 8 (rank of low-rank matrices)
- lora_alpha = 16 (scaling factor)
- lora_dropout = 0.1 (dropout for regularization)

GPU: H200 (143GB VRAM)
Estimated Memory: ~40-50GB (similar to linear probing)
"""

_base_ = ["../_base_/default_runtime.py"]

# =============================================================================
# MISC SETTINGS (OFFICIAL)
# =============================================================================
batch_size = 16  # Will be overridden by script for GPU-specific settings
batch_size_val = 1  # Matching official (None = auto 1/gpu from default_runtime.py)
batch_size_test = 1
num_worker = 16
num_worker_test = 1
mix_prob = 0.8  # OFFICIAL: MixUp augmentation enabled
empty_cache = False
enable_amp = True
clip_grad = 3.0  # OFFICIAL: Gradient clipping for stability

# HDF5 file path - V9 Dataset (with normals and RGB)
H5_PATH = "data/Bhopal/bhopal_scene_v9_normals_rgb.h5"

# Pre-trained checkpoint path (used by LoRA model's backbone_path)
CONCERTO_CHECKPOINT = "Concerto_pretrained/concerto_large_outdoor.pth"

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
#       -> mapped to ignore_index by the dataset loader

# Sqrt-scaled inverse frequency weights from train split point distribution
# Street Light capped at 20.0 (raw=38.96, only 0.03% of points = 170K points)
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
# MODEL CONFIGURATION - OFFICIAL LoRA FINE-TUNING
# =============================================================================
model = dict(
    type="DefaultLORASegmentorV2",  # Special LoRA segmentor
    num_classes=num_classes,
    backbone_out_channels=1728,  # ENCODER CONCATENATED OUTPUT (like linear probing)
    # LoRA Configuration
    use_lora=True,
    lora_r=8,           # Rank of LoRA matrices
    lora_alpha=16,      # Scaling factor (alpha/r = 2)
    lora_dropout=0.1,   # Dropout for regularization
    # Checkpoint loading keys for released weights
    keywords="module.",
    replacements="module.backbone.",
    # Path to pre-trained weights (loaded by LoRA model)
    backbone_path=CONCERTO_CHECKPOINT,
    backbone=dict(
        type="PT-v3m2",
        in_channels=9,  # XYZ(3) + RGB(3) + normal(3)
        order=("z", "z-trans", "hilbert", "hilbert-trans"),
        stride=(2, 2, 2, 2),
        # CONCERTO LARGE ENCODER (Official) - BASE WEIGHTS FROZEN
        enc_depths=(3, 3, 3, 12, 3),
        enc_channels=(64, 128, 256, 512, 768),
        enc_num_head=(4, 8, 16, 32, 48),
        enc_patch_size=(1024, 1024, 1024, 1024, 1024),
        # No decoder for LoRA (enc_mode=True)
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
        # KEY SETTINGS FOR LoRA
        enc_mode=True,        # Encoder-only mode (no decoder)
        traceable=True,       # REQUIRED for LoRA
        mask_token=False,
        freeze_encoder=True,  # Freeze base encoder weights
    ),
    freeze_backbone=False,  # Allow LoRA adapters to train
    criteria=[
        dict(type="CrossEntropyLoss", weight=class_weight, loss_weight=1.0, ignore_index=-1),
        dict(type="LovaszLoss", mode="multiclass", loss_weight=1.0, ignore_index=-1),
    ],
)

# =============================================================================
# SCHEDULER SETTINGS (OFFICIAL)
# =============================================================================
epoch = 100
eval_epoch = 100  # Must equal epoch for proper display

optimizer = dict(type="AdamW", lr=0.002, weight_decay=0.005)  # OFFICIAL: weight_decay=0.005

scheduler = dict(
    type="OneCycleLR",
    max_lr=[0.002, 0.0002],  # OFFICIAL: [base_lr, block_lr]
    pct_start=0.04,  # OFFICIAL
    anneal_strategy="cos",
    div_factor=10.0,
    final_div_factor=100.0,  # OFFICIAL: 100 (not 1000)
)

# OFFICIAL: Blocks get reduced learning rate (LoRA adapters inside blocks)
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
            # OFFICIAL transform order (matching SemanticKITTI/Concerto)
            dict(type="RandomScale", scale=[0.2, 0.2]),
            dict(type="RandomRotate", angle=[-1, 1], axis="z", center=[0, 0, 0], p=0.5),
            dict(type="RandomScale", scale=[0.9, 1.1]),
            dict(type="RandomFlip", p=0.5),
            # OFFICIAL 9ch jitter (smaller than 4ch for color data sensitivity)
            dict(type="RandomJitter", sigma=0.001, clip=0.004),
            dict(
                type="GridSample",
                grid_size=0.01,
                hash_type="fnv",
                mode="train",
                return_grid_coord=True,
            ),
            # BHOPAL PointClip: Based on analyze_bhopal_pointcloud_range.py (10% margin)
            # Bhopal data: X=[-135,193], Y=[-99,144], Z=[-20,39] -> with margin: X=[-149,213], Y=[-109,159], Z=[-22,43]
            # SemanticKITTI/Concerto official (too small for Bhopal): (-35.2*0.2, -35.2*0.2, -4*0.2, 35.2*0.2, 35.2*0.2, 2*0.2)
            dict(type="PointClip", point_cloud_range=(-149*0.2, -109*0.2, -22*0.2, 213*0.2, 159*0.2, 43*0.2)),
            # SphereCrop REMOVED: Official Concerto nuScenes/Waymo configs comment it out.
            # Removing preserves full spatial context (critical for dense Bhopal MLS data).
            dict(type="NormalizeColor"),
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=("coord", "grid_coord", "segment"),
                feat_keys=("coord", "color", "normal"),  # OFFICIAL ORDER
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
            # BHOPAL PointClip: Based on analyze_bhopal_pointcloud_range.py (10% margin)
            # Bhopal data: X=[-135,193], Y=[-99,144], Z=[-20,39] -> with margin: X=[-149,213], Y=[-109,159], Z=[-22,43]
            # SemanticKITTI/Concerto official (too small for Bhopal): (-35.2*0.2, -35.2*0.2, -4*0.2, 35.2*0.2, 35.2*0.2, 2*0.2)
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
            # BHOPAL PointClip: Based on analyze_bhopal_pointcloud_range.py (10% margin)
            # Bhopal data: X=[-135,193], Y=[-99,144], Z=[-20,39] -> with margin: X=[-149,213], Y=[-109,159], Z=[-22,43]
            # SemanticKITTI/Concerto official (too small for Bhopal): (-35.2*0.2, -35.2*0.2, -4*0.2, 35.2*0.2, 35.2*0.2, 2*0.2)
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
# HOOKS (LoRA uses backbone_path for loading, not the hook)
# =============================================================================
# NOTE: For LoRA, the checkpoint is loaded via backbone_path in the model config.
# The CheckpointLoader hook is still included for compatibility but backbone_path
# takes precedence for LoRA models.
hooks = [
    dict(
        type="CheckpointLoader",
        keywords="module.",           # Released weights have direct keys
        replacement="module.backbone.",  # Map to backbone.*
    ),
    dict(type="IterationTimer", warmup_iter=2),
    dict(type="InformationWriter"),
    dict(type="SemSegEvaluator"),
    dict(type="CheckpointSaver", save_freq=None),
    dict(type="PreciseEvaluator", test_last=False),
]

# NOTE: For LoRA, weight is loaded via backbone_path in model config
# This is kept for hook compatibility
weight = CONCERTO_CHECKPOINT
