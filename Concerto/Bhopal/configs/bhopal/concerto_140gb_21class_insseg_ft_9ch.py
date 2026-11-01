"""
Concerto Configuration for Bhopal MLS - INSTANCE SEGMENTATION Full Fine-tuning
PG-v1m2 + FT-9ch: 21-Class Fine-Grained Taxonomy

FULL FINE-TUNING: enc_mode=False, freeze_encoder=False
Entire model trains (encoder + decoder + seg_head + bias_head)

21-Class Taxonomy (from 62 original subclasses):
  - Stairs separated from Ground (mobility-relevant)
  - 2/3/4-Wheeler + Hand-Cart (India-specific vehicle taxonomy)
  - Road Side Object split into Furniture, Shop Shade, Mannequin, Other
  - Animals (Cow/Dog/Animal) mapped to ignore (too few instances)
  - Wire as stuff (continuous linear, not instanceable)

Stuff classes (excluded from instance clustering):
  -1: padding, 0: Ground, 2: Vegetation, 5: Road, 6: Wire

Thing classes evaluated for mAP: 17
  1: Stairs, 3: Building(other), 4: Commercial,
  7: Pole(other), 8: Electric Pole, 9: Street Light,
  10: 2-Wheeler, 11: 3-Wheeler, 12: 4-Wheeler, 13: Hand-Cart,
  14: Board, 15: Fencing, 16: Human,
  17: Road Side Furniture, 18: Shop Shade, 19: Mannequin, 20: RSO(other)

GPU: H200 (143GB VRAM)
Estimated Memory: ~80-100GB (full model trainable)
"""

_base_ = ["../_base_/default_runtime.py"]

# Misc settings
batch_size = 16
batch_size_val = 1
batch_size_test = 1
num_worker = 16
num_worker_test = 1
mix_prob = 0.8
empty_cache = False
enable_amp = True
clip_grad = 3.0
evaluate = True  # REQUIRED for InsSegEvaluator

# HDF5 file path
H5_PATH = "data/Bhopal/bhopal_scene_v9_normals_rgb.h5"

# Pre-trained checkpoint
CONCERTO_CHECKPOINT = "Concerto_pretrained/concerto_large_outdoor.pth"

# =============================================================================
# 21-CLASS CONFIGURATION
# =============================================================================
num_classes = 21
segment_ignore_index = (-1, 0, 2, 5, 6)  # padding + Ground + Vegetation + Road + Wire

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
# MODEL CONFIGURATION - PG-v1m2 (PointGroup) FULL FINE-TUNING
# =============================================================================
model = dict(
    type="PG-v1m2",
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
    backbone_out_channels=64,
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
    freeze_backbone=False,
)

# =============================================================================
# SCHEDULER SETTINGS
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
        load_instance=True,
        transform=[
            dict(type="RandomScale", scale=[0.2, 0.2]),
            dict(type="RandomRotate", angle=[-1, 1], axis="z", center=[0, 0, 0], p=0.5),
            dict(type="RandomScale", scale=[0.9, 1.1]),
            dict(type="RandomFlip", p=0.5),
            dict(type="RandomJitter", sigma=0.0025, clip=0.01),
            dict(type="GridSample", grid_size=0.01, hash_type="fnv", mode="train", return_grid_coord=True),
            dict(type="PointClip", point_cloud_range=(-149*0.2, -109*0.2, -22*0.2, 213*0.2, 159*0.2, 43*0.2)),
            dict(type="NormalizeColor"),
            dict(type="InstanceParser", segment_ignore_index=segment_ignore_index, instance_ignore_index=-1),
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=("coord", "grid_coord", "segment", "instance", "instance_centroid", "bbox"),
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
            dict(type="GridSample", grid_size=0.01, hash_type="fnv", mode="train", return_grid_coord=True),
            dict(type="PointClip", point_cloud_range=(-149*0.2, -109*0.2, -22*0.2, 213*0.2, 159*0.2, 43*0.2)),
            dict(type="NormalizeColor"),
            dict(type="InstanceParser", segment_ignore_index=segment_ignore_index, instance_ignore_index=-1),
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=("coord", "grid_coord", "segment", "instance",
                      "origin_coord", "origin_segment", "origin_instance",
                      "instance_centroid", "bbox", "name"),
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
            dict(type="GridSample", grid_size=0.01, hash_type="fnv", mode="train", return_grid_coord=True),
            dict(type="PointClip", point_cloud_range=(-149*0.2, -109*0.2, -22*0.2, 213*0.2, 159*0.2, 43*0.2)),
            dict(type="NormalizeColor"),
            dict(type="InstanceParser", segment_ignore_index=segment_ignore_index, instance_ignore_index=-1),
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=("coord", "grid_coord", "segment", "instance",
                      "origin_coord", "origin_segment", "origin_instance",
                      "instance_centroid", "bbox", "name"),
                feat_keys=("coord", "color", "normal"),
                offset_keys_dict=dict(offset="coord", origin_offset="origin_coord"),
            ),
        ],
        test_mode=False,
        ignore_index=-1,
    ),
)

# =============================================================================
# HOOKS
# =============================================================================
hooks = [
    dict(type="CheckpointLoader", keywords="module.", replacement="module.backbone."),
    dict(type="IterationTimer", warmup_iter=2),
    dict(type="InformationWriter"),
    dict(type="InsSegEvaluator", segment_ignore_index=segment_ignore_index, instance_ignore_index=-1),
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

# Pre-trained weights
weight = CONCERTO_CHECKPOINT
