"""
Concerto Configuration for Bhopal MLS - LINEAR PROBING (Official Style)
LP-3ch: XYZ Only - Geometry Only (zeros for color/normal)

TRUE LINEAR PROBING: enc_mode=True, only seg_head trains
Tests geometry-only performance (zeros for color and normal channels).

Note: Official Concerto always uses 9 channels. For this experiment,
we use coord(3) + zeros(3) + zeros(3) = 9 channels (3 meaningful: XYZ only).

Date: 2026-01-26
Status: ALIGNED WITH OFFICIAL CONCERTO CONFIG
"""

_base_ = ["../_base_/default_runtime.py"]

# =============================================================================
# MISC SETTINGS (OFFICIAL)
# =============================================================================
batch_size = 16
batch_size_val = 1  # Matching official (None = auto 1/gpu from default_runtime.py)
batch_size_test = 1
num_worker = 16
num_worker_test = 1
mix_prob = 0.8
empty_cache = False
enable_amp = True

H5_PATH = "data/Bhopal/bhopal_scene_v9_normals_rgb.h5"
CONCERTO_CHECKPOINT = "Concerto_pretrained/concerto_large_outdoor.pth"

# =============================================================================
# CLASS CONFIGURATION
# =============================================================================
num_classes = 12
class_names = [
    "Ground", "Vegetation", "Building", "Road", "Wire", "Pole and Tower",
    "Wire-Structure Connector", "Vehicle", "Board", "Fencing",
    "Human and Animal", "Road Side Object",
]

class_groups = [
    list(range(0, 4)), list(range(4, 8)), list(range(8, 13)), [13], [14],
    list(range(15, 19)), [19], list(range(20, 35)), list(range(35, 38)),
    list(range(38, 43)), list(range(43, 50)), list(range(50, 62)),
]

class_weight = [
    4.9549, 4.6483, 1.0000, 2.1667, 4.3169, 8.0928,
    10.5119, 3.3875, 9.6172, 15.8514, 8.9056, 4.8175,
]

# =============================================================================
# MODEL CONFIGURATION - OFFICIAL LINEAR PROBING (enc_mode=True)
# =============================================================================
model = dict(
    type="DefaultSegmentorV2",
    num_classes=num_classes,
    backbone_out_channels=1728,
    backbone=dict(
        type="PT-v3m2",
        in_channels=9,  # OFFICIAL: Always 9 channels
        order=("z", "z-trans", "hilbert", "hilbert-trans"),
        stride=(2, 2, 2, 2),
        enc_depths=(3, 3, 3, 12, 3),
        enc_channels=(64, 128, 256, 512, 768),
        enc_num_head=(4, 8, 16, 32, 48),
        enc_patch_size=(1024, 1024, 1024, 1024, 1024),
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
        enc_mode=True,
        freeze_encoder=False,
    ),
    freeze_backbone=True,
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
# DATASET CONFIGURATION - GEOMETRY ONLY (coord + zeros + zeros)
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
        use_normals=True,   # Load but will be zeroed
        use_rgb=True,       # Load but will be zeroed
        remap_classes=True,
        class_groups=class_groups,
        class_names=class_names,
        transform=[
            # OFFICIAL transform order (matching SemanticKITTI/Concerto)
            dict(type="RandomScale", scale=[0.2, 0.2]),
            dict(type="RandomRotate", angle=[-1, 1], axis="z", center=[0, 0, 0], p=0.5),
            dict(type="RandomScale", scale=[0.9, 1.1]),
            dict(type="RandomFlip", p=0.5),
            # OFFICIAL 3ch jitter (larger than 9ch - geometry only)
            dict(type="RandomJitter", sigma=0.0025, clip=0.01),
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
            dict(type="ZeroColor"),   # Zero RGB
            dict(type="ZeroNormal"),  # Zero normals → geometry only
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
            # BHOPAL PointClip: Based on analyze_bhopal_pointcloud_range.py (10% margin)
            # Bhopal data: X=[-135,193], Y=[-99,144], Z=[-20,39] -> with margin: X=[-149,213], Y=[-109,159], Z=[-22,43]
            # SemanticKITTI/Concerto official (too small for Bhopal): (-35.2*0.2, -35.2*0.2, -4*0.2, 35.2*0.2, 35.2*0.2, 2*0.2)
            dict(type="PointClip", point_cloud_range=(-149*0.2, -109*0.2, -22*0.2, 213*0.2, 159*0.2, 43*0.2)),
            dict(type="ZeroColor"),
            dict(type="ZeroNormal"),
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
            dict(type="ZeroColor"),
            dict(type="ZeroNormal"),
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
# HOOKS (For Released Weights: concerto_large_outdoor.pth)
# =============================================================================
hooks = [
    dict(
        type="CheckpointLoader",
        keywords="module.",           # Released weights have direct keys (embedding.*, enc.*)
        replacement="module.backbone.",  # Map to backbone.embedding.*, backbone.enc.*
    ),
    dict(type="IterationTimer", warmup_iter=2),
    dict(type="InformationWriter"),
    dict(type="SemSegEvaluator"),
    dict(type="CheckpointSaver", save_freq=None),
    dict(type="PreciseEvaluator", test_last=False),
]

weight = CONCERTO_CHECKPOINT
