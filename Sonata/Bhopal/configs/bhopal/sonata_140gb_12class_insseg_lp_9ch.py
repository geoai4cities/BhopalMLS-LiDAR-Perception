"""
Sonata Configuration for Bhopal MLS - INSTANCE SEGMENTATION Linear Probe
PG-v1m2 + LP-9ch: 9-Channel Input (XYZ + RGB + normals)

LINEAR PROBING: enc_mode=True, freeze_backbone=True
Only seg_head + bias_head train

Based on:
- Bhopal semantic LP: sonata_140gb_12class_linear_probe_9ch.py
- Bhopal Concerto instance LP: concerto_140gb_12class_insseg_lp_9ch.py

Instance Segmentation uses PG-v1m2 (PointGroup) which adds:
- bias_head: Predicts offset from each point to its instance centroid
- Ball query clustering at inference to group points into instances

Architecture (enc_mode=True):
  Encoder: 48+96+192+384+512 = 1232-dim concatenated output (FROZEN)
  seg_head: 1232 -> 12 classes (TRAINABLE)
  bias_head: 1232 -> 3 offset (TRAINABLE)

Stuff classes (excluded from instance clustering):
  -1: padding, 0: Ground, 1: Vegetation, 3: Road, 4: Wire

GPU: H200 (143GB VRAM)
Estimated Memory: ~30-40GB (encoder-only, no decoder)
"""

_base_ = ["../_base_/default_runtime.py"]

# Misc settings
batch_size = 16
batch_size_val = 1
batch_size_test = 1
num_worker = 16
num_worker_test = 1
mix_prob = 0.8
empty_cache = True
enable_amp = True
evaluate = True  # REQUIRED for InsSegEvaluator

# HDF5 file path
H5_PATH = "data/Bhopal/bhopal_scene_v9_normals_rgb.h5"

# Pre-trained checkpoint
SONATA_CHECKPOINT = "sonata_pretrained/pretrain-sonata-v1m1-0-base.pth"

# Class configuration
num_classes = 12
segment_ignore_index = (-1, 0, 1, 3, 4)  # padding + Ground + Vegetation + Road + Wire (stuff classes)

class_names = [
    "Ground", "Vegetation", "Building", "Road", "Wire",
    "Pole and Tower", "Wire-Structure Connector", "Vehicle",
    "Board", "Fencing", "Human and Animal", "Road Side Object",
]

class_groups = [
    list(range(0, 4)), list(range(4, 8)), list(range(8, 13)),
    [13], [14], list(range(15, 19)), [19],
    list(range(20, 35)), list(range(35, 38)), list(range(38, 43)),
    list(range(43, 50)), list(range(50, 62)),
]

class_weight = [
    4.9549, 4.6483, 1.0000, 2.1667, 4.3169,
    8.0928, 10.5119, 3.3875, 9.6172, 15.8514,
    8.9056, 4.8175,
]

# =============================================================================
# MODEL CONFIGURATION - PG-v1m2 (PointGroup) LINEAR PROBING
# =============================================================================
model = dict(
    type="PG-v1m2",
    backbone=dict(
        type="PT-v3m2",
        in_channels=9,
        order=("z", "z-trans", "hilbert", "hilbert-trans"),
        stride=(2, 2, 2, 2),
        # SONATA ENCODER (Official) - FROZEN
        enc_depths=(3, 3, 3, 12, 3),
        enc_channels=(48, 96, 192, 384, 512),
        enc_num_head=(3, 6, 12, 24, 32),
        enc_patch_size=(1024, 1024, 1024, 1024, 1024),
        # No decoder params needed for enc_mode=True
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
        enc_mode=True,           # LP: Encoder-only mode (no decoder)
        freeze_encoder=False,    # Handled by freeze_backbone
    ),
    backbone_out_channels=1232,  # Sum of all encoder channels (48+96+192+384+512)
    semantic_num_classes=num_classes,
    semantic_ignore_index=-1,
    segment_ignore_index=segment_ignore_index,
    instance_ignore_index=-1,
    cluster_thresh=1.5,
    cluster_closed_points=600,
    cluster_propose_points=200,
    cluster_min_points=50,
    voxel_size=0.01,             # MUST match grid_size
    criteria=[
        dict(type="CrossEntropyLoss", weight=class_weight, loss_weight=1.0, ignore_index=-1),
        dict(type="LovaszLoss", mode="multiclass", loss_weight=1.0, ignore_index=-1),
    ],
    freeze_backbone=True,        # FREEZE entire backbone (linear probing)
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
            dict(type="RandomJitter", sigma=0.001, clip=0.004),
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
# HOOKS (Sonata pre-training checkpoint mapping)
# =============================================================================
hooks = [
    dict(type="CheckpointLoader", keywords="module.student.backbone", replacement="module.backbone"),
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
weight = SONATA_CHECKPOINT
