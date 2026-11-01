"""
Utonia Configuration for Kerala MLS - INSTANCE SEGMENTATION (PointGroup)
PG-v1m2 + FT-9ch: 9-Channel Input (XYZ + RGB + normals)

FULL FINE-TUNING: enc_mode=False, freeze_encoder=False, entire model trains

Based on:
- Kerala semantic FT: utonia_140gb_12class_finetune_9ch.py
- Concerto instance FT: concerto_140gb_12class_insseg_ft_9ch.py

Instance Segmentation uses PG-v1m2 (PointGroup) which adds:
- bias_head: Predicts offset from each point to its instance centroid
- Ball query clustering at inference to group points into instances

Architecture (enc_mode=False, freeze_encoder=False):
  Backbone: PT-v3m3 (Utonia) - NOT PT-v3m2 (Concerto)
  Encoder: 54->108->216->432->576 (TRAINABLE, lower LR)
  Decoder: 384->192->96->96 (TRAINABLE)
  seg_head: 96 -> 12 classes (TRAINABLE)
  bias_head: 96 -> 3 offset (TRAINABLE)

Utonia-specific differences from Concerto:
- PT-v3m3 backbone with Point3DRoPE (rope_base=10)
- Smaller encoder channels: (54,108,216,432,576) vs Concerto (64,128,256,512,768)
- Deeper decoder: dec_depths=(3,3,3,3) vs Concerto (2,2,2,2)
- Different decoder channels: (96,96,192,384) vs Concerto (64,96,192,384)
- dec_num_head=(8,8,16,32) - head_dim must be divisible by 3 for RoPE
- Smaller jitter: sigma=0.001, clip=0.004 vs Concerto sigma=0.0025, clip=0.01
- Additional params: shift_coords=None, jitter_coords=None, rescale_coords=None

Stuff classes (excluded from instance clustering):
  -1: padding, 0: Ground, 1: Vegetation, 3: Road, 4: Wire

GPU: H200 (143GB VRAM)
Estimated Memory: ~80-90GB (full encoder+decoder training)
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
H5_PATH = "data/Kerala/Kerala_scene_v1_corridor.h5"

# Pre-trained checkpoint
UTONIA_CHECKPOINT = "Utonia_pretrained/utonia.pth"

# Class configuration
num_classes = 12
segment_ignore_index = (-1, 0, 1, 3, 4)  # padding + Ground + Vegetation + Road + Wire (stuff classes)

class_names = [
    "Ground", "Vegetation", "Building", "Road", "Wire",
    "Pole and Tower", "Wire-Structure Connector", "Vehicle",
    "Board", "Fencing", "Human and Animal", "Road Side Object",
]

class_groups = [
    list(range(0, 4)), list(range(4, 8)), list(range(8, 12)),
    [12], [13], list(range(14, 17)), list(range(17, 19)),
    list(range(19, 29)), list(range(29, 32)), list(range(32, 37)),
    list(range(37, 44)), list(range(44, 48)),
]

class_weight = [
    2.03, 1.00, 3.20, 1.71, 10.97,
    11.34, 100.44, 10.41, 12.46, 4.74,
    20.02, 76.34,
]

# Model: PG-v1m2 (PointGroup) - FULL FINE-TUNING
model = dict(
    type="PG-v1m2",
    backbone=dict(
        type="PT-v3m3",
        in_channels=9,
        order=("z", "z-trans", "hilbert", "hilbert-trans"),
        stride=(2, 2, 2, 2),
        # UTONIA ENCODER (Official) - TRAINABLE
        enc_depths=(3, 3, 3, 12, 3),
        enc_channels=(54, 108, 216, 432, 576),
        enc_num_head=(3, 6, 12, 24, 32),
        enc_patch_size=(1024, 1024, 1024, 1024, 1024),
        # UTONIA DECODER (Official) - TRAINABLE
        dec_depths=(3, 3, 3, 3),
        dec_channels=(96, 96, 192, 384),
        dec_num_head=(8, 8, 16, 32),  # head_dim must be divisible by 3 for RoPE: 96/8=12, 192/16=12, 384/32=12
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
    backbone_out_channels=96,
    semantic_num_classes=num_classes,
    semantic_ignore_index=-1,
    segment_ignore_index=segment_ignore_index,
    instance_ignore_index=-1,
    cluster_thresh=1.5,
    cluster_closed_points=600,
    cluster_propose_points=200,
    cluster_min_points=50,
    voxel_size=0.01,  # MUST match grid_size (Kerala uses 0.01, not ScanNet 0.02)
    criteria=[
        dict(type="CrossEntropyLoss", weight=class_weight, loss_weight=1.0, ignore_index=-1),
        dict(type="LovaszLoss", mode="multiclass", loss_weight=1.0, ignore_index=-1),
    ],
    freeze_backbone=False,
)

# Scheduler
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

# Dataset
dataset_type = "KeralaSceneHDF5Dataset"

# PointClip — Bhopal-wide PCR for clean BM<->KM transfer (matches TR/PL).
# KM-native PCR (commented below) clipped here at Y_min and Z edges.
# Kerala_scene_v1_corridor.h5 bounds: X[-78.27, 116.93], Y[-143.83, 90.79],
# Z[-27.94, 46.75]  ->  ceil(|raw|*1.1) -> *0.2
# _point_clip_range = (-87*0.2, -159*0.2, -31*0.2, 129*0.2, 100*0.2, 52*0.2)
_point_clip_range = (-29.8, -21.8, -4.4, 42.6, 31.8, 8.6)

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
            dict(type="PointClip", point_cloud_range=_point_clip_range),
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
            dict(type="PointClip", point_cloud_range=_point_clip_range),
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
            dict(type="PointClip", point_cloud_range=_point_clip_range),
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

# Hooks
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
weight = UTONIA_CHECKPOINT
