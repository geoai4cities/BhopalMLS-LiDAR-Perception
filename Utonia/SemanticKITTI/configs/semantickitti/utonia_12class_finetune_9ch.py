"""
Utonia Configuration for SemanticKITTI - FINE-TUNING (12 Classes)
FT-9ch: 9-Channel Input (XYZ + RGB + normals) - Full Fine-tuning

FINE-TUNING: enc_mode=False, full encoder+decoder trains
Based on: utonia_finetune_9ch.py (19-class) with unified 12-class mapping

Date: 2026-02-25
Status: ALIGNED WITH OFFICIAL UTONIA CONFIG

NOTE (2026-06-24) — DO NOT CHANGE: decoder is (96,96,192,384)/heads(8,8,16,32)/out=96,
the closest RoPE-valid match to the official PTv3-large decoder (64,96,192,384); the
official finest stage 64 is impossible on Utonia's RoPE (PT-v3m3) backbone because
Point3DRoPE requires head_dim div-by-3. This 12c config is wired into the cross-city
(Bhopal<->SK) merging experiments; its settings are intentionally frozen for those. The
19c FT/DP were aligned to this same (96,96,192,384) decoder on 2026-06-24. (Note the 12c
DP still uses the older (54,108,216,432) mirror decoder — also kept as-is; see that file.)

UNIFIED 12-CLASS CONFIG for Cross-Dataset Compatibility:
- Enables testing Bhopal-trained models on SemanticKITTI
- Enables testing SemanticKITTI-trained models on Bhopal
- Uses the same 12-class schema as Bhopal MLS dataset

Classes NOT present in SemanticKITTI:
- Class 4 (Wire): No power lines in KITTI - will be IGNORED
- Class 6 (Wire-Structure Connector): No insulators in KITTI - will be IGNORED
- Class 11 (Road Side Object): No equivalent - will be IGNORED

Key differences from Linear Probing:
- enc_mode=False (uses full decoder)
- freeze_backbone=False (all params trainable)
- Lower learning rate for backbone
"""

_base_ = ["../_base_/default_runtime.py"]

# =============================================================================
# MISC SETTINGS - Single 40GB GPU (fine-tuning trains all params + decoder)
# =============================================================================
batch_size = 4  # Fine-tuning needs more memory (all params + gradients)
batch_size_val = 1  # Matching official (None = auto 1/gpu from default_runtime.py)
batch_size_test = 1
num_worker = 16
mix_prob = 0.8
empty_cache = True  # Help with memory for fine-tuning
enable_amp = True
gradient_accumulation_steps = 16  # Effective batch size: 4 * 16 = 64

DATA_ROOT = "data/SemanticKITTI"
# Use released outdoor checkpoint - has embedding.*, enc.* keys (no prefix)
UTONIA_CHECKPOINT = "Utonia_pretrained/utonia.pth"

# =============================================================================
# UNIFIED 12-CLASS CONFIGURATION (Compatible with Bhopal)
# =============================================================================
num_classes = 12
class_names = [
    "Ground",                    # 0: KITTI sidewalk, other-ground, terrain
    "Vegetation",                # 1: KITTI vegetation, trunk
    "Building",                  # 2: KITTI building
    "Road",                      # 3: KITTI road, parking
    "Wire",                      # 4: NOT IN KITTI (ignored)
    "Pole and Tower",            # 5: KITTI pole
    "Wire-Structure Connector",  # 6: NOT IN KITTI (ignored)
    "Vehicle",                   # 7: KITTI car, bicycle, motorcycle, truck, other-vehicle
    "Board",                     # 8: KITTI traffic-sign
    "Fencing",                   # 9: KITTI fence
    "Human and Animal",          # 10: KITTI person, bicyclist, motorcyclist
    "Road Side Object",          # 11: NOT IN KITTI (ignored)
]

# Learning map: SemanticKITTI original label -> Unified 12-class
learning_map = {
    0: -1,    # unlabeled -> ignore
    1: -1,    # outlier -> ignore
    10: 7,    # car -> Vehicle
    11: 7,    # bicycle -> Vehicle
    13: 7,    # bus -> Vehicle
    15: 7,    # motorcycle -> Vehicle
    16: 7,    # on-rails -> Vehicle
    18: 7,    # truck -> Vehicle
    20: 7,    # other-vehicle -> Vehicle
    30: 10,   # person -> Human and Animal
    31: 10,   # bicyclist -> Human and Animal
    32: 10,   # motorcyclist -> Human and Animal
    40: 3,    # road -> Road
    44: 3,    # parking -> Road
    48: 0,    # sidewalk -> Ground
    49: 0,    # other-ground -> Ground
    50: 2,    # building -> Building
    51: 9,    # fence -> Fencing
    52: -1,   # other-structure -> ignore
    60: 3,    # lane-marking -> Road
    70: 1,    # vegetation -> Vegetation
    71: 1,    # trunk -> Vegetation
    72: 0,    # terrain -> Ground
    80: 5,    # pole -> Pole and Tower
    81: 8,    # traffic-sign -> Board
    99: -1,   # other-object -> ignore
    252: 7,   # moving-car -> Vehicle
    253: 10,  # moving-bicyclist -> Human and Animal
    254: 10,  # moving-person -> Human and Animal
    255: 10,  # moving-motorcyclist -> Human and Animal
    256: 7,   # moving-on-rails -> Vehicle
    257: 7,   # moving-bus -> Vehicle
    258: 7,   # moving-truck -> Vehicle
    259: 7,   # moving-other-vehicle -> Vehicle
}

# Classes NOT in SemanticKITTI
missing_classes = [4, 6, 11]  # Wire, Wire-Structure Connector, Road Side Object

class_weight = [
    1.9385,   # 0: Ground
    1.3212,   # 1: Vegetation
    2.0198,   # 2: Building
    1.6155,   # 3: Road
    1.0,      # 4: Wire (NOT IN KITTI)
    5.8585,   # 5: Pole and Tower
    1.0,      # 6: Wire-Structure Connector (NOT IN KITTI)
    3.1557,   # 7: Vehicle
    7.3929,   # 8: Board
    2.6261,   # 9: Fencing
    7.9937,   # 10: Human and Animal
    1.0,      # 11: Road Side Object (NOT IN KITTI)
]

# =============================================================================
# MODEL CONFIGURATION - FINE-TUNING (enc_mode=False)
# =============================================================================
model = dict(
    type="DefaultSegmentorV2",
    num_classes=num_classes,
    backbone_out_channels=96,  # Decoder output for fine-tuning
    backbone=dict(
        type="PT-v3m3",
        in_channels=9,  # coord(3) + color(3) + normal(3)
        order=("z", "z-trans", "hilbert", "hilbert-trans"),
        stride=(2, 2, 2, 2),
        # ENCODER
        enc_depths=(3, 3, 3, 12, 3),
        enc_channels=(54, 108, 216, 432, 576),
        enc_num_head=(3, 6, 12, 24, 32),
        enc_patch_size=(1024, 1024, 1024, 1024, 1024),
        # DECODER
        dec_depths=(3, 3, 3, 3),
        dec_channels=(96, 96, 192, 384),
        dec_num_head=(8, 8, 16, 32),  # Point3DRoPE: head_dim must be div-by-3 (96/8=12, 192/16=12, 384/32=12)
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
        rope_base=10,
        shift_coords=None,
        jitter_coords=None,
        rescale_coords=None,
        enc_mode=False,  # FINE-TUNING: Full encoder+decoder
        freeze_encoder=False,
    ),
    freeze_backbone=False,  # FINE-TUNING: All params trainable
    criteria=[
        dict(type="CrossEntropyLoss", weight=class_weight, loss_weight=1.0, ignore_index=-1),
        dict(type="LovaszLoss", mode="multiclass", loss_weight=1.0, ignore_index=-1),
    ],
)

# =============================================================================
# SCHEDULER SETTINGS - FINE-TUNING (lower backbone lr)
# =============================================================================
epoch = 50
eval_epoch = 50

optimizer = dict(type="AdamW", lr=0.002, weight_decay=0.005)

scheduler = dict(
    type="OneCycleLR",
    max_lr=[0.002, 0.0002],  # Backbone gets 10% lr
    pct_start=0.04,
    anneal_strategy="cos",
    div_factor=10.0,
    final_div_factor=100.0,
)

param_dicts = [dict(keyword="block", lr=0.0002)]

# =============================================================================
# DATASET CONFIGURATION
# =============================================================================
dataset_type = "SemanticKITTIImagePointDataset"
ignore_index = -1

# OFFICIAL KITTI PointClip — restored 2026-05-19, reverting the earlier
# Bhopal-wide PCR experiment. Bhopal-wide PCR + dropped SphereCrop pushed
# per-step voxel volume ~400x and made 100-epoch SK training >230 h.
# Cross-data SK->BM swap is unaffected: PCR is a dataset-side preprocessing
# transform applied at inference time, not baked into the checkpoint, so the
# BM-wide PCR can still be used at eval against Bhopal data.
PCR = (-35.2 * 0.2, -35.2 * 0.2, -4 * 0.2, 35.2 * 0.2, 35.2 * 0.2, 2 * 0.2)
# = (-7.04, -7.04, -0.8, 7.04, 7.04, 0.4)

data = dict(
    num_classes=num_classes,
    ignore_index=ignore_index,
    names=class_names,

    train=dict(
        type=dataset_type,
        split="train",
        if_img=False,
        data_root=DATA_ROOT,
        learning_map=learning_map,
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
            # Official KITTI SphereCrop: two-stage point cloud size limiting
            dict(type="SphereCrop", sample_rate=0.8, mode="random"),
            dict(type="SphereCrop", point_max=120000, mode="random"),
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
        split="val",
        if_img=False,
        data_root=DATA_ROOT,
        learning_map=learning_map,
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
        split="val",
        if_img=False,
        data_root=DATA_ROOT,
        learning_map=learning_map,
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
        ignore_index=ignore_index,
    ),
)

# =============================================================================
# HOOKS - FINE-TUNING (strict=False for partial loading)
# =============================================================================
hooks = [
    dict(
        type="CheckpointLoader",
        keywords="module.",  # For released outdoor checkpoint (embedding.*, enc.* keys)
        replacement="module.backbone.",  # Adds backbone. prefix after module.
        strict=False,  # Decoder will be randomly initialized
    ),
    dict(type="IterationTimer", warmup_iter=2),
    dict(type="InformationWriter"),
    dict(type="SemSegEvaluator"),
    dict(type="CheckpointSaver", save_freq=None),
    dict(type="PreciseEvaluator", test_last=False),
]

weight = UTONIA_CHECKPOINT
