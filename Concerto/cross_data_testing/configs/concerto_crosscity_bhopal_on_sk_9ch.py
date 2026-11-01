"""
Cross-City Test Config: Bhopal Model -> SK Data (with Coordinate Alignment)

Tests a BhopalMLS-trained Concerto FT-9ch model on SemanticKITTI data.
Uses Bhopal's wider PointClip range so SK data stays within the Bhopal
model's learned coordinate distribution.

=============================================================================
COORDINATE ALIGNMENT (Why this config exists)
=============================================================================
Bhopal model was trained with coords in [-29.8, 42.6] (after 0.2x + PointClip).
SK data after 0.2x has coords in [-16, 16] — this fits WITHIN Bhopal's range.

So no additional rescaling is needed for BM->SK (unlike SK->BM which requires
rescaling because Bhopal's range exceeds SK's PointClip).

Key insight: SK data naturally fits inside Bhopal's PointClip range:
  - SK X: [-16, 16] inside Bhopal [-29.8, 42.6]  ✓
  - SK Y: [-16, 16] inside Bhopal [-21.8, 31.8]  ✓
  - SK Z: [-0.25, 0.55] inside Bhopal [-4.4, 8.6] ✓

grid_coord for SK (~3200) is smaller than Bhopal's training (~7300).
This is a partial mismatch but much less severe than the 5x issue in
the reverse direction.
=============================================================================

Dataset: SemanticKITTIImagePointDataset (target data)
Model: Concerto Large FT (same architecture as training)
Input: 9ch (XYZ + RGB + normals)
Classes: 12 (unified mapping)
"""

_base_ = ["../../SemanticKITTI/configs/_base_/default_runtime.py"]

# =============================================================================
# MISC SETTINGS (test-only, these are not critical)
# =============================================================================
batch_size = 1
batch_size_val = 1
batch_size_test = 1
num_worker = 16
num_worker_test = 1
mix_prob = 0.0
empty_cache = False
enable_amp = True

DATA_ROOT = "data/SemanticKITTI"

# =============================================================================
# CLASS CONFIGURATION (same 12 unified classes)
# =============================================================================
num_classes = 12
class_names = [
    "Ground",                    # 0
    "Vegetation",                # 1
    "Building",                  # 2
    "Road",                      # 3
    "Wire",                      # 4: NOT IN KITTI (ignored in learning_map)
    "Pole and Tower",            # 5
    "Wire-Structure Connector",  # 6: NOT IN KITTI (ignored in learning_map)
    "Vehicle",                   # 7
    "Board",                     # 8
    "Fencing",                   # 9
    "Human and Animal",          # 10
    "Road Side Object",          # 11: NOT IN KITTI (ignored in learning_map)
]

# SemanticKITTI label -> unified 12-class mapping
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

# =============================================================================
# MODEL CONFIGURATION (same architecture, loaded from Bhopal checkpoint)
# =============================================================================
model = dict(
    type="DefaultSegmentorV2",
    num_classes=num_classes,
    backbone_out_channels=64,
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
    freeze_backbone=False,
    criteria=[
        dict(type="CrossEntropyLoss", loss_weight=1.0, ignore_index=-1),
        dict(type="LovaszLoss", mode="multiclass", loss_weight=1.0, ignore_index=-1),
    ],
)

# Scheduler (not used for testing, but required by config system)
epoch = 1
eval_epoch = 1
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
# COORDINATE ALIGNMENT PARAMETERS
# =============================================================================
# Bhopal model was trained with this PointClip (after 0.2x scale):
# Bhopal data: X=[-135,193], Y=[-99,144], Z=[-20,39]
# With 10% margin -> X=[-149,213], Y=[-109,159], Z=[-22,43]
BHOPAL_POINT_CLIP = (-149*0.2, -109*0.2, -22*0.2, 213*0.2, 159*0.2, 43*0.2)
# = (-29.8, -21.8, -4.4, 42.6, 31.8, 8.6)

# No additional scaling needed: SK coords [-16,16] fit within Bhopal's [-29.8, 42.6]

# =============================================================================
# DATASET CONFIGURATION
# =============================================================================
dataset_type = "SemanticKITTIImagePointDataset"
ignore_index = -1

data = dict(
    num_classes=num_classes,
    ignore_index=ignore_index,
    names=class_names,

    # Train/Val: minimal stubs (not used during testing)
    train=dict(
        type=dataset_type,
        split="train",
        if_img=False,
        data_root=DATA_ROOT,
        learning_map=learning_map,
        transform=[
            dict(type="RandomScale", scale=[0.2, 0.2]),
            dict(type="GridSample", grid_size=0.01, hash_type="fnv", mode="train", return_grid_coord=True),
            dict(type="PointClip", point_cloud_range=BHOPAL_POINT_CLIP),
            dict(type="NormalizeColor"),
            dict(type="ToTensor"),
            dict(type="Collect", keys=("coord", "grid_coord", "segment"), feat_keys=("coord", "color", "normal")),
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
            dict(type="GridSample", grid_size=0.01, hash_type="fnv", mode="train", return_grid_coord=True, return_inverse=True),
            dict(type="PointClip", point_cloud_range=BHOPAL_POINT_CLIP),
            dict(type="NormalizeColor"),
            dict(type="ToTensor"),
            dict(type="Collect", keys=("coord", "grid_coord", "segment", "origin_segment", "inverse"), feat_keys=("coord", "color", "normal")),
        ],
        test_mode=False,
        ignore_index=ignore_index,
    ),

    # =========================================================================
    # TEST: SK data with Bhopal-aligned coordinate transforms
    # =========================================================================
    test=dict(
        type=dataset_type,
        split="val",  # SK val set (sequence 08)
        if_img=False,
        data_root=DATA_ROOT,
        learning_map=learning_map,
        transform=[
            # 1. Standard 0.2x coordinate scaling (same as all Concerto configs)
            dict(type="RandomScale", scale=[0.2, 0.2]),
            # 2. Bhopal model's PointClip range — SK coords fit within it
            #    SK after 0.2x: X [-16,16], Y [-16,16], Z [-0.25, 0.55]
            #    All within Bhopal's [-29.8, 42.6] x [-21.8, 31.8] x [-4.4, 8.6]
            dict(type="PointClip", point_cloud_range=BHOPAL_POINT_CLIP),
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

# Hooks (not used for testing, but required by config system)
hooks = [
    dict(type="IterationTimer", warmup_iter=2),
    dict(type="InformationWriter"),
    dict(type="SemSegEvaluator"),
    dict(type="CheckpointSaver", save_freq=None),
]
