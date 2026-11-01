"""
Cross-City Test Config: Utonia BM Model -> SemanticKITTI Data

Tests a Utonia/Bhopal-trained FT-9ch model on SemanticKITTI (sequence 08 val)
for the cross-city BM <-> SK transfer cell.

=============================================================================
COORDINATE ALIGNMENT
=============================================================================
BM Utonia training PointClip (after 0.2x scale):
  X=[-29.8, 42.6], Y=[-21.8, 31.8], Z=[-4.4, 8.6]

SK after 0.2x scale (post-2026-05-19 SK training also uses Bhopal-wide PCR,
so applying it here matches SK training too — clean checkpoint swap in both
directions, mirroring TR/PL/KL/CH).

Caveat: SK data is geometrically smaller than BM, so SK grid_coord under
the Bhopal-wide PCR maxes at ~3200 vs BM training's ~7300 — a partial
serialization mismatch but no clipping in either direction.
=============================================================================

Model:    Utonia FT-9ch (PT-v3m3, BM-12 schema) — same arch as Bhopal training
          (dec_num_head=(8, 8, 16, 32))
Source:   Utonia/Bhopal/outputs/utonia_bhopal_finetune_9ch/model/model_best.pth
Target:   data/SemanticKITTI (sequence 08 val)
Input:    9ch (XYZ + RGB + normal); SK has no RGB so RGB channels are
          zero-padded by the SK dataset class — BM-trained weights see no
          color signal at eval (architectural caveat, not a config bug).
Classes:  12 unified BM-12 schema. SK raw labels are mapped to BM-12 via
          `learning_map`. BM-12 classes absent in SK (4 Wire, 6 Wire-Struct
          Connector, 11 Road Side Object) are mapped to ignore; per-class
          IoU on those reports as 0. For fair mIoU, report over the 9
          shared classes: {0, 1, 2, 3, 5, 7, 8, 9, 10}.
"""

_base_ = ["../../Bhopal/configs/_base_/default_runtime.py"]

# =============================================================================
# MISC SETTINGS (test-only stubs — required by the config system)
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
# CLASS CONFIGURATION (12 unified classes)
# =============================================================================
num_classes = 12
class_names = [
    "Ground",                    # 0: SK sidewalk(48), other-ground(49), terrain(72)
    "Vegetation",                # 1: SK vegetation(70), trunk(71)
    "Building",                  # 2: SK building(50)
    "Road",                      # 3: SK road(40), parking(44), lane-marking(60)
    "Wire",                      # 4: ABSENT in SK
    "Pole and Tower",            # 5: SK pole(80)
    "Wire-Structure Connector",  # 6: ABSENT in SK
    "Vehicle",                   # 7: SK car/bicycle/bus/motorcycle/on-rails/truck/other-vehicle
                                 #    + moving-* equivalents
    "Board",                     # 8: SK traffic-sign(81)
    "Fencing",                   # 9: SK fence(51)
    "Human and Animal",          # 10: SK person/bicyclist/motorcyclist + moving-* equivalents
    "Road Side Object",          # 11: ABSENT in SK
]

# SemanticKITTI raw label -> Unified BM-12 (same mapping used at SK training).
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
# MODEL CONFIGURATION (must match Utonia BM FT-9ch training arch exactly)
# =============================================================================
model = dict(
    type="DefaultSegmentorV2",
    num_classes=num_classes,
    backbone_out_channels=96,
    backbone=dict(
        type="PT-v3m3",
        in_channels=9,
        order=("z", "z-trans", "hilbert", "hilbert-trans"),
        stride=(2, 2, 2, 2),
        enc_depths=(3, 3, 3, 12, 3),
        enc_channels=(54, 108, 216, 432, 576),
        enc_num_head=(3, 6, 12, 24, 32),
        enc_patch_size=(1024, 1024, 1024, 1024, 1024),
        dec_depths=(3, 3, 3, 3),
        dec_channels=(96, 96, 192, 384),
        dec_num_head=(8, 8, 16, 32),  # Point3DRoPE: head_dim div-by-3
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
    freeze_backbone=False,
    criteria=[
        dict(type="CrossEntropyLoss", loss_weight=1.0, ignore_index=-1),
        dict(type="LovaszLoss", mode="multiclass", loss_weight=1.0, ignore_index=-1),
    ],
)

# Scheduler (not used for testing — required by the config system)
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
# COORDINATE ALIGNMENT — BM training PCR (Bhopal-wide; SK data fits inside)
# =============================================================================
# Reference: SK native training PCR (used by Utonia SK FT-9ch training; not
# applied here — kept for the BM-vs-SK range comparison in the docstring).
#   SK_NATIVE_POINT_CLIP = (-35.2*0.2, -35.2*0.2, -4*0.2, 35.2*0.2, 35.2*0.2, 2*0.2)
#                        = (-7.04,    -7.04,    -0.8,   7.04,     7.04,    0.4)
#
# Applied PCR — BM training (raw range +10% margin -> (-149,-109,-22) to
# (213,159,43); after 0.2x):
BHOPAL_POINT_CLIP = (-149*0.2, -109*0.2, -22*0.2, 213*0.2, 159*0.2, 43*0.2)
# = (-29.8, -21.8, -4.4, 42.6, 31.8, 8.6)
#
# SK data fits entirely inside this BM PCR — no clipping. After 2026-05-19,
# SK training also uses this same Bhopal-wide PCR, so the reverse direction
# (utonia_crosscity_sk_on_bhopal_9ch.py) is now a clean checkpoint swap too
# (the old 0.215x + CenterShift workaround was dropped).

# =============================================================================
# DATASET CONFIGURATION
# =============================================================================
dataset_type = "SemanticKITTIImagePointDataset"
ignore_index = -1

data = dict(
    num_classes=num_classes,
    ignore_index=ignore_index,
    names=class_names,

    # train/val: minimal stubs (not used at test time)
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
    # TEST: SK val (sequence 08) with BM-aligned PCR (SK fits inside, no clip)
    # =========================================================================
    test=dict(
        type=dataset_type,
        split="val",
        if_img=False,
        data_root=DATA_ROOT,
        learning_map=learning_map,
        transform=[
            dict(type="RandomScale", scale=[0.2, 0.2]),
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

# Hooks (not used by test.py, but required by the config system)
hooks = [
    dict(type="IterationTimer", warmup_iter=2),
    dict(type="InformationWriter"),
    dict(type="SemSegEvaluator"),
    dict(type="CheckpointSaver", save_freq=None),
]
