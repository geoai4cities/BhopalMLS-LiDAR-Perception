"""
LitePT-Large Configuration for Bhopal MLS Dataset - 4ch (coord + strength)
Model: LitePT-L (~85.9M params)
Input: 4 channels (XYZ coordinates + intensity/strength)
Classes: 21 Bhopal semantic classes (fine-grained)
Training: From scratch (no pretrained weights)
"""

_base_ = ["../_base_/default_runtime.py"]

# =============================================================================
# MISC SETTINGS
# =============================================================================
batch_size = 12
num_worker = 16
mix_prob = 0.8
empty_cache = False
enable_amp = True

H5_PATH = "data/Bhopal/bhopal_scene_v9_normals_rgb.h5"

save_path = "exp/bhopal/litept_L_21class_4ch"

# =============================================================================
# CLASS CONFIGURATION
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
    [22, 23, 32, 33, 34],               # 10: 2-Wheeler
    [25, 27, 28, 30],                   # 11: 3-Wheeler
    [20, 21, 26, 29, 31],               # 12: 4-Wheeler
    [24],                                # 13: Hand-Cart
    [35, 36, 37],                        # 14: Board
    [38, 39, 40, 41, 42],               # 15: Fencing
    [43, 44, 45, 46],                   # 16: Human
    [53],                                # 17: Road Side Furniture
    [52],                                # 18: Shop Shade
    [57],                                # 19: Mannequin
    [50, 51, 54, 55, 56, 58, 59, 60, 61],  # 20: Road Side Obj (other)
]
# NOTE: Original IDs 47 (Animal), 48 (Dog), 49 (Cow) are NOT in any group

class_weight = [
    4.8805, 7.7263, 3.6122, 1.8059, 1.0000, 1.7738, 3.6638, 7.1549, 7.1867,
    20.0000, 3.5210, 8.8729, 5.1955, 11.8747, 7.5868, 16.5094, 7.4137, 7.1182,
    7.2990, 16.3963, 13.9078,
]

# =============================================================================
# MODEL CONFIGURATION - LitePT-Large (~85.9M params)
# =============================================================================
model = dict(
    type="DefaultSegmentorV2",
    num_classes=num_classes,
    backbone_out_channels=72,
    backbone=dict(
        type="LitePT",
        in_channels=4,
        order=["z", "z-trans", "hilbert", "hilbert-trans"],
        stride=(2, 2, 2, 2),
        enc_depths=(3, 3, 3, 12, 3),
        enc_channels=(72, 144, 288, 576, 864),
        enc_num_head=(4, 8, 16, 32, 48),
        enc_patch_size=(1024, 1024, 1024, 1024, 1024),
        enc_conv=(True, True, True, False, False),
        enc_attn=(False, False, False, True, True),
        enc_rope_freq=(100.0, 100.0, 100.0, 100.0, 100.0),
        dec_depths=(0, 0, 0, 0),
        dec_channels=(72, 144, 288, 576),
        dec_num_head=(4, 8, 16, 32),
        dec_patch_size=(1024, 1024, 1024, 1024),
        dec_conv=(False, False, False, False),
        dec_attn=(False, False, False, False),
        dec_rope_freq=(100.0, 100.0, 100.0, 100.0),
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        drop_path=0.3,
        shuffle_orders=True,
        pre_norm=True,
        enc_mode=False,
    ),
    criteria=[
        dict(type="CrossEntropyLoss", weight=class_weight, loss_weight=1.0, ignore_index=-1),
        dict(type="LovaszLoss", mode="multiclass", loss_weight=1.0, ignore_index=-1),
    ],
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
# DATASET CONFIGURATION - 4ch (coord + strength)
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
        use_intensity=True,
        use_gps_time=False,
        use_normals=False,
        use_rgb=False,
        remap_classes=True,
        class_groups=class_groups,
        class_names=class_names,
        transform=[
            dict(type="RandomScale", scale=[0.2, 0.2]),
            dict(type="RandomRotate", angle=[-1, 1], axis="z", center=[0, 0, 0], p=0.5),
            dict(type="RandomScale", scale=[0.9, 1.1]),
            dict(type="RandomFlip", p=0.5),
            dict(type="RandomJitter", sigma=0.005, clip=0.02),
            dict(
                type="GridSample",
                grid_size=0.01,
                hash_type="fnv",
                mode="train",
                return_grid_coord=True,
            ),
            dict(type="PointClip", point_cloud_range=(-149*0.2, -109*0.2, -22*0.2, 213*0.2, 159*0.2, 43*0.2)),
            dict(type="SphereCrop", sample_rate=0.8, mode="random"),
            dict(type="SphereCrop", point_max=120000, mode="random"),
            dict(type="ToTensor"),
            dict(type="Update", keys_dict={"grid_size": 0.01}),
            dict(
                type="Collect",
                keys=("coord", "grid_coord", "segment", "grid_size"),
                feat_keys=("coord", "strength"),
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
        use_intensity=True,
        use_gps_time=False,
        use_normals=False,
        use_rgb=False,
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
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=("coord", "grid_coord", "segment", "origin_segment", "inverse"),
                feat_keys=("coord", "strength"),
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
        use_intensity=True,
        use_gps_time=False,
        use_normals=False,
        use_rgb=False,
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
                    feat_keys=("coord", "strength"),
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
# HOOKS (from-scratch training, no ModelHook needed)
# =============================================================================
hooks = [
    dict(type="CheckpointLoader"),
    dict(type="IterationTimer", warmup_iter=2),
    dict(type="InformationWriter"),
    dict(type="SemSegEvaluator"),
    dict(type="CheckpointSaver", save_freq=None),
    dict(type="PreciseEvaluator", test_last=False),
]
