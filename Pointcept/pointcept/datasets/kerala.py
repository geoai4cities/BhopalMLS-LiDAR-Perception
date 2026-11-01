"""
Scene-Level HDF5 Dataset Loader for Kerala MLS Data
Compatible with Pointcept framework and PTv3/Concerto/Sonata.

Per-point semantic + instance segmentation. Scene-level data created by
the Ke-MLS scene builder (Ke-MLS release).

Kerala V1 corridor: 48 training classes remapped to 12 parent classes.
71 paths, split: train 1-38 / val 39-51 / test 52-71.
No RGB (all zeros in H5); normals computed via Open3D KNN-PCA.

HDF5 Structure:
    scenes/points              [N, max_pts, 3]    float32  - XYZ (scene-centered)
    scenes/normals             [N, max_pts, 3]    float32  - Unit normals
    scenes/rgb                 [N, max_pts, 3]    uint8    - Zeros (no real RGB)
    scenes/intensity           [N, max_pts]       float32  - Raw 0..65535
    scenes/gps_time            [N, max_pts]       float32  - GPS timestamp
    scenes/labels              [N, max_pts]       int32    - Per-point class (0..47, -1=padding)
    scenes/global_ids          [N, max_pts]       int64    - Per-point instance ID (-1=padding)
    scenes/num_points          [N]                uint32
    metadata/scene_names       [N]                str
    metadata/source_paths      [N]                uint8   - Path number
    splits/train_indices, val_indices, test_indices

Date: 2026-04-11
"""

import h5py
import numpy as np
from pathlib import Path

from .builder import DATASETS
from .transform import Compose, TRANSFORMS

# NOTE: ClassBalancedSample transform is defined in .bhopal. Importing
# pointcept.datasets.bhopal (which happens via pointcept/datasets/__init__.py
# before this module) registers it in the TRANSFORMS registry, so it is
# reusable here by name without re-registering.


# =============================================================================
# FLEXIBLE CLASS CONFIGURATION
# =============================================================================


def get_default_learning_map():
    """
    Get the default 48-to-12 class learning map for Kerala MLS dataset.

    Matches data/Kerala/class_mapping/Kerala_class_mapping_V13_TRAINING.csv.

    Returns:
        dict: Mapping from original training label (0-47) to target label (0-11)
    """
    learning_map = {}

    # 0: Ground (training 0-3)
    for i in range(0, 4):
        learning_map[i] = 0

    # 1: Vegetation (training 4-7)
    for i in range(4, 8):
        learning_map[i] = 1

    # 2: Building (training 8-11)
    for i in range(8, 12):
        learning_map[i] = 2

    # 3: Road (training 12)
    learning_map[12] = 3

    # 4: Wire (training 13)
    learning_map[13] = 4

    # 5: Pole and Tower (training 14-16)
    for i in range(14, 17):
        learning_map[i] = 5

    # 6: Wire-Structure Connector (training 17-18)
    for i in range(17, 19):
        learning_map[i] = 6

    # 7: Vehicle (training 19-28)
    for i in range(19, 29):
        learning_map[i] = 7

    # 8: Board (training 29-31)
    for i in range(29, 32):
        learning_map[i] = 8

    # 9: Fencing (training 32-36)
    for i in range(32, 37):
        learning_map[i] = 9

    # 10: Human and Animal (training 37-43)
    for i in range(37, 44):
        learning_map[i] = 10

    # 11: Road Side Object (training 44-47)
    for i in range(44, 48):
        learning_map[i] = 11

    return learning_map


def get_default_learning_map_inv():
    """
    Get the inverse mapping from 12 target classes to a representative
    original Kerala training label (useful for visualization).
    """
    return {
        0: 0,    # Ground -> training 0
        1: 4,    # Vegetation -> training 4
        2: 8,    # Building -> training 8
        3: 12,   # Road -> training 12
        4: 13,   # Wire -> training 13
        5: 14,   # Pole and Tower -> training 14
        6: 17,   # Wire-Structure Connector -> training 17
        7: 19,   # Vehicle -> training 19
        8: 29,   # Board -> training 29
        9: 32,   # Fencing -> training 32
        10: 37,  # Human and Animal -> training 37
        11: 44,  # Road Side Object -> training 44
    }


def get_default_class_names():
    """Get the default 12 parent class names (same as Bhopal / Kerala)."""
    return [
        "Ground",                    # 0
        "Vegetation",                # 1
        "Building",                  # 2
        "Road",                      # 3
        "Wire",                      # 4
        "Pole and Tower",            # 5
        "Wire-Structure Connector",  # 6
        "Vehicle",                   # 7
        "Board",                     # 8
        "Fencing",                   # 9
        "Human and Animal",          # 10
        "Road Side Object",          # 11
    ]


def build_class_mapping_from_groups(class_groups):
    """Build a learning_map dict from class_groups format."""
    mapping = {}
    for new_label, old_labels in enumerate(class_groups):
        for old_label in old_labels:
            mapping[old_label] = new_label
    return mapping


# Kerala 48 -> 12 default configuration.
# Verified against metadata/class_mapping attribute of
# data/Kerala/Kerala_scene_v1_corridor.h5.
DEFAULT_12CLASS_CONFIG = {
    "num_classes": 12,
    "class_names": get_default_class_names(),
    "class_groups": [
        list(range(0, 4)),     # 0: Ground (train 0-3)
        list(range(4, 8)),     # 1: Vegetation (4-7)
        list(range(8, 12)),    # 2: Building (8-11)
        [12],                  # 3: Road (12)
        [13],                  # 4: Wire (13)
        list(range(14, 17)),   # 5: Pole and Tower (14-16)
        list(range(17, 19)),   # 6: Wire-Structure Connector (17-18)
        list(range(19, 29)),   # 7: Vehicle (19-28)
        list(range(29, 32)),   # 8: Board (29-31)
        list(range(32, 37)),   # 9: Fencing (32-36)
        list(range(37, 44)),   # 10: Human and Animal (37-43)
        list(range(44, 48)),   # 11: Road Side Object (44-47)
    ],
}

PARENT_CLASSES = {i: name for i, name in enumerate(DEFAULT_12CLASS_CONFIG["class_names"])}

# Stuff classes (parent IDs) for instance segmentation.
# Matches Bhopal: padding + Ground + Vegetation + Road + Wire.
DEFAULT_SEGMENT_IGNORE_INDEX = (-1, 0, 1, 3, 4)


@DATASETS.register_module()
class KeralaSceneHDF5Dataset:
    """
    Kerala MLS Scene-Level HDF5 Dataset for Semantic / Instance Segmentation.

    Compatible with Pointcept/PTv3/Concerto/Sonata frameworks.

    Loads scene-level data from HDF5 created by
    the Ke-MLS scene builder (Ke-MLS release):
    - Path-based geographic splits (train/val/test)
    - Per-point semantic labels (48 classes remapped to 12)
    - Per-point instance IDs (when load_instance=True)
    - Up to 1M points per scene (corridor subdivision with max cap)
    """

    # ================================================================
    # STATIC METHODS (SemanticKITTI API compatibility)
    # ================================================================

    @staticmethod
    def get_learning_map():
        """Get the default 48-to-12 class learning map."""
        return get_default_learning_map()

    @staticmethod
    def get_learning_map_inv():
        """Get the inverse mapping from target to representative original labels."""
        return get_default_learning_map_inv()

    @staticmethod
    def get_class_names():
        """Get the default 12 class names."""
        return get_default_class_names()

    # ================================================================
    # INITIALIZATION
    # ================================================================

    def __init__(
        self,
        h5_path: str,
        split: str = "train",
        num_points: int = None,
        transform=None,
        test_mode: bool = False,
        test_cfg=None,
        loop: int = 1,
        use_intensity: bool = True,
        use_gps_time: bool = False,
        use_normals: bool = False,
        use_rgb: bool = False,
        cache_data: bool = False,
        remap_classes: bool = True,
        learning_map: dict = None,
        class_groups: list = None,
        class_names: list = None,
        class_mapping: dict = None,
        ignore_index: int = -1,
        load_instance: bool = False,
    ):
        """
        Args:
            h5_path: Path to Kerala scene HDF5 file.
            split: 'train' / 'val' / 'test'.
            num_points: Target points per sample. None = use ALL points.
            transform: Pointcept transform pipeline.
            test_mode: Whether in test mode.
            test_cfg: Test configuration.
            loop: Number of times to loop over dataset.
            use_intensity: Include intensity as feature (data_dict['strength']).
            use_gps_time: Include GPS time as feature (z-score normalized).
            use_normals: Load normals from HDF5 as data_dict['normal'].
            use_rgb: Load RGB from HDF5 as data_dict['color']. Kerala RGB is
                     all zeros, so this yields a zero 'color' tensor — useful
                     for 9ch configs that stay compatible with the Bhopal
                     Concerto checkpoint input shape.
            cache_data: Whether to cache data in memory.
            remap_classes: Whether to remap 48 classes to target parent classes.
            learning_map: SemanticKITTI style dict.
            class_groups: Bhopal style list of lists.
            class_names: Names for target classes.
            class_mapping: DEPRECATED — use learning_map instead.
            ignore_index: Label to ignore in loss computation.
            load_instance: If True, load scenes/global_ids into data_dict['instance']
                           for PG-v1m2 / InstanceParser insseg configs.
        """
        super().__init__()
        self.h5_path = Path(h5_path)
        self.split = split
        self.num_points = num_points
        self.transform = Compose(transform) if transform is not None else None
        self.test_mode = test_mode
        self.test_cfg = test_cfg if test_mode else None

        if self.test_mode:
            if self.test_cfg:
                voxelize_cfg = self.test_cfg.get('voxelize') if hasattr(self.test_cfg, 'get') else getattr(self.test_cfg, 'voxelize', None)
                crop_cfg = self.test_cfg.get('crop') if hasattr(self.test_cfg, 'get') else getattr(self.test_cfg, 'crop', None)
                post_cfg = self.test_cfg.get('post_transform') if hasattr(self.test_cfg, 'get') else getattr(self.test_cfg, 'post_transform', None)
                aug_cfg = self.test_cfg.get('aug_transform') if hasattr(self.test_cfg, 'get') else getattr(self.test_cfg, 'aug_transform', None)

                self.test_voxelize = TRANSFORMS.build(voxelize_cfg) if voxelize_cfg else None
                self.test_crop = TRANSFORMS.build(crop_cfg) if crop_cfg else None
                self.post_transform = Compose(post_cfg) if post_cfg else None
                self.aug_transform = [Compose(aug) for aug in aug_cfg] if aug_cfg else []
            else:
                self.test_voxelize = None
                self.test_crop = None
                self.post_transform = None
                self.aug_transform = []

        self.loop = loop
        self.use_intensity = use_intensity
        self.use_gps_time = use_gps_time
        self.use_normals = use_normals
        self.use_rgb = use_rgb
        self.cache_data = cache_data
        self.remap_classes = remap_classes
        self.ignore_index = ignore_index
        self.load_instance = load_instance

        self._setup_class_mapping(learning_map, class_groups, class_names, class_mapping)

        if not self.h5_path.exists():
            raise FileNotFoundError(
                f"HDF5 file not found: {self.h5_path}\n"
                f"Please create it using:\n"
                f"  the Ke-MLS scene builder (Ke-MLS release) "
                f"--city kerala --method corridor"
            )

        self._load_dataset_info()

        features = ["XYZ"]
        if use_rgb:
            features.append("RGB(normalized)")
        if use_intensity:
            features.append("intensity(normalized)")
        if use_gps_time:
            features.append("gps_time(normalized)")
        if use_normals:
            features.append("normals")

        print(f"Loaded Kerala Scene HDF5 dataset for split '{split}':")
        print(f"  - Scenes: {len(self.indices):,}")
        if self.num_points is None:
            print(f"  - Points: ALL (no subsampling)")
        else:
            print(f"  - Target points: {self.num_points:,}")
        print(f"  - Max points/scene: {self.max_points_per_scene:,}")
        print(f"  - Classes: {self.num_classes} ({self._class_config_source})")
        print(f"  - Class names: {self.target_class_names}")
        print(f"  - Features: {' + '.join(features)}")
        print(f"  - Total channels: {self.get_num_channels()}")
        print(f"  - Ignore index: {self.ignore_index}")
        if self.load_instance:
            print(f"  - Instance loading: ENABLED (from scenes/global_ids)")

    def _setup_class_mapping(self, learning_map, class_groups, class_names, class_mapping):
        """Setup class mapping from various input formats."""
        if not self.remap_classes:
            self.class_mapping = None
            self.num_classes = 48
            self.target_class_names = [f"Class_{i}" for i in range(48)]
            self._class_config_source = "original 48 fine-grained"
            return

        if learning_map is not None:
            self.class_mapping = learning_map
            target_values = [v for v in learning_map.values() if v != self.ignore_index]
            self.num_classes = len(set(target_values))

            if class_names is not None:
                if len(class_names) != self.num_classes:
                    raise ValueError(
                        f"class_names length ({len(class_names)}) must match "
                        f"num_classes ({self.num_classes})"
                    )
                self.target_class_names = class_names
            else:
                self.target_class_names = [f"Class_{i}" for i in range(self.num_classes)]

            self._class_config_source = f"custom {self.num_classes}-class learning_map"

        elif class_groups is not None:
            self.class_mapping = build_class_mapping_from_groups(class_groups)
            self.num_classes = len(class_groups)

            if class_names is not None:
                if len(class_names) != len(class_groups):
                    raise ValueError(
                        f"class_names length ({len(class_names)}) must match "
                        f"class_groups length ({len(class_groups)})"
                    )
                self.target_class_names = class_names
            else:
                self.target_class_names = [f"Class_{i}" for i in range(self.num_classes)]

            self._class_config_source = f"custom {self.num_classes}-class groups"

        elif class_mapping is not None:
            self.class_mapping = class_mapping
            target_values = [v for v in class_mapping.values() if v != self.ignore_index]
            self.num_classes = len(set(target_values))
            self.target_class_names = [f"Class_{i}" for i in range(self.num_classes)]
            self._class_config_source = f"custom {self.num_classes}-class mapping (deprecated)"

        else:
            self.class_mapping = build_class_mapping_from_groups(
                DEFAULT_12CLASS_CONFIG["class_groups"]
            )
            self.num_classes = DEFAULT_12CLASS_CONFIG["num_classes"]
            self.target_class_names = DEFAULT_12CLASS_CONFIG["class_names"]
            self._class_config_source = "default 12-class Kerala"

        self.label_to_name = {i: name for i, name in enumerate(self.target_class_names)}

    def _load_dataset_info(self):
        """Load dataset information and split indices from HDF5."""
        with h5py.File(self.h5_path, 'r') as h5f:
            def get_split_key(name):
                return f'splits/{name}_indices' if f'splits/{name}_indices' in h5f else f'splits/{name}'

            if self.split == 'train':
                indices = h5f[get_split_key('train')][:]
            elif self.split == 'val':
                indices = h5f[get_split_key('val')][:]
            elif self.split == 'test':
                test_key = get_split_key('test')
                if test_key not in h5f:
                    raise ValueError("Test split not found in HDF5 file")
                indices = h5f[test_key][:]
            else:
                raise ValueError(f"Invalid split: {self.split}")

            # Filter out scenes with too few points (prevents empty batch errors)
            MIN_POINTS_PER_SCENE = 1000
            if 'scenes/num_points' in h5f:
                num_points_all = h5f['scenes/num_points'][:]
                num_points_split = num_points_all[indices]
                valid_mask = num_points_split >= MIN_POINTS_PER_SCENE
                n_filtered = (~valid_mask).sum()
                if n_filtered > 0:
                    print(f"  WARNING: Filtered {n_filtered} scenes with <{MIN_POINTS_PER_SCENE} points")
                self.indices = indices[valid_mask]
            else:
                self.indices = indices

            self.max_points_per_scene = h5f.attrs.get(
                'max_points_per_scene', h5f.attrs.get('max_points', 1000000)
            )
            self.num_classes_original = h5f.attrs.get('num_classes', 48)
            self.class_names = self.target_class_names

    def get_num_channels(self):
        """Get total number of feature channels (excluding XYZ coord)."""
        channels = 0
        if self.use_rgb:
            channels += 3
        if self.use_intensity:
            channels += 1
        if self.use_gps_time:
            channels += 1
        if self.use_normals:
            channels += 3
        return channels

    def __len__(self):
        return len(self.indices) * self.loop

    def _normalize_intensity(self, intensity):
        """Normalize intensity from [0, 65535] to [0, 1]."""
        return intensity / 65535.0

    def _normalize_gps_time(self, gps_time, valid_mask):
        """Normalize GPS time using z-score (only on valid points)."""
        valid_gps = gps_time[valid_mask]
        if len(valid_gps) == 0:
            return gps_time

        mean = valid_gps.mean()
        std = valid_gps.std()
        if std < 1e-6:
            return gps_time - mean

        normalized = (gps_time - mean) / std
        normalized[~valid_mask] = 0
        return normalized

    def _remap_labels(self, labels):
        """Remap fine-grained labels to target classes."""
        remapped = np.full_like(labels, self.ignore_index)
        for old_label, new_label in self.class_mapping.items():
            remapped[labels == old_label] = new_label
        remapped[labels == self.ignore_index] = self.ignore_index
        return remapped

    def get_data(self, idx):
        """Load raw data for a scene (without transforms)."""
        idx = idx % len(self.indices)
        global_idx = self.indices[idx]

        with h5py.File(self.h5_path, 'r') as h5f:
            num_pts = int(h5f['scenes/num_points'][global_idx])

            coord = h5f['scenes/points'][global_idx, :num_pts].astype(np.float32)
            intensity_raw = h5f['scenes/intensity'][global_idx, :num_pts].astype(np.float32)

            if self.use_gps_time and 'scenes/gps_time' in h5f:
                gps_time_raw = h5f['scenes/gps_time'][global_idx, :num_pts].astype(np.float32)
            else:
                gps_time_raw = None

            if self.use_rgb and 'scenes/rgb' in h5f:
                rgb_raw = h5f['scenes/rgb'][global_idx, :num_pts].astype(np.float32)
            else:
                rgb_raw = None

            if self.use_normals and 'scenes/normals' in h5f:
                normals = h5f['scenes/normals'][global_idx, :num_pts].astype(np.float32)
            else:
                normals = None

            segment = h5f['scenes/labels'][global_idx, :num_pts].astype(np.int64)

            if self.load_instance and 'scenes/global_ids' in h5f:
                instance = h5f['scenes/global_ids'][global_idx, :num_pts].astype(np.int32)
            else:
                instance = None

            scene_name = h5f['metadata/scene_names'][global_idx]
            if isinstance(scene_name, bytes):
                scene_name = scene_name.decode('utf-8')

            # Kerala uses 'metadata/source_paths' (plural), unlike Bhopal's
            # 'metadata/bhopal_paths'.
            source_path = int(h5f['metadata/source_paths'][global_idx])

        # ================================================================
        # PREPROCESSING
        # ================================================================

        if self.remap_classes:
            segment = self._remap_labels(segment)

        intensity = self._normalize_intensity(intensity_raw)

        if self.use_gps_time and gps_time_raw is not None:
            valid_mask = segment >= 0
            gps_time = self._normalize_gps_time(gps_time_raw, valid_mask)
        else:
            gps_time = None

        # RGB: Keep as float32 [0, 255] — NormalizeColor transform handles /255
        if self.use_rgb and rgb_raw is not None:
            rgb = rgb_raw.astype(np.float32)
        else:
            rgb = None

        # ================================================================
        # POINT SAMPLING (optional - skip if num_points is None)
        # ================================================================

        if self.num_points is not None:
            if num_pts > self.num_points:
                sample_indices = np.random.choice(num_pts, self.num_points, replace=False)
            elif num_pts < self.num_points:
                sample_indices = np.random.choice(num_pts, self.num_points, replace=True)
            else:
                sample_indices = np.arange(num_pts)

            coord = coord[sample_indices]
            segment = segment[sample_indices]
            intensity = intensity[sample_indices]

            if gps_time is not None:
                gps_time = gps_time[sample_indices]
            if rgb is not None:
                rgb = rgb[sample_indices]
            if normals is not None:
                normals = normals[sample_indices]
            if instance is not None:
                instance = instance[sample_indices]

        # ================================================================
        # BUILD FEATURE VECTOR
        # ================================================================

        feat_list = []
        if self.use_intensity:
            feat_list.append(intensity[:, None])
        if gps_time is not None:
            feat_list.append(gps_time[:, None])

        if len(feat_list) > 0:
            feat = np.concatenate(feat_list, axis=1).astype(np.float32)
        else:
            feat = np.zeros((len(coord), 0), dtype=np.float32)

        # ================================================================
        # BUILD DATA DICT
        # ================================================================

        data_dict = {
            'coord': coord,
            'feat': feat,
            'segment': segment,
            'name': f"{scene_name}_kerala{source_path}",
        }

        # For Pointcept compatibility ('strength' is Pointcept convention for
        # LiDAR intensity — matches Bhopal loader, SemanticKITTI, etc.).
        if self.use_intensity:
            data_dict['strength'] = intensity[:, None]

        if self.use_rgb:
            if rgb is not None:
                data_dict['color'] = rgb  # [N, 3] float32 in [0, 255]; NormalizeColor handles /255
            else:
                data_dict['color'] = np.zeros((len(coord), 3), dtype=np.float32)

        if self.use_normals:
            if normals is not None:
                data_dict['normal'] = normals.astype(np.float32)
            else:
                data_dict['normal'] = np.zeros((len(coord), 3), dtype=np.float32)

        data_dict['grid_size'] = 0.02

        # Instance segmentation: add instance IDs
        if instance is not None:
            data_dict['instance'] = instance

        return data_dict

    def get_data_name(self, idx):
        """Get scene name for a given index."""
        idx = idx % len(self.indices)
        global_idx = self.indices[idx]

        with h5py.File(self.h5_path, 'r') as h5f:
            scene_name = h5f['metadata/scene_names'][global_idx]
            if isinstance(scene_name, bytes):
                scene_name = scene_name.decode('utf-8')
            source_path = int(h5f['metadata/source_paths'][global_idx])
        return f"{scene_name}_kerala{source_path}"

    def prepare_train_data(self, idx):
        """Prepare data for training mode."""
        data_dict = self.get_data(idx)
        if self.transform is not None:
            data_dict = self.transform(data_dict)
        return data_dict

    def prepare_test_data(self, idx):
        """Prepare data for test mode with fragment-based inference."""
        from copy import deepcopy

        data_dict = self.get_data(idx)
        if self.transform is not None:
            data_dict = self.transform(data_dict)

        result_dict = dict(
            segment=data_dict.pop("segment"),
            name=data_dict.pop("name"),
        )

        if "instance" in data_dict:
            result_dict["instance"] = data_dict.pop("instance")

        if "origin_segment" in data_dict:
            assert "inverse" in data_dict, "inverse mapping required with origin_segment"
            result_dict["origin_segment"] = data_dict.pop("origin_segment")
            result_dict["inverse"] = data_dict.pop("inverse")

        data_dict_list = []
        if self.aug_transform:
            for aug in self.aug_transform:
                data_dict_list.append(aug(deepcopy(data_dict)))
        else:
            data_dict_list.append(data_dict)

        fragment_list = []
        for data in data_dict_list:
            if self.test_voxelize is not None:
                data_part_list = self.test_voxelize(data)
            else:
                data["index"] = np.arange(data["coord"].shape[0])
                data_part_list = [data]

            for data_part in data_part_list:
                if self.test_crop is not None:
                    data_part = self.test_crop(data_part)
                else:
                    data_part = [data_part]
                fragment_list += data_part

        if self.post_transform is not None:
            for i in range(len(fragment_list)):
                fragment_list[i] = self.post_transform(fragment_list[i])

        result_dict["fragment_list"] = fragment_list
        return result_dict

    def __getitem__(self, idx):
        if self.test_mode:
            return self.prepare_test_data(idx)
        else:
            return self.prepare_train_data(idx)

    def get_class_name(self, label):
        """Get class name from label."""
        return self.label_to_name.get(label, f"Unknown_{label}")

    def get_scene_info(self, idx):
        """Get metadata for a scene."""
        idx = idx % len(self.indices)
        global_idx = self.indices[idx]

        with h5py.File(self.h5_path, 'r') as h5f:
            scene_name = h5f['metadata/scene_names'][global_idx]
            if isinstance(scene_name, bytes):
                scene_name = scene_name.decode('utf-8')

            info = {
                'scene_name': scene_name,
                'source_path': int(h5f['metadata/source_paths'][global_idx]),
                'num_points': int(h5f['scenes/num_points'][global_idx]),
            }
            if 'metadata/num_instances' in h5f:
                info['num_instances'] = int(h5f['metadata/num_instances'][global_idx])
            return info
