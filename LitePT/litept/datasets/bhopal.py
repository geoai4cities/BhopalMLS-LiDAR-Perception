"""
Scene-Level HDF5 Dataset Loader for Bhopal MLS Data
Compatible with LitePT framework (Pointcept-based)

Ported from Pointcept's bhopal.py with import paths adapted for LitePT.

HDF5 Structure (Scene V9):
    scenes/points              [N, max_pts, 3]    float32  - XYZ (scene-centered)
    scenes/intensity           [N, max_pts]       float32  - Raw values (0-65535)
    scenes/gps_time            [N, max_pts]       float32  - GPS timestamp
    scenes/labels              [N, max_pts]       int32    - Per-point class (0-61, -1=padding)
    scenes/rgb                 [N, max_pts, 3]    uint8    - RGB
    scenes/normals             [N, max_pts, 3]    float32  - Unit normals
    scenes/num_points          [N]                uint32   - Valid point count
    metadata/scene_names       [N]                str      - Scene name
    splits/train_indices, val_indices, test_indices
"""

import h5py
import numpy as np
from pathlib import Path

from .builder import DATASETS
from .transform import Compose, TRANSFORMS


# =============================================================================
# CLASS-BALANCED SAMPLING TRANSFORM
# =============================================================================

@TRANSFORMS.register_module()
class ClassBalancedSample:
    """
    Class-balanced point sampling to address class imbalance.
    Reduces dominant classes before voxelization.
    """

    def __init__(self, class_sample_rates=None, min_points=10000, seed=None):
        self.class_sample_rates = class_sample_rates or {}
        self.min_points = min_points
        self.seed = seed

        for cls, rate in self.class_sample_rates.items():
            if not 0.0 <= rate <= 1.0:
                raise ValueError(f"Sample rate for class {cls} must be in [0, 1], got {rate}")

    def __call__(self, data_dict):
        if "segment" not in data_dict or len(self.class_sample_rates) == 0:
            return data_dict

        segment = data_dict["segment"]
        n_points = len(segment)

        if n_points == 0:
            return data_dict

        keep_mask = np.ones(n_points, dtype=bool)
        rng = np.random.RandomState(self.seed) if self.seed is not None else np.random

        for class_id, sample_rate in self.class_sample_rates.items():
            if sample_rate >= 1.0:
                continue
            class_mask = (segment == class_id)
            n_class = class_mask.sum()
            if n_class == 0:
                continue
            n_keep = int(n_class * sample_rate)
            n_drop = n_class - n_keep
            if n_drop > 0:
                class_indices = np.where(class_mask)[0]
                drop_indices = rng.choice(class_indices, size=n_drop, replace=False)
                keep_mask[drop_indices] = False

        n_kept = keep_mask.sum()
        if n_kept < self.min_points and n_points >= self.min_points:
            n_restore = self.min_points - n_kept
            dropped_indices = np.where(~keep_mask)[0]
            if len(dropped_indices) > 0:
                restore_indices = rng.choice(
                    dropped_indices,
                    size=min(n_restore, len(dropped_indices)),
                    replace=False
                )
                keep_mask[restore_indices] = True

        filter_keys = ["coord", "segment", "instance", "color", "normal", "intensity",
                       "strength", "origin_coord", "origin_segment", "index",
                       "rgb", "normals"]

        for key in filter_keys:
            if key in data_dict and data_dict[key] is not None:
                arr = data_dict[key]
                if isinstance(arr, np.ndarray) and len(arr) == n_points:
                    data_dict[key] = arr[keep_mask]

        return data_dict


# =============================================================================
# CLASS MAPPING UTILITIES
# =============================================================================

def get_default_learning_map():
    """Get the default 62-to-12 class learning map for Bhopal MLS dataset."""
    learning_map = {}
    for i in range(0, 4):
        learning_map[i] = 0      # Ground
    for i in range(4, 8):
        learning_map[i] = 1      # Vegetation
    for i in range(8, 13):
        learning_map[i] = 2      # Building
    learning_map[13] = 3          # Road
    learning_map[14] = 4          # Wire
    for i in range(15, 19):
        learning_map[i] = 5      # Pole and Tower
    learning_map[19] = 6          # Wire-Structure Connector
    for i in range(20, 35):
        learning_map[i] = 7      # Vehicle
    for i in range(35, 38):
        learning_map[i] = 8      # Board
    for i in range(38, 43):
        learning_map[i] = 9      # Fencing
    for i in range(43, 50):
        learning_map[i] = 10     # Human and Animal
    for i in range(50, 62):
        learning_map[i] = 11     # Road Side Object
    return learning_map


def get_default_learning_map_inv():
    """Get the inverse mapping from 12 target classes to representative original labels."""
    return {
        0: 0, 1: 4, 2: 8, 3: 13, 4: 14, 5: 15,
        6: 19, 7: 20, 8: 35, 9: 38, 10: 43, 11: 50,
    }


def get_default_class_names():
    """Get the default 12 class names for Bhopal MLS dataset."""
    return [
        "Ground", "Vegetation", "Building", "Road", "Wire",
        "Pole and Tower", "Wire-Structure Connector", "Vehicle",
        "Board", "Fencing", "Human and Animal", "Road Side Object",
    ]


def build_class_mapping_from_groups(class_groups):
    """Build a learning_map dict from class_groups format."""
    mapping = {}
    for new_label, old_labels in enumerate(class_groups):
        for old_label in old_labels:
            mapping[old_label] = new_label
    return mapping


DEFAULT_12CLASS_CONFIG = {
    "num_classes": 12,
    "class_names": get_default_class_names(),
    "class_groups": [
        list(range(0, 4)),
        list(range(4, 8)),
        list(range(8, 13)),
        [13], [14],
        list(range(15, 19)),
        [19],
        list(range(20, 35)),
        list(range(35, 38)),
        list(range(38, 43)),
        list(range(43, 50)),
        list(range(50, 62)),
    ],
}


@DATASETS.register_module()
class BhopalSceneHDF5Dataset:
    """
    Bhopal MLS Scene-Level HDF5 Dataset for Semantic Segmentation.
    Compatible with LitePT/Pointcept frameworks.
    """

    @staticmethod
    def get_learning_map():
        return get_default_learning_map()

    @staticmethod
    def get_learning_map_inv():
        return get_default_learning_map_inv()

    @staticmethod
    def get_class_names():
        return get_default_class_names()

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
            raise FileNotFoundError(f"HDF5 file not found: {self.h5_path}")

        self._load_dataset_info()

        features = ["XYZ"]
        if use_rgb:
            features.append("RGB")
        if use_intensity:
            features.append("intensity")
        if use_gps_time:
            features.append("gps_time")
        if use_normals:
            features.append("normals")

        print(f"Loaded Bhopal Scene HDF5 dataset for split '{split}':")
        print(f"  - Scenes: {len(self.indices):,}")
        print(f"  - Classes: {self.num_classes} ({self._class_config_source})")
        print(f"  - Features: {' + '.join(features)}")
        if self.load_instance:
            print(f"  - Instance loading: ENABLED (from scenes/global_ids)")

    def _setup_class_mapping(self, learning_map, class_groups, class_names, class_mapping):
        if not self.remap_classes:
            self.class_mapping = None
            self.num_classes = 62
            self.target_class_names = [f"Class_{i}" for i in range(62)]
            self._class_config_source = "original 62 fine-grained"
            return

        if learning_map is not None:
            self.class_mapping = learning_map
            target_values = [v for v in learning_map.values() if v != self.ignore_index]
            self.num_classes = len(set(target_values))
            self.target_class_names = class_names if class_names else [f"Class_{i}" for i in range(self.num_classes)]
            self._class_config_source = f"custom {self.num_classes}-class learning_map"
        elif class_groups is not None:
            self.class_mapping = build_class_mapping_from_groups(class_groups)
            self.num_classes = len(class_groups)
            self.target_class_names = class_names if class_names else [f"Class_{i}" for i in range(self.num_classes)]
            self._class_config_source = f"custom {self.num_classes}-class groups"
        elif class_mapping is not None:
            self.class_mapping = class_mapping
            target_values = [v for v in class_mapping.values() if v != self.ignore_index]
            self.num_classes = len(set(target_values))
            self.target_class_names = [f"Class_{i}" for i in range(self.num_classes)]
            self._class_config_source = f"custom {self.num_classes}-class mapping"
        else:
            self.class_mapping = build_class_mapping_from_groups(DEFAULT_12CLASS_CONFIG["class_groups"])
            self.num_classes = DEFAULT_12CLASS_CONFIG["num_classes"]
            self.target_class_names = DEFAULT_12CLASS_CONFIG["class_names"]
            self._class_config_source = "default 12-class Bhopal"

        self.label_to_name = {i: name for i, name in enumerate(self.target_class_names)}

    def _load_dataset_info(self):
        with h5py.File(self.h5_path, 'r') as h5f:
            def get_split_key(name):
                return f'splits/{name}_indices' if f'splits/{name}_indices' in h5f else f'splits/{name}'

            if self.split == 'train':
                indices = h5f[get_split_key('train')][:]
            elif self.split == 'val':
                indices = h5f[get_split_key('val')][:]
            elif self.split == 'test':
                indices = h5f[get_split_key('test')][:]
            else:
                raise ValueError(f"Invalid split: {self.split}")

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

            self.max_points_per_scene = h5f.attrs.get('max_points_per_scene', h5f.attrs.get('max_points', 1000000))
            self.class_names = self.target_class_names

    def __len__(self):
        return len(self.indices) * self.loop

    def _normalize_intensity(self, intensity):
        return intensity / 65535.0

    def _remap_labels(self, labels):
        remapped = np.full_like(labels, self.ignore_index)
        for old_label, new_label in self.class_mapping.items():
            remapped[labels == old_label] = new_label
        return remapped

    def get_data(self, idx):
        idx = idx % len(self.indices)
        global_idx = self.indices[idx]

        with h5py.File(self.h5_path, 'r') as h5f:
            num_pts = int(h5f['scenes/num_points'][global_idx])
            coord = h5f['scenes/points'][global_idx, :num_pts].astype(np.float32)
            intensity_raw = h5f['scenes/intensity'][global_idx, :num_pts].astype(np.float32)

            if self.use_rgb and 'scenes/rgb' in h5f:
                rgb_raw = h5f['scenes/rgb'][global_idx, :num_pts].astype(np.float32)
            else:
                rgb_raw = None

            if self.use_normals and 'scenes/normals' in h5f:
                normals = h5f['scenes/normals'][global_idx, :num_pts].astype(np.float32)
            else:
                normals = None

            segment = h5f['scenes/labels'][global_idx, :num_pts].astype(np.int64)

            # Load per-point instance IDs if needed (for instance segmentation)
            if self.load_instance and 'scenes/global_ids' in h5f:
                instance = h5f['scenes/global_ids'][global_idx, :num_pts].astype(np.int32)
            else:
                instance = None

            scene_name = h5f['metadata/scene_names'][global_idx]
            if isinstance(scene_name, bytes):
                scene_name = scene_name.decode('utf-8')

            bhopal_path_key = 'metadata/bhopal_paths' if 'metadata/bhopal_paths' in h5f else 'metadata/source_path'
            bhopal_path = int(h5f[bhopal_path_key][global_idx])

        if self.remap_classes:
            segment = self._remap_labels(segment)

        intensity = self._normalize_intensity(intensity_raw)

        if self.use_rgb and rgb_raw is not None:
            rgb = rgb_raw / 255.0
        else:
            rgb = None

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
            if rgb is not None:
                rgb = rgb[sample_indices]
            if normals is not None:
                normals = normals[sample_indices]
            if instance is not None:
                instance = instance[sample_indices]

        data_dict = {
            'coord': coord,
            'segment': segment,
            'name': f"{scene_name}_bhopal{bhopal_path}",
        }

        if self.use_intensity:
            data_dict['strength'] = intensity[:, None]

        if self.use_rgb:
            data_dict['color'] = rgb.astype(np.float32) if rgb is not None else np.zeros((len(coord), 3), dtype=np.float32)

        if self.use_normals:
            data_dict['normal'] = normals.astype(np.float32) if normals is not None else np.zeros((len(coord), 3), dtype=np.float32)

        data_dict['grid_size'] = 0.02

        # Instance segmentation: add instance IDs
        if instance is not None:
            data_dict['instance'] = instance

        return data_dict

    def get_data_name(self, idx):
        idx = idx % len(self.indices)
        global_idx = self.indices[idx]
        with h5py.File(self.h5_path, 'r') as h5f:
            scene_name = h5f['metadata/scene_names'][global_idx]
            if isinstance(scene_name, bytes):
                scene_name = scene_name.decode('utf-8')
            bhopal_path_key = 'metadata/bhopal_paths' if 'metadata/bhopal_paths' in h5f else 'metadata/source_path'
            bhopal_path = int(h5f[bhopal_path_key][global_idx])
        return f"{scene_name}_bhopal{bhopal_path}"

    def prepare_train_data(self, idx):
        data_dict = self.get_data(idx)
        if self.transform is not None:
            data_dict = self.transform(data_dict)
        return data_dict

    def prepare_test_data(self, idx):
        from copy import deepcopy

        data_dict = self.get_data(idx)
        if self.transform is not None:
            data_dict = self.transform(data_dict)

        result_dict = dict(
            segment=data_dict.pop("segment"),
            name=data_dict.pop("name")
        )

        if "instance" in data_dict:
            result_dict["instance"] = data_dict.pop("instance")

        if "origin_segment" in data_dict:
            assert "inverse" in data_dict
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
