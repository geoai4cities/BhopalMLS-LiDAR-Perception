"""
Scene-Level HDF5 Dataset Loader for Paris-Lille-3D MLS Data (50-class fine taxonomy)
Compatible with Pointcept framework and PTv3/Concerto/Sonata.

Per-point semantic + instance segmentation. Scene-level data created by
data/Paris-Lille-3D/preprocess_pl3d_to_hdf5.py.

Source: Paris-Lille-3D 50-class fine taxonomy (Roynard 2017). Each
labelled point carries a 9-digit hierarchical class ID (e.g. 202020000=road,
302030600=wire). The default learning_map remaps PL fine IDs -> BhopalMLS-12
unified schema for cross-region comparability (mirrors the SK-12
finetune precedent).

No RGB (zero-padded in HDF5).  Reflectance is uint8 0-255 (cast to float32),
normalized by /255.

HDF5 Structure:
    scenes/points              [N, max_pts, 3]    float32  - XYZ scene-centered
    scenes/normals             [N, max_pts, 3]    float32  - Unit normals (sensor-oriented)
    scenes/rgb                 [N, max_pts, 3]    uint8    - Zeros (PL has no RGB)
    scenes/intensity           [N, max_pts]       float32  - PL reflectance, 0-255
    scenes/gps_time            [N, max_pts]       float32  - PL GPS timestamps (matches Bhopal V9)
    scenes/labels              [N, max_pts]       int32    - PL fine 9-digit class IDs, -1=padding (matches Bhopal V9)
    scenes/global_ids          [N, max_pts]       int32    - PL instance IDs, -1=padding
    scenes/num_points          [N]                uint32
    metadata/scene_names       [N]                str
    metadata/pl_files          [N]                uint8    - 1=Lille1, 2=Paris, 3=Lille2
    metadata/bhopal_paths      [N]                uint8    - alias of pl_files
    metadata/sensor_center     [N, 3]             float64  - per-tile sensor centroid
    splits/train_indices, val_indices, test_indices

Date: 2026-05-04
"""

import h5py
import numpy as np
from pathlib import Path

from .builder import DATASETS
from .transform import Compose, TRANSFORMS


# =============================================================================
# CLASS CONFIGURATION
# =============================================================================

# Bhopal/Kerala-shared 12-class parent taxonomy (cross-region reporting schema)
BM12_CLASS_NAMES = [
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


# Comprehensive PL3D fine 9-digit class ID -> BhopalMLS-12 learning_map.
# Built from the official classes.xml (training_50_classes/).
#
# Where the PL hierarchy uses parent or "other_X" placeholders, we map them
# to the most reasonable child cluster (e.g., "303040000 4+ wheelers"
# -> Vehicle). Truly ambiguous parents (e.g., "200000000 surface") map to
# -1 ignore, matching the convention used for SemanticKITTI's "other-*"
# classes in the SK 12c learning_map.
PL_FINE_TO_BM12 = {
    # ---- Unclassified / ambiguous parents -> ignore ----
    0:           -1,  # unclassified
    100000000:   -1,  # other (outliers)
    200000000:   -1,  # surface (parent — could be anything)
    201000000:   -1,  # other surface
    300000000:   -1,  # object (parent)
    301000000:   -1,  # other object
    302000000:   -1,  # static (parent)
    302010000:   -1,  # other static
    303000000:   -1,  # dynamic (parent)
    303010000:   -1,  # other dynamic

    # ---- Ground (BM 0): sidewalk + curb + island + grass + other_ground ----
    202000000:    0,  # ground (parent — most likely ground surface)
    202010000:    0,  # other ground
    202030000:    0,  # sidewalk
    202040000:    0,  # curb
    202050000:    0,  # island (pedestrian refuge)
    202060000:    0,  # vegetation (grass/lawn — coarse maps it to ground)

    # ---- Vegetation (BM 1): trees, bushes, hedges, plants ----
    304000000:    1,  # natural (parent)
    304010000:    1,  # other natural
    304020000:    1,  # tree
    304030000:    1,  # bush
    304040000:    1,  # potted plant
    304050000:    1,  # hedge

    # ---- Building (BM 2) ----
    203000000:    2,  # building
    302040700:    2,  # lighting console (xml has coarse=building for this)

    # ---- Road (BM 3): drivable surface ----
    202020000:    3,  # road

    # ---- Wire (BM 4): electric/telecom wires ----
    302030600:    4,  # wire

    # ---- Pole and Tower (BM 5): posts, lamps, traffic signals/signs ----
    302020000:    5,  # punctual (parent — most are posts)
    302020100:    5,  # other punctual
    302020200:    5,  # post
    302020300:    5,  # bollard
    302020400:    5,  # floor lamp
    302020500:    5,  # traffic light
    302020600:    5,  # traffic sign

    # ---- Wire-Structure Connector (BM 6): no PL equivalent ----

    # ---- Vehicle (BM 7): all 2-/4-wheelers ----
    303030000:    7,  # 2 wheelers (parent)
    303030100:    7,  # other 2 wheels
    303030200:    7,  # bicycle (parent)
    303030201:    7, 303030202: 7, 303030203: 7, 303030204: 7,  # bicycle variants
    303030300:    7,  # scooter (parent)
    303030301:    7, 303030302: 7, 303030303: 7, 303030304: 7,
    303030400:    7,  # moped (parent)
    303030401:    7, 303030402: 7, 303030403: 7, 303030404: 7,
    303030500:    7,  # motorbike (parent)
    303030501:    7, 303030502: 7, 303030503: 7, 303030504: 7,
    303040000:    7,  # 4+ wheelers (parent)
    303040100:    7,  # other 4+ wheels
    303040200:    7,  # car (parent)
    303040201:    7, 303040202: 7, 303040203: 7, 303040204: 7,  # car variants
    303040300:    7,  # van (parent)
    303040301:    7, 303040302: 7, 303040303: 7, 303040304: 7,
    303040400:    7,  # truck (parent)
    303040401:    7, 303040402: 7, 303040403: 7, 303040404: 7,
    303040500:    7,  # bus (parent)
    303040501:    7, 303040502: 7, 303040503: 7, 303040504: 7,

    # ---- Board (BM 8): advertising signboards ----
    302020700:    8,  # signboard

    # ---- Fencing (BM 9): barrier / grid / chain / low wall / wired fence ----
    302030000:    9,  # linear (parent)
    302030100:    9,  # other linear
    302030200:    9,  # barrier
    302030300:    9,  # roasting (wired fence)
    302030400:    9,  # grid
    302030500:    9,  # chain
    302030700:    9,  # low wall

    # ---- Human and Animal (BM 10): pedestrians ----
    303020000:   10,  # pedestrian (parent)
    303020100:   10,  # other pedestrian
    303020200:   10, 303020300: 10, 303020400: 10, 303020500: 10,
    303020600:   10, 303020700: 10, 303020800: 10, 303020900: 10,
    303021000:   10,  # wheelchair

    # ---- Road Side Object (BM 11): everything else street furniture ----
    302020800:   11,  # mailbox
    302020900:   11,  # trash can (fixed)
    302021000:   11,  # meter
    302021100:   11,  # bicycle terminal
    302021200:   11,  # bicycle rack
    302021300:   11,  # statue
    302040000:   11,  # extended (parent)
    302040100:   11,  # other extended
    302040200:   11,  # shelter
    302040300:   11,  # kiosk
    302040400:   11,  # scaffolding
    302040500:   11,  # bench
    302040600:   11,  # distribution box
    302040800:   11,  # windmill
    303050000:   11,  # furniture (parent)
    303050100:   11,  # other furniture
    303050200:   11,  # table
    303050300:   11,  # chair
    303050400:   11,  # stool
    303050500:   11,  # trash can (mobile)
    303050600:   11,  # waste
}


def get_default_learning_map():
    """PL fine 9-digit class ID -> BhopalMLS-12 unified schema."""
    return dict(PL_FINE_TO_BM12)


def get_default_learning_map_inv():
    """Inverse: BM-12 target -> representative PL fine class ID."""
    return {
        0:  202030000,  # Ground             -> sidewalk
        1:  304020000,  # Vegetation         -> tree
        2:  203000000,  # Building           -> building
        3:  202020000,  # Road               -> road
        4:  302030600,  # Wire               -> wire
        5:  302020200,  # Pole and Tower     -> post
        6:  -1,         # Wire-Str-Conn      -> N/A in PL
        7:  303040204,  # Vehicle            -> parked car
        8:  302020700,  # Board              -> signboard
        9:  302030200,  # Fencing            -> barrier
        10: 303020300,  # Human and Animal   -> walking pedestrian
        11: 302020900,  # Road Side Object   -> trash can (fixed)
    }


def get_default_class_names():
    return list(BM12_CLASS_NAMES)


def build_class_mapping_from_groups(class_groups):
    """Build a learning_map dict from class_groups format."""
    mapping = {}
    for new_label, old_labels in enumerate(class_groups):
        for old_label in old_labels:
            mapping[old_label] = new_label
    return mapping


@DATASETS.register_module()
class ParisLilleSceneHDF5Dataset:
    """
    Paris-Lille-3D Scene-Level HDF5 Dataset for Semantic / Instance Segmentation.

    Defaults match the BM-12 reporting schema.  Pass
    `remap_classes=False` to keep the native PL fine 9-digit class IDs (useful
    for native PL-50 benchmarks reported in supplementary).
    """

    # ================================================================
    # STATIC METHODS (SemanticKITTI API compatibility)
    # ================================================================

    @staticmethod
    def get_learning_map():
        return get_default_learning_map()

    @staticmethod
    def get_learning_map_inv():
        return get_default_learning_map_inv()

    @staticmethod
    def get_class_names():
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
            h5_path: Path to PL scene HDF5 (50-class fine version).
            split: 'train' | 'val' | 'test' (val == test in PL: Lille2.ply).
            num_points: Target points per sample. None = use ALL points.
            transform: Pointcept transform pipeline.
            test_mode: Whether in test mode.
            test_cfg: Test configuration.
            loop: Number of times to loop over dataset.
            use_intensity: Add normalized reflectance to data_dict['strength'].
            use_gps_time: Add z-score-normalized GPS time as feature.
            use_normals: Load HDF5 'normals' as data_dict['normal'].
            use_rgb: Load HDF5 'rgb' as data_dict['color']. PL RGB is zeros,
                     so this yields a zero color tensor — useful for 9ch
                     configs that match Bhopal/Kerala input shape.
            cache_data: Whether to cache data in memory.
            remap_classes: If True (default), apply learning_map (PL fine
                           -> BM-12 by default). If False, return native PL
                           fine 9-digit class IDs.
            learning_map: Override the default PL fine -> BM-12 mapping.
            class_groups: Bhopal-style list-of-lists alternative.
            class_names: Override class names.
            class_mapping: DEPRECATED — use learning_map.
            ignore_index: Label to ignore in loss computation.
            load_instance: If True, load 'global_ids' as data_dict['instance']
                           for instance segmentation configs (PG-v1m2 etc).
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
                f"Create it with:\n"
                f"  python data/Paris-Lille-3D/preprocess_pl3d_to_hdf5.py "
                f"--output {self.h5_path}"
            )

        self._load_dataset_info()

        features = ["XYZ"]
        if use_rgb:
            features.append("RGB(zeros)")
        if use_intensity:
            features.append("reflectance(/255)")
        if use_gps_time:
            features.append("gps_time(z-score)")
        if use_normals:
            features.append("normals")

        print(f"Loaded Paris-Lille-3D Scene HDF5 dataset for split '{split}':")
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

    def _setup_class_mapping(self, learning_map, class_groups, class_names, class_mapping):
        if not self.remap_classes:
            self.class_mapping = None
            self.num_classes = -1   # native PL fine: too many possible IDs to enumerate
            self.target_class_names = []
            self._class_config_source = "PL native fine 9-digit class IDs (no remap)"
            self.label_to_name = {}
            return

        if learning_map is not None:
            self.class_mapping = dict(learning_map)
            target_values = [v for v in self.class_mapping.values() if v != self.ignore_index]
            # num_classes = max(target) + 1 (NOT len(unique targets)). This is
            # the model output dimension. We intentionally allow targets that
            # are absent in the training data — e.g. BM 6 Wire-Structure
            # Connector has no PL fine-class equivalent so no fine ID maps to
            # 6, but the FT model still needs a 12-way classifier so the
            # checkpoint is compatible for BM<->PL cross-region inference.
            self.num_classes = (max(target_values) + 1) if target_values else 0

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
            # Default: PL fine -> BM12
            self.class_mapping = get_default_learning_map()
            self.num_classes = len(BM12_CLASS_NAMES)
            self.target_class_names = list(BM12_CLASS_NAMES)
            self._class_config_source = "default PL fine -> BM-12"

        self.label_to_name = {i: n for i, n in enumerate(self.target_class_names)}

    def _load_dataset_info(self):
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
                'max_points_per_scene', h5f.attrs.get('max_points', 1050000)
            )
            self.class_names = self.target_class_names

    def get_num_channels(self):
        channels = 0
        if self.use_rgb: channels += 3
        if self.use_intensity: channels += 1
        if self.use_gps_time: channels += 1
        if self.use_normals: channels += 3
        return channels

    def __len__(self):
        return len(self.indices) * self.loop

    def _normalize_intensity(self, intensity):
        """PL reflectance is uint8 (0-255). Normalize to [0, 1]."""
        return intensity / 255.0

    def _normalize_gps_time(self, gps_time, valid_mask):
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
        """
        Remap PL fine 9-digit class IDs via class_mapping. Unmapped -> ignore.

        Implementation note: PL fine IDs are very sparse in [0, ~3e8], so we
        iterate over the (small) class_mapping dict rather than building a
        dense lookup table. This is identical to Bhopal/Kerala's approach.
        """
        remapped = np.full_like(labels, self.ignore_index)
        for old_label, new_label in self.class_mapping.items():
            remapped[labels == old_label] = new_label
        # Preserve padding
        remapped[labels == self.ignore_index] = self.ignore_index
        return remapped

    def get_data(self, idx):
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

            # PL fine class IDs are int32 in HDF5 (9-digit hierarchical up to ~3e8
            # fits in int32). Cast to int64 here so the learning_map remap below
            # produces int64 segments — matches the rest of the Pointcept pipeline.
            segment = h5f['scenes/labels'][global_idx, :num_pts].astype(np.int64)

            if self.load_instance and 'scenes/global_ids' in h5f:
                instance = h5f['scenes/global_ids'][global_idx, :num_pts].astype(np.int32)
            else:
                instance = None

            scene_name = h5f['metadata/scene_names'][global_idx]
            if isinstance(scene_name, bytes):
                scene_name = scene_name.decode('utf-8')

            if 'metadata/pl_files' in h5f:
                source_path = int(h5f['metadata/pl_files'][global_idx])
            elif 'metadata/bhopal_paths' in h5f:
                source_path = int(h5f['metadata/bhopal_paths'][global_idx])
            elif 'metadata/source_paths' in h5f:
                source_path = int(h5f['metadata/source_paths'][global_idx])
            else:
                source_path = 0

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

        if self.use_rgb and rgb_raw is not None:
            rgb = rgb_raw.astype(np.float32)
        else:
            rgb = None

        # ================================================================
        # POINT SAMPLING
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
            'name': f"{scene_name}_pl{source_path}",
        }

        if self.use_intensity:
            data_dict['strength'] = intensity[:, None]

        if self.use_rgb:
            if rgb is not None:
                data_dict['color'] = rgb
            else:
                data_dict['color'] = np.zeros((len(coord), 3), dtype=np.float32)

        if self.use_normals:
            if normals is not None:
                data_dict['normal'] = normals.astype(np.float32)
            else:
                data_dict['normal'] = np.zeros((len(coord), 3), dtype=np.float32)

        data_dict['grid_size'] = 0.02

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
            if 'metadata/pl_files' in h5f:
                source_path = int(h5f['metadata/pl_files'][global_idx])
            elif 'metadata/bhopal_paths' in h5f:
                source_path = int(h5f['metadata/bhopal_paths'][global_idx])
            else:
                source_path = 0
        return f"{scene_name}_pl{source_path}"

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
        return self.label_to_name.get(label, f"Unknown_{label}")

    def get_scene_info(self, idx):
        idx = idx % len(self.indices)
        global_idx = self.indices[idx]

        with h5py.File(self.h5_path, 'r') as h5f:
            scene_name = h5f['metadata/scene_names'][global_idx]
            if isinstance(scene_name, bytes):
                scene_name = scene_name.decode('utf-8')

            info = {
                'scene_name': scene_name,
                'num_points': int(h5f['scenes/num_points'][global_idx]),
            }
            if 'metadata/pl_files' in h5f:
                info['pl_file'] = int(h5f['metadata/pl_files'][global_idx])
            if 'metadata/num_instances' in h5f:
                info['num_instances'] = int(h5f['metadata/num_instances'][global_idx])
            return info
