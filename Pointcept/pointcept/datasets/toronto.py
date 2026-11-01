"""
Scene-Level HDF5 Dataset Loader for Toronto-3D MLS Data (9-class taxonomy)
Compatible with Pointcept (PTv3 / Concerto / Sonata).

Per-point semantic segmentation (Toronto-3D has no instance labels).
Scene-level data is created by
  data/Toronto_3D/preprocess_toronto_to_hdf5.py.

Source: Toronto-3D (Tan et al. 2020), https://github.com/WeikaiTan/Toronto-3D
9 native classes (0=Unclassified, 1=Road, 2=Road_markings, 3=Natural,
4=Building, 5=Utility_line, 6=Pole, 7=Car, 8=Fence).

The default class_groups remap collapses these 9 native classes into the
unified BhopalMLS-12 schema for the cross-region experiments. The mapping
follows the dataset-mapping decisions:
    TR-0 (Unclassified)    -> -1 ignore
    TR-1 (Road, mixed*)    -> BM-3 Road
    TR-2 (Road_markings)   -> BM-3 Road            (folded into Road)
    TR-3 (Natural)         -> BM-1 Vegetation
    TR-4 (Building)        -> BM-2 Building
    TR-5 (Utility_line)    -> BM-4 Wire
    TR-6 (Pole)            -> BM-5 Pole and Tower
    TR-7 (Car)             -> BM-7 Vehicle
    TR-8 (Fence)           -> BM-9 Fencing

* Toronto-3D's class 1 "Road" merges paved road, sidewalks, curbs, and
  parking lots into a single label; we send the entire bucket to BM-3
  Road (single-bucket simplification accepted as a known limitation —
  BM-on-TR cross-eval will see Bhopal sidewalks/curbs predicted as BM-0
  Ground get counted as wrong on TR ground, since TR's GT has them
  merged).

Absent in TR (will be unevaluable on TR test): BM-0 Ground, BM-6
Wire-Structure-Connector, BM-8 Board, BM-10 Human/Animal, BM-11 Road
Side Object. Per-class IoU for these classes will be NaN/0 on TR; mIoU
should be reported over the 7 present classes (1,2,3,4,5,7,9) per the
paper plan.

Real RGB is preserved end-to-end (Toronto has uchar 0-255 R,G,B per
point, unlike PL3D which is zero-padded).

This class is a thin subclass of ParisLilleSceneHDF5Dataset — the HDF5
schema is identical (scene-level points/normals/rgb/intensity/gps_time/
labels/global_ids + metadata/* + splits/*), so the parent class's
get_data / prepare_train_data / prepare_test_data / __getitem__ work
without modification. Only the source-attribution prefix in scene names
and the default learning_map differ.

HDF5 Structure:
    scenes/points              [N, max_pts, 3]    float32  - XYZ scene-centered
    scenes/normals             [N, max_pts, 3]    float32  - Unit normals (sensor-oriented)
    scenes/rgb                 [N, max_pts, 3]    uint8    - Real Toronto RGB 0-255
    scenes/intensity           [N, max_pts]       float32  - scalar_Intensity
    scenes/gps_time            [N, max_pts]       float32  - scalar_GPSTime
    scenes/labels              [N, max_pts]       int32    - TR native 9-class IDs (0..8), -1=padding
    scenes/global_ids          [N, max_pts]       int32    - all -1 (Toronto has no instances)
    scenes/num_points          [N]                uint32
    metadata/scene_names       [N]                str
    metadata/tr_files          [N]                uint8    - 1=L001, 2=L002, 3=L003, 4=L004
    metadata/bhopal_paths      [N]                uint8    - alias of tr_files
    metadata/pl_files          [N]                uint8    - alias of tr_files
    metadata/sensor_center     [N, 3]             float64  - per-tile sensor centroid
    splits/train_indices, val_indices, test_indices    (val == test, both = L002)

Date: 2026-05-05
"""

from .builder import DATASETS
from .paris_lille import (
    BM12_CLASS_NAMES,
    ParisLilleSceneHDF5Dataset,
    build_class_mapping_from_groups,
)


# =============================================================================
# CLASS CONFIGURATION
# =============================================================================

# Toronto-3D native 9-class taxonomy (matches Mavericks_classes_9.txt with
# label 1 renamed from "Ground" to "Road" to match the official paper).
TR9_CLASS_NAMES = [
    "Unclassified",   # 0
    "Road",           # 1
    "Road_markings",  # 2
    "Natural",        # 3
    "Building",       # 4
    "Utility_line",   # 5
    "Pole",           # 6
    "Car",            # 7
    "Fence",          # 8
]


# Toronto native 9-class -> BM-12 unified, expressed as Bhopal-style class_groups.
# Empty groups (BM 0, 6, 8, 10, 11) correspond to BM classes that are absent
# in Toronto's labels. The dataset's _setup_class_mapping infers
# num_classes = len(class_groups) = 12, so the model output dimension still
# matches Bhopal-12 — needed for checkpoint compatibility in TR<->BM cross-eval.
TR9_TO_BM12_GROUPS = [
    [],          # 0: BM Ground             (absent in TR — TR-1 includes
                 #     sidewalks/curbs but is mapped to BM Road per the
                 #     user-confirmed cross-eval decision)
    [3],         # 1: BM Vegetation         <- TR Natural
    [4],         # 2: BM Building           <- TR Building
    [1, 2],      # 3: BM Road               <- TR Road + Road_markings
    [5],         # 4: BM Wire               <- TR Utility_line
    [6],         # 5: BM Pole and Tower     <- TR Pole
    [],          # 6: BM Wire-Str-Conn      (absent in TR)
    [7],         # 7: BM Vehicle            <- TR Car
    [],          # 8: BM Board              (absent in TR)
    [8],         # 9: BM Fencing            <- TR Fence
    [],          # 10: BM Human and Animal  (absent in TR)
    [],          # 11: BM Road Side Object  (absent in TR)
]


# Convenience dict form of the same mapping. Mirrors PL_FINE_TO_BM12's API
# so configs that prefer the dict form can use this directly via
# `learning_map=TORONTO_FINE_TO_BM12`. NOTE: when used as `learning_map`,
# the dataset class infers num_classes = max(targets)+1 = 10 (since the
# largest mapped target is 9). For BM-12 model parity prefer
# `class_groups=TR9_TO_BM12_GROUPS` in configs — that explicitly gives
# num_classes = 12 with empty groups for the BM classes absent in Toronto.
TORONTO_FINE_TO_BM12 = build_class_mapping_from_groups(TR9_TO_BM12_GROUPS)


def get_default_class_groups():
    """Return the default TR-9 -> BM-12 class_groups list (deep-copied)."""
    return [list(g) for g in TR9_TO_BM12_GROUPS]


def get_default_class_names():
    return list(BM12_CLASS_NAMES)


# =============================================================================
# DATASET CLASS
# =============================================================================

@DATASETS.register_module()
class Toronto3DSceneHDF5Dataset(ParisLilleSceneHDF5Dataset):
    """
    Toronto-3D Scene-Level HDF5 Dataset.

    Identical mechanics to ParisLilleSceneHDF5Dataset (same HDF5 schema);
    only the source attribution in scene names and the default class
    mapping differ. Pass `class_groups=TR9_TO_BM12_GROUPS` (recommended)
    or `learning_map=TORONTO_FINE_TO_BM12` in configs to remap the 9
    native classes to BM-12 unified at training time.

    Defaults to the BM-12 schema for the cross-region experiments.
    """

    @staticmethod
    def get_learning_map():
        """Default TR-9 -> BM-12 mapping (dict form). For BM-12 model parity
        prefer `class_groups=` in configs — see TORONTO_FINE_TO_BM12 docstring."""
        return dict(TORONTO_FINE_TO_BM12)

    @staticmethod
    def get_learning_map_inv():
        """Inverse: BM-12 target -> representative TR fine class ID (for visualization)."""
        return {
            0:  -1,   # Ground            -> N/A in TR
            1:   3,   # Vegetation        -> Natural
            2:   4,   # Building          -> Building
            3:   1,   # Road              -> Road  (TR-1; TR-2 also feeds this)
            4:   5,   # Wire              -> Utility_line
            5:   6,   # Pole and Tower    -> Pole
            6:  -1,   # Wire-Str-Conn     -> N/A in TR
            7:   7,   # Vehicle           -> Car
            8:  -1,   # Board             -> N/A in TR
            9:   8,   # Fencing           -> Fence
            10: -1,   # Human and Animal  -> N/A in TR
            11: -1,   # Road Side Object  -> N/A in TR
        }

    @staticmethod
    def get_class_names():
        return get_default_class_names()

    def __init__(self, *args, class_groups=None, learning_map=None, **kwargs):
        # If neither remap form is provided, default to the TR-9 -> BM-12
        # class_groups (cross-region convention).
        # Without this default, the parent class would fall back to
        # PL_FINE_TO_BM12, which has no overlap with TR's 9 raw labels.
        if class_groups is None and learning_map is None and kwargs.get("class_mapping") is None:
            class_groups = get_default_class_groups()
        super().__init__(
            *args,
            class_groups=class_groups,
            learning_map=learning_map,
            **kwargs,
        )
        # The parent's __init__ prints "Loaded Paris-Lille-3D Scene HDF5
        # dataset...". Print a Toronto-specific banner so logs are clear
        # when both PL and TR datasets are referenced (e.g. cross-eval).
        print(f"  -> instantiated as Toronto3DSceneHDF5Dataset (TR-9 -> BM-12 class_groups; real RGB)")

    def get_data(self, idx):
        # Parent's get_data hardcodes "_pl{N}" in data_dict["name"]; rewrite
        # to "_tr{N}" so per-scene result files / log lines disambiguate
        # Toronto scenes from Paris-Lille scenes.
        data_dict = super().get_data(idx)
        if "name" in data_dict and "_pl" in data_dict["name"]:
            data_dict["name"] = data_dict["name"].replace("_pl", "_tr", 1)
        return data_dict

    def get_data_name(self, idx):
        # Override the source-attribution prefix ("tr" instead of "pl").
        idx = idx % len(self.indices)
        global_idx = self.indices[idx]

        import h5py
        with h5py.File(self.h5_path, 'r') as h5f:
            scene_name = h5f['metadata/scene_names'][global_idx]
            if isinstance(scene_name, bytes):
                scene_name = scene_name.decode('utf-8')
            if 'metadata/tr_files' in h5f:
                source_path = int(h5f['metadata/tr_files'][global_idx])
            elif 'metadata/bhopal_paths' in h5f:
                source_path = int(h5f['metadata/bhopal_paths'][global_idx])
            else:
                source_path = 0
        return f"{scene_name}_tr{source_path}"
