#!/usr/bin/env python3
"""
Paris-Lille-3D PLY (50-class fine taxonomy) -> Bhopal-V9-compatible HDF5.

Uses the FINE 50-class data from
  data/Paris-Lille-3D/training_50_classes/
(per the official Roynard 2017 dataset description PDF). The fine taxonomy
explicitly labels Road, Wire, sidewalk, curb, etc. — far richer than the
coarse 10-class subset that lumps Road into Ground and discards Wire.

Key advantages of 50-class over 10-class:
  - Road (202020000)  44% of points, explicit class -> BM-12 Road
  - Wire (302030600)  rare but explicit                -> BM-12 Wire
  - Per-point x_origin/y_origin/z_origin = LiDAR sensor position, used for
    proper sensor-centric normal orientation (not tile-centroid heuristic)
  - Real GPS_time (double) instead of zero placeholders
  - label (uint32) instance IDs available for future instance segmentation

PLY fields (per the official spec):
  x, y, z              (float32) point position (offset to fit in float32)
  x_origin, y_origin, z_origin (float32) LiDAR sensor position when acquired
  GPS_time             (float64) acquisition timestamp
  reflectance          (uint8)   intensity, 0-255
  label                (uint32)  per-object instance ID
  class                (uint32)  per-point fine class ID (9-digit hierarchical)

Files (per official Table 3):
  Lille1.ply  71.3M pts  1349 objects  39 classes  (training)
  Paris.ply   45.7M pts   629 objects  41 classes  (training)
  Lille2.ply  26.8M pts   501 objects  29 classes  (validation/test)
  Total      143.1M pts  2479 objects  50 classes total

The fine `class` field is stored unchanged in the HDF5 so the dataset class
can apply different learning_maps at load time. The default learning_map
(in pointcept.datasets.paris_lille) maps PL fine -> BhopalMLS-12 schema for
the cross-region table cross-region row comparability.

Output: data/Paris-Lille-3D/paris_lille_scene_v1_normals.h5

Date: 2026-05-04
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import h5py
import numpy as np
from tqdm import tqdm

try:
    from plyfile import PlyData
except ImportError:
    print("ERROR: plyfile not installed (pip install plyfile)", file=sys.stderr); sys.exit(1)

try:
    import open3d as o3d
    HAS_OPEN3D = True
except ImportError:
    HAS_OPEN3D = False

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
PL_DATA_ROOT = Path("data/Paris-Lille-3D/training_50_classes")

# (PLY filename, source-id integer, split bucket)
PL_FILES = [
    ("Lille1.ply", 1, "train"),
    ("Paris.ply",  2, "train"),
    ("Lille2.ply", 3, "val"),    # Lille2 also serves as test (no labelled test set)
]

# Tiling parameters
BASE_TILE_M = 5.0
MIN_TILE_M = 0.625                 # adaptive subdivision floor

# Per-scene point thresholds (mirror Bhopal V8)
MIN_POINTS_THRESHOLD = 3_000
MAX_POINTS_THRESHOLD = 1_000_000
MAX_POINTS_BUFFER    = int(MAX_POINTS_THRESHOLD * 1.05)   # = 1,050,000

# Normal estimation
DEFAULT_NORMAL_K = 30


# ---------------------------------------------------------------------------
# Logging / memory helpers
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def log_mem(msg=""):
    if HAS_PSUTIL:
        proc = psutil.Process()
        mem = proc.memory_info().rss / 1024**3
        avail = psutil.virtual_memory().available / 1024**3
        logging.info(f"  [Memory] {msg}: Process={mem:.1f}GB, Available={avail:.1f}GB")


# ---------------------------------------------------------------------------
# Normal computation — Open3D KNN + sensor-origin orientation
# ---------------------------------------------------------------------------
def compute_normals_with_sensor(points, sensor_origin, k_neighbors=DEFAULT_NORMAL_K):
    """
    Open3D KNN normals oriented toward the LiDAR sensor position.

    For each point p with normal n and sensor origin s:
      if (s - p) . n < 0  => flip n  (so n points toward the sensor)

    This is the textbook orientation rule for MLS data and is correct for
    every surface (horizontal AND vertical), unlike the road-centric heuristic
    used for Bhopal which only knows tile centroid as a proxy for the sensor.

    `sensor_origin` is a (N, 3) float32 array of per-point sensor positions
    (Paris-Lille-3D's x_origin/y_origin/z_origin fields).
    """
    if not HAS_OPEN3D:
        logging.warning("Open3D not available, returning zero normals")
        return np.zeros_like(points, dtype=np.float32)

    n = len(points)
    if n < 3:
        return np.zeros_like(points, dtype=np.float32)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=k_neighbors))

    normals = np.asarray(pcd.normals).astype(np.float32)

    # Replace NaNs / zero-length with [0, 0, 1]
    norm_lens = np.linalg.norm(normals, axis=-1)
    invalid = np.isnan(normals).any(axis=-1) | (norm_lens < 1e-6)
    normals[invalid] = [0.0, 0.0, 1.0]

    # Orient toward sensor: flip when (sensor - point) . normal < 0
    to_sensor = (sensor_origin - points).astype(np.float32)
    dots = np.sum(normals * to_sensor, axis=-1)
    flip = dots < 0
    normals[flip] = -normals[flip]

    # Re-normalize (defensive)
    norms = np.linalg.norm(normals, axis=-1, keepdims=True)
    norms = np.maximum(norms, 1e-8)
    return (normals / norms).astype(np.float32)


# ---------------------------------------------------------------------------
# PLY loading
# ---------------------------------------------------------------------------
def load_ply(ply_path: Path):
    """
    Returns dict with arrays:
      xyz       (N, 3) float32
      origin    (N, 3) float32
      gps_time  (N,)   float32  (matches Bhopal V9 dtype)
      refl      (N,)   float32  (cast from uint8 0..255)
      label     (N,)   int32    (instance ID; matches Bhopal V9 global_ids dtype)
      cls       (N,)   int32    (PL fine 9-digit class IDs up to ~3e8 fit in int32; matches Bhopal V9 labels dtype)
    """
    logging.info(f"  Loading {ply_path.name} ...")
    pd = PlyData.read(str(ply_path))
    v = pd["vertex"].data
    fields = v.dtype.names
    logging.info(f"    fields: {fields}")
    n = len(v)
    logging.info(f"    n={n:,}")

    xyz = np.stack([v["x"], v["y"], v["z"]], axis=-1).astype(np.float32)

    # Sensor origin per-point
    if "x_origin" in fields:
        origin = np.stack([v["x_origin"], v["y_origin"], v["z_origin"]], axis=-1).astype(np.float32)
    else:
        # Defensive fallback: zero origin (will degrade normal orientation)
        logging.warning(f"    no x_origin in PLY; falling back to zeros (normals may flip)")
        origin = np.zeros((n, 3), dtype=np.float32)

    # GPS time. Stored as float32 to match Bhopal V9. PL GPS_time is in
    # ~19000-19500 seconds range; float32 has ~7 significant digits =
    # sub-millisecond precision, which is plenty for our z-score-normalized
    # GPS feature usage.
    if "GPS_time" in fields:
        gps_time = v["GPS_time"].astype(np.float32)
    elif "gps_time" in fields:
        gps_time = v["gps_time"].astype(np.float32)
    else:
        gps_time = np.zeros(n, dtype=np.float32)

    # Reflectance (uint8 0..255 per spec)
    if "reflectance" in fields:
        refl = v["reflectance"].astype(np.float32)
    else:
        refl = np.zeros(n, dtype=np.float32)

    # Per-point instance ID. int32 matches Bhopal V9 /scenes/global_ids dtype
    # for cross-loader parity. PL instance labels are uint32 but values are
    # well below 2^31 in practice.
    if "label" in fields:
        label = v["label"].astype(np.int32)
    else:
        label = np.full(n, -1, dtype=np.int32)

    # Per-point fine class ID. PL fine 9-digit IDs max ~3e8, fits in int32
    # (2.14e9 max). int32 matches Bhopal V9 labels dtype for cross-loader parity.
    if "class" in fields:
        cls = v["class"].astype(np.int32)
    else:
        cls = np.full(n, -1, dtype=np.int32)

    logging.info(f"    XYZ range: X=[{xyz[:,0].min():.1f}, {xyz[:,0].max():.1f}]"
                 f"  Y=[{xyz[:,1].min():.1f}, {xyz[:,1].max():.1f}]"
                 f"  Z=[{xyz[:,2].min():.1f}, {xyz[:,2].max():.1f}]")
    logging.info(f"    unique fine classes: {len(np.unique(cls))}")
    logging.info(f"    unique instance labels: {len(np.unique(label))}")
    return dict(xyz=xyz, origin=origin, gps_time=gps_time, refl=refl, label=label, cls=cls)


# ---------------------------------------------------------------------------
# 2D adaptive XY tiling
# ---------------------------------------------------------------------------
def make_tiles(xyz, base_tile_m=BASE_TILE_M):
    """
    Tile the cloud into uniform XY tiles of size base_tile_m.
    Returns list of (key=(tx,ty), idx=indices_into_xyz).
    """
    ix = np.floor(xyz[:, 0] / base_tile_m).astype(np.int64)
    iy = np.floor(xyz[:, 1] / base_tile_m).astype(np.int64)

    key = (ix.astype(np.int64) << 32) ^ (iy.astype(np.int64) & 0xFFFFFFFF)
    order = np.argsort(key, kind="stable")
    key_s = key[order]

    boundaries = np.concatenate(([0], np.where(np.diff(key_s) != 0)[0] + 1, [len(key_s)]))
    tiles = []
    for s, e in zip(boundaries[:-1], boundaries[1:]):
        if e - s < MIN_POINTS_THRESHOLD:
            continue
        idx = order[s:e]
        tx = ix[idx[0]]
        ty = iy[idx[0]]
        tiles.append(((int(tx), int(ty)), idx))
    return tiles


def adaptive_subdivide(xyz, idx, tile_m, depth=0):
    """
    Recursively subdivide a tile until each sub-tile has <= MAX_POINTS_THRESHOLD
    points OR tile_m has halved past MIN_TILE_M.
    Returns list of (sub_idx, sub_tile_m).
    """
    n = len(idx)
    if n <= MAX_POINTS_THRESHOLD or tile_m / 2.0 < MIN_TILE_M:
        return [(idx, tile_m)]

    half = tile_m / 2.0
    sx = xyz[idx, 0]
    sy = xyz[idx, 1]
    x0 = np.floor(sx / tile_m).min() * tile_m
    y0 = np.floor(sy / tile_m).min() * tile_m
    qx = ((sx - x0) >= half).astype(np.int8)
    qy = ((sy - y0) >= half).astype(np.int8)
    quad = qx * 2 + qy   # 0..3

    out = []
    for q in range(4):
        sub_local = np.where(quad == q)[0]
        if len(sub_local) < MIN_POINTS_THRESHOLD:
            continue
        sub_idx = idx[sub_local]
        out.extend(adaptive_subdivide(xyz, sub_idx, half, depth + 1))
    return out


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", required=True, help="HDF5 output path")
    ap.add_argument("--normal-k", type=int, default=DEFAULT_NORMAL_K)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit-tiles-per-file", type=int, default=0,
                    help="for smoke testing: only emit the first N tiles per PLY")
    args = ap.parse_args()

    if not HAS_OPEN3D:
        logging.error("Open3D required (pip install open3d). Aborting.")
        sys.exit(1)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    logging.info("=" * 80)
    logging.info("Paris-Lille-3D 50-class -> BM-12 scene preprocessing")
    logging.info("=" * 80)
    logging.info(f"Output:               {out_path}")
    logging.info(f"Source PLYs (50cls):  {PL_DATA_ROOT}")
    logging.info(f"Base tile size:       {BASE_TILE_M}m")
    logging.info(f"Min tile size:        {MIN_TILE_M}m")
    logging.info(f"Min points/tile:      {MIN_POINTS_THRESHOLD:,}")
    logging.info(f"Max points/tile:      {MAX_POINTS_THRESHOLD:,}")
    logging.info(f"HDF5 buffer:          {MAX_POINTS_BUFFER:,}")
    logging.info(f"Normal K-neighbors:   {args.normal_k}")
    logging.info("=" * 80)

    # ---- pass 1: tile each PLY in memory ----
    all_descriptors = []
    pl_data_cache = {}

    for ply_name, pl_id, split in PL_FILES:
        ply_path = PL_DATA_ROOT / ply_name
        if not ply_path.exists():
            logging.warning(f"Missing PLY: {ply_path} — skipping")
            continue
        log_mem(f"Before loading {ply_name}")
        data = load_ply(ply_path)
        log_mem(f"After loading {ply_name}")
        pl_data_cache[pl_id] = data

        tiles = make_tiles(data["xyz"], BASE_TILE_M)
        logging.info(f"  Initial 5m tiles ({ply_name}): {len(tiles)}")

        finalised = []
        for (tx, ty), idx in tiles:
            for sub_idx, sub_tile_m in adaptive_subdivide(data["xyz"], idx, BASE_TILE_M):
                finalised.append({
                    "pl_id": pl_id,
                    "parent_scene": ply_name.replace(".ply", ""),
                    "idx": sub_idx,
                    "tile_m": sub_tile_m,
                    "split": split,
                    "tx": tx, "ty": ty,
                })
        if args.limit_tiles_per_file > 0:
            finalised = finalised[: args.limit_tiles_per_file]
        logging.info(f"  Finalised tiles after adaptive subdivision: {len(finalised)}")

        all_descriptors.extend(finalised)

    num_scenes = len(all_descriptors)
    if num_scenes == 0:
        logging.error("No tiles produced. Aborting.")
        sys.exit(1)

    logging.info("=" * 80)
    logging.info(f"Total scenes to write: {num_scenes}")
    logging.info("=" * 80)

    # ---- pass 2: stream tiles into HDF5 ----
    with h5py.File(out_path, "w") as h5f:
        # global attrs
        h5f.attrs["dataset_name"]                = "Paris-Lille-3D Scene v1 (50-class)"
        h5f.attrs["version"]                      = "1.0"
        h5f.attrs["description"]                  = (
            "Paris-Lille-3D MLS scene-level dataset (Roynard 2017 50-class fine "
            "taxonomy). 5m XY tiles + adaptive 2x2 quad-tree subdivision. Normals "
            "computed via Open3D KNN, oriented toward the per-point LiDAR sensor "
            "origin (x_origin/y_origin/z_origin). RGB is zero-padded (PL has no "
            "RGB) for 9ch input compat with Bhopal/Kerala. Native PL fine class "
            "IDs (9-digit hierarchical, up to ~3e8) stored in scenes/labels — the "
            "ParisLilleSceneHDF5Dataset loader applies a learning_map to remap to "
            "BM-12 at training time."
        )
        h5f.attrs["coordinate_system"]            = "RGF93/Lambert93 with offset (per-section)"
        h5f.attrs["source_taxonomy"]              = "Paris-Lille-3D 50-class fine"
        h5f.attrs["segmentation_method"]          = "spatial_xy_quadtree_with_sensor_normals"
        h5f.attrs["max_points_per_scene"]         = MAX_POINTS_BUFFER
        h5f.attrs["min_points_threshold"]         = MIN_POINTS_THRESHOLD
        h5f.attrs["max_points_threshold"]         = MAX_POINTS_THRESHOLD
        h5f.attrs["base_segment_size_m"]          = BASE_TILE_M
        h5f.attrs["min_segment_size_m"]           = MIN_TILE_M
        h5f.attrs["subdivision_enabled"]          = True
        h5f.attrs["has_normals"]                  = True
        h5f.attrs["normal_k_neighbors"]           = args.normal_k
        h5f.attrs["normal_method"]                = "open3d_knn_pca_sensor_centric"
        h5f.attrs["normal_orientation"]           = "toward per-point LiDAR sensor origin (x_origin/y_origin/z_origin)"
        h5f.attrs["has_rgb"]                      = True
        h5f.attrs["rgb_source"]                   = "zeros (PL3D has no RGB)"
        h5f.attrs["rgb_coverage_note"]            = "0% — PL3D has no RGB; zeros stored for 9ch input compat"
        h5f.attrs["has_gps_time"]                 = True
        h5f.attrs["has_instance_labels"]          = True
        h5f.attrs["ignore_index"]                 = -1
        h5f.attrs["random_seed"]                  = args.seed
        h5f.attrs["creation_date"]                = time.strftime("%Y-%m-%d %H:%M:%S")
        # Bhopal V9 has num_classes=62 (fine) and num_parent_classes=12.
        # PL uses 50-class fine taxonomy (Roynard) and the same 12 parent
        # classes (BM-12 schema applied via PL_FINE_TO_BM12 learning_map).
        h5f.attrs["num_classes"]                  = 50  # PL3D fine-taxonomy size
        h5f.attrs["num_parent_classes"]           = 12  # BM-12 reporting schema
        # Building-class sampling is N/A for PL (label distribution far less
        # skewed than Bhopal's 65% Building). Stored as empty for parity.
        h5f.attrs["building_classes"]             = "[]"
        h5f.attrs["building_sample_rate"]         = 1.0
        # Split manifest (which PLY file IDs go to each split)
        h5f.attrs["train_paths"]                  = str(sorted({pid for _, pid, sp in PL_FILES if sp == "train"}))
        h5f.attrs["val_paths"]                    = str(sorted({pid for _, pid, sp in PL_FILES if sp == "val"}))
        h5f.attrs["test_paths"]                   = str(sorted({pid for _, pid, sp in PL_FILES if sp == "val"}))  # val == test
        # PL filename map (uint8 pl_files value -> filename)
        pl_file_map = {pid: fname for fname, pid, _sp in PL_FILES}
        h5f.attrs["pl_file_names"]                = json.dumps(pl_file_map)
        # Class mapping JSON — PL fine 9-digit -> BM-12 (mirrors the dataset class default)
        try:
            from pointcept.datasets.paris_lille import PL_FINE_TO_BM12
            h5f.attrs["class_mapping"]            = json.dumps({
                "fine_to_bm12": {str(k): int(v) for k, v in PL_FINE_TO_BM12.items()},
                "bm12_names": [
                    "Ground", "Vegetation", "Building", "Road", "Wire",
                    "Pole and Tower", "Wire-Structure Connector", "Vehicle",
                    "Board", "Fencing", "Human and Animal", "Road Side Object",
                ],
            })
        except Exception as _e:
            logging.warning(f"  Could not write class_mapping attr: {_e}")

        # /scenes/
        scenes = h5f.create_group("scenes")
        chunk = (1, min(MAX_POINTS_BUFFER, 65536))
        chunk3 = (1, min(MAX_POINTS_BUFFER, 65536), 3)

        ds_points    = scenes.create_dataset("points",    (num_scenes, MAX_POINTS_BUFFER, 3),
                                             dtype=np.float32, compression="gzip", compression_opts=4, chunks=chunk3)
        ds_points.attrs["description"] = "XYZ scene-centered (subtract metadata/center)"

        ds_normals   = scenes.create_dataset("normals",   (num_scenes, MAX_POINTS_BUFFER, 3),
                                             dtype=np.float32, compression="gzip", compression_opts=4, chunks=chunk3)
        ds_normals.attrs["description"] = "Surface normals; oriented toward per-point sensor origin"

        ds_rgb       = scenes.create_dataset("rgb",       (num_scenes, MAX_POINTS_BUFFER, 3),
                                             dtype=np.uint8, compression="gzip", compression_opts=4, chunks=chunk3)
        ds_rgb.attrs["description"] = "All zeros — PL3D has no RGB"

        ds_intensity = scenes.create_dataset("intensity", (num_scenes, MAX_POINTS_BUFFER),
                                             dtype=np.float32, compression="gzip", compression_opts=4, chunks=chunk)
        ds_intensity.attrs["description"] = "PL3D reflectance (uint8 0-255) cast to float32"

        ds_gps_time  = scenes.create_dataset("gps_time",  (num_scenes, MAX_POINTS_BUFFER),
                                             dtype=np.float32, compression="gzip", compression_opts=4, chunks=chunk)
        ds_gps_time.attrs["description"] = "PL3D GPS_time (seconds, float32 to match Bhopal V9)"

        # `labels` stores the FINE 9-digit class IDs (PL native). The dataset
        # class applies a learning_map to remap to BM-12 (or any target) at
        # load time. int32 matches Bhopal V9 (PL max ID ~3e8 fits easily).
        ds_labels    = scenes.create_dataset("labels",    (num_scenes, MAX_POINTS_BUFFER),
                                             dtype=np.int32, compression="gzip", compression_opts=4, chunks=chunk)
        ds_labels.attrs["description"] = "PL3D fine class IDs (9-digit), -1 = padding"

        # `global_ids` stores the per-point INSTANCE labels (PL `label` field).
        ds_global_ids= scenes.create_dataset("global_ids",(num_scenes, MAX_POINTS_BUFFER),
                                             dtype=np.int32, compression="gzip", compression_opts=4, chunks=chunk)
        ds_global_ids.attrs["description"] = "PL3D `label` per-object instance IDs, -1 = padding"

        ds_num_pts   = scenes.create_dataset("num_points", (num_scenes,), dtype=np.uint32)
        ds_orig_pts  = scenes.create_dataset("original_num_points", (num_scenes,), dtype=np.uint32)

        # /metadata/
        meta = h5f.create_group("metadata")
        dt_str = h5py.string_dtype(encoding="utf-8")
        ds_scene_names    = meta.create_dataset("scene_names",    (num_scenes,), dtype=dt_str)
        ds_parent_scene   = meta.create_dataset("parent_scene",   (num_scenes,), dtype=dt_str)
        ds_pl_files       = meta.create_dataset("pl_files",       (num_scenes,), dtype=np.uint8)
        # alias for cross-loader compatibility (BhopalSceneHDF5Dataset checks both keys)
        ds_bhopal_paths   = meta.create_dataset("bhopal_paths",   (num_scenes,), dtype=np.uint8)
        ds_bounds         = meta.create_dataset("bounds",         (num_scenes, 6), dtype=np.float64)
        ds_center         = meta.create_dataset("center",         (num_scenes, 3), dtype=np.float64)
        ds_sensor_center  = meta.create_dataset("sensor_center",  (num_scenes, 3), dtype=np.float64)
        ds_num_inst       = meta.create_dataset("num_instances",  (num_scenes,), dtype=np.uint32)
        ds_seg_size       = meta.create_dataset("segment_size_m", (num_scenes,), dtype=np.float32)

        # /class_info/ — mirrors Bhopal V9 schema for inspection / cross-loader
        # parity. PL has its own 50-class fine taxonomy plus the BM-12 parent
        # schema applied via PL_FINE_TO_BM12 in the dataset class.
        cls_grp = h5f.create_group("class_info")

        # Bhopal-12 parent classes (same as Concerto/SK 12c reporting schema)
        BM12_NAMES = [
            "Ground", "Vegetation", "Building", "Road", "Wire", "Pole and Tower",
            "Wire-Structure Connector", "Vehicle", "Board", "Fencing",
            "Human and Animal", "Road Side Object",
        ]
        BM12_COLORS = np.array([
            [128, 128, 128],   # 0 Ground
            [  0, 128,   0],   # 1 Vegetation
            [128,   0,   0],   # 2 Building
            [ 64,  64,  64],   # 3 Road
            [255, 255,   0],   # 4 Wire
            [255,   0, 255],   # 5 Pole and Tower
            [255, 128,   0],   # 6 Wire-Structure Connector
            [  0,   0, 255],   # 7 Vehicle
            [  0, 255, 255],   # 8 Board
            [128,   0, 128],   # 9 Fencing
            [255,   0,   0],   # 10 Human and Animal
            [  0, 128, 128],   # 11 Road Side Object
        ], dtype=np.uint8)
        cls_grp.create_dataset("parent_names",  data=np.asarray(BM12_NAMES, dtype=dt_str))
        cls_grp.create_dataset("parent_colors", data=BM12_COLORS)

        # PL fine taxonomy: list of 9-digit class IDs we expect
        # (pulled from the dataset class's PL_FINE_TO_BM12 to keep them in sync).
        try:
            from pointcept.datasets.paris_lille import PL_FINE_TO_BM12
            fine_ids = sorted(PL_FINE_TO_BM12.keys())
            fine_targets = [PL_FINE_TO_BM12[k] for k in fine_ids]
            cls_grp.create_dataset("fine_class_ids",     data=np.asarray(fine_ids, dtype=np.int32))
            cls_grp.create_dataset("fine_to_bm12",       data=np.asarray(fine_targets, dtype=np.int32))
        except Exception as _e:
            logging.warning(f"  Could not import PL_FINE_TO_BM12 to write class_info: {_e}")

        # ---- write each scene ----
        train_idx, val_idx, test_idx = [], [], []
        scenes_per_pl_file = {pid: 0 for _, pid, _ in PL_FILES}
        total_pts_written = 0
        actual_max = 0

        for s_idx, descr in enumerate(tqdm(all_descriptors, desc="writing scenes")):
            pl_id = descr["pl_id"]
            data = pl_data_cache[pl_id]
            sel = descr["idx"]

            pts = data["xyz"][sel]
            origin = data["origin"][sel]
            gps_time = data["gps_time"][sel]
            refl = data["refl"][sel]
            label = data["label"][sel]
            cls = data["cls"][sel]

            n_pts = len(sel)
            actual_max = max(actual_max, n_pts)

            # bounds, center
            x_min, y_min, z_min = pts.min(axis=0)
            x_max, y_max, z_max = pts.max(axis=0)
            center = pts.mean(axis=0)
            sensor_center = origin.mean(axis=0)
            pts_centered = (pts - center).astype(np.float32)
            origin_centered = (origin - center).astype(np.float32)

            # normals (on centered coords w/ centered sensor origin so orientation
            # vector is preserved): note that subtracting the same constant from
            # both points and origins leaves (origin - point) unchanged.
            try:
                nrm = compute_normals_with_sensor(
                    pts_centered, origin_centered, k_neighbors=args.normal_k)
            except Exception as e:
                logging.warning(f"  Normal estimation failed (scene {s_idx}): {e}")
                nrm = np.zeros_like(pts_centered, dtype=np.float32)

            # zero-pad to buffer
            buf3f = np.zeros((MAX_POINTS_BUFFER, 3), dtype=np.float32)
            buf3u = np.zeros((MAX_POINTS_BUFFER, 3), dtype=np.uint8)
            buf1f = np.zeros((MAX_POINTS_BUFFER,),    dtype=np.float32)
            buf1f_gps = np.zeros((MAX_POINTS_BUFFER,), dtype=np.float32)
            buf1i32_lab = np.full((MAX_POINTS_BUFFER,), -1, dtype=np.int32)
            buf1i_gid   = np.full((MAX_POINTS_BUFFER,), -1, dtype=np.int32)

            buf3f[:n_pts] = pts_centered
            ds_points[s_idx]    = buf3f
            buf3f[:n_pts] = nrm
            ds_normals[s_idx]   = buf3f
            ds_rgb[s_idx]       = buf3u                          # all zeros
            buf1f[:n_pts] = refl
            ds_intensity[s_idx] = buf1f
            buf1f_gps[:n_pts] = gps_time
            ds_gps_time[s_idx]  = buf1f_gps
            buf1i32_lab[:n_pts] = cls
            ds_labels[s_idx]    = buf1i32_lab
            buf1i_gid[:n_pts] = label.astype(np.int32, copy=False)
            ds_global_ids[s_idx]= buf1i_gid

            ds_num_pts[s_idx]    = n_pts
            ds_orig_pts[s_idx]   = n_pts

            scene_name = f"{descr['parent_scene']}_t{s_idx:05d}"
            ds_scene_names[s_idx]  = scene_name
            ds_parent_scene[s_idx] = descr["parent_scene"]
            ds_pl_files[s_idx]     = pl_id
            ds_bhopal_paths[s_idx] = pl_id
            ds_bounds[s_idx]       = [float(x_min), float(y_min), float(z_min),
                                       float(x_max), float(y_max), float(z_max)]
            ds_center[s_idx]       = center.astype(np.float64)
            ds_sensor_center[s_idx]= sensor_center.astype(np.float64)
            ds_num_inst[s_idx]     = int(np.unique(label[label >= 0]).size)
            ds_seg_size[s_idx]     = descr["tile_m"]

            scenes_per_pl_file[pl_id] += 1
            total_pts_written += n_pts

            if descr["split"] == "train":
                train_idx.append(s_idx)
            elif descr["split"] == "val":
                val_idx.append(s_idx)
                test_idx.append(s_idx)

        # /splits/
        spl = h5f.create_group("splits")
        spl.create_dataset("train_indices", data=np.asarray(train_idx, dtype=np.uint32))
        spl.create_dataset("val_indices",   data=np.asarray(val_idx,   dtype=np.uint32))
        spl.create_dataset("test_indices",  data=np.asarray(test_idx,  dtype=np.uint32))

        # final attrs
        h5f.attrs["num_scenes"]    = num_scenes
        h5f.attrs["actual_max_points"] = actual_max
        h5f.attrs["total_points"] = total_pts_written
        h5f.attrs["train_size"]   = len(train_idx)
        h5f.attrs["val_size"]     = len(val_idx)
        h5f.attrs["test_size"]    = len(test_idx)

    logging.info("=" * 80)
    logging.info(f"DONE. {num_scenes} scenes written.")
    logging.info(f"  Train: {len(train_idx)}   Val: {len(val_idx)}   Test: {len(test_idx)}")
    logging.info(f"  Tiles per PL file: {scenes_per_pl_file}")
    logging.info(f"  Total points: {total_pts_written:,}")
    logging.info(f"  Actual max points / scene: {actual_max:,}  (buffer: {MAX_POINTS_BUFFER:,})")
    logging.info(f"  Output: {out_path}")
    logging.info("=" * 80)


if __name__ == "__main__":
    main()
