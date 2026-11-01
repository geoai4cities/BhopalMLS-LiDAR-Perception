#!/usr/bin/env python3
"""
Toronto-3D PLY (9-class taxonomy) -> Bhopal-V9-compatible HDF5.

Source: Toronto-3D (Tan et al. 2020), https://github.com/WeikaiTan/Toronto-3D

PLY fields (binary_little_endian, all four files L001-L004 share this schema):
  x, y, z              (double)  point position in UTM zone 17N
  red, green, blue     (uchar)   real RGB
  scalar_Intensity     (float)   reflectance
  scalar_GPSTime       (float)   acquisition timestamp (used for sensor track)
  scalar_ScanAngleRank (float)   scan angle in degrees (unused — could feed
                                  a future 10ch input)
  scalar_Label         (float)   class ID 0..8 (encoded as float; cast to int)

Class IDs (per the official Toronto-3D README and Mavericks_classes_9.txt):
  0  Unclassified            (mapped to -1 ignore at load time)
  1  Road                    (paved road + sidewalks + curbs + parking lots —
                              all merged into one label by the dataset authors)
  2  Road_markings           (lane lines, arrows, crosswalks)
  3  Natural                 (trees, shrubs)
  4  Building
  5  Utility_line            (power / telecom wires)
  6  Pole                    (poles, traffic signs, lamp posts)
  7  Car
  8  Fence

Splits (per Tan et al. 2020 benchmark protocol):
  L001 -> train
  L002 -> val + test          (the official Toronto-3D test section; reused
                               as val since the dataset has no separate val,
                               mirroring the PL3D Lille2 precedent in
                               preprocess_pl3d_to_hdf5.py)
  L003 -> train
  L004 -> train

Splits cover ~71M training points and ~10M val/test points, ample for an
paper cross-city cell.

Coordinate handling:
  - Apply UTM_OFFSET = [627285, 4841948, 0] to subtract from the raw doubles
    BEFORE float32 cast (Toronto-3D coords are ~6.3e5 / 4.8e6 — too big for
    float32 without losing sub-millimetre precision).
  - After offset, per-scene XY tiling at 5m, adaptive 2x2 quadtree subdivide.
  - Per-scene XYZ centering (subtract scene centroid) — same as PL3D.

Normals:
  - Open3D KNN(k=30) PCA, oriented toward an estimated sensor origin.
  - Toronto-3D PLY has no explicit sensor origin field, but does expose
    scalar_GPSTime per point. Mobile-mapping LiDAR rotates rapidly so a
    GPS-time bin (~50ms) groups one rotation worth of returns; the sensor
    was approximately at the XY centroid of that bin, z just above the
    scene's max Z. Per-point sensor origin = (gps_bin_xy_centroid, z_top+1m).
    This is more accurate than a single tile-centroid (Bhopal's heuristic
    for data without GPS time) and serviceable in lieu of true x_origin
    fields (PL3D's gold standard).

RGB:
  - Real RGB (uchar 0..255). NOT zero-padded (unlike PL3D).
  - Toronto coverage: 100% per-point RGB.

Output: data/Toronto_3D/toronto_scene_v1_normals_rgb.h5

Date: 2026-05-05
"""

import argparse
import json
import logging
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
TR_DATA_ROOT = Path("data/Toronto_3D")

# (PLY filename, source-id integer, split bucket)
TR_FILES = [
    ("L001.ply", 1, "train"),
    ("L002.ply", 2, "val"),     # L002 also serves as test (no labelled test set)
    ("L003.ply", 3, "train"),
    ("L004.ply", 4, "train"),
]

# UTM zone 17N origin used by the Toronto-3D authors (per README data prep tip).
# Subtracting this brings the doubles into a few-hundred-metres range so the
# float32 cast preserves sub-mm precision.
UTM_OFFSET = np.array([627285.0, 4841948.0, 0.0], dtype=np.float64)

# Tiling parameters (mirror PL3D)
BASE_TILE_M = 5.0
MIN_TILE_M = 0.625

# Per-scene point thresholds (mirror PL3D / Bhopal V9)
MIN_POINTS_THRESHOLD = 3_000
MAX_POINTS_THRESHOLD = 1_000_000
MAX_POINTS_BUFFER    = int(MAX_POINTS_THRESHOLD * 1.05)   # = 1,050,000

# Normals
DEFAULT_NORMAL_K = 30

# Sensor-origin estimation: bin width (seconds) for GPS-time grouping.
# Toronto-3D's MLS scanner rotates ~10 Hz; 0.05s bins cover roughly half a
# rotation, which is a tight enough cluster to approximate the sensor's XY
# position at that instant.
GPS_BIN_SECONDS = 0.05

# Toronto-3D 9-class taxonomy (matches Mavericks_classes_9.txt with label 1
# renamed from "Ground" to "Road" to match the official paper terminology).
TR9_NAMES = [
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
# Sensor-origin estimation from GPS time (Toronto-3D-specific)
# ---------------------------------------------------------------------------
def estimate_sensor_origin_from_gps(xyz, gps_time, bin_seconds=GPS_BIN_SECONDS):
    """
    Per-point sensor origin estimated from GPS-time-binned XY centroids.

    For each point with timestamp t, find the centroid of all points with
    gps_time in the same bin as t. That XY centroid approximates the sensor's
    XY position at time t (mobile-mapping LiDAR scans 360 deg fast enough
    that a ~50ms time bin captures one rotation worth of returns; the
    geometric centroid of those returns is approximately the sensor's
    horizontal location).

    Z is fixed at scene_max_z + 1m so normals get pushed "up" off horizontal
    surfaces (the road/ground) — same intent as PL3D's per-point z_origin.

    Returns: (N, 3) float32 sensor origins (in input coord frame).
    """
    n = len(xyz)
    if n == 0:
        return np.zeros((0, 3), dtype=np.float32)

    t_min = gps_time.min()
    bin_idx = ((gps_time - t_min) / bin_seconds).astype(np.int64)

    # Compute per-bin XY centroid via grouped sums
    unique_bins, inverse = np.unique(bin_idx, return_inverse=True)
    n_bins = len(unique_bins)
    sums = np.zeros((n_bins, 2), dtype=np.float64)
    counts = np.zeros(n_bins, dtype=np.int64)
    np.add.at(sums, inverse, xyz[:, :2].astype(np.float64))
    np.add.at(counts, inverse, 1)
    centroids = (sums / counts[:, None]).astype(np.float32)

    sensor = np.zeros((n, 3), dtype=np.float32)
    sensor[:, :2] = centroids[inverse]
    sensor[:, 2] = float(xyz[:, 2].max()) + 1.0
    return sensor


# ---------------------------------------------------------------------------
# Normals (mirrors PL3D's compute_normals_with_sensor)
# ---------------------------------------------------------------------------
def compute_normals_with_sensor(points, sensor_origin, k_neighbors=DEFAULT_NORMAL_K):
    """Open3D KNN normals oriented toward per-point sensor positions."""
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

    norm_lens = np.linalg.norm(normals, axis=-1)
    invalid = np.isnan(normals).any(axis=-1) | (norm_lens < 1e-6)
    normals[invalid] = [0.0, 0.0, 1.0]

    to_sensor = (sensor_origin - points).astype(np.float32)
    dots = np.sum(normals * to_sensor, axis=-1)
    flip = dots < 0
    normals[flip] = -normals[flip]

    norms = np.linalg.norm(normals, axis=-1, keepdims=True)
    norms = np.maximum(norms, 1e-8)
    return (normals / norms).astype(np.float32)


# ---------------------------------------------------------------------------
# PLY loading
# ---------------------------------------------------------------------------
def load_ply(ply_path: Path):
    """
    Returns dict with arrays:
      xyz       (N, 3) float32  (UTM-offset already applied)
      sensor    (N, 3) float32  (estimated from GPS time)
      gps_time  (N,)   float32
      refl      (N,)   float32  (scalar_Intensity)
      rgb       (N, 3) uint8    (real RGB)
      label     (N,)   int32    (TR class IDs 0..8; instance labels absent)
      cls       (N,)   int32    (alias of label for cross-loader parity with
                                  Bhopal V9's scenes/labels)
    """
    logging.info(f"  Loading {ply_path.name} ...")
    pd = PlyData.read(str(ply_path))
    v = pd["vertex"].data
    fields = v.dtype.names
    logging.info(f"    fields: {fields}")
    n = len(v)
    logging.info(f"    n={n:,}")

    # Apply UTM offset on the doubles, THEN cast to float32. Reverse order
    # would lose ~1m of precision (627285 doesn't fit in float32 with mm
    # accuracy; 627285 - 627285 = 0 does).
    x = v["x"].astype(np.float64) - UTM_OFFSET[0]
    y = v["y"].astype(np.float64) - UTM_OFFSET[1]
    z = v["z"].astype(np.float64) - UTM_OFFSET[2]
    xyz = np.stack([x, y, z], axis=-1).astype(np.float32)

    if "scalar_GPSTime" in fields:
        gps_time = v["scalar_GPSTime"].astype(np.float32)
    elif "GPS_time" in fields:
        gps_time = v["GPS_time"].astype(np.float32)
    else:
        logging.warning(f"    no GPSTime in PLY; sensor track will fall back to scene centroid")
        gps_time = np.zeros(n, dtype=np.float32)

    if "scalar_Intensity" in fields:
        refl = v["scalar_Intensity"].astype(np.float32)
    else:
        refl = np.zeros(n, dtype=np.float32)

    if all(f in fields for f in ("red", "green", "blue")):
        rgb = np.stack([v["red"], v["green"], v["blue"]], axis=-1).astype(np.uint8)
    else:
        logging.warning(f"    no RGB in PLY; falling back to zeros")
        rgb = np.zeros((n, 3), dtype=np.uint8)

    # Label (class). Toronto-3D stores the class id in scalar_Label as float;
    # cast to int32 directly (values are small integers 0..8).
    if "scalar_Label" in fields:
        label = v["scalar_Label"].astype(np.int32)
    elif "class" in fields:
        label = v["class"].astype(np.int32)
    else:
        label = np.full(n, -1, dtype=np.int32)

    # Per-point sensor origin (Toronto has no explicit field; estimate from GPS).
    if (gps_time != 0).any():
        sensor = estimate_sensor_origin_from_gps(xyz, gps_time)
    else:
        # Fallback: scene centroid XY at z_top+1m
        cxy = xyz[:, :2].mean(axis=0)
        z_top = float(xyz[:, 2].max()) + 1.0
        sensor = np.zeros((n, 3), dtype=np.float32)
        sensor[:, 0] = cxy[0]
        sensor[:, 1] = cxy[1]
        sensor[:, 2] = z_top

    logging.info(f"    XYZ range (UTM-offset): X=[{xyz[:,0].min():.1f}, {xyz[:,0].max():.1f}]"
                 f"  Y=[{xyz[:,1].min():.1f}, {xyz[:,1].max():.1f}]"
                 f"  Z=[{xyz[:,2].min():.1f}, {xyz[:,2].max():.1f}]")
    logging.info(f"    unique class labels: {np.unique(label).tolist()}")

    return dict(xyz=xyz, sensor=sensor, gps_time=gps_time, refl=refl,
                rgb=rgb, label=label, cls=label)


# ---------------------------------------------------------------------------
# 2D adaptive XY tiling (identical to PL3D)
# ---------------------------------------------------------------------------
def make_tiles(xyz, base_tile_m=BASE_TILE_M):
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
    quad = qx * 2 + qy

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
    ap.add_argument("--only", type=str, default="",
                    help="comma-separated PLY basenames to include (smoke testing). "
                         "Example: --only L004.ply")
    args = ap.parse_args()

    if not HAS_OPEN3D:
        logging.error("Open3D required (pip install open3d). Aborting.")
        sys.exit(1)

    only_set = {s.strip() for s in args.only.split(",") if s.strip()} if args.only else None

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    logging.info("=" * 80)
    logging.info("Toronto-3D 9-class -> BM-12-compatible scene preprocessing")
    logging.info("=" * 80)
    logging.info(f"Output:               {out_path}")
    logging.info(f"Source PLYs:          {TR_DATA_ROOT}")
    logging.info(f"UTM_OFFSET:           {UTM_OFFSET.tolist()}")
    logging.info(f"Base tile size:       {BASE_TILE_M}m")
    logging.info(f"Min tile size:        {MIN_TILE_M}m")
    logging.info(f"Min points/tile:      {MIN_POINTS_THRESHOLD:,}")
    logging.info(f"Max points/tile:      {MAX_POINTS_THRESHOLD:,}")
    logging.info(f"HDF5 buffer:          {MAX_POINTS_BUFFER:,}")
    logging.info(f"Normal K-neighbors:   {args.normal_k}")
    logging.info(f"GPS bin (s):          {GPS_BIN_SECONDS}")
    if only_set:
        logging.info(f"Restricted to:        {sorted(only_set)}")
    logging.info("=" * 80)

    # ---- pass 1: load + tile each PLY (one at a time, drop after tiling) ----
    # Toronto-3D PLYs total ~78M points but L003 alone is ~40M; we keep one
    # PLY's arrays in memory at a time, write its tiles to disk staging, then
    # release before loading the next file. To match the PL3D pattern (which
    # cached all PLYs in memory) we do the same here — Toronto's worst-case
    # working set is 4 * (xyz + sensor + rgb + intensity + gps + label) ~=
    # 4 * 78M * (3*4 + 3*4 + 3 + 4 + 4 + 4) = ~12 GB, fits on workstation RAM.
    all_descriptors = []
    tr_data_cache = {}

    for ply_name, tr_id, split in TR_FILES:
        if only_set is not None and ply_name not in only_set:
            continue
        ply_path = TR_DATA_ROOT / ply_name
        if not ply_path.exists():
            logging.warning(f"Missing PLY: {ply_path} — skipping")
            continue
        log_mem(f"Before loading {ply_name}")
        data = load_ply(ply_path)
        log_mem(f"After loading {ply_name}")
        tr_data_cache[tr_id] = data

        tiles = make_tiles(data["xyz"], BASE_TILE_M)
        logging.info(f"  Initial 5m tiles ({ply_name}): {len(tiles)}")

        finalised = []
        for (tx, ty), idx in tiles:
            for sub_idx, sub_tile_m in adaptive_subdivide(data["xyz"], idx, BASE_TILE_M):
                finalised.append({
                    "tr_id": tr_id,
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
        h5f.attrs["dataset_name"]                = "Toronto-3D Scene v1 (9-class)"
        h5f.attrs["version"]                      = "1.0"
        h5f.attrs["description"]                  = (
            "Toronto-3D MLS scene-level dataset (Tan et al. 2020, 9-class "
            "taxonomy). 5m XY tiles + adaptive 2x2 quad-tree subdivision. "
            "Normals via Open3D KNN, oriented toward per-point sensor origins "
            "estimated from GPS-time-binned XY centroids (Toronto has no "
            "explicit x_origin/y_origin/z_origin fields). Real RGB stored "
            "(uchar 0-255). Native 9-class IDs (0..8) preserved in "
            "scenes/labels — the Toronto3DSceneHDF5Dataset loader applies a "
            "class_groups remap to BM-12 at training time. L001+L003+L004 = "
            "train, L002 = val + test (mirrors PL3D's Lille2 dual-use)."
        )
        h5f.attrs["coordinate_system"]            = (
            f"UTM zone 17N with offset {UTM_OFFSET.tolist()}"
        )
        h5f.attrs["source_taxonomy"]              = "Toronto-3D 9-class"
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
        h5f.attrs["normal_orientation"]           = (
            "toward GPS-time-binned XY centroid + scene_max_z + 1m"
        )
        h5f.attrs["has_rgb"]                      = True
        h5f.attrs["rgb_source"]                   = "Toronto-3D real RGB (uchar 0-255)"
        h5f.attrs["rgb_coverage_note"]            = "100% per-point real RGB"
        h5f.attrs["has_gps_time"]                 = True
        h5f.attrs["has_instance_labels"]          = False  # Toronto has no instance IDs
        h5f.attrs["ignore_index"]                 = -1
        h5f.attrs["random_seed"]                  = args.seed
        h5f.attrs["creation_date"]                = time.strftime("%Y-%m-%d %H:%M:%S")
        h5f.attrs["num_classes"]                  = 9   # Toronto 9-class native
        h5f.attrs["num_parent_classes"]           = 12  # BM-12 reporting schema
        h5f.attrs["building_classes"]             = "[]"
        h5f.attrs["building_sample_rate"]         = 1.0
        h5f.attrs["utm_offset"]                   = UTM_OFFSET.tolist()
        h5f.attrs["gps_bin_seconds"]              = GPS_BIN_SECONDS
        # Split manifest
        h5f.attrs["train_paths"]                  = str(sorted({tid for _, tid, sp in TR_FILES if sp == "train"}))
        h5f.attrs["val_paths"]                    = str(sorted({tid for _, tid, sp in TR_FILES if sp == "val"}))
        h5f.attrs["test_paths"]                   = str(sorted({tid for _, tid, sp in TR_FILES if sp == "val"}))  # val == test
        # Filename map (uint8 tr_id -> filename)
        tr_file_map = {tid: fname for fname, tid, _sp in TR_FILES}
        h5f.attrs["tr_file_names"]                = json.dumps(tr_file_map)
        h5f.attrs["pl_file_names"]                = json.dumps(tr_file_map)  # alias for cross-loader parity
        # Class-mapping JSON: TR raw 9 -> BM-12 (matches Toronto3DSceneHDF5Dataset default).
        # Format mirrors the Bhopal/Kerala class_groups convention used by all BM<->X cross-eval configs.
        TR9_TO_BM12_GROUPS = [
            [],          # 0: BM Ground (absent in TR — TR-1 'Road' includes
                         #     sidewalks/curbs/parking but is mapped to BM Road
                         #     per the user-confirmed cross-eval decision)
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
        h5f.attrs["class_mapping"]                = json.dumps({
            "tr9_to_bm12_groups": TR9_TO_BM12_GROUPS,
            "tr9_names": TR9_NAMES,
            "bm12_names": [
                "Ground", "Vegetation", "Building", "Road", "Wire",
                "Pole and Tower", "Wire-Structure Connector", "Vehicle",
                "Board", "Fencing", "Human and Animal", "Road Side Object",
            ],
        })

        # /scenes/
        scenes = h5f.create_group("scenes")
        chunk = (1, min(MAX_POINTS_BUFFER, 65536))
        chunk3 = (1, min(MAX_POINTS_BUFFER, 65536), 3)

        ds_points    = scenes.create_dataset("points",    (num_scenes, MAX_POINTS_BUFFER, 3),
                                             dtype=np.float32, compression="gzip", compression_opts=4, chunks=chunk3)
        ds_points.attrs["description"] = "XYZ scene-centered (UTM-offset already applied; subtract metadata/center for per-scene-centered coords)"

        ds_normals   = scenes.create_dataset("normals",   (num_scenes, MAX_POINTS_BUFFER, 3),
                                             dtype=np.float32, compression="gzip", compression_opts=4, chunks=chunk3)
        ds_normals.attrs["description"] = "Surface normals; oriented toward GPS-time-binned sensor centroid"

        ds_rgb       = scenes.create_dataset("rgb",       (num_scenes, MAX_POINTS_BUFFER, 3),
                                             dtype=np.uint8, compression="gzip", compression_opts=4, chunks=chunk3)
        ds_rgb.attrs["description"] = "Real RGB (uchar 0-255) from Toronto-3D PLY red/green/blue fields"

        ds_intensity = scenes.create_dataset("intensity", (num_scenes, MAX_POINTS_BUFFER),
                                             dtype=np.float32, compression="gzip", compression_opts=4, chunks=chunk)
        ds_intensity.attrs["description"] = "Toronto-3D scalar_Intensity (float, sensor-native)"

        ds_gps_time  = scenes.create_dataset("gps_time",  (num_scenes, MAX_POINTS_BUFFER),
                                             dtype=np.float32, compression="gzip", compression_opts=4, chunks=chunk)
        ds_gps_time.attrs["description"] = "Toronto-3D scalar_GPSTime (seconds, float32)"

        ds_labels    = scenes.create_dataset("labels",    (num_scenes, MAX_POINTS_BUFFER),
                                             dtype=np.int32, compression="gzip", compression_opts=4, chunks=chunk)
        ds_labels.attrs["description"] = "Toronto-3D 9-class native IDs (0..8), -1 = padding"

        # `global_ids` carries instance IDs in the cross-loader Bhopal V9 schema
        # but Toronto has no instance labels — fill with -1 (padding sentinel).
        ds_global_ids= scenes.create_dataset("global_ids",(num_scenes, MAX_POINTS_BUFFER),
                                             dtype=np.int32, compression="gzip", compression_opts=4, chunks=chunk)
        ds_global_ids.attrs["description"] = "Toronto-3D has no instance labels — all entries -1"

        ds_num_pts   = scenes.create_dataset("num_points", (num_scenes,), dtype=np.uint32)
        ds_orig_pts  = scenes.create_dataset("original_num_points", (num_scenes,), dtype=np.uint32)

        # /metadata/
        meta = h5f.create_group("metadata")
        dt_str = h5py.string_dtype(encoding="utf-8")
        meta.create_dataset("scene_names",    (num_scenes,), dtype=dt_str)
        meta.create_dataset("parent_scene",   (num_scenes,), dtype=dt_str)
        meta.create_dataset("tr_files",       (num_scenes,), dtype=np.uint8)
        # Aliases for cross-loader compatibility (BhopalSceneHDF5Dataset checks
        # bhopal_paths; ParisLilleSceneHDF5Dataset prefers pl_files then
        # bhopal_paths). Keeping both makes the Toronto HDF5 loadable by
        # either dataset class without modification.
        meta.create_dataset("bhopal_paths",   (num_scenes,), dtype=np.uint8)
        meta.create_dataset("pl_files",       (num_scenes,), dtype=np.uint8)
        meta.create_dataset("bounds",         (num_scenes, 6), dtype=np.float64)
        meta.create_dataset("center",         (num_scenes, 3), dtype=np.float64)
        meta.create_dataset("sensor_center",  (num_scenes, 3), dtype=np.float64)
        meta.create_dataset("num_instances",  (num_scenes,), dtype=np.uint32)
        meta.create_dataset("segment_size_m", (num_scenes,), dtype=np.float32)

        # /class_info/ — mirrors PL3D / Bhopal V9 schema for inspection.
        cls_grp = h5f.create_group("class_info")

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
        cls_grp.create_dataset("native_names",  data=np.asarray(TR9_NAMES, dtype=dt_str))
        cls_grp.create_dataset("native_class_ids", data=np.arange(9, dtype=np.int32))

        # ---- write each scene ----
        train_idx, val_idx, test_idx = [], [], []
        scenes_per_tr_file = {tid: 0 for _, tid, _ in TR_FILES}
        total_pts_written = 0
        actual_max = 0

        for s_idx, descr in enumerate(tqdm(all_descriptors, desc="writing scenes")):
            tr_id = descr["tr_id"]
            data = tr_data_cache[tr_id]
            sel = descr["idx"]

            pts    = data["xyz"][sel]
            sensor = data["sensor"][sel]
            gps_time = data["gps_time"][sel]
            refl   = data["refl"][sel]
            rgb    = data["rgb"][sel]
            label  = data["label"][sel]

            n_pts = len(sel)
            actual_max = max(actual_max, n_pts)

            x_min, y_min, z_min = pts.min(axis=0)
            x_max, y_max, z_max = pts.max(axis=0)
            center = pts.mean(axis=0)
            sensor_center = sensor.mean(axis=0)
            pts_centered = (pts - center).astype(np.float32)
            sensor_centered = (sensor - center).astype(np.float32)

            try:
                nrm = compute_normals_with_sensor(
                    pts_centered, sensor_centered, k_neighbors=args.normal_k)
            except Exception as e:
                logging.warning(f"  Normal estimation failed (scene {s_idx}): {e}")
                nrm = np.zeros_like(pts_centered, dtype=np.float32)

            buf3f = np.zeros((MAX_POINTS_BUFFER, 3), dtype=np.float32)
            buf3u = np.zeros((MAX_POINTS_BUFFER, 3), dtype=np.uint8)
            buf1f = np.zeros((MAX_POINTS_BUFFER,),   dtype=np.float32)
            buf1f_gps = np.zeros((MAX_POINTS_BUFFER,), dtype=np.float32)
            buf1i32_lab = np.full((MAX_POINTS_BUFFER,), -1, dtype=np.int32)
            buf1i_gid   = np.full((MAX_POINTS_BUFFER,), -1, dtype=np.int32)

            buf3f[:n_pts] = pts_centered
            ds_points[s_idx]    = buf3f
            buf3f[:n_pts] = nrm
            ds_normals[s_idx]   = buf3f
            buf3u[:n_pts] = rgb
            ds_rgb[s_idx]       = buf3u
            buf1f[:n_pts] = refl
            ds_intensity[s_idx] = buf1f
            buf1f_gps[:n_pts] = gps_time
            ds_gps_time[s_idx]  = buf1f_gps
            buf1i32_lab[:n_pts] = label
            ds_labels[s_idx]    = buf1i32_lab
            ds_global_ids[s_idx]= buf1i_gid       # all -1 (no instances)

            ds_num_pts[s_idx]    = n_pts
            ds_orig_pts[s_idx]   = n_pts

            scene_name = f"{descr['parent_scene']}_t{s_idx:05d}"
            meta["scene_names"][s_idx]  = scene_name
            meta["parent_scene"][s_idx] = descr["parent_scene"]
            meta["tr_files"][s_idx]     = tr_id
            meta["bhopal_paths"][s_idx] = tr_id
            meta["pl_files"][s_idx]     = tr_id
            meta["bounds"][s_idx]       = [float(x_min), float(y_min), float(z_min),
                                            float(x_max), float(y_max), float(z_max)]
            meta["center"][s_idx]       = center.astype(np.float64)
            meta["sensor_center"][s_idx]= sensor_center.astype(np.float64)
            meta["num_instances"][s_idx]= 0
            meta["segment_size_m"][s_idx] = descr["tile_m"]

            scenes_per_tr_file[tr_id] += 1
            total_pts_written += n_pts

            if descr["split"] == "train":
                train_idx.append(s_idx)
            elif descr["split"] == "val":
                val_idx.append(s_idx)
                test_idx.append(s_idx)

        spl = h5f.create_group("splits")
        spl.create_dataset("train_indices", data=np.asarray(train_idx, dtype=np.uint32))
        spl.create_dataset("val_indices",   data=np.asarray(val_idx,   dtype=np.uint32))
        spl.create_dataset("test_indices",  data=np.asarray(test_idx,  dtype=np.uint32))

        h5f.attrs["num_scenes"]    = num_scenes
        h5f.attrs["actual_max_points"] = actual_max
        h5f.attrs["total_points"] = total_pts_written
        h5f.attrs["train_size"]   = len(train_idx)
        h5f.attrs["val_size"]     = len(val_idx)
        h5f.attrs["test_size"]    = len(test_idx)

    logging.info("=" * 80)
    logging.info(f"DONE. {num_scenes} scenes written.")
    logging.info(f"  Train: {len(train_idx)}   Val: {len(val_idx)}   Test: {len(test_idx)}")
    logging.info(f"  Tiles per TR file: {scenes_per_tr_file}")
    logging.info(f"  Total points: {total_pts_written:,}")
    logging.info(f"  Actual max points / scene: {actual_max:,}  (buffer: {MAX_POINTS_BUFFER:,})")
    logging.info(f"  Output: {out_path}")
    logging.info("=" * 80)


if __name__ == "__main__":
    main()
