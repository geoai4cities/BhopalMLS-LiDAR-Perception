#!/usr/bin/env python3
"""
Bhopal MLS Scene-Level V8 Dataset Creator - With Surface Normals
=================================================================

Creates SCENE-LEVEL samples with CLASS-SPECIFIC SAMPLING, ADAPTIVE SUBDIVISION,
and SURFACE NORMAL ESTIMATION for better compatibility with Concerto pre-training.

V8 Strategy (builds on V7):
- All V7 features: Building sampling, adaptive subdivision, safety downsampling
- NEW: Surface normals computed using Open3D KNN-based PCA (k=30)
- NEW: Road-centric normal orientation (Z-up for horizontal, toward road for vertical)
- NEW: 7-channel input support: XYZ (3) + intensity (1) + normal (3)

V9 Strategy (builds on V8):
- All V8 features plus RGB support
- RGB values extracted from V9 source files (~35-40% coverage, Bhopal_2/8: 0%)
- 10-channel input support: XYZ (3) + RGB (3) + intensity (1) + normal (3)
- Improved Concerto compatibility with RGB + normal channels

Normal Computation (Road-centric for MLS data):
1. Open3D estimate_normals() with KNN k=30
2. Hybrid orientation based on surface type:
   - Horizontal surfaces (|nz| > 0.5): Z-up (ground, roofs point upward)
   - Vertical surfaces (|nz| <= 0.5): toward road center in XY plane
3. Normalize to unit length

Why Add Normals?
- Official Concerto uses 9-channel input: XYZ + RGB + normal
- Our LiDAR has no RGB, but we can compute normals from point cloud geometry
- Normals improve semantic understanding of surfaces (horizontal vs vertical)
- Better transfer learning from Concerto outdoor pre-trained checkpoint

Building Classes (keep 90%, drop 10%):
- 8: Building
- 9: Residential
- 10: Commercial
- 11: Industrial
- 12: Special

Input:
- Scene definitions: Bhopal_scene_division_v5/Bhopal_N/S*_*.json
- V1 scene definitions: Bhopal_scene_division_v1/Bhopal_N/S*.txt
- Point data: Combined_NPZ_Files_V14_Training_And_Unlabeled/Bhopal_N.npz

Output Structure:
    bhopal_scene_v9.h5
    ├── scenes/
    │   ├── points              [N, max_pts, 3]    float32 (scene-centered)
    │   ├── rgb                 [N, max_pts, 3]    uint8 (RGB from V9, ~35-40% coverage)
    │   ├── normals             [N, max_pts, 3]    float32 (unit normals, road-centric)
    │   ├── intensity           [N, max_pts]       float32 (raw 0-65535)
    │   ├── gps_time            [N, max_pts]       float32
    │   ├── labels              [N, max_pts]       int32 (0-61, -1=padding)
    │   ├── global_ids          [N, max_pts]       int32 (-1=padding)
    │   ├── num_points          [N]                uint32
    │   └── original_num_points [N]                uint32 (before sampling)
    ├── metadata/
    │   ├── scene_names         [N]                str
    │   ├── bhopal_paths        [N]                uint8
    │   ├── bounds              [N, 6]             float64 (xmin,ymin,zmin,xmax,ymax,zmax)
    │   ├── center              [N, 3]             float64
    │   ├── num_instances       [N]                uint32
    │   ├── segment_size_m      [N]                float32 (5.0 or 2.5)
    │   └── parent_scene        [N]                str (original V5 scene name)
    ├── splits/
    │   ├── train_indices       [T]                uint32
    │   ├── val_indices         [V]                uint32
    │   └── test_indices        [E]                uint32
    ├── class_info/
    │   ├── names               [62]               str
    │   ├── colors              [62, 3]            uint8
    │   ├── parent_ids          [62]               uint8
    │   ├── xxzz_codes          [62]               str
    │   ├── parent_names        [12]               str
    │   └── parent_colors       [12, 3]            uint8
    └── Attributes:
        ├── dataset_name, version, description, segmentation_method
        ├── num_scenes, max_points_per_scene, total_points
        ├── num_classes (62), num_parent_classes (12)
        ├── coordinate_system, creation_date, random_seed
        ├── train_size, val_size, test_size
        ├── train_paths, val_paths, test_paths
        ├── ignore_index (-1)
        ├── building_classes, building_sample_rate (0.90)
        ├── min_points_threshold (3000), max_points_threshold (1000000)
        ├── subdivision_enabled (True), base_segment_size_m (5.0)
        ├── has_normals (True), normal_k_neighbors (30)
        ├── has_rgb (True), rgb_source (V9), rgb_coverage (~35-40%)
        └── class_mapping (JSON)

Usage:
    python3 19b_create_scene_V8_dataset.py --output ../bhopal_scene_v9.h5 \
        --train-paths 1 2 3 4 --val-paths 5 6 --test-paths 7 8

    # Custom thresholds (0.85 = keep 85% of building points)
    python3 19b_create_scene_V8_dataset.py --output ../bhopal_scene_v9.h5 \
        --building-sample-rate 0.85 --min-points 3000 --max-points 1000000 \
        --train-paths 1 2 3 4 --val-paths 5 6 --test-paths 7 8

    # Different normal k-neighbors
    python3 19b_create_scene_V8_dataset.py --output ../bhopal_scene_v9.h5 \
        --normal-k 50 --train-paths 1 2 3 4 --val-paths 5 6 --test-paths 7 8

    # Show info
    python3 19b_create_scene_V8_dataset.py --info bhopal_scene_v9.h5

Date: 2026-01-25
Version: 9.0
Author: Pyare Lal
"""

import os
import sys
import csv
import gc
import time
import traceback
import warnings
from collections import defaultdict
import h5py
import numpy as np
import json
import re
import argparse
import logging
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
import pandas as pd

# Suppress pandas casting warnings
warnings.filterwarnings('ignore', category=RuntimeWarning, message='invalid value encountered in cast')

# Open3D for normal estimation
try:
    import open3d as o3d
    HAS_OPEN3D = True
except ImportError:
    HAS_OPEN3D = False
    # Note: Warning will be logged in create_scene_v8_hdf5_dataset() after logging is configured

# Memory monitoring
try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False


def log_memory_usage(msg=""):
    """Log current memory usage if psutil is available."""
    if HAS_PSUTIL:
        process = psutil.Process()
        mem_gb = process.memory_info().rss / (1024 ** 3)
        avail_gb = psutil.virtual_memory().available / (1024 ** 3)
        logging.info(f"  [Memory] {msg}: Process={mem_gb:.1f}GB, Available={avail_gb:.1f}GB")
    return None

# Configuration
BASE_DIR = Path("data/Bhopal/source")  # holds Combined_NPZ_Files_V14_Training_And_Unlabeled/ (= npz/ of the release), the scene_division folders and class_mapping/
COMBINED_NPZ_DIR = BASE_DIR / "Combined_NPZ_Files_V14_Training_And_Unlabeled"
SCENE_V5_DIR = BASE_DIR / "Bhopal_scene_division_v5"
SCENE_V1_DIR = BASE_DIR / "Bhopal_scene_division_v1"
MAPPING_CSV = BASE_DIR / "class_mapping" / "Bhopal_class_mapping_V13_TRAINING.csv"

# V8 Sampling Configuration (same as V7)
BUILDING_CLASSES = [8, 9, 10, 11, 12]  # Building, Residential, Commercial, Industrial, Special
DEFAULT_BUILDING_SAMPLE_RATE = 0.90   # Keep 90% of building classes (drop 10%)
DEFAULT_MIN_POINTS = 3000             # Exclude scenes with fewer points after sampling
DEFAULT_MAX_POINTS = 1_000_000        # Subdivide scenes exceeding this threshold

# V8 Normal Estimation Configuration
DEFAULT_NORMAL_K_NEIGHBORS = 30       # KNN neighbors for normal estimation (Concerto default)

# Dataset metadata
DATASET_NAME = "Bhopal MLS Scene V9"
DATASET_VERSION = "9.0"
DATASET_DESCRIPTION = (
    "Mobile Laser Scanning (MLS) dataset for urban semantic segmentation. "
    "Collected in Bhopal, India. Contains 62 fine-grained classes organized "
    "into 12 parent categories. Scene-level segments based on TRUE SPATIAL "
    "CUTTING along road axis (5m base, adaptive subdivision for large scenes). "
    "Building classes (8,9,10,11,12) sampled at 90% (drop 10%). Scenes <3K points excluded. "
    "All scenes kept under 1M points via recursive subdivision. "
    "V8: Surface normals computed using Open3D KNN-based PCA (k=30) with "
    "road-centric orientation (horizontal=Z-up, vertical=toward road center). "
    "V9 NEW: RGB values extracted from V9 source files (~35-40% coverage, Bhopal_2/8 have 0%). "
    "10-channel input for Concerto pre-training: XYZ (3) + RGB (3) + intensity (1) + normal (3)."
)
COORDINATE_SYSTEM = "UTM Zone 43N (EPSG:32643)"

# 12 Parent class definitions with colors
PARENT_CLASSES = {
    0: {"name": "Ground", "color": [128, 128, 128]},
    1: {"name": "Vegetation", "color": [0, 128, 0]},
    2: {"name": "Building", "color": [128, 0, 0]},
    3: {"name": "Road", "color": [64, 64, 64]},
    4: {"name": "Wire", "color": [255, 255, 0]},
    5: {"name": "Pole and Tower", "color": [255, 0, 255]},
    6: {"name": "Wire-Structure Connector", "color": [255, 128, 0]},
    7: {"name": "Vehicle", "color": [0, 0, 255]},
    8: {"name": "Board", "color": [0, 255, 255]},
    9: {"name": "Fencing", "color": [128, 0, 128]},
    10: {"name": "Human and Animal", "color": [255, 0, 0]},
    11: {"name": "Road Side Object", "color": [0, 128, 128]},
}

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


def compute_normals(points, k_neighbors=30):
    """
    Compute surface normals using Open3D KNN-based PCA with road-centric orientation.

    Road-centric orientation for MLS (Mobile Laser Scanner) data:
    - The scanner is mounted on a vehicle driving along the road
    - Most surfaces are either horizontal (ground, roofs) or vertical (walls, poles)
    - We use a hybrid approach based on surface orientation:

    Orientation Strategy:
    1. Horizontal surfaces (|nz| > 0.5): Z-up (normals point upward)
       - Ground/road surfaces → UP toward scanner
       - Roof surfaces → UP (consistent with ground)
    2. Vertical surfaces (|nz| <= 0.5): toward road center in XY plane
       - Building walls → toward road/scanner
       - Poles, signs, fencing → toward road/scanner

    This is more physically meaningful than pure center-based orientation
    because it produces consistent normals for the same surface type.

    Args:
        points: (N, 3) array of XYZ coordinates (scene-centered)
        k_neighbors: Number of neighbors for KNN (default: 30)

    Returns:
        normals: (N, 3) array of unit normal vectors
    """
    if not HAS_OPEN3D:
        logging.warning("Open3D not available, returning zero normals")
        return np.zeros_like(points, dtype=np.float32)

    num_points = len(points)
    if num_points < 3:
        return np.zeros_like(points, dtype=np.float32)

    # Create Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))

    # Estimate normals using KNN
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamKNN(knn=k_neighbors)
    )

    # Get normals as numpy array
    normals = np.asarray(pcd.normals).astype(np.float32)

    # Check for invalid normals (NaN or zero-length from Open3D)
    normal_lengths = np.linalg.norm(normals, axis=-1)
    invalid_mask = np.isnan(normals).any(axis=-1) | (normal_lengths < 1e-6)

    # =========================================================================
    # Road-centric hybrid orientation
    # =========================================================================

    # Classify surfaces by orientation:
    # - Horizontal: |nz| > 0.5 (ground, roofs, horizontal surfaces)
    # - Vertical: |nz| <= 0.5 (walls, poles, signs, fencing)
    is_horizontal = np.abs(normals[:, 2]) > 0.5

    # --- Horizontal surfaces: Z-up (flip to point upward) ---
    # Ground surfaces should have normals pointing UP toward the scanner
    # Roofs should also point UP for consistency
    horizontal_flip_mask = is_horizontal & (normals[:, 2] < 0)
    normals[horizontal_flip_mask] = -normals[horizontal_flip_mask]

    # --- Vertical surfaces: orient toward road center (XY plane) ---
    # The scanner was on the road, so vertical surfaces should have
    # normals pointing toward the road centerline (scene center in XY)
    vertical_mask = ~is_horizontal & ~invalid_mask
    if np.any(vertical_mask):
        # Direction toward center in XY plane only (ignore Z)
        xy_to_center = -points[vertical_mask, :2]  # [N_vert, 2]

        # Normalize XY direction
        xy_norms = np.linalg.norm(xy_to_center, axis=-1, keepdims=True)
        xy_norms = np.maximum(xy_norms, 1e-8)
        xy_to_center_normalized = xy_to_center / xy_norms

        # Get XY components of normals for vertical surfaces
        normal_xy = normals[vertical_mask, :2]  # [N_vert, 2]

        # Dot product in XY plane
        dot_xy = np.sum(normal_xy * xy_to_center_normalized, axis=-1)

        # Flip where normal points away from center in XY plane
        vertical_flip_mask = dot_xy < 0
        # Get indices in original array
        vertical_indices = np.where(vertical_mask)[0]
        flip_indices = vertical_indices[vertical_flip_mask]
        normals[flip_indices] = -normals[flip_indices]

    # Set invalid normals to [0, 0, 1] (default up direction)
    normals[invalid_mask] = [0.0, 0.0, 1.0]

    # Ensure unit length (should already be, but verify)
    norms = np.linalg.norm(normals, axis=-1, keepdims=True)
    norms = np.maximum(norms, 1e-8)
    normals = normals / norms

    return normals.astype(np.float32)


def load_class_mapping():
    """Load class mapping from CSV and build complete class info."""
    df = pd.read_csv(MAPPING_CSV)

    names = []
    colors = np.zeros((62, 3), dtype=np.uint8)
    parent_ids = np.zeros(62, dtype=np.uint8)
    xxzz_codes = []

    label_to_xxzz = {}
    label_to_name = {}

    for _, row in df.iterrows():
        label = int(row['training_class_code'])
        xx = int(row['XX'])
        zz = int(row['ZZ'])
        xxzz = f"{xx:02d}{zz:02d}"
        name = row['class_name']

        parent_id = xx - 1
        names.append(name)
        xxzz_codes.append(xxzz)
        parent_ids[label] = parent_id
        colors[label] = PARENT_CLASSES[parent_id]["color"]

        label_to_xxzz[label] = xxzz
        label_to_name[label] = name

    return {
        'names': names,
        'colors': colors,
        'parent_ids': parent_ids,
        'xxzz_codes': xxzz_codes,
        'num_classes': 62,
        'num_parent_classes': 12,
        'label_to_xxzz': label_to_xxzz,
        'label_to_name': label_to_name
    }


def natural_sort_key(path):
    """Sort scene files numerically: S1_0001 < S2_0001 < S10_0001"""
    match = re.match(r'S(\d+)_(\d+)', path.stem)
    if match:
        return (int(match.group(1)), int(match.group(2)))
    return (0, 0)


def parse_v1_scene_name(v5_scene_name):
    """Extract V1 scene name from V5 scene name: S1_0001 -> S1"""
    match = re.match(r'(S\d+)_\d+', v5_scene_name)
    if match:
        return match.group(1)
    return None


def load_v5_scene_definitions(bhopal_num):
    """Load V5 scene definitions from JSON files."""
    bhopal_name = f"Bhopal_{bhopal_num}"
    scene_dir = SCENE_V5_DIR / bhopal_name

    if not scene_dir.exists():
        logging.warning(f"V5 scene directory not found: {scene_dir}")
        return []

    scenes = []
    for scene_file in sorted(scene_dir.glob("S*_*.json"), key=natural_sort_key):
        with open(scene_file, 'r') as f:
            scene_data = json.load(f)

        v1_scene = parse_v1_scene_name(scene_data['scene_name'])
        scene_data['v1_scene'] = v1_scene
        scenes.append(scene_data)

    return scenes


def load_v1_scene_instances(bhopal_num, v1_scene_name):
    """Load instance IDs from V1 scene file."""
    bhopal_name = f"Bhopal_{bhopal_num}"
    scene_file = SCENE_V1_DIR / bhopal_name / f"{v1_scene_name}.txt"

    if not scene_file.exists():
        return set()

    instance_ids = set()
    with open(scene_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            instance_ids.add(int(row['Instance_V13']))
    return instance_ids


def load_combined_npz_with_index(bhopal_num):
    """Load combined NPZ file and build instance index.

    Returns data dict with: points, intensity, gps_time, labels, global_ids, rgb
    RGB is loaded from V14.3 NPZ files (extracted from V9 source).

    Uses memory-mapping where possible to reduce peak memory usage.
    """
    npz_path = COMBINED_NPZ_DIR / f"Bhopal_{bhopal_num}.npz"

    if not npz_path.exists():
        logging.warning(f"NPZ file not found: {npz_path}")
        return None, None

    logging.info(f"Loading {npz_path}...")

    # Force garbage collection before loading large file
    gc.collect()

    # Use mmap_mode='r' for memory-mapped reading (reduces peak memory)
    # Note: mmap doesn't work with compressed NPZ, so we load normally but carefully
    npz_data = np.load(npz_path, allow_pickle=True)

    # Load arrays one at a time and force gc between large arrays
    logging.info(f"  Loading points...")
    data = {'points': np.array(npz_data['points'])}
    gc.collect()

    logging.info(f"  Loading intensity...")
    data['intensity'] = np.array(npz_data['intensity'])
    gc.collect()

    logging.info(f"  Loading gps_time...")
    data['gps_time'] = np.array(npz_data['gps_time'])
    gc.collect()

    logging.info(f"  Loading labels...")
    data['labels'] = np.array(npz_data['labels'])
    gc.collect()

    logging.info(f"  Loading global_ids...")
    data['global_ids'] = np.array(npz_data['global_ids'])
    gc.collect()

    # Load RGB from V14.3 NPZ (V9 source)
    logging.info(f"  Loading RGB...")
    if 'rgb' in npz_data:
        data['rgb'] = np.array(npz_data['rgb'])
        rgb_nonzero = np.sum(np.any(data['rgb'] > 0, axis=1))
        rgb_pct = (rgb_nonzero / len(data['rgb'])) * 100 if len(data['rgb']) > 0 else 0
        logging.info(f"  RGB loaded: {rgb_pct:.1f}% coverage ({rgb_nonzero:,} points)")
    else:
        # No RGB in NPZ, fill with zeros
        data['rgb'] = np.zeros((len(data['points']), 3), dtype=np.uint8)
        logging.info(f"  RGB: not available in NPZ, using zeros")
    gc.collect()

    logging.info(f"  Building instance index...")
    instance_global_ids = npz_data['instance_global_ids']
    instance_indices = npz_data['instance_indices']

    instance_index = {}
    for i, global_id in enumerate(instance_global_ids):
        start_idx, end_idx = instance_indices[i]
        instance_index[int(global_id)] = (int(start_idx), int(end_idx))

    logging.info(f"  Loaded {len(data['points']):,} points, {len(instance_index):,} instances")

    # Close NPZ file handle explicitly
    npz_data.close()
    del npz_data
    gc.collect()

    return data, instance_index


def gather_v1_scene_points(instance_ids, data, instance_index):
    """Gather all points for a V1 scene's instances, including RGB."""
    all_points = []
    all_intensity = []
    all_gps_time = []
    all_labels = []
    all_global_ids = []
    all_rgb = []

    for inst_id in instance_ids:
        if inst_id in instance_index:
            start, end = instance_index[inst_id]
            all_points.append(data['points'][start:end].copy())
            all_intensity.append(data['intensity'][start:end].copy())
            all_gps_time.append(data['gps_time'][start:end].copy())
            all_labels.append(data['labels'][start:end].copy())
            all_global_ids.append(data['global_ids'][start:end].copy())
            all_rgb.append(data['rgb'][start:end].copy())

    if not all_points:
        return None

    return {
        'points': np.concatenate(all_points, axis=0),
        'intensity': np.concatenate(all_intensity, axis=0),
        'gps_time': np.concatenate(all_gps_time, axis=0),
        'labels': np.concatenate(all_labels, axis=0),
        'global_ids': np.concatenate(all_global_ids, axis=0),
        'rgb': np.concatenate(all_rgb, axis=0)
    }


def project_points(points, direction, origin):
    """Project points onto road direction axis."""
    dx, dy = direction
    ox, oy = origin[0], origin[1]
    return (points[:, 0] - ox) * dx + (points[:, 1] - oy) * dy


def apply_class_sampling(points, intensity, gps_time, labels, global_ids, rgb,
                         building_classes, building_sample_rate, rng):
    """
    Apply class-specific sampling: keep 100% of non-building, sample building classes.
    Now includes RGB data.
    """
    num_pts = len(labels)

    # Create mask for points to keep
    keep_mask = np.zeros(num_pts, dtype=bool)

    # Keep all non-building points
    non_building_mask = ~np.isin(labels, building_classes)
    keep_mask |= non_building_mask

    # Sample building points
    building_mask = np.isin(labels, building_classes)
    building_indices = np.where(building_mask)[0]

    if len(building_indices) > 0:
        num_keep = int(len(building_indices) * building_sample_rate)
        if num_keep > 0:
            sampled_indices = rng.choice(building_indices, size=num_keep, replace=False)
            keep_mask[sampled_indices] = True

    # Apply mask
    return (
        points[keep_mask],
        intensity[keep_mask],
        gps_time[keep_mask],
        labels[keep_mask],
        global_ids[keep_mask],
        rgb[keep_mask]
    )


def filter_points_by_projection(v1_data, projections, proj_start, proj_end, is_last=False):
    """Filter V1 scene points to V5 scene bounds, including RGB."""
    if is_last:
        mask = (projections >= proj_start) & (projections <= proj_end)
    else:
        mask = (projections >= proj_start) & (projections < proj_end)

    num_pts = np.sum(mask)
    if num_pts == 0:
        return None

    return {
        'points': v1_data['points'][mask],
        'intensity': v1_data['intensity'][mask],
        'gps_time': v1_data['gps_time'][mask],
        'labels': v1_data['labels'][mask],
        'global_ids': v1_data['global_ids'][mask],
        'rgb': v1_data['rgb'][mask]
    }


def estimate_sampled_points(class_counts, building_classes, building_sample_rate):
    """Estimate point count after class sampling."""
    total = 0
    for cls, count in enumerate(class_counts):
        if cls in building_classes:
            total += int(count * building_sample_rate)
        else:
            total += count
    return total


def compute_subdivisions(estimated_points, max_points_threshold, base_segment_m=5.0):
    """
    Compute how many subdivisions are needed to keep under max_points.

    Returns:
        list of (relative_start, relative_end, segment_size_m) for each subdivision
    """
    if estimated_points <= max_points_threshold:
        # No subdivision needed
        return [(0.0, 1.0, base_segment_m)]

    # Calculate number of subdivisions needed
    num_parts = 1
    current_segment = base_segment_m
    pts_per_part = estimated_points

    while pts_per_part > max_points_threshold:
        num_parts *= 2
        current_segment /= 2
        pts_per_part = estimated_points / num_parts

    # Generate subdivision ranges
    subdivisions = []
    for i in range(num_parts):
        rel_start = i / num_parts
        rel_end = (i + 1) / num_parts
        subdivisions.append((rel_start, rel_end, current_segment))

    return subdivisions


def create_scene_v8_hdf5_dataset(
    output_path,
    train_paths=None,
    val_paths=None,
    test_paths=None,
    building_sample_rate=DEFAULT_BUILDING_SAMPLE_RATE,
    min_points_threshold=DEFAULT_MIN_POINTS,
    max_points_threshold=DEFAULT_MAX_POINTS,
    normal_k_neighbors=DEFAULT_NORMAL_K_NEIGHBORS,
    random_seed=42,
    version="9.0"
):
    """
    Create scene-level V9 HDF5 dataset with class-balanced sampling,
    adaptive subdivision, surface normals, and RGB.
    """
    rng = np.random.default_rng(random_seed)

    logging.info("=" * 80)
    logging.info("BHOPAL MLS SCENE V9 DATASET CREATION (With Normals + RGB)")
    logging.info("=" * 80)
    logging.info(f"Output: {output_path}")
    logging.info(f"Building classes: {BUILDING_CLASSES}")
    logging.info(f"Building sample rate: {building_sample_rate * 100:.0f}%")
    logging.info(f"Min points threshold: {min_points_threshold:,}")
    logging.info(f"Max points threshold: {max_points_threshold:,}")
    logging.info(f"Normal K-neighbors: {normal_k_neighbors}")
    logging.info(f"Open3D available: {HAS_OPEN3D}")
    logging.info(f"Train paths: {train_paths}")
    logging.info(f"Val paths: {val_paths}")
    logging.info(f"Test paths: {test_paths}")
    logging.info("=" * 80)

    if not HAS_OPEN3D:
        logging.error("Open3D is required for V8 normal estimation!")
        logging.error("Install with: pip install open3d")
        return

    # Load class mapping
    class_info = load_class_mapping()
    logging.info(f"Loaded {class_info['num_classes']} classes")

    all_paths = set()
    if train_paths:
        all_paths.update(train_paths)
    if val_paths:
        all_paths.update(val_paths)
    if test_paths:
        all_paths.update(test_paths)

    if not all_paths:
        all_paths = set(range(1, 9))

    # =========================================================================
    # PASS 1: Scan scenes and compute subdivisions
    # =========================================================================
    logging.info("\n" + "=" * 80)
    logging.info("PASS 1: Scanning scenes and computing subdivisions...")
    logging.info("=" * 80)

    # Load class distribution for estimation
    class_dist_path = BASE_DIR / "Converted_From_NPZ_V14/H5/scenes/v5_class_distribution.h5"

    scene_plan_list = []  # List of (bhopal_num, scene_def, subdivisions)

    class_counts_all = None
    scene_name_to_idx = {}

    if class_dist_path.exists():
        logging.info(f"Using pre-computed class distribution: {class_dist_path}")
        with h5py.File(class_dist_path, 'r') as dist_h5:
            class_counts_all = dist_h5['class_counts'][:]
            scene_names_all = [dist_h5['scene_names'][i].decode() if isinstance(dist_h5['scene_names'][i], bytes)
                              else dist_h5['scene_names'][i] for i in range(len(dist_h5['scene_names']))]
            scene_name_to_idx = {name: i for i, name in enumerate(scene_names_all)}
    else:
        logging.error(f"Class distribution file not found: {class_dist_path}")
        logging.error("Please run 19d_analyze_scene_class_distribution.py first")
        return

    total_scenes_planned = 0
    subdivision_stats = {5.0: 0, 2.5: 0, 1.25: 0, 0.625: 0}

    for bhopal_num in sorted(all_paths):
        scene_defs = load_v5_scene_definitions(bhopal_num)
        if not scene_defs:
            continue

        for scene_def in scene_defs:
            scene_name = scene_def['scene_name']

            # Estimate points after sampling
            if scene_name in scene_name_to_idx:
                idx = scene_name_to_idx[scene_name]
                est_pts = estimate_sampled_points(
                    class_counts_all[idx], BUILDING_CLASSES, building_sample_rate
                )
            else:
                # Fallback: ~67% of points are building, so with 90% retention:
                # 33% non-building + 67% * 0.9 = 33% + 60.3% = 93.3%
                est_pts = int(scene_def['num_points'] * 0.93)

            # Skip if below minimum
            if est_pts < min_points_threshold:
                continue

            # Compute subdivisions needed
            subdivisions = compute_subdivisions(est_pts, max_points_threshold)

            for rel_start, rel_end, seg_size in subdivisions:
                scene_plan_list.append({
                    'bhopal_num': bhopal_num,
                    'scene_def': scene_def,
                    'rel_start': rel_start,
                    'rel_end': rel_end,
                    'segment_size_m': seg_size,
                    'estimated_pts': est_pts / len(subdivisions),
                    'parent_scene': scene_name,
                    'subdivision_idx': subdivisions.index((rel_start, rel_end, seg_size)),
                    'total_subdivisions': len(subdivisions)
                })
                subdivision_stats[seg_size] = subdivision_stats.get(seg_size, 0) + 1
                total_scenes_planned += 1

        logging.info(f"  Bhopal_{bhopal_num}: planned scenes added")

    num_scenes = total_scenes_planned
    if num_scenes == 0:
        logging.error("No scenes passed the filters!")
        return

    logging.info(f"\nTotal scenes planned: {num_scenes}")
    logging.info(f"Subdivision breakdown:")
    for seg_size in sorted(subdivision_stats.keys(), reverse=True):
        if subdivision_stats[seg_size] > 0:
            logging.info(f"  {seg_size}m segments: {subdivision_stats[seg_size]}")

    # Estimate max points - use max_points_threshold since we cap scenes at this value
    # Add small buffer for safety
    max_points = int(max_points_threshold * 1.05)  # 5% buffer (safety downsample caps at threshold)
    logging.info(f"HDF5 max_points buffer: {max_points:,}")

    # =========================================================================
    # PASS 2: Stream scenes to HDF5
    # =========================================================================
    logging.info("\n" + "=" * 80)
    logging.info("PASS 2: Creating HDF5 with adaptive subdivision and normals...")
    logging.info("=" * 80)

    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or '.', exist_ok=True)

    scene_to_path = []
    total_points_written = 0
    actual_max_points = 0

    with h5py.File(output_path, 'w') as h5f:
        # /scenes/ group
        scenes_group = h5f.create_group('scenes')
        chunk_size = min(max_points, 65536)

        points_ds = scenes_group.create_dataset(
            'points', shape=(num_scenes, max_points, 3), dtype=np.float32,
            compression='gzip', compression_opts=4, chunks=(1, chunk_size, 3)
        )
        points_ds.attrs['description'] = 'XYZ coordinates, scene-centered'

        # V8 NEW: Normals dataset
        normals_ds = scenes_group.create_dataset(
            'normals', shape=(num_scenes, max_points, 3), dtype=np.float32,
            compression='gzip', compression_opts=4, chunks=(1, chunk_size, 3)
        )
        normals_ds.attrs['description'] = f'Surface normals (road-centric: horizontal=Z-up, vertical=toward_road, k={normal_k_neighbors})'

        # V9 NEW: RGB dataset
        rgb_ds = scenes_group.create_dataset(
            'rgb', shape=(num_scenes, max_points, 3), dtype=np.uint8,
            compression='gzip', compression_opts=4, chunks=(1, chunk_size, 3)
        )
        rgb_ds.attrs['description'] = 'RGB values from V9 source files (~35-40% coverage, 0=missing)'

        intensity_ds = scenes_group.create_dataset(
            'intensity', shape=(num_scenes, max_points), dtype=np.float32,
            compression='gzip', compression_opts=4, chunks=(1, chunk_size)
        )
        intensity_ds.attrs['description'] = 'LiDAR return intensity, raw values (0-65535)'

        gps_time_ds = scenes_group.create_dataset(
            'gps_time', shape=(num_scenes, max_points), dtype=np.float32,
            compression='gzip', compression_opts=4, chunks=(1, chunk_size)
        )
        gps_time_ds.attrs['description'] = 'GPS timestamp'

        labels_ds = scenes_group.create_dataset(
            'labels', shape=(num_scenes, max_points), dtype=np.int32,
            compression='gzip', compression_opts=4, chunks=(1, chunk_size)
        )
        labels_ds.attrs['description'] = 'Per-point semantic class (0-61, -1=padding)'

        global_ids_ds = scenes_group.create_dataset(
            'global_ids', shape=(num_scenes, max_points), dtype=np.int32,
            compression='gzip', compression_opts=4, chunks=(1, chunk_size)
        )
        global_ids_ds.attrs['description'] = 'Per-point global instance ID (-1=padding)'

        num_points_ds = scenes_group.create_dataset(
            'num_points', shape=(num_scenes,), dtype=np.uint32
        )

        original_num_points_ds = scenes_group.create_dataset(
            'original_num_points', shape=(num_scenes,), dtype=np.uint32
        )
        original_num_points_ds.attrs['description'] = 'Original point count before sampling'

        # /metadata/ group
        metadata_group = h5f.create_group('metadata')

        dt = h5py.special_dtype(vlen=str)
        scene_names_ds = metadata_group.create_dataset('scene_names', shape=(num_scenes,), dtype=dt)
        parent_scene_ds = metadata_group.create_dataset('parent_scene', shape=(num_scenes,), dtype=dt)
        bhopal_paths_ds = metadata_group.create_dataset('bhopal_paths', shape=(num_scenes,), dtype=np.uint8)
        bounds_ds = metadata_group.create_dataset('bounds', shape=(num_scenes, 6), dtype=np.float64)
        center_ds = metadata_group.create_dataset('center', shape=(num_scenes, 3), dtype=np.float64)
        num_instances_ds = metadata_group.create_dataset('num_instances', shape=(num_scenes,), dtype=np.uint32)
        segment_size_ds = metadata_group.create_dataset('segment_size_m', shape=(num_scenes,), dtype=np.float32)

        # Process ONE Bhopal path at a time
        scene_idx = 0
        data = None
        instance_index = None

        # Group plans by bhopal_num and v1_scene for efficient processing
        plans_by_bhopal = defaultdict(list)
        for plan in scene_plan_list:
            plans_by_bhopal[plan['bhopal_num']].append(plan)

        for bhopal_num in sorted(plans_by_bhopal.keys()):
            logging.info(f"\nProcessing Bhopal_{bhopal_num}...")
            log_memory_usage(f"Before loading Bhopal_{bhopal_num}")

            bhopal_plans = plans_by_bhopal[bhopal_num]
            if not bhopal_plans:
                continue

            try:
                data, instance_index = load_combined_npz_with_index(bhopal_num)
                log_memory_usage(f"After loading Bhopal_{bhopal_num}")
                if data is None:
                    logging.warning(f"  Skipping Bhopal_{bhopal_num} - no data")
                    continue

                # Group plans by V1 scene
                v1_groups = defaultdict(list)
                for plan in bhopal_plans:
                    v1_name = plan['scene_def']['v1_scene']
                    v1_groups[v1_name].append(plan)

                v1_progress = tqdm(sorted(v1_groups.keys()), desc=f"  V1 scenes in Bhopal_{bhopal_num}")

                for v1_name in v1_progress:
                    v1_plans = v1_groups[v1_name]

                    v1_instances = load_v1_scene_instances(bhopal_num, v1_name)
                    if not v1_instances:
                        for plan in v1_plans:
                            # Write empty scene
                            scene_name = plan['scene_def']['scene_name']
                            if plan['total_subdivisions'] > 1:
                                scene_name = f"{scene_name}_{chr(97 + plan['subdivision_idx'])}"
                            num_points_ds[scene_idx] = 0
                            original_num_points_ds[scene_idx] = 0
                            scene_names_ds[scene_idx] = scene_name
                            parent_scene_ds[scene_idx] = plan['parent_scene']
                            bhopal_paths_ds[scene_idx] = bhopal_num
                            segment_size_ds[scene_idx] = plan['segment_size_m']
                            scene_to_path.append(bhopal_num)
                            scene_idx += 1
                        continue

                    v1_data = gather_v1_scene_points(v1_instances, data, instance_index)
                    if v1_data is None:
                        for plan in v1_plans:
                            scene_name = plan['scene_def']['scene_name']
                            if plan['total_subdivisions'] > 1:
                                scene_name = f"{scene_name}_{chr(97 + plan['subdivision_idx'])}"
                            num_points_ds[scene_idx] = 0
                            original_num_points_ds[scene_idx] = 0
                            scene_names_ds[scene_idx] = scene_name
                            parent_scene_ds[scene_idx] = plan['parent_scene']
                            bhopal_paths_ds[scene_idx] = bhopal_num
                            segment_size_ds[scene_idx] = plan['segment_size_m']
                            scene_to_path.append(bhopal_num)
                            scene_idx += 1
                        continue

                    # Get projection info from first plan
                    first_plan = v1_plans[0]
                    direction = first_plan['scene_def']['direction']
                    origin = first_plan['scene_def']['origin']
                    projections = project_points(v1_data['points'], direction, origin)

                    for plan in v1_plans:
                        # Get scene_name early so it's available for error logging
                        scene_def = plan['scene_def']
                        scene_name = scene_def['scene_name']
                        if plan['total_subdivisions'] > 1:
                            scene_name = f"{scene_name}_{chr(97 + plan['subdivision_idx'])}"

                        try:
                            rel_start = plan['rel_start']
                            rel_end = plan['rel_end']

                            # Calculate actual projection bounds for this subdivision
                            full_proj_start = scene_def['proj_start']
                            full_proj_end = scene_def['proj_end']
                            proj_range = full_proj_end - full_proj_start

                            sub_proj_start = full_proj_start + rel_start * proj_range
                            sub_proj_end = full_proj_start + rel_end * proj_range

                            is_last = (rel_end >= 0.999)  # Last subdivision

                            filtered = filter_points_by_projection(
                                v1_data, projections,
                                sub_proj_start,
                                sub_proj_end,
                                is_last=is_last
                            )

                            if filtered is None:
                                num_points_ds[scene_idx] = 0
                                original_num_points_ds[scene_idx] = 0
                                scene_names_ds[scene_idx] = scene_name
                                parent_scene_ds[scene_idx] = plan['parent_scene']
                                bhopal_paths_ds[scene_idx] = bhopal_num
                                segment_size_ds[scene_idx] = plan['segment_size_m']
                                scene_to_path.append(bhopal_num)
                                scene_idx += 1
                                continue

                            original_num_pts = len(filtered['labels'])

                            # Log progress for large scenes
                            if original_num_pts > 500000:
                                logging.debug(f"  Processing large scene {scene_name}: {original_num_pts:,} points")

                            # Apply class-specific sampling (now includes RGB)
                            pts, intensity, gps_t, scene_labels, scene_global_ids, scene_rgb = apply_class_sampling(
                                filtered['points'],
                                filtered['intensity'],
                                filtered['gps_time'],
                                filtered['labels'],
                                filtered['global_ids'],
                                filtered['rgb'],
                                BUILDING_CLASSES,
                                building_sample_rate,
                                rng
                            )

                            num_pts = len(scene_labels)

                            # Safety downsampling: if still over max after class sampling
                            # This handles non-uniform point distribution along road axis
                            if num_pts > max_points_threshold:
                                logging.debug(f"  Safety downsample {scene_name}: {num_pts:,} -> {max_points_threshold:,}")
                                keep_indices = rng.choice(num_pts, size=max_points_threshold, replace=False)
                                keep_indices.sort()  # Maintain spatial order
                                pts = pts[keep_indices]
                                intensity = intensity[keep_indices]
                                gps_t = gps_t[keep_indices]
                                scene_labels = scene_labels[keep_indices]
                                scene_global_ids = scene_global_ids[keep_indices]
                                scene_rgb = scene_rgb[keep_indices]
                                num_pts = max_points_threshold

                            # Skip if below threshold
                            if num_pts < min_points_threshold:
                                num_points_ds[scene_idx] = 0
                                original_num_points_ds[scene_idx] = original_num_pts
                                scene_names_ds[scene_idx] = scene_name
                                parent_scene_ds[scene_idx] = plan['parent_scene']
                                bhopal_paths_ds[scene_idx] = bhopal_num
                                segment_size_ds[scene_idx] = plan['segment_size_m']
                                scene_to_path.append(bhopal_num)
                                scene_idx += 1
                                # Clean up before continue
                                del filtered, pts, intensity, gps_t, scene_labels, scene_global_ids, scene_rgb
                                continue

                            # Calculate bounds and center
                            x_min, y_min, z_min = pts.min(axis=0)
                            x_max, y_max, z_max = pts.max(axis=0)
                            center = pts.mean(axis=0)

                            # Center the points
                            pts_centered = (pts - center).astype(np.float32)

                            # V8 NEW: Compute surface normals on centered points
                            try:
                                scene_normals = compute_normals(pts_centered, k_neighbors=normal_k_neighbors)
                            except Exception as norm_err:
                                logging.warning(f"  Normal estimation failed for {scene_name}: {norm_err}")
                                scene_normals = np.zeros_like(pts_centered, dtype=np.float32)

                            # Count unique instances
                            unique_instances = len(np.unique(scene_global_ids))

                            # Track actual max
                            actual_max_points = max(actual_max_points, num_pts)

                            # Zero-pad and write
                            pts_padded = np.zeros((max_points, 3), dtype=np.float32)
                            normals_padded = np.zeros((max_points, 3), dtype=np.float32)
                            rgb_padded = np.zeros((max_points, 3), dtype=np.uint8)
                            intensity_padded = np.zeros(max_points, dtype=np.float32)
                            gps_time_padded = np.zeros(max_points, dtype=np.float32)
                            labels_padded = np.full(max_points, -1, dtype=np.int32)
                            global_ids_padded = np.full(max_points, -1, dtype=np.int32)

                            pts_padded[:num_pts] = pts_centered
                            normals_padded[:num_pts] = scene_normals
                            rgb_padded[:num_pts] = scene_rgb
                            intensity_padded[:num_pts] = intensity.astype(np.float32)
                            gps_time_padded[:num_pts] = gps_t.astype(np.float32)
                            labels_padded[:num_pts] = scene_labels.astype(np.int32)
                            global_ids_padded[:num_pts] = scene_global_ids.astype(np.int32)

                            # Write to HDF5
                            points_ds[scene_idx] = pts_padded
                            normals_ds[scene_idx] = normals_padded
                            rgb_ds[scene_idx] = rgb_padded
                            intensity_ds[scene_idx] = intensity_padded
                            gps_time_ds[scene_idx] = gps_time_padded
                            labels_ds[scene_idx] = labels_padded
                            global_ids_ds[scene_idx] = global_ids_padded
                            num_points_ds[scene_idx] = num_pts
                            original_num_points_ds[scene_idx] = original_num_pts

                            scene_names_ds[scene_idx] = scene_name
                            parent_scene_ds[scene_idx] = plan['parent_scene']
                            bhopal_paths_ds[scene_idx] = bhopal_num
                            bounds_ds[scene_idx] = [float(x_min), float(y_min), float(z_min),
                                                    float(x_max), float(y_max), float(z_max)]
                            center_ds[scene_idx] = center.astype(np.float64)
                            num_instances_ds[scene_idx] = unique_instances
                            segment_size_ds[scene_idx] = plan['segment_size_m']

                            scene_to_path.append(bhopal_num)
                            total_points_written += num_pts
                            scene_idx += 1

                            # Free memory from padded arrays and filtered data
                            del filtered, pts_padded, normals_padded, rgb_padded, intensity_padded, gps_time_padded
                            del labels_padded, global_ids_padded
                            del pts, intensity, gps_t, scene_labels, scene_global_ids, scene_rgb
                            del pts_centered, scene_normals

                        except Exception as e:
                            logging.error(f"  Error processing scene {scene_name}: {e}")
                            traceback.print_exc()
                            # Write empty scene properly (consistent with other empty cases)
                            try:
                                num_points_ds[scene_idx] = 0
                                original_num_points_ds[scene_idx] = 0
                                scene_names_ds[scene_idx] = scene_name
                                parent_scene_ds[scene_idx] = plan['parent_scene']
                                bhopal_paths_ds[scene_idx] = bhopal_num
                                segment_size_ds[scene_idx] = plan['segment_size_m']
                            except Exception as write_err:
                                logging.error(f"  Failed to write empty scene: {write_err}")
                            scene_to_path.append(bhopal_num)
                            scene_idx += 1
                            # Force garbage collection to clean up partial allocations
                            gc.collect()

                    del v1_data, projections
                    gc.collect()

            except Exception as e:
                logging.error(f"Error processing Bhopal_{bhopal_num}: {e}")
                traceback.print_exc()
                continue

            finally:
                # Aggressively free memory before loading next Bhopal path
                if data is not None:
                    # Delete each array in data dict individually
                    for key in list(data.keys()):
                        del data[key]
                    del data
                    data = None
                if instance_index is not None:
                    instance_index.clear()
                    del instance_index
                    instance_index = None
                # Force garbage collection multiple times to ensure memory is freed
                for _ in range(5):
                    gc.collect()
                time.sleep(2)  # Give OS time to reclaim memory
                log_memory_usage(f"After freeing Bhopal_{bhopal_num}")
                logging.info(f"  Freed memory for Bhopal_{bhopal_num}")

        # /splits/ group
        splits_group = h5f.create_group('splits')

        train_indices = [i for i, p in enumerate(scene_to_path) if train_paths and p in train_paths]
        val_indices = [i for i, p in enumerate(scene_to_path) if val_paths and p in val_paths]
        test_indices = [i for i, p in enumerate(scene_to_path) if test_paths and p in test_paths]

        splits_group.create_dataset('train_indices', data=np.array(train_indices, dtype=np.uint32))
        splits_group.create_dataset('val_indices', data=np.array(val_indices, dtype=np.uint32))
        if test_indices:
            splits_group.create_dataset('test_indices', data=np.array(test_indices, dtype=np.uint32))

        # /class_info/ group
        class_group = h5f.create_group('class_info')

        names_ds = class_group.create_dataset('names', shape=(62,), dtype=dt)
        for i, name in enumerate(class_info['names']):
            names_ds[i] = name

        class_group.create_dataset('colors', data=class_info['colors'])
        class_group.create_dataset('parent_ids', data=class_info['parent_ids'])

        xxzz_ds = class_group.create_dataset('xxzz_codes', shape=(62,), dtype=dt)
        for i, code in enumerate(class_info['xxzz_codes']):
            xxzz_ds[i] = code

        parent_names_ds = class_group.create_dataset('parent_names', shape=(12,), dtype=dt)
        for i in range(12):
            parent_names_ds[i] = PARENT_CLASSES[i]["name"]

        parent_colors = np.array([PARENT_CLASSES[i]["color"] for i in range(12)], dtype=np.uint8)
        class_group.create_dataset('parent_colors', data=parent_colors)

        # Root attributes
        h5f.attrs['dataset_name'] = DATASET_NAME
        h5f.attrs['version'] = version
        h5f.attrs['description'] = DATASET_DESCRIPTION
        h5f.attrs['segmentation_method'] = 'spatial_cutting_v9_with_rgb_and_normals'
        h5f.attrs['num_scenes'] = num_scenes
        h5f.attrs['max_points_per_scene'] = actual_max_points
        h5f.attrs['num_classes'] = 62
        h5f.attrs['num_parent_classes'] = 12
        h5f.attrs['coordinate_system'] = COORDINATE_SYSTEM
        h5f.attrs['creation_date'] = datetime.now().isoformat()
        h5f.attrs['random_seed'] = random_seed
        h5f.attrs['train_size'] = len(train_indices)
        h5f.attrs['val_size'] = len(val_indices)
        h5f.attrs['test_size'] = len(test_indices)
        h5f.attrs['train_paths'] = str(train_paths) if train_paths else ""
        h5f.attrs['val_paths'] = str(val_paths) if val_paths else ""
        h5f.attrs['test_paths'] = str(test_paths) if test_paths else ""
        h5f.attrs['ignore_index'] = -1
        h5f.attrs['total_points'] = int(total_points_written)

        # V7-specific attributes (carried over)
        h5f.attrs['building_classes'] = str(BUILDING_CLASSES)
        h5f.attrs['building_sample_rate'] = building_sample_rate
        h5f.attrs['min_points_threshold'] = min_points_threshold
        h5f.attrs['max_points_threshold'] = max_points_threshold
        h5f.attrs['subdivision_enabled'] = True
        h5f.attrs['base_segment_size_m'] = 5.0

        # V8-specific attributes
        h5f.attrs['has_normals'] = True
        h5f.attrs['normal_k_neighbors'] = normal_k_neighbors
        h5f.attrs['normal_method'] = 'open3d_knn_pca_road_centric'
        h5f.attrs['normal_orientation'] = 'horizontal=Z-up, vertical=toward_road_center'

        # V9-specific attributes (RGB)
        h5f.attrs['has_rgb'] = True
        h5f.attrs['rgb_source'] = 'V9_source_files'
        h5f.attrs['rgb_coverage_note'] = '~35-40% (Bhopal_2/8: 0%), Bhopal_2/8 have 0%, zeros=missing'

        # Store class mapping as JSON attribute
        class_mapping_json = json.dumps({
            'label_to_xxzz': class_info['label_to_xxzz'],
            'label_to_name': class_info['label_to_name']
        })
        h5f.attrs['class_mapping'] = class_mapping_json

    gc.collect()
    logging.info("HDF5 file written and closed")

    # Summary
    logging.info(f"\n{'=' * 80}")
    logging.info("DATASET CREATION COMPLETE")
    logging.info(f"{'=' * 80}")

    file_size_gb = os.path.getsize(output_path) / (1024 ** 3)
    logging.info(f"Output: {output_path}")
    logging.info(f"File size: {file_size_gb:.2f} GB")
    logging.info(f"Total scenes: {num_scenes:,}")
    logging.info(f"Total points: {total_points_written:,}")
    logging.info(f"Actual max points/scene: {actual_max_points:,}")
    logging.info(f"Train: {len(train_indices):,}, Val: {len(val_indices):,}, Test: {len(test_indices):,}")
    logging.info(f"Normal estimation: k={normal_k_neighbors} neighbors")
    logging.info(f"RGB: from V9 source files (~35-40% coverage, 0=missing)")


def print_dataset_info(hdf5_path):
    """Print comprehensive information about HDF5 dataset."""
    with h5py.File(hdf5_path, 'r') as h5f:
        print("\n" + "=" * 80)
        print("BHOPAL MLS SCENE V9 DATASET INFO")
        print("=" * 80)

        print(f"\nName: {h5f.attrs.get('dataset_name', 'N/A')}")
        print(f"Version: {h5f.attrs.get('version', 'N/A')}")
        print(f"Segmentation: {h5f.attrs.get('segmentation_method', 'N/A')}")
        print(f"Creation Date: {h5f.attrs.get('creation_date', 'N/A')}")

        print(f"\nV7 Sampling & Subdivision Config:")
        print(f"  Building classes: {h5f.attrs.get('building_classes', 'N/A')}")
        print(f"  Building sample rate: {h5f.attrs.get('building_sample_rate', 'N/A')}")
        print(f"  Min points threshold: {h5f.attrs.get('min_points_threshold', 'N/A')}")
        print(f"  Max points threshold: {h5f.attrs.get('max_points_threshold', 'N/A')}")
        print(f"  Subdivision enabled: {h5f.attrs.get('subdivision_enabled', 'N/A')}")
        print(f"  Base segment size: {h5f.attrs.get('base_segment_size_m', 'N/A')}m")

        print(f"\nV8 Normal Estimation Config:")
        print(f"  Has normals: {h5f.attrs.get('has_normals', False)}")
        print(f"  Normal K-neighbors: {h5f.attrs.get('normal_k_neighbors', 'N/A')}")
        print(f"  Normal method: {h5f.attrs.get('normal_method', 'N/A')}")
        print(f"  Normal orientation: {h5f.attrs.get('normal_orientation', 'N/A')}")

        print(f"\nV9 RGB Config:")
        print(f"  Has RGB: {h5f.attrs.get('has_rgb', False)}")
        print(f"  RGB source: {h5f.attrs.get('rgb_source', 'N/A')}")
        print(f"  RGB coverage: {h5f.attrs.get('rgb_coverage_note', 'N/A')}")

        print(f"\nScenes: {h5f.attrs.get('num_scenes', 'N/A'):,}")
        print(f"Max Points/Scene: {h5f.attrs.get('max_points_per_scene', 'N/A'):,}")
        print(f"Total Points: {h5f.attrs.get('total_points', 'N/A'):,}")
        print(f"Classes: {h5f.attrs.get('num_classes', 'N/A')} fine / "
              f"{h5f.attrs.get('num_parent_classes', 'N/A')} parent")

        print(f"\nSplits:")
        print(f"  Train: {h5f.attrs.get('train_size', 0):,} scenes (paths: {h5f.attrs.get('train_paths', '')})")
        print(f"  Val: {h5f.attrs.get('val_size', 0):,} scenes (paths: {h5f.attrs.get('val_paths', '')})")
        print(f"  Test: {h5f.attrs.get('test_size', 0):,} scenes (paths: {h5f.attrs.get('test_paths', '')})")

        # Check for normals dataset
        if 'scenes/normals' in h5f:
            print(f"\nNormals Dataset: PRESENT")
            print(f"  Shape: {h5f['scenes/normals'].shape}")
            print(f"  Dtype: {h5f['scenes/normals'].dtype}")
        else:
            print(f"\nNormals Dataset: NOT PRESENT")

        # Check for RGB dataset
        if 'scenes/rgb' in h5f:
            print(f"\nRGB Dataset: PRESENT")
            print(f"  Shape: {h5f['scenes/rgb'].shape}")
            print(f"  Dtype: {h5f['scenes/rgb'].dtype}")
        else:
            print(f"\nRGB Dataset: NOT PRESENT")

        if 'scenes/num_points' in h5f:
            num_points = h5f['scenes/num_points'][:]
            valid_mask = num_points > 0
            valid_points = num_points[valid_mask]
            print(f"\nPoint Statistics per Scene (after sampling):")
            print(f"  Valid scenes: {np.sum(valid_mask):,}")
            print(f"  Min: {np.min(valid_points):,}")
            print(f"  Max: {np.max(valid_points):,}")
            print(f"  Mean: {np.mean(valid_points):,.0f}")
            print(f"  Median: {np.median(valid_points):,.0f}")
            print(f"  Total: {np.sum(valid_points):,}")

        if 'metadata/segment_size_m' in h5f and 'scenes/num_points' in h5f:
            segment_sizes = h5f['metadata/segment_size_m'][:]
            num_points = h5f['scenes/num_points'][:]
            valid_mask = num_points > 0
            valid_sizes = segment_sizes[valid_mask]
            print(f"\nSegment Size Distribution:")
            for size in sorted(np.unique(valid_sizes), reverse=True):
                count = np.sum(valid_sizes == size)
                print(f"  {size:.2f}m: {count:,} scenes")

        if 'scenes/original_num_points' in h5f:
            orig_num_points = h5f['scenes/original_num_points'][:]
            valid_orig = orig_num_points[orig_num_points > 0]
            print(f"\nOriginal Point Statistics (before sampling):")
            print(f"  Min: {np.min(valid_orig):,}")
            print(f"  Max: {np.max(valid_orig):,}")
            print(f"  Mean: {np.mean(valid_orig):,.0f}")
            print(f"  Total: {np.sum(valid_orig):,}")

        print("\n" + "=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description='Create Bhopal MLS Scene V9 HDF5 dataset with normals and RGB',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Create dataset with default settings (90% building retention, 3K min, 1M max, k=30)
  python3 19b_create_scene_V8_dataset.py --output ../bhopal_scene_v9.h5 \\
      --train-paths 1 2 3 4 --val-paths 5 6 --test-paths 7 8

  # Custom thresholds (0.85 = keep 85% of building points)
  python3 19b_create_scene_V8_dataset.py --output ../bhopal_scene_v9.h5 \\
      --building-sample-rate 0.85 --min-points 5000 --max-points 800000 \\
      --train-paths 1 2 3 4 --val-paths 5 6 --test-paths 7 8

  # Different normal k-neighbors
  python3 19b_create_scene_V8_dataset.py --output ../bhopal_scene_v9.h5 \\
      --normal-k 50 --train-paths 1 2 3 4 --val-paths 5 6 --test-paths 7 8

  # Show info
  python3 19b_create_scene_V8_dataset.py --info bhopal_scene_v9.h5
        """
    )

    parser.add_argument('--output', type=str, default='bhopal_scene_v9.h5',
                       help='Output HDF5 file path')
    parser.add_argument('--building-sample-rate', type=float, default=DEFAULT_BUILDING_SAMPLE_RATE,
                       help=f'Sampling rate for building classes (default: {DEFAULT_BUILDING_SAMPLE_RATE})')
    parser.add_argument('--min-points', type=int, default=DEFAULT_MIN_POINTS,
                       help=f'Minimum points threshold (default: {DEFAULT_MIN_POINTS})')
    parser.add_argument('--max-points', type=int, default=DEFAULT_MAX_POINTS,
                       help=f'Maximum points threshold - subdivide if exceeded (default: {DEFAULT_MAX_POINTS})')
    parser.add_argument('--normal-k', type=int, default=DEFAULT_NORMAL_K_NEIGHBORS,
                       help=f'K neighbors for normal estimation (default: {DEFAULT_NORMAL_K_NEIGHBORS})')
    parser.add_argument('--train-paths', type=int, nargs='+', default=None,
                       help='Bhopal paths for training (e.g., 1 2 3 4)')
    parser.add_argument('--val-paths', type=int, nargs='+', default=None,
                       help='Bhopal paths for validation (e.g., 5 6)')
    parser.add_argument('--test-paths', type=int, nargs='+', default=None,
                       help='Bhopal paths for testing (e.g., 7 8)')
    parser.add_argument('--random-seed', type=int, default=42,
                       help='Random seed (default: 42)')
    parser.add_argument('--version', type=str, default='9.0',
                       help='Dataset version (default: 9.0)')
    parser.add_argument('--info', type=str, default=None,
                       help='Show info about existing HDF5 dataset')

    args = parser.parse_args()

    if args.info:
        if not os.path.exists(args.info):
            logging.error(f"File not found: {args.info}")
            return 1
        print_dataset_info(args.info)
        return 0

    if args.train_paths is None or args.val_paths is None:
        logging.error("Both --train-paths and --val-paths must be specified")
        return 1

    create_scene_v8_hdf5_dataset(
        output_path=args.output,
        train_paths=args.train_paths,
        val_paths=args.val_paths,
        test_paths=args.test_paths,
        building_sample_rate=args.building_sample_rate,
        min_points_threshold=args.min_points,
        max_points_threshold=args.max_points,
        normal_k_neighbors=args.normal_k,
        random_seed=args.random_seed,
        version=args.version
    )

    print_dataset_info(args.output)
    return 0


if __name__ == '__main__':
    sys.exit(main())
