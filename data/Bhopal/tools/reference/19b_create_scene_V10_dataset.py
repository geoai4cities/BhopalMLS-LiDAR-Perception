#!/usr/bin/env python3
"""
Bhopal MLS Scene-Level V10 Dataset Creator - With Moving Classes
=================================================================

Creates SCENE-LEVEL samples with CLASS-SPECIFIC SAMPLING, ADAPTIVE SUBDIVISION,
SURFACE NORMAL ESTIMATION, RGB, and MOVING CLASS support.

V10 Strategy (builds on V9):
- All V9 features: Building sampling, adaptive subdivision, normals, RGB
- NEW: Moving classes included from Unlabelled NPZ (original codes 3-8)
- NEW: 68 classes (62 training + 6 moving), 12 parent categories
- NEW: Moving classes merged into existing parents (NOT a separate parent):
    - Moving Vehicle (62), Mobike Rider Moving (67) → parent 7 "Vehicle"
    - Moving Humans (63), Moving Female (64), Moving Male (65), Moving Animal (66) → parent 10 "Human and Animal"
- NEW: Spatial assignment of moving points to scenes (not in V1 scene files)
- NEW: Label remapping: original code 3→62, 4→63, 5→64, 6→65, 7→66, 8→67
- Global IDs from Unlabelled NPZ preserved (NXXZZYYYY format, XX=00)

Moving Classes (merged into existing parents):
- 62: Moving Vehicle       → parent 7 "Vehicle"         (original code 3)
- 63: Moving Humans        → parent 10 "Human and Animal" (original code 4)
- 64: Moving Female        → parent 10 "Human and Animal" (original code 5)
- 65: Moving Male          → parent 10 "Human and Animal" (original code 6)
- 66: Moving Animal        → parent 10 "Human and Animal" (original code 7)
- 67: Mobike Rider Moving  → parent 7 "Vehicle"         (original code 8)

Excluded from original XX=0:
- 0: Unlabeled         (original code 0)
- 1: Noise Points      (original code 1)
- 2: High Noise Points (original code 2)

Spatial Assignment of Moving Points:
- Moving class instances are NOT in V1 scene definition files
- Instead, we use XY bounding box of each V1 scene's training points
- Moving points within the XY bounds (+ padding) are assigned to that scene
- Then standard V5 projection filtering applies along road axis

Building Classes (keep 90%, drop 10%):
- 8: Building
- 9: Residential
- 10: Commercial
- 11: Industrial
- 12: Special

Input:
- Scene definitions: Bhopal_scene_division_v5/Bhopal_N/S*_*.json
- V1 scene definitions: Bhopal_scene_division_v1/Bhopal_N/S*.txt
- Training data: Combined_NPZ_Files_V14_Training_And_Unlabeled/Bhopal_N.npz
- Moving data: Combined_NPZ_Files_V14_Training_And_Unlabeled/Bhopal_N_Unlabelled.npz

Output Structure:
    bhopal_scene_v10.h5
    ├── scenes/
    │   ├── points              [N, max_pts, 3]    float32 (scene-centered)
    │   ├── rgb                 [N, max_pts, 3]    uint8 (RGB from V9, ~35-40% coverage)
    │   ├── normals             [N, max_pts, 3]    float32 (unit normals, road-centric)
    │   ├── intensity           [N, max_pts]       float32 (raw 0-65535)
    │   ├── gps_time            [N, max_pts]       float32
    │   ├── labels              [N, max_pts]       int32 (0-67, -1=padding)
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
    │   ├── names               [68]               str
    │   ├── colors              [68, 3]            uint8
    │   ├── parent_ids          [68]               uint8
    │   ├── xxzz_codes          [68]               str
    │   ├── parent_names        [12]               str
    │   └── parent_colors       [12, 3]            uint8
    └── Attributes:
        ├── dataset_name, version, description, segmentation_method
        ├── num_scenes, max_points_per_scene, total_points
        ├── num_classes (68), num_parent_classes (12)
        ├── coordinate_system, creation_date, random_seed
        ├── train_size, val_size, test_size
        ├── train_paths, val_paths, test_paths
        ├── ignore_index (-1)
        ├── building_classes, building_sample_rate (0.90)
        ├── min_points_threshold (3000), max_points_threshold (1000000)
        ├── subdivision_enabled (True), base_segment_size_m (5.0)
        ├── has_normals (True), normal_k_neighbors (30)
        ├── has_rgb (True), rgb_source (V9), rgb_coverage (~35-40%)
        ├── has_moving_classes (True), moving_classes (62-67)
        ├── moving_spatial_padding_m (5.0)
        └── class_mapping (JSON)

Usage:
    python3 19b_create_scene_V10_dataset.py --output ../bhopal_scene_v10.h5 \
        --train-paths 1 2 3 4 --val-paths 5 6 --test-paths 7 8

    # Custom thresholds
    python3 19b_create_scene_V10_dataset.py --output ../bhopal_scene_v10.h5 \
        --building-sample-rate 0.85 --min-points 3000 --max-points 1000000 \
        --train-paths 1 2 3 4 --val-paths 5 6 --test-paths 7 8

    # Show info
    python3 19b_create_scene_V10_dataset.py --info bhopal_scene_v10.h5

Date: 2026-04-09
Version: 10.0
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
MAPPING_CSV = BASE_DIR / "class_mapping" / "Bhopal_class_mapping_V13_TRAINING_v10h5.csv"

# V10 Sampling Configuration
BUILDING_CLASSES = [8, 9, 10, 11, 12]  # Building, Residential, Commercial, Industrial, Special
DEFAULT_BUILDING_SAMPLE_RATE = 0.90   # Keep 90% of building classes (drop 10%)
DEFAULT_MIN_POINTS = 3000             # Exclude scenes with fewer points after sampling
DEFAULT_MAX_POINTS = 1_000_000        # Subdivide scenes exceeding this threshold

# V10 Normal Estimation Configuration
DEFAULT_NORMAL_K_NEIGHBORS = 30       # KNN neighbors for normal estimation (Concerto default)

# V10 Moving Classes Configuration
# Original label codes in Unlabelled NPZ → new training class codes
MOVING_LABEL_REMAP = {
    3: 62,   # Moving Vehicle
    4: 63,   # Moving Humans
    5: 64,   # Moving Female
    6: 65,   # Moving Male
    7: 66,   # Moving Animal
    8: 67,   # Mobike Rider Moving
}
MOVING_ORIGINAL_CODES = set(MOVING_LABEL_REMAP.keys())  # {3, 4, 5, 6, 7, 8}
EXCLUDED_UNLABELLED_CODES = {0, 1, 2}  # Unlabeled, Noise Points, High Noise Points
DEFAULT_MOVING_SPATIAL_PADDING = 5.0  # meters padding for XY bounding box match

# Moving class → parent mapping (merged into existing parents)
# 62 Moving Vehicle, 67 Mobike Rider Moving → parent 7 (Vehicle)
# 63 Moving Humans, 64 Moving Female, 65 Moving Male, 66 Moving Animal → parent 10 (Human and Animal)
MOVING_PARENT_MAP = {
    62: 7,   # Moving Vehicle       → Vehicle
    63: 10,  # Moving Humans        → Human and Animal
    64: 10,  # Moving Female        → Human and Animal
    65: 10,  # Moving Male          → Human and Animal
    66: 10,  # Moving Animal        → Human and Animal
    67: 7,   # Mobike Rider Moving  → Vehicle
}

# Number of classes
NUM_CLASSES = 68
NUM_PARENT_CLASSES = 12

# Dataset metadata
DATASET_NAME = "Bhopal MLS Scene V10"
DATASET_VERSION = "10.0"
DATASET_DESCRIPTION = (
    "Mobile Laser Scanning (MLS) dataset for urban semantic segmentation. "
    "Collected in Bhopal, India. Contains 68 fine-grained classes organized "
    "into 12 parent categories. Scene-level segments based on TRUE SPATIAL "
    "CUTTING along road axis (5m base, adaptive subdivision for large scenes). "
    "Building classes (8,9,10,11,12) sampled at 90% (drop 10%). Scenes <3K points excluded. "
    "All scenes kept under 1M points via recursive subdivision. "
    "Surface normals computed using Open3D KNN-based PCA (k=30) with "
    "road-centric orientation (horizontal=Z-up, vertical=toward road center). "
    "RGB values extracted from V9 source files (~35-40% coverage, Bhopal_2/8 have 0%). "
    "V10 NEW: Moving classes (62-67) from Unlabelled NPZ, spatially assigned to scenes. "
    "Moving classes merged into existing parents: 62,67 → Vehicle (parent 7); "
    "63,64,65,66 → Human and Animal (parent 10). 12 parent categories (same as V9). "
    "10-channel input for Concerto pre-training: XYZ (3) + RGB (3) + intensity (1) + normal (3)."
)
COORDINATE_SYSTEM = "UTM Zone 43N (EPSG:32643)"

# 12 Parent class definitions with colors (same as V9)
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

    Orientation Strategy:
    1. Horizontal surfaces (|nz| > 0.5): Z-up (normals point upward)
    2. Vertical surfaces (|nz| <= 0.5): toward road center in XY plane

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

    # Road-centric hybrid orientation
    is_horizontal = np.abs(normals[:, 2]) > 0.5

    # Horizontal surfaces: Z-up (flip to point upward)
    horizontal_flip_mask = is_horizontal & (normals[:, 2] < 0)
    normals[horizontal_flip_mask] = -normals[horizontal_flip_mask]

    # Vertical surfaces: orient toward road center (XY plane)
    vertical_mask = ~is_horizontal & ~invalid_mask
    if np.any(vertical_mask):
        xy_to_center = -points[vertical_mask, :2]
        xy_norms = np.linalg.norm(xy_to_center, axis=-1, keepdims=True)
        xy_norms = np.maximum(xy_norms, 1e-8)
        xy_to_center_normalized = xy_to_center / xy_norms

        normal_xy = normals[vertical_mask, :2]
        dot_xy = np.sum(normal_xy * xy_to_center_normalized, axis=-1)

        vertical_flip_mask = dot_xy < 0
        vertical_indices = np.where(vertical_mask)[0]
        flip_indices = vertical_indices[vertical_flip_mask]
        normals[flip_indices] = -normals[flip_indices]

    # Set invalid normals to [0, 0, 1] (default up direction)
    normals[invalid_mask] = [0.0, 0.0, 1.0]

    # Ensure unit length
    norms = np.linalg.norm(normals, axis=-1, keepdims=True)
    norms = np.maximum(norms, 1e-8)
    normals = normals / norms

    return normals.astype(np.float32)


def load_class_mapping():
    """Load class mapping from V10 CSV and build complete class info."""
    df = pd.read_csv(MAPPING_CSV)

    names = []
    colors = np.zeros((NUM_CLASSES, 3), dtype=np.uint8)
    parent_ids = np.zeros(NUM_CLASSES, dtype=np.uint8)
    xxzz_codes = []

    label_to_xxzz = {}
    label_to_name = {}

    for _, row in df.iterrows():
        label = int(row['training_class_code'])
        xx = int(row['XX'])
        zz = int(row['ZZ'])
        xxzz = f"{xx:02d}{zz:02d}"
        name = row['class_name']

        # Moving classes use MOVING_PARENT_MAP instead of xx-1
        if label in MOVING_PARENT_MAP:
            parent_id = MOVING_PARENT_MAP[label]
        else:
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
        'num_classes': NUM_CLASSES,
        'num_parent_classes': NUM_PARENT_CLASSES,
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
    """Load combined training NPZ file and build instance index.

    Returns data dict with: points, intensity, gps_time, labels, global_ids, rgb
    """
    npz_path = COMBINED_NPZ_DIR / f"Bhopal_{bhopal_num}.npz"

    if not npz_path.exists():
        logging.warning(f"NPZ file not found: {npz_path}")
        return None, None

    logging.info(f"Loading training NPZ: {npz_path}...")
    gc.collect()

    npz_data = np.load(npz_path, allow_pickle=True)

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

    logging.info(f"  Loading RGB...")
    if 'rgb' in npz_data:
        data['rgb'] = np.array(npz_data['rgb'])
        rgb_nonzero = np.sum(np.any(data['rgb'] > 0, axis=1))
        rgb_pct = (rgb_nonzero / len(data['rgb'])) * 100 if len(data['rgb']) > 0 else 0
        logging.info(f"  RGB loaded: {rgb_pct:.1f}% coverage ({rgb_nonzero:,} points)")
    else:
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

    logging.info(f"  Loaded {len(data['points']):,} training points, {len(instance_index):,} instances")

    npz_data.close()
    del npz_data
    gc.collect()

    return data, instance_index


def load_moving_points(bhopal_num):
    """Load moving class points from Unlabelled NPZ file.

    Filters to moving classes only (original codes 3-8), excludes 0,1,2.
    Remaps labels: 3→62, 4→63, 5→64, 6→65, 7→66, 8→67.
    Preserves original global_ids (NXXZZYYYY format with XX=00).

    Returns:
        moving_data: dict with points, intensity, gps_time, labels (remapped),
                     global_ids (original), rgb arrays. Or None if no data.
    """
    npz_path = COMBINED_NPZ_DIR / f"Bhopal_{bhopal_num}_Unlabelled.npz"

    if not npz_path.exists():
        logging.info(f"  No Unlabelled NPZ for Bhopal_{bhopal_num}: {npz_path}")
        return None

    logging.info(f"Loading moving classes from: {npz_path}...")
    gc.collect()

    npz_data = np.load(npz_path, allow_pickle=True)

    # Load labels first to filter
    labels_raw = np.array(npz_data['labels'])

    # Create mask for moving classes only (original codes 3-8)
    moving_mask = np.isin(labels_raw, list(MOVING_ORIGINAL_CODES))
    num_moving = np.sum(moving_mask)

    if num_moving == 0:
        logging.info(f"  No moving class points found in Bhopal_{bhopal_num}")
        npz_data.close()
        del npz_data, labels_raw
        gc.collect()
        return None

    # Load and filter arrays
    logging.info(f"  Filtering {num_moving:,} moving points from {len(labels_raw):,} total...")

    moving_data = {}
    moving_data['points'] = np.array(npz_data['points'])[moving_mask]
    gc.collect()

    moving_data['intensity'] = np.array(npz_data['intensity'])[moving_mask]
    gc.collect()

    moving_data['gps_time'] = np.array(npz_data['gps_time'])[moving_mask]
    gc.collect()

    # Remap labels: original code → training class code
    remapped_labels = labels_raw[moving_mask].copy()
    for orig_code, new_code in MOVING_LABEL_REMAP.items():
        remapped_labels[remapped_labels == orig_code] = new_code
    moving_data['labels'] = remapped_labels.astype(np.uint16)
    del remapped_labels
    gc.collect()

    # Preserve original global_ids (NXXZZYYYY format with XX=00)
    moving_data['global_ids'] = np.array(npz_data['global_ids'])[moving_mask]
    gc.collect()

    # Load RGB
    if 'rgb' in npz_data:
        moving_data['rgb'] = np.array(npz_data['rgb'])[moving_mask]
    else:
        moving_data['rgb'] = np.zeros((num_moving, 3), dtype=np.uint8)
    gc.collect()

    # Log per-class counts
    unique_labels, counts = np.unique(moving_data['labels'], return_counts=True)
    for lbl, cnt in zip(unique_labels, counts):
        logging.info(f"    Class {lbl}: {cnt:,} points")

    logging.info(f"  Loaded {num_moving:,} moving points for Bhopal_{bhopal_num}")

    npz_data.close()
    del npz_data, labels_raw, moving_mask
    gc.collect()

    return moving_data


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


def gather_moving_points_for_v1_scene(v1_data, moving_data, moving_used_mask, padding_m=5.0):
    """
    Spatially filter moving points that fall within a V1 scene's XY extent.

    Uses the XY bounding box of the V1 scene's training points with padding
    to find moving points that belong to this scene. Each moving point is
    assigned to at most one V1 scene (first match wins) to avoid duplication.

    Args:
        v1_data: dict with training points for this V1 scene
        moving_data: dict with ALL moving points for this Bhopal path
        moving_used_mask: boolean array tracking which moving points are already assigned
        padding_m: XY padding in meters around the training data bounds

    Returns:
        dict with filtered moving points, or None if no matches
    """
    if moving_data is None or len(moving_data['points']) == 0:
        return None

    # Compute XY bounding box of training data
    train_xy = v1_data['points'][:, :2]
    xy_min = train_xy.min(axis=0) - padding_m
    xy_max = train_xy.max(axis=0) + padding_m

    # Filter moving points within XY bounds AND not already assigned
    moving_xy = moving_data['points'][:, :2]
    xy_mask = (
        ~moving_used_mask &
        (moving_xy[:, 0] >= xy_min[0]) & (moving_xy[:, 0] <= xy_max[0]) &
        (moving_xy[:, 1] >= xy_min[1]) & (moving_xy[:, 1] <= xy_max[1])
    )

    num_matched = np.sum(xy_mask)
    if num_matched == 0:
        return None

    # Mark these points as used so they won't be assigned to another V1 scene
    moving_used_mask[xy_mask] = True

    return {
        'points': moving_data['points'][xy_mask],
        'intensity': moving_data['intensity'][xy_mask],
        'gps_time': moving_data['gps_time'][xy_mask],
        'labels': moving_data['labels'][xy_mask],
        'global_ids': moving_data['global_ids'][xy_mask],
        'rgb': moving_data['rgb'][xy_mask]
    }


def merge_training_and_moving(v1_data, moving_filtered):
    """
    Merge training V1 scene data with spatially-filtered moving points.

    Args:
        v1_data: dict with training points for this V1 scene
        moving_filtered: dict with filtered moving points, or None

    Returns:
        merged dict with all data fields concatenated
    """
    if moving_filtered is None or len(moving_filtered['points']) == 0:
        return v1_data

    merged = {}
    for key in ['points', 'intensity', 'gps_time', 'labels', 'global_ids', 'rgb']:
        merged[key] = np.concatenate([v1_data[key], moving_filtered[key]], axis=0)

    return merged


def project_points(points, direction, origin):
    """Project points onto road direction axis."""
    dx, dy = direction
    ox, oy = origin[0], origin[1]
    return (points[:, 0] - ox) * dx + (points[:, 1] - oy) * dy


def apply_class_sampling(points, intensity, gps_time, labels, global_ids, rgb,
                         building_classes, building_sample_rate, rng):
    """
    Apply class-specific sampling: keep 100% of non-building, sample building classes.
    Moving classes (62-67) are kept at 100%.
    """
    num_pts = len(labels)

    # Create mask for points to keep
    keep_mask = np.zeros(num_pts, dtype=bool)

    # Keep all non-building points (including moving classes)
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
        return [(0.0, 1.0, base_segment_m)]

    num_parts = 1
    current_segment = base_segment_m
    pts_per_part = estimated_points

    while pts_per_part > max_points_threshold:
        num_parts *= 2
        current_segment /= 2
        pts_per_part = estimated_points / num_parts

    subdivisions = []
    for i in range(num_parts):
        rel_start = i / num_parts
        rel_end = (i + 1) / num_parts
        subdivisions.append((rel_start, rel_end, current_segment))

    return subdivisions


def write_empty_scene(scene_idx, scene_name, parent_scene, bhopal_num, segment_size_m,
                      num_points_ds, original_num_points_ds, scene_names_ds,
                      parent_scene_ds, bhopal_paths_ds, segment_size_ds,
                      original_num_pts=0):
    """Write an empty scene entry to HDF5 datasets."""
    num_points_ds[scene_idx] = 0
    original_num_points_ds[scene_idx] = original_num_pts
    scene_names_ds[scene_idx] = scene_name
    parent_scene_ds[scene_idx] = parent_scene
    bhopal_paths_ds[scene_idx] = bhopal_num
    segment_size_ds[scene_idx] = segment_size_m


def create_scene_v10_hdf5_dataset(
    output_path,
    train_paths=None,
    val_paths=None,
    test_paths=None,
    building_sample_rate=DEFAULT_BUILDING_SAMPLE_RATE,
    min_points_threshold=DEFAULT_MIN_POINTS,
    max_points_threshold=DEFAULT_MAX_POINTS,
    normal_k_neighbors=DEFAULT_NORMAL_K_NEIGHBORS,
    moving_spatial_padding=DEFAULT_MOVING_SPATIAL_PADDING,
    random_seed=42,
    version="10.0"
):
    """
    Create scene-level V10 HDF5 dataset with class-balanced sampling,
    adaptive subdivision, surface normals, RGB, and moving classes.
    """
    rng = np.random.default_rng(random_seed)

    logging.info("=" * 80)
    logging.info("BHOPAL MLS SCENE V10 DATASET CREATION (With Moving Classes)")
    logging.info("=" * 80)
    logging.info(f"Output: {output_path}")
    logging.info(f"Building classes: {BUILDING_CLASSES}")
    logging.info(f"Building sample rate: {building_sample_rate * 100:.0f}%")
    logging.info(f"Min points threshold: {min_points_threshold:,}")
    logging.info(f"Max points threshold: {max_points_threshold:,}")
    logging.info(f"Normal K-neighbors: {normal_k_neighbors}")
    logging.info(f"Moving classes: {list(MOVING_LABEL_REMAP.values())} (from original {list(MOVING_LABEL_REMAP.keys())})")
    logging.info(f"Moving spatial padding: {moving_spatial_padding}m")
    logging.info(f"Open3D available: {HAS_OPEN3D}")
    logging.info(f"Train paths: {train_paths}")
    logging.info(f"Val paths: {val_paths}")
    logging.info(f"Test paths: {test_paths}")
    logging.info("=" * 80)

    if not HAS_OPEN3D:
        logging.error("Open3D is required for normal estimation!")
        logging.error("Install with: pip install open3d")
        return

    # Load class mapping
    class_info = load_class_mapping()
    logging.info(f"Loaded {class_info['num_classes']} classes, {class_info['num_parent_classes']} parent categories")

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

    scene_plan_list = []

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

            # Estimate points after sampling (conservative: moving points add ~2-5%)
            if scene_name in scene_name_to_idx:
                idx = scene_name_to_idx[scene_name]
                est_pts = estimate_sampled_points(
                    class_counts_all[idx], BUILDING_CLASSES, building_sample_rate
                )
                # Add ~5% for moving classes (conservative estimate)
                est_pts = int(est_pts * 1.05)
            else:
                est_pts = int(scene_def['num_points'] * 0.98)

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

    # Estimate max points
    max_points = int(max_points_threshold * 1.05)  # 5% buffer
    logging.info(f"HDF5 max_points buffer: {max_points:,}")

    # =========================================================================
    # PASS 2: Stream scenes to HDF5
    # =========================================================================
    logging.info("\n" + "=" * 80)
    logging.info("PASS 2: Creating HDF5 with moving classes, normals, and RGB...")
    logging.info("=" * 80)

    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or '.', exist_ok=True)

    scene_to_path = []
    total_points_written = 0
    total_moving_points_written = 0
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

        normals_ds = scenes_group.create_dataset(
            'normals', shape=(num_scenes, max_points, 3), dtype=np.float32,
            compression='gzip', compression_opts=4, chunks=(1, chunk_size, 3)
        )
        normals_ds.attrs['description'] = f'Surface normals (road-centric: horizontal=Z-up, vertical=toward_road, k={normal_k_neighbors})'

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
        labels_ds.attrs['description'] = 'Per-point semantic class (0-67, -1=padding)'

        global_ids_ds = scenes_group.create_dataset(
            'global_ids', shape=(num_scenes, max_points), dtype=np.int32,
            compression='gzip', compression_opts=4, chunks=(1, chunk_size)
        )
        global_ids_ds.attrs['description'] = 'Per-point global instance ID (-1=padding, NXXZZYYYY format)'

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
        moving_data = None

        # Group plans by bhopal_num
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
                # Load training data
                data, instance_index = load_combined_npz_with_index(bhopal_num)
                log_memory_usage(f"After loading training Bhopal_{bhopal_num}")
                if data is None:
                    logging.warning(f"  Skipping Bhopal_{bhopal_num} - no training data")
                    continue

                # Load moving class data
                moving_data = load_moving_points(bhopal_num)
                log_memory_usage(f"After loading moving Bhopal_{bhopal_num}")

                if moving_data is not None:
                    logging.info(f"  Moving points available: {len(moving_data['points']):,}")
                    # Track which moving points have been assigned to a V1 scene
                    moving_used_mask = np.zeros(len(moving_data['points']), dtype=bool)
                else:
                    logging.info(f"  No moving points for Bhopal_{bhopal_num}")
                    moving_used_mask = None

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
                            scene_name = plan['scene_def']['scene_name']
                            if plan['total_subdivisions'] > 1:
                                scene_name = f"{scene_name}_{chr(97 + plan['subdivision_idx'])}"
                            write_empty_scene(
                                scene_idx, scene_name, plan['parent_scene'], bhopal_num,
                                plan['segment_size_m'], num_points_ds, original_num_points_ds,
                                scene_names_ds, parent_scene_ds, bhopal_paths_ds, segment_size_ds
                            )
                            scene_to_path.append(bhopal_num)
                            scene_idx += 1
                        continue

                    v1_data = gather_v1_scene_points(v1_instances, data, instance_index)
                    if v1_data is None:
                        for plan in v1_plans:
                            scene_name = plan['scene_def']['scene_name']
                            if plan['total_subdivisions'] > 1:
                                scene_name = f"{scene_name}_{chr(97 + plan['subdivision_idx'])}"
                            write_empty_scene(
                                scene_idx, scene_name, plan['parent_scene'], bhopal_num,
                                plan['segment_size_m'], num_points_ds, original_num_points_ds,
                                scene_names_ds, parent_scene_ds, bhopal_paths_ds, segment_size_ds
                            )
                            scene_to_path.append(bhopal_num)
                            scene_idx += 1
                        continue

                    # V10 NEW: Spatially filter moving points for this V1 scene
                    moving_filtered = gather_moving_points_for_v1_scene(
                        v1_data, moving_data, moving_used_mask,
                        padding_m=moving_spatial_padding
                    )

                    # Merge training + moving points
                    v1_merged = merge_training_and_moving(v1_data, moving_filtered)

                    if moving_filtered is not None:
                        logging.debug(f"    {v1_name}: {len(moving_filtered['points']):,} moving points merged")

                    # Get projection info from first plan
                    first_plan = v1_plans[0]
                    direction = first_plan['scene_def']['direction']
                    origin = first_plan['scene_def']['origin']
                    projections = project_points(v1_merged['points'], direction, origin)

                    for plan in v1_plans:
                        scene_def = plan['scene_def']
                        scene_name = scene_def['scene_name']
                        if plan['total_subdivisions'] > 1:
                            scene_name = f"{scene_name}_{chr(97 + plan['subdivision_idx'])}"

                        try:
                            rel_start = plan['rel_start']
                            rel_end = plan['rel_end']

                            full_proj_start = scene_def['proj_start']
                            full_proj_end = scene_def['proj_end']
                            proj_range = full_proj_end - full_proj_start

                            sub_proj_start = full_proj_start + rel_start * proj_range
                            sub_proj_end = full_proj_start + rel_end * proj_range

                            is_last = (rel_end >= 0.999)

                            filtered = filter_points_by_projection(
                                v1_merged, projections,
                                sub_proj_start,
                                sub_proj_end,
                                is_last=is_last
                            )

                            if filtered is None:
                                write_empty_scene(
                                    scene_idx, scene_name, plan['parent_scene'], bhopal_num,
                                    plan['segment_size_m'], num_points_ds, original_num_points_ds,
                                    scene_names_ds, parent_scene_ds, bhopal_paths_ds, segment_size_ds
                                )
                                scene_to_path.append(bhopal_num)
                                scene_idx += 1
                                continue

                            original_num_pts = len(filtered['labels'])

                            if original_num_pts > 500000:
                                logging.debug(f"  Processing large scene {scene_name}: {original_num_pts:,} points")

                            # Apply class-specific sampling (moving classes kept at 100%)
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

                            # Safety downsampling
                            if num_pts > max_points_threshold:
                                logging.debug(f"  Safety downsample {scene_name}: {num_pts:,} -> {max_points_threshold:,}")
                                keep_indices = rng.choice(num_pts, size=max_points_threshold, replace=False)
                                keep_indices.sort()
                                pts = pts[keep_indices]
                                intensity = intensity[keep_indices]
                                gps_t = gps_t[keep_indices]
                                scene_labels = scene_labels[keep_indices]
                                scene_global_ids = scene_global_ids[keep_indices]
                                scene_rgb = scene_rgb[keep_indices]
                                num_pts = max_points_threshold

                            # Skip if below threshold
                            if num_pts < min_points_threshold:
                                write_empty_scene(
                                    scene_idx, scene_name, plan['parent_scene'], bhopal_num,
                                    plan['segment_size_m'], num_points_ds, original_num_points_ds,
                                    scene_names_ds, parent_scene_ds, bhopal_paths_ds, segment_size_ds,
                                    original_num_pts=original_num_pts
                                )
                                scene_to_path.append(bhopal_num)
                                scene_idx += 1
                                del filtered, pts, intensity, gps_t, scene_labels, scene_global_ids, scene_rgb
                                continue

                            # Calculate bounds and center
                            x_min, y_min, z_min = pts.min(axis=0)
                            x_max, y_max, z_max = pts.max(axis=0)
                            center = pts.mean(axis=0)

                            # Center the points
                            pts_centered = (pts - center).astype(np.float32)

                            # Compute surface normals
                            try:
                                scene_normals = compute_normals(pts_centered, k_neighbors=normal_k_neighbors)
                            except Exception as norm_err:
                                logging.warning(f"  Normal estimation failed for {scene_name}: {norm_err}")
                                scene_normals = np.zeros_like(pts_centered, dtype=np.float32)

                            # Count unique instances
                            unique_instances = len(np.unique(scene_global_ids))

                            # Count moving points in this scene
                            scene_moving_count = np.sum(np.isin(scene_labels, list(MOVING_LABEL_REMAP.values())))
                            total_moving_points_written += scene_moving_count

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

                            # Free memory
                            del filtered, pts_padded, normals_padded, rgb_padded, intensity_padded, gps_time_padded
                            del labels_padded, global_ids_padded
                            del pts, intensity, gps_t, scene_labels, scene_global_ids, scene_rgb
                            del pts_centered, scene_normals

                        except Exception as e:
                            logging.error(f"  Error processing scene {scene_name}: {e}")
                            traceback.print_exc()
                            try:
                                write_empty_scene(
                                    scene_idx, scene_name, plan['parent_scene'], bhopal_num,
                                    plan['segment_size_m'], num_points_ds, original_num_points_ds,
                                    scene_names_ds, parent_scene_ds, bhopal_paths_ds, segment_size_ds
                                )
                            except Exception as write_err:
                                logging.error(f"  Failed to write empty scene: {write_err}")
                            scene_to_path.append(bhopal_num)
                            scene_idx += 1
                            gc.collect()

                    del v1_data, v1_merged, projections, moving_filtered
                    gc.collect()

                # Log moving points assignment summary
                if moving_used_mask is not None:
                    assigned = np.sum(moving_used_mask)
                    total_moving = len(moving_used_mask)
                    unassigned = total_moving - assigned
                    logging.info(f"  Moving points summary for Bhopal_{bhopal_num}: "
                                 f"{assigned:,} assigned, {unassigned:,} unassigned out of {total_moving:,}")

            except Exception as e:
                logging.error(f"Error processing Bhopal_{bhopal_num}: {e}")
                traceback.print_exc()
                continue

            finally:
                # Aggressively free memory before loading next Bhopal path
                if data is not None:
                    for key in list(data.keys()):
                        del data[key]
                    del data
                    data = None
                if instance_index is not None:
                    instance_index.clear()
                    del instance_index
                    instance_index = None
                if moving_data is not None:
                    for key in list(moving_data.keys()):
                        del moving_data[key]
                    del moving_data
                    moving_data = None
                moving_used_mask = None
                for _ in range(5):
                    gc.collect()
                time.sleep(2)
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

        names_ds = class_group.create_dataset('names', shape=(NUM_CLASSES,), dtype=dt)
        for i, name in enumerate(class_info['names']):
            names_ds[i] = name

        class_group.create_dataset('colors', data=class_info['colors'])
        class_group.create_dataset('parent_ids', data=class_info['parent_ids'])

        xxzz_ds = class_group.create_dataset('xxzz_codes', shape=(NUM_CLASSES,), dtype=dt)
        for i, code in enumerate(class_info['xxzz_codes']):
            xxzz_ds[i] = code

        parent_names_ds = class_group.create_dataset('parent_names', shape=(NUM_PARENT_CLASSES,), dtype=dt)
        for i in range(NUM_PARENT_CLASSES):
            parent_names_ds[i] = PARENT_CLASSES[i]["name"]

        parent_colors = np.array([PARENT_CLASSES[i]["color"] for i in range(NUM_PARENT_CLASSES)], dtype=np.uint8)
        class_group.create_dataset('parent_colors', data=parent_colors)

        # Root attributes
        h5f.attrs['dataset_name'] = DATASET_NAME
        h5f.attrs['version'] = version
        h5f.attrs['description'] = DATASET_DESCRIPTION
        h5f.attrs['segmentation_method'] = 'spatial_cutting_v10_with_moving_rgb_normals'
        h5f.attrs['num_scenes'] = num_scenes
        h5f.attrs['max_points_per_scene'] = actual_max_points
        h5f.attrs['num_classes'] = NUM_CLASSES
        h5f.attrs['num_parent_classes'] = NUM_PARENT_CLASSES
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
        h5f.attrs['total_moving_points'] = int(total_moving_points_written)

        # Building sampling attributes
        h5f.attrs['building_classes'] = str(BUILDING_CLASSES)
        h5f.attrs['building_sample_rate'] = building_sample_rate
        h5f.attrs['min_points_threshold'] = min_points_threshold
        h5f.attrs['max_points_threshold'] = max_points_threshold
        h5f.attrs['subdivision_enabled'] = True
        h5f.attrs['base_segment_size_m'] = 5.0

        # Normal estimation attributes
        h5f.attrs['has_normals'] = True
        h5f.attrs['normal_k_neighbors'] = normal_k_neighbors
        h5f.attrs['normal_method'] = 'open3d_knn_pca_road_centric'
        h5f.attrs['normal_orientation'] = 'horizontal=Z-up, vertical=toward_road_center'

        # RGB attributes
        h5f.attrs['has_rgb'] = True
        h5f.attrs['rgb_source'] = 'V9_source_files'
        h5f.attrs['rgb_coverage_note'] = '~35-40% (Bhopal_2/8: 0%), zeros=missing'

        # V10 Moving classes attributes
        h5f.attrs['has_moving_classes'] = True
        h5f.attrs['moving_classes'] = str(list(MOVING_LABEL_REMAP.values()))
        h5f.attrs['moving_label_remap'] = json.dumps({str(k): v for k, v in MOVING_LABEL_REMAP.items()})
        h5f.attrs['moving_source'] = 'Bhopal_N_Unlabelled.npz (original codes 3-8)'
        h5f.attrs['moving_spatial_padding_m'] = moving_spatial_padding
        h5f.attrs['moving_assignment_method'] = 'XY bounding box of V1 scene training points with padding'
        h5f.attrs['moving_parent_mapping'] = '62,67 → Vehicle (parent 7); 63,64,65,66 → Human and Animal (parent 10)'
        h5f.attrs['excluded_unlabelled_codes'] = str(list(EXCLUDED_UNLABELLED_CODES))

        # Store class mapping as JSON attribute
        class_mapping_json = json.dumps({
            'label_to_xxzz': {str(k): v for k, v in class_info['label_to_xxzz'].items()},
            'label_to_name': {str(k): v for k, v in class_info['label_to_name'].items()}
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
    logging.info(f"Total moving points: {total_moving_points_written:,}")
    moving_pct = (total_moving_points_written / max(total_points_written, 1)) * 100
    logging.info(f"Moving points percentage: {moving_pct:.2f}%")
    logging.info(f"Actual max points/scene: {actual_max_points:,}")
    logging.info(f"Train: {len(train_indices):,}, Val: {len(val_indices):,}, Test: {len(test_indices):,}")
    logging.info(f"Classes: {NUM_CLASSES} (68 = 62 training + 6 moving)")
    logging.info(f"Parent categories: {NUM_PARENT_CLASSES} (12 = same as V9, moving merged into Vehicle/Human&Animal)")
    logging.info(f"Normal estimation: k={normal_k_neighbors} neighbors")
    logging.info(f"RGB: from V9 source files (~35-40% coverage, 0=missing)")


def print_dataset_info(hdf5_path):
    """Print comprehensive information about HDF5 dataset."""
    with h5py.File(hdf5_path, 'r') as h5f:
        print("\n" + "=" * 80)
        print("BHOPAL MLS SCENE V10 DATASET INFO (12 Parents, 68 Fine Classes)")
        print("=" * 80)

        print(f"\nName: {h5f.attrs.get('dataset_name', 'N/A')}")
        print(f"Version: {h5f.attrs.get('version', 'N/A')}")
        print(f"Segmentation: {h5f.attrs.get('segmentation_method', 'N/A')}")
        print(f"Creation Date: {h5f.attrs.get('creation_date', 'N/A')}")

        print(f"\nSampling & Subdivision Config:")
        print(f"  Building classes: {h5f.attrs.get('building_classes', 'N/A')}")
        print(f"  Building sample rate: {h5f.attrs.get('building_sample_rate', 'N/A')}")
        print(f"  Min points threshold: {h5f.attrs.get('min_points_threshold', 'N/A')}")
        print(f"  Max points threshold: {h5f.attrs.get('max_points_threshold', 'N/A')}")
        print(f"  Subdivision enabled: {h5f.attrs.get('subdivision_enabled', 'N/A')}")
        print(f"  Base segment size: {h5f.attrs.get('base_segment_size_m', 'N/A')}m")

        print(f"\nNormal Estimation Config:")
        print(f"  Has normals: {h5f.attrs.get('has_normals', False)}")
        print(f"  Normal K-neighbors: {h5f.attrs.get('normal_k_neighbors', 'N/A')}")
        print(f"  Normal method: {h5f.attrs.get('normal_method', 'N/A')}")
        print(f"  Normal orientation: {h5f.attrs.get('normal_orientation', 'N/A')}")

        print(f"\nRGB Config:")
        print(f"  Has RGB: {h5f.attrs.get('has_rgb', False)}")
        print(f"  RGB source: {h5f.attrs.get('rgb_source', 'N/A')}")
        print(f"  RGB coverage: {h5f.attrs.get('rgb_coverage_note', 'N/A')}")

        print(f"\nV10 Moving Classes Config:")
        print(f"  Has moving classes: {h5f.attrs.get('has_moving_classes', False)}")
        print(f"  Moving classes: {h5f.attrs.get('moving_classes', 'N/A')}")
        print(f"  Moving source: {h5f.attrs.get('moving_source', 'N/A')}")
        print(f"  Spatial padding: {h5f.attrs.get('moving_spatial_padding_m', 'N/A')}m")
        print(f"  Assignment method: {h5f.attrs.get('moving_assignment_method', 'N/A')}")
        print(f"  Excluded codes: {h5f.attrs.get('excluded_unlabelled_codes', 'N/A')}")
        print(f"  Total moving points: {h5f.attrs.get('total_moving_points', 'N/A'):,}")

        total_pts = h5f.attrs.get('total_points', 0)
        moving_pts = h5f.attrs.get('total_moving_points', 0)
        if total_pts > 0:
            print(f"  Moving percentage: {(moving_pts / total_pts) * 100:.2f}%")

        print(f"\nScenes: {h5f.attrs.get('num_scenes', 'N/A'):,}")
        print(f"Max Points/Scene: {h5f.attrs.get('max_points_per_scene', 'N/A'):,}")
        print(f"Total Points: {h5f.attrs.get('total_points', 'N/A'):,}")
        print(f"Classes: {h5f.attrs.get('num_classes', 'N/A')} fine / "
              f"{h5f.attrs.get('num_parent_classes', 'N/A')} parent")

        print(f"\nSplits:")
        print(f"  Train: {h5f.attrs.get('train_size', 0):,} scenes (paths: {h5f.attrs.get('train_paths', '')})")
        print(f"  Val: {h5f.attrs.get('val_size', 0):,} scenes (paths: {h5f.attrs.get('val_paths', '')})")
        print(f"  Test: {h5f.attrs.get('test_size', 0):,} scenes (paths: {h5f.attrs.get('test_paths', '')})")

        # Check datasets
        for ds_name in ['normals', 'rgb']:
            ds_path = f'scenes/{ds_name}'
            if ds_path in h5f:
                print(f"\n{ds_name.capitalize()} Dataset: PRESENT")
                print(f"  Shape: {h5f[ds_path].shape}")
                print(f"  Dtype: {h5f[ds_path].dtype}")
            else:
                print(f"\n{ds_name.capitalize()} Dataset: NOT PRESENT")

        if 'scenes/num_points' in h5f:
            num_points = h5f['scenes/num_points'][:]
            valid_mask = num_points > 0
            valid_points = num_points[valid_mask]
            print(f"\nPoint Statistics per Scene (after sampling):")
            print(f"  Valid scenes: {np.sum(valid_mask):,}")
            if len(valid_points) > 0:
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
            if len(valid_orig) > 0:
                print(f"\nOriginal Point Statistics (before sampling):")
                print(f"  Min: {np.min(valid_orig):,}")
                print(f"  Max: {np.max(valid_orig):,}")
                print(f"  Mean: {np.mean(valid_orig):,.0f}")
                print(f"  Total: {np.sum(valid_orig):,}")

        # V10: Show moving class distribution
        if 'scenes/labels' in h5f and 'scenes/num_points' in h5f:
            print(f"\nMoving Class Distribution (sampled from first 100 scenes):")
            num_points = h5f['scenes/num_points'][:]
            valid_indices = np.where(num_points > 0)[0][:100]
            moving_class_counts = defaultdict(int)
            for idx in valid_indices:
                n = num_points[idx]
                scene_labels = h5f['scenes/labels'][idx, :n]
                for mc in MOVING_LABEL_REMAP.values():
                    moving_class_counts[mc] += int(np.sum(scene_labels == mc))

            if any(v > 0 for v in moving_class_counts.values()):
                # Load class names from mapping
                try:
                    class_mapping_str = h5f.attrs.get('class_mapping', '{}')
                    cm = json.loads(class_mapping_str)
                    label_to_name = cm.get('label_to_name', {})
                except Exception:
                    label_to_name = {}

                for mc in sorted(MOVING_LABEL_REMAP.values()):
                    name = label_to_name.get(str(mc), f"Class {mc}")
                    print(f"  {mc} ({name}): {moving_class_counts[mc]:,} points")
            else:
                print(f"  No moving class points found in sampled scenes")

        print("\n" + "=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description='Create Bhopal MLS Scene V10 HDF5 dataset with moving classes, normals, and RGB',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Create dataset with default settings
  python3 19b_create_scene_V10_dataset.py --output ../bhopal_scene_v10.h5 \\
      --train-paths 1 2 3 4 --val-paths 5 6 --test-paths 7 8

  # Custom thresholds
  python3 19b_create_scene_V10_dataset.py --output ../bhopal_scene_v10.h5 \\
      --building-sample-rate 0.85 --min-points 5000 --max-points 800000 \\
      --train-paths 1 2 3 4 --val-paths 5 6 --test-paths 7 8

  # Custom moving spatial padding
  python3 19b_create_scene_V10_dataset.py --output ../bhopal_scene_v10.h5 \\
      --moving-padding 10.0 --train-paths 1 2 3 4 --val-paths 5 6 --test-paths 7 8

  # Show info
  python3 19b_create_scene_V10_dataset.py --info bhopal_scene_v10.h5
        """
    )

    parser.add_argument('--output', type=str, default='bhopal_scene_v10.h5',
                       help='Output HDF5 file path')
    parser.add_argument('--building-sample-rate', type=float, default=DEFAULT_BUILDING_SAMPLE_RATE,
                       help=f'Sampling rate for building classes (default: {DEFAULT_BUILDING_SAMPLE_RATE})')
    parser.add_argument('--min-points', type=int, default=DEFAULT_MIN_POINTS,
                       help=f'Minimum points threshold (default: {DEFAULT_MIN_POINTS})')
    parser.add_argument('--max-points', type=int, default=DEFAULT_MAX_POINTS,
                       help=f'Maximum points threshold - subdivide if exceeded (default: {DEFAULT_MAX_POINTS})')
    parser.add_argument('--normal-k', type=int, default=DEFAULT_NORMAL_K_NEIGHBORS,
                       help=f'K neighbors for normal estimation (default: {DEFAULT_NORMAL_K_NEIGHBORS})')
    parser.add_argument('--moving-padding', type=float, default=DEFAULT_MOVING_SPATIAL_PADDING,
                       help=f'XY padding in meters for moving class spatial assignment (default: {DEFAULT_MOVING_SPATIAL_PADDING})')
    parser.add_argument('--train-paths', type=int, nargs='+', default=None,
                       help='Bhopal paths for training (e.g., 1 2 3 4)')
    parser.add_argument('--val-paths', type=int, nargs='+', default=None,
                       help='Bhopal paths for validation (e.g., 5 6)')
    parser.add_argument('--test-paths', type=int, nargs='+', default=None,
                       help='Bhopal paths for testing (e.g., 7 8)')
    parser.add_argument('--random-seed', type=int, default=42,
                       help='Random seed (default: 42)')
    parser.add_argument('--version', type=str, default='10.0',
                       help='Dataset version (default: 10.0)')
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

    create_scene_v10_hdf5_dataset(
        output_path=args.output,
        train_paths=args.train_paths,
        val_paths=args.val_paths,
        test_paths=args.test_paths,
        building_sample_rate=args.building_sample_rate,
        min_points_threshold=args.min_points,
        max_points_threshold=args.max_points,
        normal_k_neighbors=args.normal_k,
        moving_spatial_padding=args.moving_padding,
        random_seed=args.random_seed,
        version=args.version
    )

    if os.path.exists(args.output):
        print_dataset_info(args.output)
    else:
        logging.error(f"Output file was not created: {args.output}")
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
