#!/usr/bin/env python3
"""Partition BhopalMLS test scenes by train-coverage overlap .

For each test scene, compute the fraction of its observed-geometry AABB that
intersects the union of train-scene AABBs (same geometry basis as exp_01).
Scenes with zero overlap form the leakage-free "disjoint" subset; the rest are
"overlapping". Also records per-scene distance to the nearest train trajectory
and per-subset GT class composition (62 fine -> 12 parent), so the re-scoring
step can check class coverage.

Outputs: results/test_scene_partition.csv, results/partition_summary.json
"""

import csv
import json
from pathlib import Path

import h5py
import numpy as np
import shapely
from shapely.geometry import Point, box
from shapely.ops import unary_union

REPO = Path(__file__).resolve().parents[2]  # repository root
H5_PATH = REPO / "data/Bhopal/bhopal_scene_v9_normals_rgb.h5"
GPKG_DIR = REPO / "data/Bhopal/Bhopal_Path_vector_data"
OUT = Path(__file__).parent / "results"

TRAIN_PATHS = {1, 2, 3, 4}
N_FINE = 62
FINE_TO_PARENT = np.empty(N_FINE, dtype=np.int64)
for parent, fines in enumerate([range(0, 4), range(4, 8), range(8, 13), [13], [14],
                                range(15, 19), [19], range(20, 35), range(35, 38),
                                range(38, 43), range(43, 50), range(50, 62)]):
    FINE_TO_PARENT[list(fines)] = parent
PARENT_NAMES = ["Ground", "Vegetation", "Building", "Road", "Wire", "Pole and Tower",
                "Wire-Structure Connector", "Vehicle", "Board", "Fencing",
                "Human and Animal", "Road Side Object"]


def train_trajectory():
    import geopandas as gpd

    geoms = []
    for i in sorted(TRAIN_PATHS):
        gdf = gpd.read_file(GPKG_DIR / f"Bhopal_{i}.gpkg", layer="bhopal_paths_utm_final")
        geoms.append(shapely.force_2d(unary_union(gdf.geometry.values)))
    return unary_union(geoms)


def main():
    OUT.mkdir(exist_ok=True)
    with h5py.File(H5_PATH, "r") as f:
        bounds = f["metadata/bounds"][:]
        center = f["metadata/center"][:]
        path_id = f["metadata/bhopal_paths"][:]
        scene_names = [s.decode() if isinstance(s, bytes) else s
                       for s in f["metadata/scene_names"][:]]
        num_points = f["metadata/num_points"][:] if "num_points" in f["metadata"] \
            else f["scenes/num_points"][:]
        train_idx = f["splits/train_indices"][:]
        test_idx = f["splits/test_indices"][:]

    valid = ~np.all(center == 0.0, axis=1)
    train_union = unary_union([box(*bounds[i, 0:2], *bounds[i, 3:5])
                               for i in train_idx if valid[i]])
    traj = train_trajectory()

    rows = []
    for i in test_idx:
        i = int(i)
        if not valid[i]:
            rows.append({"h5_index": i, "scene_name": scene_names[i],
                         "path": int(path_id[i]),
                         "pred_name": f"{scene_names[i]}_bhopal{int(path_id[i])}",
                         "n_points": int(num_points[i]), "overlap_frac": None,
                         "dist_to_train_traj_m": None, "subset": "no_geometry"})
            continue
        aabb = box(*bounds[i, 0:2], *bounds[i, 3:5])
        frac = aabb.intersection(train_union).area / aabb.area
        d = traj.distance(Point(center[i, :2]))
        rows.append({"h5_index": i, "scene_name": scene_names[i],
                     "path": int(path_id[i]),
                     "pred_name": f"{scene_names[i]}_bhopal{int(path_id[i])}",
                     "n_points": int(num_points[i]),
                     "overlap_frac": round(float(frac), 4),
                     "dist_to_train_traj_m": round(float(d), 2),
                     "subset": "disjoint" if frac == 0.0 else "overlapping"})

    with open(OUT / "test_scene_partition.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)

    # per-subset GT class composition (read labels row-wise, valid points only)
    comp = {s: np.zeros(N_FINE, dtype=np.int64) for s in ("disjoint", "overlapping")}
    ignored = {s: 0 for s in comp}
    with h5py.File(H5_PATH, "r") as f:
        labels = f["scenes/labels"]
        for r in rows:
            if r["subset"] not in comp:
                continue
            lab = labels[r["h5_index"], :r["n_points"]]
            ok = (lab >= 0) & (lab < N_FINE)
            comp[r["subset"]] += np.bincount(lab[ok], minlength=N_FINE)
            ignored[r["subset"]] += int((~ok).sum())

    summary = {"h5": str(H5_PATH), "criterion": "scene AABB vs union of train AABBs"}
    for s in ("disjoint", "overlapping"):
        sub = [r for r in rows if r["subset"] == s]
        parent = np.zeros(12, dtype=np.int64)
        for c in range(N_FINE):
            parent[FINE_TO_PARENT[c]] += comp[s][c]
        summary[s] = {
            "scenes": len(sub),
            "points": int(sum(r["n_points"] for r in sub)),
            "paths": {p: sum(1 for r in sub if r["path"] == p) for p in (7, 8)},
            "ignored_points": ignored[s],
            "parent_class_points": {PARENT_NAMES[k]: int(parent[k]) for k in range(12)},
            "parent_classes_present": int((parent > 0).sum()),
            "fine_classes_present": int((comp[s] > 0).sum()),
        }
    summary["no_geometry_scenes"] = [r["pred_name"] for r in rows
                                     if r["subset"] == "no_geometry"]
    fracs = np.array([r["overlap_frac"] for r in rows if r["overlap_frac"] is not None])
    summary["overlap_frac_distribution"] = {
        "mean": float(fracs.mean()),
        "pct_scenes_over_50pct_overlap": float((fracs > 0.5).mean() * 100),
        "pct_scenes_over_10pct_overlap": float((fracs > 0.1).mean() * 100),
    }

    with open(OUT / "partition_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
