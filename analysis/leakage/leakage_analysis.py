#!/usr/bin/env python3
"""Spatial-leakage analysis of the BhopalMLS path-based split .

Adapts the proximity metric of Lilja et al., CVPR 2024 ("Localization is All
You Evaluate") to a handheld MLS dataset: instead of frame poses, we measure
(1) trajectory-to-trajectory proximity, (2) scene-centre proximity, and
(3) observed-geometry overlap via scene AABBs and sensor-range-buffered
corridors. Outputs metrics JSON + map figure + CDF figure into results/.

Inputs (all EPSG:32643):
  - data/Bhopal/Bhopal_Path_vector_data/Bhopal_{1..8}.gpkg  (path trajectories)
  - data/Bhopal/bhopal_scene_v9_normals_rgb.h5              (benchmark split + scene metadata)
"""

import json
from pathlib import Path

import h5py
import numpy as np
import shapely
from scipy.spatial import cKDTree
from shapely.geometry import box
from shapely.ops import unary_union

import matplotlib

matplotlib.use("Agg")
import matplotlib.patheffects
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

REPO = Path(__file__).resolve().parents[2]  # repository root
H5_PATH = REPO / "data/Bhopal/bhopal_scene_v9_normals_rgb.h5"
GPKG_DIR = REPO / "data/Bhopal/Bhopal_Path_vector_data"
OUT = Path(__file__).parent / "results"

SPLIT_OF_PATH = {1: "train", 2: "train", 3: "train", 4: "train",
                 5: "val", 6: "val", 7: "test", 8: "test"}
SPLIT_COLORS = {"train": "#2E86AB", "val": "#A23B72", "test": "#F18F01"}
THRESHOLDS_M = [5.0, 10.0, 20.0, 25.0, 50.0]
SAMPLE_STEP_M = 1.0


def load_trajectories():
    """Per-path 2D LineString geometry from the gpkg files."""
    import geopandas as gpd

    paths = {}
    for i in range(1, 9):
        gdf = gpd.read_file(GPKG_DIR / f"Bhopal_{i}.gpkg", layer="bhopal_paths_utm_final")
        geom = shapely.force_2d(unary_union(gdf.geometry.values))
        paths[i] = geom
    return paths


def load_scenes():
    with h5py.File(H5_PATH, "r") as f:
        center = f["metadata/center"][:]
        bounds = f["metadata/bounds"][:]
        path_id = f["metadata/bhopal_paths"][:]
        split_idx = {s: f[f"splits/{s}_indices"][:] for s in ("train", "val", "test")}
    split = np.full(len(path_id), "none", dtype=object)
    for s, idx in split_idx.items():
        split[idx] = s
    # sanity: split assignment must equal the path-based rule
    for s, idx in split_idx.items():
        got = {SPLIT_OF_PATH[int(p)] for p in np.unique(path_id[idx])}
        assert got == {s}, f"split {s} contains paths from {got}"
    valid = ~np.all(center == 0.0, axis=1)  # a minority of scenes lack geo metadata
    return center, bounds, path_id, split, valid


def sample_along(geom, step=SAMPLE_STEP_M):
    """Points every `step` metres along a (Multi)LineString."""
    lines = getattr(geom, "geoms", [geom])
    pts = []
    for ln in lines:
        d = np.arange(0.0, ln.length, step)
        pts.extend(ln.interpolate(x) for x in d)
    return shapely.points([[p.x, p.y] for p in pts])


def cdf_stats(dists):
    return {
        "n": int(len(dists)),
        "median_m": float(np.median(dists)),
        "pct_within": {f"{t:g}m": float(np.mean(dists <= t) * 100) for t in THRESHOLDS_M},
    }


def main():
    OUT.mkdir(exist_ok=True)
    paths = load_trajectories()
    center, bounds, path_id, split, valid = load_scenes()

    split_traj = {
        s: unary_union([g for p, g in paths.items() if SPLIT_OF_PATH[p] == s])
        for s in ("train", "val", "test")
    }

    results = {"h5": str(H5_PATH), "thresholds_m": THRESHOLDS_M}

    # ---- A. trajectory proximity: eval-split samples vs train trajectory ----
    traj = {}
    for s in ("val", "test"):
        samples = sample_along(split_traj[s])
        d = shapely.distance(samples, split_traj["train"])
        traj[s] = d
        results[f"trajectory_{s}_vs_train"] = cdf_stats(d)
    d = shapely.distance(sample_along(split_traj["test"]),
                         unary_union([split_traj["train"], split_traj["val"]]))
    results["trajectory_test_vs_trainval"] = cdf_stats(d)

    # ---- B. scene-centre proximity (Lilja-style sample metric) ----
    xy = center[:, :2]
    tree = cKDTree(xy[(split == "train") & valid])
    for s in ("val", "test"):
        d, _ = tree.query(xy[(split == s) & valid])
        results[f"scene_centre_{s}_vs_train"] = cdf_stats(d)

    # ---- C. observed-geometry overlap ----
    # sensor reach estimated from the data: scenes are ~5 m along-track chunks,
    # so the AABB extent is dominated by cross-track observed geometry
    half_extent = np.max(bounds[:, 3:5] - bounds[:, 0:2], axis=1)[valid] / 2.0
    r_med, r_p90 = float(np.median(half_extent)), float(np.percentile(half_extent, 90))
    results["sensor_half_extent_m"] = {"median": r_med, "p90": r_p90}

    aabb = {
        s: unary_union([box(*bounds[i, 0:2], *bounds[i, 3:5])
                        for i in np.where((split == s) & valid)[0]])
        for s in ("train", "val", "test")
    }
    results["aabb_union_overlap"] = {
        s: {
            "area_m2": float(aabb[s].area),
            "overlap_with_train_m2": float(aabb[s].intersection(aabb["train"]).area),
            "overlap_with_train_pct": float(aabb[s].intersection(aabb["train"]).area
                                            / aabb[s].area * 100),
        }
        for s in ("val", "test")
    }

    results["corridor_overlap"] = {}
    for r in [10.0, 20.0, r_med]:
        buf = {s: split_traj[s].buffer(r) for s in ("train", "val", "test")}
        results["corridor_overlap"][f"r={r:.1f}m"] = {
            s: float(buf[s].intersection(buf["train"]).area / buf[s].area * 100)
            for s in ("val", "test")
        }

    # ---- figures ----
    overlap_zone = split_traj["test"].buffer(r_med).intersection(split_traj["train"].buffer(r_med))

    from matplotlib.patches import Polygon as MplPolygon

    fig, ax = plt.subplots(figsize=(7.5, 7.5))
    for z in getattr(overlap_zone, "geoms", [overlap_zone]):
        ax.add_patch(MplPolygon(np.asarray(z.exterior.coords), closed=True,
                                facecolor="#C0392B", alpha=0.25, edgecolor="#C0392B",
                                hatch="///", linewidth=0.5, zorder=1))
    for p, geom in paths.items():
        s = SPLIT_OF_PATH[p]
        for ln in getattr(geom, "geoms", [geom]):
            x, y = ln.xy
            ax.plot(x, y, color=SPLIT_COLORS[s], linewidth=2.0, zorder=2)
        mid = geom.interpolate(0.5, normalized=True)
        # bare path numbers: the main paper says "Path 01--04", never "P1"
        ax.annotate(f"{p}", (mid.x, mid.y), fontsize=10, fontweight="bold",
                    color=SPLIT_COLORS[s], zorder=3,
                    path_effects=[matplotlib.patheffects.withStroke(linewidth=2.5,
                                                                    foreground="white")])
    handles = [Line2D([], [], color=SPLIT_COLORS[s], lw=2.5,
                      label={"train": "Train (paths 1-4)", "val": "Val (paths 5-6)",
                             "test": "Test (paths 7-8)"}[s])
               for s in ("train", "val", "test")]
    handles.append(Patch(facecolor="#C0392B", alpha=0.25, hatch="///",
                         edgecolor="#C0392B",
                         label=f"Test/train shared coverage (r = {r_med:.0f} m)"))
    # Legend below the map: placed inside the axes it sat on top of the P1
    # train path and the upper-left corner of the shared-coverage zone.
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.10),
              ncol=2, frameon=False, fontsize=9, handlelength=2.2,
              columnspacing=1.6, borderaxespad=0.0)
    ax.set_aspect("equal")
    ax.set_xlabel("Easting (m, UTM 43N)")
    ax.set_ylabel("Northing (m, UTM 43N)")
    # Plain tick labels: the default offset notation printed a bare "1e6" in the
    # corner the legend used to occupy, which reads as a stray annotation.
    ax.ticklabel_format(style="plain", useOffset=False)
    ax.set_xticks(ax.get_xticks()[::2])
    ax.set_yticks(ax.get_yticks()[::2])
    ax.tick_params(labelsize=8)
    ax.grid(alpha=0.2, linewidth=0.5)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"map_split_overlap.{ext}", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.5, 4.0))
    for s in ("val", "test"):
        d = np.sort(traj[s])
        ax.plot(d, np.arange(1, len(d) + 1) / len(d) * 100,
                color=SPLIT_COLORS[s], lw=2.0)
        ax.annotate(s.capitalize(), (d[int(len(d) * 0.6)], 60 if s == "test" else 40),
                    color=SPLIT_COLORS[s], fontsize=10, fontweight="bold")
    for t in [5, 25]:
        ax.axvline(t, color="#999999", lw=0.8, ls="--", zorder=0)
        ax.annotate(f"{t} m", (t, 96), fontsize=8, color="#777777", ha="left")
    ax.set_xscale("log")
    ax.set_xlabel("Distance to nearest Train trajectory (m)")
    ax.set_ylabel("Fraction of eval samples (%)")
    ax.set_ylim(0, 100)
    ax.grid(alpha=0.2, linewidth=0.5)
    handles = [Line2D([], [], color=SPLIT_COLORS[s], lw=2.0, label=s.capitalize())
               for s in ("val", "test")]
    ax.legend(handles=handles, loc="lower right", frameon=False, fontsize=9)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"proximity_cdf.{ext}", dpi=200)
    plt.close(fig)

    with open(OUT / "leakage_metrics.json", "w") as f:
        json.dump(results, f, indent=2)

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
