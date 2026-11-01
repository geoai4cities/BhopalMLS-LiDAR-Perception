#!/usr/bin/env python3
"""Build factor-controlled train subsets for exp_03b .

Four ablation train sets over BhopalMLS v9, equal LABELED-point budget
(25% of full train labels ~= Paris-Lille-3D scale):

  B1 size-matched:        random scene subsample (seed 42)
  B2 variety-collapsed:   labels of non-dominant fine subclasses -> ignore
                          (via config class_groups); random scenes until the
                          DOMINANT-subclass labeled budget matches B1
  B3 geometry-regularized: scenes ranked by corridor regularity
                          (wide road + straight trajectory), most regular first
  B4 subtype-swapped:     B2, except Vehicle keeps *Car* rather than its
                          dominant subtype (Mobike / Two-Wheeler)

B4 separates the two readings of B2's result. B2 collapses each class to its
dominant fine subclass, and BhopalMLS's dominant Vehicle subtype is the
two-wheeler, which is exactly what the target benchmarks do not contain. B2's
collapse could therefore mean "intra-class variety carries the transfer
advantage" or, far more weakly, "we retained the one subtype the targets lack".
B4 retains Car, the subtype the targets are largely made of, and changes
nothing else. If B4 still transfers like B2, variety is causal; if B4 recovers
toward B1, the effect is subtype-target match and the claim must narrow.

B4 walks scenes in the SAME permutation as B2, so the only difference between
the two is which subtype's points count toward the budget.

Emits per-ablation lightweight HDF5s next to the v9 file (external links to
scenes/metadata/class_info + own splits group) so NO dataset code changes and
no 23 GB copies are needed. Also writes results/ablation_subsets.json with
the dominant-subclass table (for the B2/B4 configs) and subset stats.

Usage:
  python3 build_ablation_subsets.py                    # B1,B2,B3 (as shipped)
  python3 build_ablation_subsets.py --arms B4_subtype  # add B4, others untouched

Selected arms are merged into results/ablation_subsets.json; arms not selected
keep their existing entries byte-for-byte, so rebuilding B4 can never disturb
the three subsets whose models are already trained and reported.
"""

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import shapely
from shapely.ops import unary_union

REPO = Path(__file__).resolve().parents[2]  # repository root
H5 = REPO / "data/Bhopal/bhopal_scene_v9_normals_rgb.h5"
GPKG_DIR = REPO / "data/Bhopal/Bhopal_Path_vector_data"
OUT = Path(__file__).parent / "results"

BUDGET_FRAC = 0.25
SEED = 42
N_FINE = 62
ROAD_FINE = 13
BUILDING_FINE = set(range(8, 13))
PARENT_GROUPS = [list(range(0, 4)), list(range(4, 8)), list(range(8, 13)),
                 [13], [14], list(range(15, 19)), [19], list(range(20, 35)),
                 list(range(35, 38)), list(range(38, 43)), list(range(43, 50)),
                 list(range(50, 62))]
PARENT_NAMES = ["Ground", "Vegetation", "Building", "Road", "Wire", "Pole and Tower",
                "Wire-Structure Connector", "Vehicle", "Board", "Fencing",
                "Human and Animal", "Road Side Object"]

VEHICLE_PARENT = 7          # index into PARENT_GROUPS / PARENT_NAMES
B4_VEHICLE_FINE = 21        # 'Car' — the subtype the target benchmarks are made of
ALL_ARMS = ["B1_size", "B2_variety", "B3_regular", "B4_subtype"]


def trajectory_curvature(paths_geom, centers, path_ids, window=20.0):
    """Per-scene heading spread (deg) of its path within +-window m of the scene."""
    import geopandas as gpd  # noqa: F401  (fiona backend warm-up)
    curv = np.full(len(centers), np.nan)
    samples = {}
    for p, geom in paths_geom.items():
        lines = getattr(geom, "geoms", [geom])
        pts, headings, chain = [], [], []
        off = 0.0
        for ln in lines:
            d = np.arange(0.0, ln.length, 2.0)
            xy = np.array([ln.interpolate(x).coords[0][:2] for x in d])
            if len(xy) < 3:
                off += ln.length
                continue
            h = np.arctan2(np.diff(xy[:, 1]), np.diff(xy[:, 0]))
            pts.append(xy[:-1])
            headings.append(h)
            chain.append(d[:-1] + off)
            off += ln.length
        samples[p] = (np.vstack(pts), np.concatenate(headings), np.concatenate(chain))
    for i, (c, p) in enumerate(zip(centers, path_ids)):
        if p not in samples:
            continue
        xy, h, chain = samples[p]
        j = np.argmin(np.linalg.norm(xy - c[:2], axis=1))
        m = np.abs(chain - chain[j]) <= window
        hh = np.unwrap(h[m])
        curv[i] = np.degrees(hh.max() - hh.min()) if m.sum() >= 3 else np.nan
    return curv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default="B1_size,B2_variety,B3_regular",
                    help="comma-separated subset of " + ",".join(ALL_ARMS))
    args = ap.parse_args()
    want = [a.strip() for a in args.arms.split(",") if a.strip()]
    unknown = [a for a in want if a not in ALL_ARMS]
    if unknown:
        raise SystemExit(f"unknown arm(s): {unknown}; known: {ALL_ARMS}")
    print("building:", ", ".join(want))

    OUT.mkdir(exist_ok=True)
    rng = np.random.default_rng(SEED)

    with h5py.File(H5, "r") as f:
        train_idx = f["splits/train_indices"][:]
        val_idx = f["splits/val_indices"][:]
        test_idx = f["splits/test_indices"][:]
        num_points = f["scenes/num_points"][:]
        center = f["metadata/center"][:]
        path_id = f["metadata/bhopal_paths"][:]
        root_attrs = dict(f.attrs)

    total_train = int(num_points[train_idx].sum())
    budget = int(total_train * BUDGET_FRAC)
    print(f"train scenes {len(train_idx)}, labeled points {total_train:,}, "
          f"budget {budget:,} ({BUDGET_FRAC:.0%})")

    # ---- per-scene fine-class counts + road width (one pass over train) ----
    counts = np.zeros((len(train_idx), N_FINE), dtype=np.int64)
    road_width = np.full(len(train_idx), np.nan)
    with h5py.File(H5, "r") as f:
        labels_ds, points_ds = f["scenes/labels"], f["scenes/points"]
        for k, gi in enumerate(train_idx):
            n = int(num_points[gi])
            lab = labels_ds[gi, :n]
            ok = (lab >= 0) & (lab < N_FINE)
            counts[k] = np.bincount(lab[ok], minlength=N_FINE)
            road_mask = lab == ROAD_FINE
            if road_mask.sum() >= 500:
                xy = points_ds[gi, :n, :2][road_mask]
                xy = xy - xy.mean(0)
                cov = np.cov(xy.T)
                w, v = np.linalg.eigh(cov)
                proj = xy @ v[:, 0]  # minor axis
                road_width[k] = np.percentile(proj, 95) - np.percentile(proj, 5)
            if (k + 1) % 200 == 0:
                print(f"  scanned {k + 1}/{len(train_idx)} scenes")

    # ---- dominant fine subclass per parent (train point counts) ----
    total_counts = counts.sum(0)
    dominant = []
    for parent, fines in enumerate(PARENT_GROUPS):
        dom = int(np.array(fines)[np.argmax(total_counts[fines])])
        dominant.append(dom)
        share = total_counts[dom] / max(total_counts[fines].sum(), 1)
        print(f"  parent {PARENT_NAMES[parent]:26s} dominant fine {dom:2d} "
              f"({share:.0%} of parent points)")
    dom_mask = np.zeros(N_FINE, dtype=bool)
    dom_mask[dominant] = True
    dom_points_per_scene = counts[:, dom_mask].sum(1)

    # ---- B4: same retained-subtype rule, Vehicle swapped to Car ----
    dominant_b4 = list(dominant)
    dominant_b4[VEHICLE_PARENT] = B4_VEHICLE_FINE
    b4_mask = np.zeros(N_FINE, dtype=bool)
    b4_mask[dominant_b4] = True
    b4_points_per_scene = counts[:, b4_mask].sum(1)
    veh_fines = PARENT_GROUPS[VEHICLE_PARENT]
    veh_total = max(total_counts[veh_fines].sum(), 1)
    print(f"  B4 swaps Vehicle {dominant[VEHICLE_PARENT]} -> {B4_VEHICLE_FINE}: "
          f"dominant subtype holds {total_counts[dominant[VEHICLE_PARENT]] / veh_total:.0%} "
          f"of Vehicle points, Car holds {total_counts[B4_VEHICLE_FINE] / veh_total:.0%}")

    # ---- curvature from trajectories ----
    import geopandas as gpd
    paths_geom = {}
    for p in sorted(set(int(x) for x in path_id[train_idx])):
        gdf = gpd.read_file(GPKG_DIR / f"Bhopal_{p}.gpkg", layer="bhopal_paths_utm_final")
        paths_geom[p] = shapely.force_2d(unary_union(gdf.geometry.values))
    curv = trajectory_curvature(paths_geom, center[train_idx], path_id[train_idx])

    def take_until(order, weights, cap):
        picked, tot = [], 0
        for k in order:
            if tot >= cap:
                break
            picked.append(k)
            tot += int(weights[k])
        return np.array(picked), tot

    n_pts = num_points[train_idx].astype(np.int64)

    # Draw both permutations unconditionally so the rng stream is identical
    # regardless of which arms were requested: B1's and B2's scene orders must
    # not depend on the --arms flag.
    perm_b1 = rng.permutation(len(train_idx))
    perm_b2 = rng.permutation(len(train_idx))

    # B1: random scenes, full labels
    b1, b1_tot = take_until(perm_b1, n_pts, budget)
    # B2: random scenes, only dominant-subclass labels count toward budget
    b2, b2_tot = take_until(perm_b2, dom_points_per_scene, budget)
    # B4: B2's scene order exactly, Car counted for Vehicle instead of Mobike
    b4, b4_tot = take_until(perm_b2, b4_points_per_scene, budget)
    # B3: most regular first (wide + straight); scenes without road width sink
    z = lambda x: (x - np.nanmean(x)) / np.nanstd(x)
    score = np.where(np.isnan(road_width), -10.0, z(road_width)) \
        - np.where(np.isnan(curv), 2.0, z(curv))
    b3, b3_tot = take_until(np.argsort(-score), n_pts, budget)

    subsets = {"B1_size": b1, "B2_variety": b2, "B3_regular": b3, "B4_subtype": b4}
    labeled = {"B1_size": b1_tot, "B2_variety": b2_tot, "B3_regular": b3_tot,
               "B4_subtype": b4_tot}

    # Regression guard. B1/B2/B3 are already trained and reported, so a change
    # to the rng path or the budget rule that silently moved them would
    # invalidate the published table. Every arm is recomputed on every run even
    # when not selected, so this check is free.
    prev = json.loads((OUT / "ablation_subsets.json").read_text()) \
        if (OUT / "ablation_subsets.json").is_file() else {}
    for name in ("B1_size", "B2_variety", "B3_regular"):
        if name in prev and "scenes" in prev[name]:
            got, exp = len(subsets[name]), prev[name]["scenes"]
            assert got == exp, (f"{name} moved: {got} scenes now vs {exp} recorded. "
                                f"The published arms must not change; fix before writing.")
    print("regression guard: B1/B2/B3 reproduce their recorded scene counts")

    subsets = {k: v for k, v in subsets.items() if k in want}

    # Start from the existing report so arms that were not rebuilt keep their
    # recorded stats exactly as they were when their models were trained.
    report_path = OUT / "ablation_subsets.json"
    report = json.loads(report_path.read_text()) if report_path.is_file() else {}
    report.update({
        "h5": str(H5), "seed": SEED, "budget_frac": BUDGET_FRAC,
        "train_scenes": int(len(train_idx)), "train_points": total_train,
        "budget_labeled_points": budget,
        "dominant_fine_per_parent": {PARENT_NAMES[i]: dominant[i] for i in range(12)},
        "b2_class_groups": [[d] for d in dominant],
        "b4_class_groups": [[d] for d in dominant_b4],
        "b4_vehicle_fine": B4_VEHICLE_FINE,
    })
    for name, sel in subsets.items():
        gi = np.sort(train_idx[sel])
        pcs = counts[sel].sum(0)
        parent_pts = {PARENT_NAMES[p]: int(pcs[PARENT_GROUPS[p]].sum()) for p in range(12)}
        report[name] = {
            "scenes": int(len(sel)), "total_points": int(n_pts[sel].sum()),
            "labeled_points": int(labeled[name]),
            "paths": {int(p): int((path_id[gi] == p).sum())
                      for p in sorted(set(int(x) for x in path_id[gi]))},
            "parent_class_points_full": parent_pts,
            "mean_road_width_m": float(np.nanmean(road_width[sel])),
            "mean_curvature_deg": float(np.nanmean(curv[sel])),
        }
        h5_out = H5.parent / f"bhopal_scene_v9_abl{name}.h5"
        with h5py.File(h5_out, "w") as f:
            for grp in ("scenes", "metadata", "class_info"):
                f[grp] = h5py.ExternalLink(H5.name, f"/{grp}")
            sp = f.create_group("splits")
            sp.create_dataset("train_indices", data=gi)
            sp.create_dataset("val_indices", data=val_idx)
            sp.create_dataset("test_indices", data=test_idx)
            for k, v in root_attrs.items():
                try:
                    f.attrs[k] = v
                except Exception:
                    pass
            f.attrs["ablation"] = name
            f.attrs["parent_h5"] = H5.name
        report[name]["h5_out"] = str(h5_out)
        print(f"{name}: {len(sel)} scenes, {labeled[name]:,} labeled pts -> {h5_out.name}")

    report["full_train_reference"] = {
        "mean_road_width_m": float(np.nanmean(road_width)),
        "mean_curvature_deg": float(np.nanmean(curv)),
    }
    report_path.write_text(json.dumps(report, indent=2))
    print("written:", report_path)


if __name__ == "__main__":
    main()
