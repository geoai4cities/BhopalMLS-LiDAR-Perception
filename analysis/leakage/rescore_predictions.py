#!/usr/bin/env python3
"""Re-score cached Pointcept test predictions on leakage subsets (exp_02).

Takes one or more Pointcept experiment dirs containing result/<pred_name>_pred.npy
files for the BhopalMLS v9 test split, and recomputes semantic-seg metrics
separately for the 'disjoint' (leakage-free) and 'overlapping' test subsets
from partition_test_scenes.py, plus the full test set as a reproduction check
against the logged official number.

Supports both benchmark class spaces via --classes:
  12 (default): DEFAULT_12CLASS_CONFIG parent grouping (all 62 fine mapped)
  21          : 21-class grouping from the *_21c configs (fine 47-49
                Animal/Dog/Cow are unmapped -> ignored, matching training)

Usage:
  rescore_predictions.py [--classes 21] --exp-dir <path> [--exp-dir ...]
                         [--label <name> ...] [--out <report.json>]
"""

import argparse
import csv
import json
from pathlib import Path

import h5py
import numpy as np

H5_PATH = Path(__file__).resolve().parents[2] / Path("data/Bhopal/bhopal_scene_v9_normals_rgb.h5")
PARTITION = Path(__file__).parent / "results/test_scene_partition.csv"
OUT_DIR = Path(__file__).parent / "results"

N_FINE = 62

CLASS_CONFIGS = {
    12: {
        "names": ["Ground", "Vegetation", "Building", "Road", "Wire",
                  "Pole and Tower", "Wire-Structure Connector", "Vehicle",
                  "Board", "Fencing", "Human and Animal", "Road Side Object"],
        "groups": [list(range(0, 4)), list(range(4, 8)), list(range(8, 13)),
                   [13], [14], list(range(15, 19)), [19], list(range(20, 35)),
                   list(range(35, 38)), list(range(38, 43)), list(range(43, 50)),
                   list(range(50, 62))],
    },
    21: {
        "names": ["Ground", "Stairs", "Vegetation", "Building (other)",
                  "Commercial", "Road", "Wire", "Pole (other)", "Electric Pole",
                  "Street Light", "2-Wheeler", "3-Wheeler", "4-Wheeler",
                  "Hand-Cart", "Board", "Fencing", "Human",
                  "Road Side Furniture", "Shop Shade", "Mannequin",
                  "Road Side Obj (other)"],
        "groups": [[0, 1, 2], [3], [4, 5, 6, 7], [8, 9, 11, 12], [10], [13],
                   [14], [15, 17, 19], [16], [18], [22, 23, 32, 33, 34],
                   [25, 27, 28, 30], [20, 21, 26, 29, 31], [24], [35, 36, 37],
                   [38, 39, 40, 41, 42], [43, 44, 45, 46], [53], [52], [57],
                   [50, 51, 54, 55, 56, 58, 59, 60, 61]],
    },
}


def build_mapping(groups):
    m = np.full(N_FINE, -1, dtype=np.int64)
    for target, fines in enumerate(groups):
        m[list(fines)] = target
    return m


def load_partition():
    rows = list(csv.DictReader(open(PARTITION)))
    for r in rows:
        r["h5_index"] = int(r["h5_index"])
        r["n_points"] = int(r["n_points"])
    return rows


def metrics_from_confusion(conf, names):
    n = len(names)
    tp = np.diag(conf).astype(float)
    fp = conf.sum(0) - tp
    fn = conf.sum(1) - tp
    denom = tp + fp + fn
    iou = np.where(denom > 0, tp / np.maximum(denom, 1), np.nan)
    acc = np.where(conf.sum(1) > 0, tp / np.maximum(conf.sum(1), 1), np.nan)
    return {
        "mIoU": float(np.nanmean(iou) * 100),
        "mAcc": float(np.nanmean(acc) * 100),
        "allAcc": float(tp.sum() / conf.sum() * 100),
        "per_class_IoU": {names[k]: (None if np.isnan(iou[k]) else float(iou[k] * 100))
                          for k in range(n)},
    }


def rescore(exp_dir, rows, labels_ds, mapping, names):
    n_cls = len(names)
    result_dir = exp_dir / "result"
    conf = {s: np.zeros((n_cls, n_cls), dtype=np.int64)
            for s in ("disjoint", "overlapping", "full")}
    missing, mismatched = [], []
    for r in rows:
        pf = result_dir / f"{r['pred_name']}_pred.npy"
        if not pf.exists():
            missing.append(r["pred_name"])
            continue
        pred = np.load(pf)
        gt_fine = labels_ds[r["h5_index"], :r["n_points"]]
        if len(pred) != len(gt_fine):
            mismatched.append((r["pred_name"], len(pred), len(gt_fine)))
            continue
        ok = (gt_fine >= 0) & (gt_fine < N_FINE)
        gt = mapping[gt_fine[ok]]
        pr = pred[ok].astype(np.int64)
        keep = (gt >= 0) & (pr >= 0) & (pr < n_cls)
        c = np.bincount(gt[keep] * n_cls + pr[keep], minlength=n_cls * n_cls) \
            .reshape(n_cls, n_cls)
        conf["full"] += c
        if r["subset"] in conf:
            conf[r["subset"]] += c
    out = {s: metrics_from_confusion(conf[s], names) for s in conf}
    out["n_missing_pred"] = len(missing)
    out["missing_pred"] = missing[:10]
    out["n_length_mismatch"] = len(mismatched)
    out["length_mismatch"] = mismatched[:10]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp-dir", action="append", required=True)
    ap.add_argument("--label", action="append", default=None,
                    help="display name per --exp-dir, same order")
    ap.add_argument("--classes", type=int, default=12, choices=(12, 21))
    ap.add_argument("--out", default=None,
                    help="report filename (default rescore_report[_21c].json)")
    args = ap.parse_args()
    labels_arg = args.label or [Path(d).name for d in args.exp_dir]
    assert len(labels_arg) == len(args.exp_dir)

    cfg = CLASS_CONFIGS[args.classes]
    mapping = build_mapping(cfg["groups"])
    out_name = args.out or ("rescore_report.json" if args.classes == 12
                            else "rescore_report_21c.json")

    rows = load_partition()
    report = {}
    with h5py.File(H5_PATH, "r") as f:
        labels_ds = f["scenes/labels"]
        for name, d in zip(labels_arg, args.exp_dir):
            print(f"== {name}: {d}")
            report[name] = rescore(Path(d), rows, labels_ds, mapping, cfg["names"])
            r = report[name]
            print(f"   full {r['full']['mIoU']:.2f}  "
                  f"disjoint {r['disjoint']['mIoU']:.2f}  "
                  f"overlapping {r['overlapping']['mIoU']:.2f}  "
                  f"(missing {r['n_missing_pred']}, mismatch {r['n_length_mismatch']})")
            # A length mismatch means a cached prediction does not correspond to
            # the scene it is named for: the scene is silently dropped from the
            # confusion matrix, so the mIoU above is computed over a subset.
            # Pointcept asserts on this; do not let it pass as a footnote.
            if r["n_length_mismatch"]:
                raise SystemExit(
                    f"!! {name}: {r['n_length_mismatch']} scene(s) whose cached "
                    f"prediction length disagrees with the H5 (e.g. "
                    f"{r['length_mismatch'][:3]}). The cache is stale or belongs "
                    f"to a different dataset version - re-run test.py for this "
                    f"model instead of scoring a subset.")

    OUT_DIR.mkdir(exist_ok=True)
    out_path = OUT_DIR / out_name
    if out_path.exists():
        prev = json.load(open(out_path))
        prev.update(report)
        report = prev
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"written: {out_path}")


if __name__ == "__main__":
    main()
