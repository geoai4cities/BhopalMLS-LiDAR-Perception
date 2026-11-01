#!/usr/bin/env python3
"""
Phase 4 — CORAL DA: Stage 3 — Evaluate trained heads on PL val features.

Predicts labels with both:
  - coral_head.pkl     (trained on CORAL-aligned source)
  - identity_head.pkl  (trained on raw source — reproducibility check)

Reports:
  - nc=11 mIoU + per-class IoU
  - common-7 mIoU (intersection of all 4 datasets: Veg, Bldg, Road, Wire, Pole, Veh, Fence)

Outputs:
    results/{arch}/eval_metrics.json
    results/coral_da_supp.tex   (cross-arch supp table; appended on second invocation)
"""
import argparse
import json
import os
import sys
from collections import OrderedDict

import joblib
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Same mapping as in coral_align_and_train.py
BM12_TO_NC11 = np.array([0, 1, 2, 3, 4, 5, -1, 6, 7, 8, 9, 10], dtype=np.int64)
NC11_NAMES = ["Ground", "Vegetation", "Building", "Road", "Wire", "Pole&Tower",
              "Vehicle", "Board", "Fencing", "Human&Animal", "Road-Side-Obj"]
# common-7 in BM-12: {1 Veg, 2 Bldg, 3 Road, 4 Wire, 5 Pole, 7 Veh, 9 Fence}
# In nc=11 space: BM12_TO_NC11 maps these BM-ids: 1→1, 2→2, 3→3, 4→4, 5→5, 7→6, 9→8
COMMON7_NC11_IDS = np.array([1, 2, 3, 4, 5, 6, 8], dtype=np.int64)


def remap_bm12_to_nc11(labels_bm12):
    out = np.full_like(labels_bm12, -1, dtype=np.int64)
    valid = (labels_bm12 >= 0) & (labels_bm12 < len(BM12_TO_NC11))
    out[valid] = BM12_TO_NC11[labels_bm12[valid]]
    return out


def compute_per_class_iou(preds, gts, n_classes):
    """Standard per-class IoU (TP / (TP + FP + FN)). Returns array [n_classes] in [0, 1]."""
    iou = np.zeros(n_classes, dtype=np.float64)
    for c in range(n_classes):
        pred_c = preds == c
        gt_c = gts == c
        tp = (pred_c & gt_c).sum()
        fp = (pred_c & ~gt_c).sum()
        fn = (~pred_c & gt_c).sum()
        denom = tp + fp + fn
        iou[c] = tp / denom if denom > 0 else np.nan
    return iou


def evaluate(F_val, labels_nc11, clf, common7_ids):
    """Return dict with miou_nc11, per_class_iou_nc11, miou_common7."""
    keep = labels_nc11 >= 0
    F_val_k = F_val[keep]
    labels_k = labels_nc11[keep]
    preds = clf.predict(F_val_k.astype(np.float32))

    n_cls = 11
    per_iou = compute_per_class_iou(preds, labels_k, n_cls)
    miou_11 = float(np.nanmean(per_iou))

    # common-7 mIoU = mean over the 7 ids in common
    miou_7 = float(np.nanmean(per_iou[common7_ids]))

    return {
        "miou_nc11": miou_11 * 100,
        "per_class_iou_nc11": [None if np.isnan(v) else float(v) * 100 for v in per_iou],
        "class_names_nc11": NC11_NAMES,
        "miou_common7": miou_7 * 100,
        "common7_indices_in_nc11": common7_ids.tolist(),
        "n_eval_pts": int(F_val_k.shape[0]),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arch", required=True, choices=["concerto", "utonia"])
    parser.add_argument("--out-dir", default=os.path.join(SCRIPT_DIR, "results"))
    args = parser.parse_args()

    arch_dir = os.path.join(args.out_dir, args.arch)
    val_path = os.path.join(arch_dir, "features_pl_val.npz")
    if not os.path.isfile(val_path):
        raise FileNotFoundError(f"Missing PL val features: {val_path}")

    print(f"=== Loading PL val features + heads ({args.arch}) ===")
    val = np.load(val_path)
    F_val = val["features"].astype(np.float32)
    labels_val_bm12 = val["labels"]
    labels_val_nc11 = remap_bm12_to_nc11(labels_val_bm12)
    print(f"  F_val: {F_val.shape}  ignore_count={int((labels_val_nc11 < 0).sum()):,}  "
          f"valid={int((labels_val_nc11 >= 0).sum()):,}")

    coral_head_path = os.path.join(arch_dir, "coral_head.pkl")
    id_head_path = os.path.join(arch_dir, "identity_head.pkl")
    clf_coral = joblib.load(coral_head_path)
    clf_id = joblib.load(id_head_path) if os.path.isfile(id_head_path) else None

    print(f"\n=== Eval: identity head (no CORAL) ===")
    metrics = OrderedDict()
    if clf_id is not None:
        metrics["identity"] = evaluate(F_val, labels_val_nc11, clf_id, COMMON7_NC11_IDS)
        print(f"  mIoU nc=11   = {metrics['identity']['miou_nc11']:.2f}")
        print(f"  mIoU common-7 = {metrics['identity']['miou_common7']:.2f}")
    else:
        print("  (skipped: identity_head.pkl not found)")

    print(f"\n=== Eval: CORAL-aligned head ===")
    metrics["coral"] = evaluate(F_val, labels_val_nc11, clf_coral, COMMON7_NC11_IDS)
    print(f"  mIoU nc=11   = {metrics['coral']['miou_nc11']:.2f}")
    print(f"  mIoU common-7 = {metrics['coral']['miou_common7']:.2f}")

    if "identity" in metrics:
        d11 = metrics["coral"]["miou_nc11"] - metrics["identity"]["miou_nc11"]
        d7 = metrics["coral"]["miou_common7"] - metrics["identity"]["miou_common7"]
        print(f"\n  Δ (CORAL - identity): nc=11 {d11:+.2f}  common-7 {d7:+.2f}")

    # Save metrics
    out_path = os.path.join(arch_dir, "eval_metrics.json")
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\n  wrote: {out_path}")

    # Update / write the cross-arch supp .tex (overwrite-and-rebuild from per-arch json files)
    write_supp_tex(args.out_dir)


def write_supp_tex(out_dir):
    """Rebuild the cross-arch supp table from per-arch eval_metrics.json files."""
    rows = []
    for arch in ["concerto", "utonia"]:
        m_path = os.path.join(out_dir, arch, "eval_metrics.json")
        if not os.path.isfile(m_path):
            continue
        with open(m_path) as f:
            m = json.load(f)
        rows.append((arch.capitalize(), m))

    if not rows:
        return

    tex_path = os.path.join(out_dir, "coral_da_supp.tex")
    # Vanilla baselines from RESULTS.md §E.5 (BM→PL)
    VANILLA = {
        "Concerto": {"nc11": 48.84, "common7": 60.70},
        "Utonia":   {"nc11": 55.68, "common7": 65.70},
    }
    with open(tex_path, "w") as f:
        f.write("% Phase 4 CORAL DA results (post-hoc, frozen FT-9ch encoder + retrained linear head)\n")
        f.write("% Source: BM (FT-9ch); Target: PL (Lille2 val); class set: nc=11 (PL native, drops WSC)\n")
        f.write("% Columns: vanilla = full FT-9ch seg head; retrain = post-hoc linear head on raw features\n")
        f.write("%          (A=I, isolates head-retrain effect); +CORAL = post-hoc head on CORAL-aligned features\n")
        f.write("\\begin{table}[t]\n\\centering\n\\small\n")
        f.write("\\caption{Post-hoc CORAL Domain Adaptation (R3-M4) on BM$\\to$PL with frozen FT-9ch "
                "encoder. \\textit{vanilla}: full-FT seg head. \\textit{retrain}: post-hoc linear head "
                "on raw source features (A=I, isolates head-retrain effect). \\textit{+CORAL}: "
                "post-hoc linear head on CORAL-aligned source features. Columns: nc=11 (PL native, drops "
                "WSC) / common-7 mIoU. CORAL hurts $-$3 to $-$6\\,pp over the identity-retrained "
                "baseline; covariance alignment alone cannot close the BM$\\leftrightarrow$PL gap, "
                "indicating it is structural rather than a 2nd-order distribution shift.}\n")
        f.write("\\label{tab:coral_da}\n")
        f.write("\\begin{tabular}{l|ccc|ccc}\n\\toprule\n")
        f.write(" & \\multicolumn{3}{c|}{\\textbf{nc=11 mIoU}} & \\multicolumn{3}{c}{\\textbf{common-7 mIoU}} \\\\\n")
        f.write("\\textbf{Method} & vanilla & retrain & + CORAL & vanilla & retrain & + CORAL \\\\\n")
        f.write("\\midrule\n")
        for arch_name, m in rows:
            v = VANILLA.get(arch_name, {"nc11": 0, "common7": 0})
            i = m.get("identity", {"miou_nc11": 0, "miou_common7": 0})
            c = m["coral"]
            f.write(f"BM$\\to$PL  ({arch_name}) "
                    f"& {v['nc11']:.2f} & {i['miou_nc11']:.2f} & {c['miou_nc11']:.2f} "
                    f"& {v['common7']:.2f} & {i['miou_common7']:.2f} & {c['miou_common7']:.2f} \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")
        f.write("\\end{table}\n")
    print(f"  wrote: {tex_path}")


if __name__ == "__main__":
    main()
