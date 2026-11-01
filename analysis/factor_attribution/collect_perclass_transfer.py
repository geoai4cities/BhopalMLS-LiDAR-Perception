#!/usr/bin/env python3
"""Collect per-class IoUs from cross-region test logs .

Walks {Utonia,Concerto,Sonata}/cross_data_testing/outputs/{a}_on_{b}/ft_9ch*/
logs, extracts the final per-class IoU block (BM-12 space), infers each
BM<->target pair's intersection mask (classes structurally zero in every run
of the pair), recomputes intersection mIoU, validates against paper Table
`tab:cross_region_1cm`, and emits the per-class asymmetry decomposition
Delta_c = IoU(BM->X) - IoU(X->BM).

Outputs: results/perclass_transfer.json, results/asymmetry_table.md
"""

import json
import re
import sys
from pathlib import Path

import numpy as np

M = Path(__file__).resolve().parents[2]  # repository root
OUT = Path(__file__).parent / "results"

CLASSES = ["Ground", "Vegetation", "Building", "Road", "Wire", "Pole and Tower",
           "Wire-Structure Connector", "Vehicle", "Board", "Fencing",
           "Human and Animal", "Road Side Object"]
TARGETS = {"kerala": "KM", "tr": "TR", "pl": "PL", "sk": "SK",
           "chandigarh": "CH"}
# paper Table tab:cross_region_1cm, Utonia(L)/Concerto(L) mIoU, for validation.
# 2026-08-22: Concerto(L) added. Its absence is exactly how the stale Concerto
# SK->BM rows survived in tab:cross_city_full -- the March old-PCR run stayed in
# the paper after the 05-25 native-KITTI rerun superseded it, and nothing here
# ever compared the two. Every pair the paper reports must be listed.
PAPER = {("Utonia", "ft_9ch", "BM->KM"): 56.78, ("Utonia", "ft_9ch", "KM->BM"): 33.96,
         ("Utonia", "ft_9ch", "BM->TR"): 77.27, ("Utonia", "ft_9ch", "TR->BM"): 51.65,
         ("Utonia", "ft_9ch", "BM->PL"): 55.68, ("Utonia", "ft_9ch", "PL->BM"): 27.06,
         ("Utonia", "ft_9ch", "BM->SK"): 40.05, ("Utonia", "ft_9ch", "SK->BM"): 35.54,
         ("Concerto", "ft_9ch_indoor", "BM->KM"): 49.23,
         ("Concerto", "ft_9ch_indoor", "KM->BM"): 32.70,
         ("Concerto", "ft_9ch_indoor", "BM->TR"): 75.20,
         ("Concerto", "ft_9ch_indoor", "TR->BM"): 39.11,
         ("Concerto", "ft_9ch_indoor", "BM->PL"): 48.84,
         ("Concerto", "ft_9ch_indoor", "PL->BM"): 31.07,
         ("Concerto", "ft_9ch_indoor", "BM->SK"): 32.07,
         ("Concerto", "ft_9ch_indoor", "SK->BM"): 12.29}

LINE = re.compile(r"Class_(\d+) - (.+?) Result: iou/accuracy ([0-9.]+)/([0-9.]+)")


def parse_log(exp_dir):
    """Last per-class block from test.log (fallback train.log). -> iou[12], acc[12]"""
    for logname in ("test.log", "train.log"):
        f = exp_dir / logname
        if not f.exists():
            continue
        iou = np.full(12, np.nan)
        acc = np.full(12, np.nan)
        found = False
        for line in f.read_text(errors="ignore").splitlines():
            m = LINE.search(line)
            if m:
                k = int(m.group(1))
                if k < 12:
                    iou[k], acc[k] = float(m.group(3)), float(m.group(4))
                    found = True
        if found:
            return iou * 100, acc * 100
    return None, None


def main():
    OUT.mkdir(exist_ok=True)
    runs = {}  # (model, variant, direction) -> {"iou": [...], "acc": [...]}
    for model in ("Utonia", "Concerto", "Sonata"):
        base = M / model / "cross_data_testing/outputs"
        if not base.exists():
            continue
        for d in sorted(base.iterdir()):
            m = re.fullmatch(r"(bhopal)_on_(\w+)|(\w+)_on_(bhopal)", d.name)
            if not m:
                continue
            if m.group(1):  # bhopal_on_X
                tgt, direction = m.group(2), "out"
            else:  # X_on_bhopal
                tgt, direction = m.group(3), "in"
            if tgt not in TARGETS:
                continue
            for var_dir in sorted(d.iterdir()):
                if not var_dir.is_dir() or not var_dir.name.startswith("ft"):
                    continue
                iou, acc = parse_log(var_dir)
                if iou is None:
                    continue
                code = TARGETS[tgt]
                arrow = f"BM->{code}" if direction == "out" else f"{code}->BM"
                runs[(model, var_dir.name, arrow)] = {"iou": iou.tolist(),
                                                      "acc": acc.tolist()}

    # intersection mask per target: classes with iou==0 AND acc==0 in EVERY run
    # of that pair (both directions, all models/variants)
    masks = {}
    for code in sorted(set(TARGETS.values())):   # sorted: keep the JSON diff stable
        pair_runs = [v for (mo, va, ar), v in runs.items() if code in ar]
        if not pair_runs:
            continue
        structural_zero = np.ones(12, dtype=bool)
        for v in pair_runs:
            iou = np.array(v["iou"])
            acc = np.array(v["acc"])
            structural_zero &= (iou == 0.0) & (acc == 0.0)
        masks[code] = (~structural_zero).tolist()

    report = {"class_names": CLASSES, "masks": masks, "runs": {}}
    lines = ["# exp_03a — per-class cross-region transfer decomposition\n"]
    lines.append("Intersection masks (inferred: classes structurally zero in every "
                 "run of a pair are excluded):\n")
    for code, mk in sorted(masks.items()):
        inc = [CLASSES[i] for i in range(12) if mk[i]]
        lines.append(f"- **{code}** (n_c={sum(mk)}): {', '.join(inc)}")

    lines.append("\n## Validation vs paper Table (Utonia ft_9ch, Concerto ft_9ch_indoor)\n")
    lines.append("| model | pair | paper mIoU | recomputed | match |")
    lines.append("|---|---|---|---|---|")
    bad = []
    for (mo, va, ar), paper_v in PAPER.items():
        v = runs.get((mo, va, ar))
        if v is None:
            lines.append(f"| {mo} | {ar} | {paper_v} | MISSING | — |")
            bad.append(f"{mo} {ar}: no run on disk")
            continue
        code = ar.replace("BM->", "").replace("->BM", "")
        mk = np.array(masks[code])
        val = float(np.mean(np.array(v["iou"])[mk]))
        ok = abs(val - paper_v) < 0.06
        lines.append(f"| {mo} | {ar} | {paper_v:.2f} | {val:.2f} | "
                     f"{'OK' if ok else 'DIFF'} |")
        if not ok:
            bad.append(f"{mo} {ar}: paper {paper_v:.2f} vs disk {val:.2f} "
                       f"({val - paper_v:+.2f})")
    if bad:
        # Loud, because a quiet DIFF row in a markdown file is what let the
        # stale Concerto SK rows sit in the paper from 05-25 to 08-22.
        print("\n*** PAPER/DISK MISMATCH -- the .tex quotes runs that are "
              "not the ones on disk:", file=sys.stderr)
        for b in bad:
            print("      " + b, file=sys.stderr)
        print("    Resolve before building; do not just rerun this "
              "script.\n", file=sys.stderr)

    for (mo, va, ar), v in sorted(runs.items()):
        code = ar.replace("BM->", "").replace("->BM", "")
        mk = np.array(masks[code])
        report["runs"][f"{mo}|{va}|{ar}"] = {
            **v, "mIoU_intersection": float(np.mean(np.array(v["iou"])[mk])),
            "n_c": int(mk.sum()),
        }

    lines.append("\n## Per-class asymmetry Δ_c = IoU(BM→X) − IoU(X→BM)\n")
    for model in ("Utonia", "Concerto", "Sonata"):
        variants = sorted({va for (mo, va, ar) in runs if mo == model})
        for va in variants:
            pairs = sorted({ar.replace("BM->", "").replace("->BM", "")
                            for (mo, v_, ar) in runs
                            if mo == model and v_ == va})
            pairs = [c for c in pairs
                     if (model, va, f"BM->{c}") in runs
                     and (model, va, f"{c}->BM") in runs]
            if not pairs:
                continue
            lines.append(f"\n### {model} {va}\n")
            lines.append("| class | " + " | ".join(
                f"{c}: out / in / Δ" for c in pairs) + " |")
            lines.append("|---|" + "---|" * len(pairs))
            for k, cname in enumerate(CLASSES):
                cells = []
                for c in pairs:
                    if not masks[c][k]:
                        cells.append("·")
                        continue
                    o = runs[(model, va, f"BM->{c}")]["iou"][k]
                    i = runs[(model, va, f"{c}->BM")]["iou"][k]
                    cells.append(f"{o:.1f} / {i:.1f} / {o - i:+.1f}")
                lines.append(f"| {cname} | " + " | ".join(cells) + " |")

    (OUT / "perclass_transfer.json").write_text(json.dumps(report, indent=2))
    (OUT / "asymmetry_table.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
