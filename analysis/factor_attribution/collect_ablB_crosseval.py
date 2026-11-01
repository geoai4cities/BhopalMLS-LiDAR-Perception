#!/usr/bin/env python3
"""exp_03b step 3: turn the ablation cross-region evals into the attribution table.

Reads   Utonia/cross_data_testing/outputs/abl<ARM>_on_<T>/ft_9ch/logs/test_*.log
Against exp_03a's perclass_transfer.json (paper full-model BM->X and X->BM, plus
        the per-pair intersection masks, so every number is computed on exactly
        the class set the paper's Table 5 uses -- n_c = KM 12, CH 12, PL 11,
        SK 9, TR 7).

The attribution metric is how much of the asymmetry an ablation removes:

    gap        = mIoU(BM->X, full) - mIoU(X->BM, full)      # what needs explaining
    closed_pp  = mIoU(BM->X, full) - mIoU(BM->X, arm)       # what the arm removed
    closed_%   = 100 * closed_pp / gap

B1_size is the size control: B2/B3 are read against B1, not only against full,
because all three arms share B1's labeled budget. A factor "matters" when its
arm closes markedly more of the gap than B1 does at the same budget.

Usage:  python3 collect_ablB_crosseval.py [--json perclass_transfer.json] [--out results/]
"""
import argparse
import glob
import json
import os
import re
import sys

# Per-architecture wiring. The Concerto tree suffixes its FT dir with the
# pretraining variant, and the paper's Concerto(L) rows are the *_indoor ones
# (bare-name dirs are FT-from-outdoor = Concerto(O)); see NOTES.md.
ARCHS = {
    "utonia": {
        "xdir": "Utonia/cross_data_testing/outputs",
        "subdir": "ft_9ch",
        "refkey": "Utonia|ft_9ch",
    },
    "concerto": {
        "xdir": "Concerto/cross_data_testing/outputs",
        "subdir": "ft_9ch_indoor",
        "refkey": "Concerto|ft_9ch_indoor",
    },
}
ARMS = ["FULL", "B1_size", "B2_variety", "B3_regular", "B4_subtype"]
TARGETS = [("kerala", "KM"), ("tr", "TR"), ("pl", "PL"), ("sk", "SK"), ("chandigarh", "CH")]

RE_OVERALL = re.compile(r"Val result: mIoU/mAcc/allAcc ([\d.]+)/([\d.]+)/([\d.]+)")
RE_CLASS = re.compile(r"Class_(\d+) - (.+?) Result: iou/accuracy ([\d.]+)/([\d.]+)")


def pct(x):
    """Logs print IoU as a fraction; exp_03a stored percentages."""
    return x * 100.0 if x <= 1.0 else x


def parse_log(path):
    """Last evaluation block in a test log -> (overall mIoU, {class_idx: iou})."""
    overall, cls = None, {}
    with open(path, errors="ignore") as fh:
        for line in fh:
            m = RE_OVERALL.search(line)
            if m:
                overall = pct(float(m.group(1)))
                cls = {}          # a later block supersedes an earlier one
                continue
            m = RE_CLASS.search(line)
            if m:
                cls[int(m.group(1))] = pct(float(m.group(3)))
    return overall, cls


def latest_log(arm, target_dir, xdir, subdir):
    logs = sorted(glob.glob(os.path.join(xdir, f"abl{arm}_on_{target_dir}", subdir, "logs", "test_*.log")))
    return logs[-1] if logs else None


def masked_miou(iou_by_idx, mask):
    vals = [iou_by_idx[i] for i, keep in enumerate(mask) if keep and i in iou_by_idx]
    if len(vals) != sum(mask):
        return None, len(vals)
    return sum(vals) / len(vals), len(vals)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                   "perclass_transfer.json"))
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "results"))
    ap.add_argument("--arch", default="utonia", choices=sorted(ARCHS))
    args = ap.parse_args()
    XDIR = ARCHS[args.arch]["xdir"]
    SUBDIR = ARCHS[args.arch]["subdir"]
    REFKEY = ARCHS[args.arch]["refkey"]

    if not os.path.isfile(args.json):
        sys.exit(f"exp_03a reference not found: {args.json}\n"
                 f"copy it from the repo: results/perclass_transfer.json")
    ref = json.load(open(args.json))
    names, masks, runs = ref["class_names"], ref["masks"], ref["runs"]

    out = {"targets": {}, "class_names": names}
    lines = [f"# exp_03b — factor attribution from cross-region evals ({args.arch})", "",
             "mIoU on each pair's intersection class set (same masks as exp_03a / paper Table 5).",
             "`closed %` = share of the full model's BM->X advantage that the ablation removes.", ""]

    for tdir, tkey in TARGETS:
        mask = masks.get(tkey)
        if mask is None:
            continue
        fwd = runs.get(f"{REFKEY}|BM->{tkey}")
        rev = runs.get(f"{REFKEY}|{tkey}->BM")
        if not fwd or not rev:
            continue
        full_fwd = fwd["mIoU_intersection"]
        back = rev["mIoU_intersection"]
        gap = full_fwd - back

        rows, per_class = [], {}
        for arm in ARMS:
            log = latest_log(arm, tdir, XDIR, SUBDIR)
            if log is None:
                continue
            overall, cls = parse_log(log)
            if overall is None:
                rows.append((arm, None, None, None, "log has no result block"))
                continue
            mi, n = masked_miou(cls, mask)
            if mi is None:
                rows.append((arm, None, None, None, f"only {n}/{sum(mask)} masked classes in log"))
                continue
            closed_pp = full_fwd - mi
            rows.append((arm, mi, closed_pp, 100.0 * closed_pp / gap if gap else float("nan"),
                         os.path.basename(log)))
            per_class[arm] = {names[i]: cls.get(i) for i, k in enumerate(mask) if k}

        if not rows:
            continue

        lines += [f"## BM -> {tkey}  (n_c={sum(mask)})", "",
                  f"- paper full model  BM->{tkey} = **{full_fwd:.2f}**, {tkey}->BM = **{back:.2f}**, "
                  f"asymmetry gap = **{gap:.2f}** pp", "",
                  "| arm | BM->X mIoU | removed (pp) | closed % | source log |",
                  "|---|---|---|---|---|"]
        for arm, mi, cp, cpct, note in rows:
            if mi is None:
                lines.append(f"| {arm} | — | — | — | {note} |")
            else:
                lines.append(f"| {arm} | {mi:.2f} | {cp:+.2f} | {cpct:.1f}% | {note} |")
        lines.append("")

        out["targets"][tkey] = {
            "n_c": sum(mask), "full_forward": full_fwd, "reverse": back, "gap": gap,
            "arms": {r[0]: {"miou": r[1], "removed_pp": r[2], "closed_pct": r[3]} for r in rows},
            "per_class": per_class,
        }

    if not out["targets"]:
        sys.exit(f"no {args.arch} ablation cross-eval logs found yet under " + XDIR)

    # per-class view: where each arm's loss concentrates, vs exp_03a's Delta_c story
    lines += ["## Per-class BM->X IoU by arm", ""]
    for tkey, blk in out["targets"].items():
        arms = [a for a in ARMS if a in blk["per_class"]]
        if len(arms) < 2:
            continue
        lines += [f"### {tkey}", "", "| class | " + " | ".join(arms) + " |",
                  "|---|" + "---|" * len(arms)]
        for cname in blk["per_class"][arms[0]]:
            vals = " | ".join(f"{blk['per_class'][a].get(cname):.2f}"
                              if blk["per_class"][a].get(cname) is not None else "—" for a in arms)
            lines.append(f"| {cname} | {vals} |")
        lines.append("")

    os.makedirs(args.out, exist_ok=True)
    tag = "" if args.arch == "utonia" else f"_{args.arch}"
    md = os.path.join(args.out, f"ablation_attribution{tag}.md")
    js = os.path.join(args.out, f"ablation_attribution{tag}.json")
    open(md, "w").write("\n".join(lines) + "\n")
    json.dump(out, open(js, "w"), indent=2)
    print("\n".join(lines))
    print(f"\nwrote {md}\nwrote {js}")


if __name__ == "__main__":
    main()
