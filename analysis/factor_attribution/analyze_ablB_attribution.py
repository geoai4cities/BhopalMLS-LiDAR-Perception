#!/usr/bin/env python3
"""exp_03b step 4: read the finished attribution table with the controls it needs.

`collect_ablB_crosseval.py` (run on the H200 host) reports each arm's BM->X mIoU and the
share of the full model's advantage it removes. That headline number has two
confounds this script removes:

  1. **In-domain capability.** An arm that is simply a weaker model loses transfer
     for free. We therefore report transfer penalty *net of* the same arm's
     in-domain penalty on the Bhopal test split.
  2. **One class.** B2's loss is concentrated on Vehicle, whose dominant BhopalMLS
     subtype is the two-wheeler, not the car. We recompute every arm with Vehicle
     dropped from the intersection mask to check the effect survives.

Inputs (both already in results/):
  ablation_attribution.json  — arm mIoU + per-class IoU, built on the H200 host
  perclass_transfer.json     — exp_03a reference: full-model BM->X / X->BM + masks

Usage: python3 analyze_ablB_attribution.py [--out results/ablation_robustness.md]
"""
import argparse
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ALL_ARMS = ["B1_size", "B2_variety", "B3_regular", "B4_subtype"]
LABEL = {"B1_size": "B1 size", "B2_variety": "B2 variety", "B3_regular": "B3 geometry",
         "B4_subtype": "B4 subtype"}

# In-domain BhopalMLS test mIoU, 12-class, precise (TTA) evaluation.
# FULL: exp_02 results/summary_table.md (utonia_ft full-test 71.28, matches its log).
# arms: outputs/utonia_bhopal_finetune_9ch_abl<ARM>/logs/test_*.log, test.py line 340.
INDOMAIN = {"FULL": 71.28, "B1_size": 67.99, "B2_variety": 61.18, "B3_regular": 68.61,
            "B4_subtype": 56.34}
# B4 is the WEAKEST arm in-domain, below even B2. That is expected rather than
# alarming: BhopalMLS's Vehicle class is 52% two-wheeler and only 23% car, so an
# arm supervised on cars alone covers less of what Bhopal actually contains. It
# also makes the in-domain normalisation the decisive statistic for B4 -- a low
# raw transfer number would be ambiguous, but the RATIO is not. If B4 loses less
# transfer than it loses in-domain (ratio < 1) it is better abroad than at home,
# which is what subtype-target match predicts and what no other arm shows.

DROP = "Vehicle"


def masked_mean(iou_by_name, names):
    vals = [iou_by_name[n] for n in names if iou_by_name.get(n) is not None]
    return (sum(vals) / len(vals)) if len(vals) == len(names) else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--attr", default=os.path.join(HERE, "results/ablation_attribution.json"))
    ap.add_argument("--ref", default=os.path.join(HERE, "results/perclass_transfer.json"))
    ap.add_argument("--out", default=os.path.join(HERE, "results/ablation_robustness.md"))
    args = ap.parse_args()

    attr = json.load(open(args.attr))
    ref = json.load(open(args.ref))
    cls_names = ref["class_names"]

    # Only report arms that actually have results, so this runs unchanged while
    # a new arm is still training.
    present = {a for blk in attr["targets"].values() for a in blk["arms"]
               if blk["arms"][a].get("miou") is not None}
    global ARMS
    ARMS = [a for a in ALL_ARMS if a in present]
    missing = [a for a in ALL_ARMS if a not in present]
    if missing:
        print(f"note: no results yet for {', '.join(missing)} — omitted\n")

    L = ["# exp_03b — attribution with controls",
         "",
         "Built by `analyze_ablB_attribution.py` from `ablation_attribution.json`",
         "(the H200 host cross-evals) and `perclass_transfer.json` (exp_03a reference).",
         "All three arms carry the same ~135M labelled-point budget, so they differ",
         "only in *what kind* of data is kept.",
         ""]

    # ---- 1. headline -------------------------------------------------------
    L += ["## 1. BM->X transfer, full class intersection", "",
          "| pair | n_c | full BM->X | X->BM | gap | B1 size | B2 variety | B3 geometry |",
          "|---|---|---|---|---|---|---|---|"]
    order = [t for t in ("KM", "TR", "PL", "SK") if t in attr["targets"]]
    for t in order:
        b = attr["targets"][t]
        cells = []
        for a in ARMS:
            m = b["arms"].get(a, {}).get("miou")
            cells.append("—" if m is None else f"{m:.2f} ({m - b['full_forward']:+.2f})")
        L.append(f"| BM$\\to${t} | {b['n_c']} | {b['full_forward']:.2f} | {b['reverse']:.2f} | "
                 f"{b['gap']:.2f} | " + " | ".join(cells) + " |")
    L += ["", "Parentheses: change vs the full model. `gap` is what needs explaining.", ""]

    # ---- 2. arm vs the size control ---------------------------------------
    L += ["## 2. Read against B1, the size control (same budget)", "",
          "| pair | B2 − B1 | B3 − B1 |", "|---|---|---|"]
    d21, d31 = [], []
    for t in order:
        arms = attr["targets"][t]["arms"]
        b1 = arms.get("B1_size", {}).get("miou")
        b2 = arms.get("B2_variety", {}).get("miou")
        b3 = arms.get("B3_regular", {}).get("miou")
        if None in (b1, b2, b3):
            continue
        d21.append(b2 - b1)
        d31.append(b3 - b1)
        L.append(f"| BM$\\to${t} | {b2 - b1:+.2f} | {b3 - b1:+.2f} |")
    L.append(f"| **mean** | **{sum(d21)/len(d21):+.2f}** | **{sum(d31)/len(d31):+.2f}** |")
    L.append("")

    # ---- 3. net of in-domain capability ------------------------------------
    L += ["## 3. Transfer penalty net of in-domain penalty", "",
          "An arm that is simply a weaker model loses transfer for free. Column 2 is",
          "the arm's own in-domain loss on the BhopalMLS test split; column 4 is the",
          "part of the transfer loss that in-domain capability does *not* explain.",
          "Reported over all four pairs and over KM/TR/PL alone, because SK's own",
          "asymmetry gap is 4.50 pp and both control arms *beat* the full model there",
          "(see the caveat in NOTES.md).", ""]

    def dtr_mean(a, pairs):
        d = [attr["targets"][t]["arms"][a]["miou"] - attr["targets"][t]["full_forward"]
             for t in pairs if attr["targets"][t]["arms"].get(a, {}).get("miou") is not None]
        return sum(d) / len(d) if d else None

    for scope, pairs in (("all four", order), ("KM/TR/PL", [t for t in order if t != "SK"])):
        L += [f"**{scope}**", "",
              "| arm | in-domain (Δ vs full) | mean transfer Δ | net transfer penalty | ratio |",
              "|---|---|---|---|---|"]
        for a in ARMS:
            if a not in INDOMAIN:
                continue          # not precise-evaluated yet
            dins = INDOMAIN[a] - INDOMAIN["FULL"]
            dtr = dtr_mean(a, pairs)
            if dtr is None:
                continue
            L.append(f"| {LABEL[a]} | {INDOMAIN[a]:.2f} ({dins:+.2f}) | {dtr:+.2f} | "
                     f"{dtr - dins:+.2f} | {dtr / dins:.2f}× |")
        L.append("")
    L += ["In-domain values are precise (TTA) 12-class test mIoU; the full model is",
          "71.28 (exp_02 `summary_table.md`).", ""]

    # ---- 4. drop Vehicle ---------------------------------------------------
    L += [f"## 4. Same table with *{DROP}* removed from every mask", "",
          f"B2's loss concentrates on {DROP}, whose dominant BhopalMLS subtype is",
          "*Mobike (Two-Wheeler)* (fine class 22) — not the car that dominates every",
          "target dataset. If the variety effect were only that one class, the",
          "contrast would vanish here.", "",
          "| pair | full | B1 size | B2 variety | B3 geometry | B2 − B1 |",
          "|---|---|---|---|---|---|"]
    nd21 = []
    for t in order:
        b = attr["targets"][t]
        mask = ref["masks"][t]
        keep = [cls_names[i] for i, k in enumerate(mask) if k and cls_names[i] != DROP]
        if len(keep) == sum(mask):
            continue  # class not in this pair's intersection
        full_iou = {cls_names[i]: v for i, v in
                    enumerate(ref["runs"][f"Utonia|ft_9ch|BM->{t}"]["iou"])}
        full_v = masked_mean(full_iou, keep)
        vals = {a: masked_mean(b["per_class"].get(a, {}), keep) for a in ARMS}
        vals = {a: v for a, v in vals.items() if v is not None}
        if full_v is None or not {"B1_size", "B2_variety"} <= set(vals):
            continue
        nd21.append(vals["B2_variety"] - vals["B1_size"])
        cells = " | ".join(f"{vals[a]:.2f} ({vals[a] - full_v:+.2f})" if a in vals else "—"
                           for a in ARMS)
        L.append(f"| BM$\\to${t} ({len(keep)}c) | {full_v:.2f} | {cells} | "
                 f"{vals['B2_variety'] - vals['B1_size']:+.2f} |")
    if nd21:
        L.append(f"| **mean B2 − B1** | | | | | **{sum(nd21)/len(nd21):+.2f}** |")
    L.append("")

    # ---- 5. where B2 loses, class by class ---------------------------------
    L += ["## 5. B2 − B1 per class (negative = variety collapse costs transfer)", ""]
    allc = [c for c in cls_names]
    hdr = "| class | " + " | ".join(order) + " |"
    L += [hdr, "|---|" + "---|" * len(order)]
    for c in allc:
        row = []
        any_v = False
        for t in order:
            pc = attr["targets"][t]["per_class"]
            b1, b2 = pc.get("B1_size", {}).get(c), pc.get("B2_variety", {}).get(c)
            if b1 is None or b2 is None:
                row.append("—")
            else:
                row.append(f"{b2 - b1:+.1f}")
                any_v = True
        if any_v:
            L.append(f"| {c} | " + " | ".join(row) + " |")
    L.append("")

    open(args.out, "w").write("\n".join(L) + "\n")
    print("\n".join(L))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
