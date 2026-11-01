#!/usr/bin/env python3
"""Summarize rescore reports: full/disjoint/overlapping table + agreement stats."""

import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

RES = Path(__file__).parent / "results"

lines = []
for tag, fname in [("12-class", "rescore_report.json"),
                   ("21-class", "rescore_report_21c.json")]:
    r = json.load(open(RES / fname))
    rows = [(k, v["full"]["mIoU"], v["disjoint"]["mIoU"], v["overlapping"]["mIoU"])
            for k, v in r.items()]
    rows.sort(key=lambda x: -x[1])
    full = np.array([x[1] for x in rows])
    dis = np.array([x[2] for x in rows])
    ovl = np.array([x[3] for x in rows])
    delta = dis - ovl
    rho_fd = spearmanr(full, dis).statistic
    rho_fo = spearmanr(full, ovl).statistic

    lines.append(f"\n## {tag} ({len(rows)} models)\n")
    lines.append("| model | full | disjoint | overlapping | Δ (disj−ovl) |")
    lines.append("|---|---|---|---|---|")
    for k, f_, d_, o_ in rows:
        lines.append(f"| {k} | {f_:.2f} | {d_:.2f} | {o_:.2f} | {d_-o_:+.2f} |")
    lines.append(f"\n- mean Δ (disjoint − overlapping): **{delta.mean():+.2f} pp** "
                 f"(range {delta.min():+.2f} … {delta.max():+.2f})")
    lines.append(f"- Spearman rank corr, full vs disjoint: **{rho_fd:.3f}**; "
                 f"full vs overlapping: {rho_fo:.3f}")
    lines.append(f"- mean |full − disjoint|: {np.abs(full-dis).mean():.2f} pp")

out = "\n".join(["# exp_02 summary — leakage-subset re-scoring, full zoo"] + lines) + "\n"
(RES / "summary_table.md").write_text(out)
print(out)
