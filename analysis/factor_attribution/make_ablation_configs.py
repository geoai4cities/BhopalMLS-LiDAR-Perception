#!/usr/bin/env python3
"""Generate the B1/B2/B3/B4 training configs from an official Bhopal FT config.

Serves both backbones via --arch: utonia (default) or concerto. Their FT configs
are structurally identical, so the same substitutions apply to each.

Each config is a byte-copy of utonia_140gb_12class_finetune_9ch.py with ONLY:
  - a provenance header,
  - H5_PATH pointing at the ablation external-link H5,
  - (B2/B4 only) class_groups replaced by single-subclass groups so all other
    fine subclasses fall to ignore_index at load time (see bhopal.py
    _remap_labels). B2 keeps each class's dominant subtype; B4 is identical
    except Vehicle keeps Car instead of the dominant two-wheeler.

Run AFTER build_ablation_subsets.py (needs results/ablation_subsets.json).

Usage:
  python3 make_ablation_configs.py                    # utonia B1,B2,B3 (as shipped)
  python3 make_ablation_configs.py --arms B4_subtype  # utonia B4 only
  python3 make_ablation_configs.py --arch concerto --arms B1_size,B2_variety,B3_regular
"""

import argparse
import json
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]  # repository root

# The two backbones' Bhopal FT configs are structurally identical (same
# top-level assignment names, same H5_PATH line, same class_groups block, same
# three class_groups=class_groups references for train/val/test), so one
# generator serves both.
ARCHS = {
    "utonia": {
        "cfg_dir": REPO / "Utonia/Bhopal/configs/bhopal",
        "base": "utonia_140gb_12class_finetune_9ch.py",
    },
    "concerto": {
        "cfg_dir": REPO / "Concerto/Bhopal/configs/bhopal",
        "base": "concerto_140gb_12class_finetune_9ch.py",
    },
}
SUBSETS = json.load(open(Path(__file__).parent / "results/ablation_subsets.json"))

ABLATIONS = {
    "B1_size": "size-matched random subsample (25% labeled budget, seed 42)",
    "B2_variety": "variety-collapsed: only dominant fine subclass per parent "
                  "is labeled, rest -> ignore (input geometry unchanged)",
    "B3_regular": "geometry-regularized: widest/straightest corridors first",
    "B4_subtype": "subtype-swapped control: as B2, but Vehicle keeps Car (the "
                  "subtype the target benchmarks are made of) instead of the "
                  "dominant two-wheeler; separates variety from subtype match",
}
# arms whose supervision is collapsed to one fine subclass per parent, and the
# ablation_subsets.json key holding their groups
GROUPS_KEY = {"B2_variety": "b2_class_groups", "B4_subtype": "b4_class_groups"}

ap = argparse.ArgumentParser()
ap.add_argument("--arms", default="B1_size,B2_variety,B3_regular")
ap.add_argument("--arch", default="utonia", choices=sorted(ARCHS))
_args = ap.parse_args()
_want = [a.strip() for a in _args.arms.split(",") if a.strip()]
assert all(a in ABLATIONS for a in _want), f"unknown arm in {_want}"

CFG_DIR = ARCHS[_args.arch]["cfg_dir"]
BASE = CFG_DIR / ARCHS[_args.arch]["base"]
STEM = BASE.stem                      # e.g. concerto_140gb_12class_finetune_9ch
assert BASE.is_file(), f"base config not found: {BASE}"
base_text = BASE.read_text()
for name, desc in ((n, d) for n, d in ABLATIONS.items() if n in _want):
    text = base_text
    h5_rel = f"data/Bhopal/bhopal_scene_v9_abl{name}.h5"
    text, n = re.subn(r'H5_PATH = "data/Bhopal/bhopal_scene_v9_normals_rgb\.h5"',
                      f'H5_PATH = "{h5_rel}"', text)
    assert n == 1, f"H5_PATH substitution failed for {name}"
    if name in GROUPS_KEY:
        groups = SUBSETS[GROUPS_KEY[name]]
        dom = SUBSETS["dominant_fine_per_parent"]
        # Collapse supervision for TRAINING ONLY. val/test must keep the full
        # 12-class grouping, otherwise B2 is scored on a different label space
        # than B1/B3 and the ablation compares nothing.
        collapsed = ("class_groups = [\n" + "".join(
            f"    [{g[0]}],  # {pname}: single fine subclass only (train)\n"
            for g, pname in zip(groups, dom.keys())) + "]\n")

        def _keep_original_as_eval(m):
            # The original block IS the full grouping. Rename it rather than
            # discarding it: `class_groups_eval = class_groups` would reference
            # a name this substitution has just deleted (NameError at import).
            return (m.group(0).replace("class_groups = [",
                                       "class_groups_eval = [  # full grouping (val/test)", 1)
                    + collapsed)

        text, n = re.subn(r"class_groups = \[\n(?:.*\n)*?\]\n",
                          _keep_original_as_eval, text, count=1)
        assert n == 1, f"class_groups substitution failed for {name}"
        assert "class_groups_eval = [" in text and "class_groups = [" in text
        # point val and test at the untouched grouping
        head, sep, tail = text.partition('    val=dict(')
        assert sep, "val=dict( not found"
        tail = tail.replace("class_groups=class_groups,", "class_groups=class_groups_eval,")
        text = head + sep + tail
        assert text.count("class_groups=class_groups_eval,") == 2, \
            "expected exactly val and test to use the eval grouping"
    header = (f'"""Factor-attribution ablation {name}: {desc}.\n'
              f'Generated from {BASE.name}; only H5_PATH'
              f'{" and class_groups" if name in GROUPS_KEY else ""} differ.\n'
              f'Subset definition: data/Bhopal/'
              f'ablation_subsets.json\n"""\n')
    out = CFG_DIR / f"{STEM}_abl{name}.py"
    out.write_text(header + text)
    print("wrote", out)
