"""The BhopalMLS label spaces, read from label_codes.json in this folder.

Label spaces
  V13 codes 0-70   annotation codes: 0-2 unlabelled/noise, 3-8 the six moving classes, 9-70 the 62 static classes
  label62 0-61     = V13 code - 9; what the v9 benchmark file and the per-path NPZ files carry
  label68 0-67     the v10 benchmark file; 62-67 are the moving classes (V13 codes 3-8)
  benchmark_12     the 12 parent classes used in the main tables
  benchmark_21     the 21-class taxonomy of the supplementary tables (Animal, Dog and Cow are ignored)

    >>> from data.Bhopal.class_mapping.taxonomy import BENCHMARK_12, groups_to_learning_map   # from the repository root
    >>> lm = groups_to_learning_map(BENCHMARK_12["groups_label62"])   # {label62: class12}
    >>> BENCHMARK_12["names"][lm[13]]
    'Road'
"""
from __future__ import annotations

import json
from pathlib import Path

_META = Path(__file__).resolve().parent / "label_codes.json"


def load_label_codes(path: Path | str = _META) -> dict:
    with open(path) as f:
        return json.load(f)


_LC = load_label_codes()
V13_CODES = _LC["v13_codes"]              # list of {code, xx, zz, name, kind}
LABEL62 = _LC["label62"]                  # list of {label, code, xxzz, name, parent12, parent12_name}
BENCHMARK_12 = _LC["benchmark_12"]        # {names: [12], groups_label62: [[...], ...]}
BENCHMARK_21 = _LC["benchmark_21"]        # {names: [21], groups_label62: [[...], ...]}


def groups_to_learning_map(groups: list[list[int]], ignore_index: int = -1) -> dict[int, int]:
    """Pointcept-style learning map: fine label -> target class; fine labels in no group map to ignore_index."""
    lm = {i: ignore_index for i in range(len(LABEL62))}
    for target, fines in enumerate(groups):
        for f in fines:
            lm[int(f)] = target
    return lm


def palette(space: str = "12c") -> list[list[int]]:
    """RGB colours (0-255) per class for '12c', '21c' or '68c'."""
    return _LC["palettes_rgb"][space]
