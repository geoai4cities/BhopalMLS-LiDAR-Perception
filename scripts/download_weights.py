#!/usr/bin/env python3
"""Download released BhopalMLS weights (fine-tuned Utonia(L), Concerto(L), Sonata(L)) from Hugging Face into the run folders the launcher reads.

  python scripts/download_weights.py --list [utonia|concerto|sonata]     what is released
  python scripts/download_weights.py utonia_bhopal_finetune_9ch [...]    one or more runs (names as in MODEL_ZOO.md)
  python scripts/download_weights.py --all concerto                       every run of one model

Each run lands in <Model>/<Dataset>/outputs/<run>/model/model_best.pth, the file that
<Model>/Bhopal/scripts/test_bhopal.sh and <Model>/<Dataset>/scripts/test.sh evaluate by default, e.g.

  python scripts/download_weights.py utonia_bhopal_finetune_9ch && Utonia/Bhopal/scripts/test_bhopal.sh ft 9ch

On the Hub every run is one model.safetensors (model weights only) under <Dataset>/<run>/ of
geoai4cities/BhopalMLS-{Utonia,Concerto,Sonata}; this script writes it back as the {"state_dict", "epoch"}
checkpoint Pointcept's tester loads. --from-dir DIR reads local copies of the Hub repositories instead
(DIR/BhopalMLS-Utonia/..., as `hf download geoai4cities/BhopalMLS-Utonia --local-dir DIR/BhopalMLS-Utonia` makes them).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ORG = "geoai4cities"
REPOS = {"Utonia": "BhopalMLS-Utonia", "Concerto": "BhopalMLS-Concerto", "Sonata": "BhopalMLS-Sonata"}
ROOT = Path(__file__).resolve().parents[1]


def catalogue(models: list[str], from_dir: Path | None) -> dict[str, tuple[str, str, str, int]]:
    """run -> (model folder, dataset folder, path in the Hub repo, bytes)"""
    runs = {}
    for arch in models:
        name = REPOS[arch]
        if from_dir:
            base = from_dir / name
            files = [(p.relative_to(base).as_posix(), p.stat().st_size) for p in base.glob("*/*/model.safetensors")]
        else:
            from huggingface_hub import HfApi
            files = [(f.path, f.size) for f in HfApi().list_repo_tree(f"{ORG}/{name}", recursive=True)
                     if f.path.endswith("/model.safetensors")]
        for path, size in files:
            dataset, run, _ = path.split("/")
            runs[run] = (arch, dataset, path, size)
    return runs


def fetch(run: str, entry: tuple[str, str, str, int], from_dir: Path | None, force: bool) -> Path:
    import torch
    from safetensors import safe_open
    from safetensors.torch import load_file

    arch, dataset, path, _ = entry
    out = ROOT / arch / dataset / "outputs" / run / "model" / "model_best.pth"
    if out.exists() and not force:
        print(f"exists, skipped (--force to replace): {out.relative_to(ROOT)}")
        return out
    if from_dir:
        src = str(from_dir / REPOS[arch] / path)
    else:
        from huggingface_hub import hf_hub_download
        src = hf_hub_download(f"{ORG}/{REPOS[arch]}", path)
    with safe_open(src, framework="pt") as f:
        meta = f.metadata() or {}
    epoch = int(meta["epoch"]) if meta.get("epoch", "").isdigit() else -1
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".pth.part")
    torch.save({"state_dict": load_file(src), "epoch": epoch}, tmp)
    tmp.replace(out)
    print(f"{run}: {out.relative_to(ROOT)}  (config {meta.get('config', '?')})")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="*", help="run names, e.g. utonia_bhopal_finetune_9ch")
    ap.add_argument("--list", nargs="?", const="all", metavar="MODEL", help="list the released runs (utonia, concerto, sonata)")
    ap.add_argument("--all", metavar="MODEL", help="download every run of one model (utonia, concerto, sonata)")
    ap.add_argument("--from-dir", type=Path, help="read local copies of the Hub repositories instead of downloading")
    ap.add_argument("--force", action="store_true", help="replace an existing model_best.pth")
    a = ap.parse_args()

    def models(sel: str | None) -> list[str]:
        if sel in (None, "all"):
            return list(REPOS)
        m = {k.lower(): k for k in REPOS}.get(sel.lower())
        if not m:
            ap.error(f"unknown model '{sel}' (utonia, concerto, sonata)")
        return [m]

    if a.list:
        runs = catalogue(models(a.list), a.from_dir)
        for run, (arch, dataset, _, size) in sorted(runs.items(), key=lambda kv: (kv[1][0], kv[1][1], kv[0])):
            print(f"{arch:9s} {dataset:15s} {run:48s} {size / 1e9:5.2f} GB")
        print(f"{len(runs)} runs")
        return 0
    if not a.runs and not a.all:
        ap.print_help()
        return 1
    wanted = models(a.all) if a.all else sorted({arch for r in a.runs for arch in REPOS if r.startswith(arch.lower() + "_")}) or list(REPOS)
    runs = catalogue(wanted, a.from_dir)
    names = sorted(r for r, e in runs.items() if e[0] in models(a.all)) if a.all else a.runs
    missing = [r for r in names if r not in runs]
    if missing:
        print(f"not released: {', '.join(missing)}  (see --list and MODEL_ZOO.md)")
        return 1
    for r in names:
        fetch(r, runs[r], a.from_dir, a.force)
    return 0


if __name__ == "__main__":
    sys.exit(main())
