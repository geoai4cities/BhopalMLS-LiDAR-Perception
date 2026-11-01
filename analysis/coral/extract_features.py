#!/usr/bin/env python3
"""
Phase 4 — CORAL DA: Stage 1 — Feature extraction.

Loads a frozen FT-9ch encoder (Concerto or Utonia) and extracts 64-dim
penultimate-layer features on a configurable split: BM train, PL train, or PL val.
Features + labels are cached to NPZ for downstream CORAL alignment + linear-head training.

Runs on CPU (the A100 host 40GB GPU is busy with Phase 3 ablations).

Per-scene: random subsample to --max-pts points (default 2000) to control wall-clock and
to keep total feature count tractable for sklearn LogisticRegression in stage 2.

Outputs:
    results/{arch}/features_{split}.npz
        features  [N, 64]   float32
        labels    [N]       int64    (BM-12 native; remap to nc=11 in stage 2)
        scene_idx [N]       int32    (per-point scene index in source dataset)
"""
import argparse
import os
import sys
import time

import numpy as np
import torch

# Resolve paths relative to this script (portable across servers)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))  # repository root
sys.path.insert(0, os.path.join(REPO_ROOT, "Pointcept"))
# Datasets and configs reference paths relative to REPO_ROOT
# (e.g. "data/Bhopal/bhopal_scene_v9_normals_rgb.h5"); cd there so they resolve.
os.chdir(REPO_ROOT)

from pointcept.utils.config import Config
from pointcept.datasets import build_dataset, collate_fn
from pointcept.models import build_model

# ----- Architecture-specific config + checkpoint paths -------------------------------------------
ARCH_SPEC = {
    "concerto": {
        "config": "Concerto/Bhopal/configs/bhopal/concerto_140gb_12class_finetune_9ch.py",
        "ckpt":   "Concerto/Bhopal/outputs/concerto_bhopal_finetune_9ch_indoor/model/model_best.pth",
        "pl_config": "Concerto/Paris-Lille-3D/configs/paris_lille/concerto_140gb_12class_finetune_9ch.py",
    },
    "utonia": {
        "config": "Utonia/Bhopal/configs/bhopal/utonia_140gb_12class_finetune_9ch.py",
        "ckpt":   "Utonia/Bhopal/outputs/utonia_bhopal_finetune_9ch/model/model_best.pth",
        "pl_config": "Utonia/Paris-Lille-3D/configs/paris_lille/utonia_140gb_12class_finetune_9ch.py",
    },
}

SPLIT_SPEC = {
    "bm_train":  {"data_section": "val", "split_override": "train", "config_key": "config"},
    "pl_train":  {"data_section": "val", "split_override": "train", "config_key": "pl_config"},
    "pl_val":    {"data_section": "val", "split_override": "val",   "config_key": "pl_config"},
}


def load_state_dict_strip_module(ckpt_path):
    """Load a checkpoint and strip 'module.' prefix from keys (for non-DDP loading)."""
    ckpt = torch.load(ckpt_path, weights_only=False, map_location="cpu")
    state = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
    cleaned = {}
    for k, v in state.items():
        if k.startswith("module."):
            k = k[len("module."):]
        cleaned[k] = v
    epoch = ckpt.get("epoch", "?")
    return cleaned, epoch


def build_model_on(cfg_model_dict, device):
    """Build the model from config dict and put on `device` ('cuda' or 'cpu')."""
    model = build_model(cfg_model_dict)
    return model.to(device).eval()


def make_dataset(arch, split):
    """Construct a dataset using the val transform pipeline + the requested split."""
    spec = SPLIT_SPEC[split]
    cfg_path = os.path.join(REPO_ROOT, ARCH_SPEC[arch][spec["config_key"]])
    if not os.path.isfile(cfg_path):
        raise FileNotFoundError(f"Config not found: {cfg_path}")
    cfg = Config.fromfile(cfg_path)

    # Use val data section as the "clean" pipeline (deterministic, no train-time aug).
    # Override the split to whatever we need.
    data_cfg = cfg.data[spec["data_section"]]
    data_cfg = dict(data_cfg)  # shallow copy
    data_cfg["split"] = spec["split_override"]
    data_cfg["test_mode"] = False  # train-style getitem (single dict, no fragments)

    dataset = build_dataset(data_cfg)
    return dataset, cfg


def extract_for_split(arch, split, max_pts, smoke_n=None, log_every=25, out_path=None,
                      ckpt_path=None, device="cuda"):
    print(f"\n{'='*80}\n[{arch}/{split}] feature extraction (device={device})\n{'='*80}")

    dataset, cfg = make_dataset(arch, split)
    print(f"  dataset size: {len(dataset)}")
    if smoke_n is not None:
        print(f"  SMOKE TEST: limiting to first {smoke_n} scenes")

    model = build_model_on(cfg.model, device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  model: {cfg.model['type']}, n_params={n_params:,}")

    state, epoch = load_state_dict_strip_module(ckpt_path)
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"  loaded ckpt (epoch={epoch}); missing={len(missing)} unexpected={len(unexpected)}")
    if len(missing) > 0:
        print(f"    first 5 missing keys: {missing[:5]}")
    if len(unexpected) > 0:
        print(f"    first 5 unexpected keys: {unexpected[:5]}")

    n_total_scenes = len(dataset) if smoke_n is None else min(len(dataset), smoke_n)

    all_feat, all_label, all_scene = [], [], []
    t0 = time.time()
    rng = np.random.default_rng(42)

    for idx in range(n_total_scenes):
        try:
            data_dict = dataset[idx]
        except Exception as e:
            print(f"  [{idx}] skip — dataset error: {type(e).__name__}: {e}")
            continue
        if isinstance(data_dict, list):
            # test_mode=True returns a list of fragments; we asked for test_mode=False so this
            # shouldn't happen, but guard anyway.
            data_dict = data_dict[0]

        # Collate single sample → batched input_dict (offset, etc.)
        input_dict = collate_fn([data_dict])
        for k, v in list(input_dict.items()):
            if isinstance(v, torch.Tensor):
                input_dict[k] = v.to(device, non_blocking=True)

        try:
            with torch.no_grad():
                out = model(input_dict, return_point=True)
        except torch.cuda.OutOfMemoryError as e:
            print(f"  [{idx}] OOM, skipping ({e})")
            torch.cuda.empty_cache() if device == "cuda" else None
            continue
        except Exception as e:
            print(f"  [{idx}] skip — forward error: {type(e).__name__}: {e}")
            continue

        if "point" not in out:
            print(f"  [{idx}] skip — no point in output")
            continue
        feat = out["point"].feat.cpu().numpy()  # [N_pts, 64]

        # Labels: voxelized labels from data_dict['segment'] (already in BM-12 ids for BM dataset,
        # in nc=11 space for PL with mapping); we keep BM-12 native here and remap in stage 2.
        if "segment" in data_dict and isinstance(data_dict["segment"], torch.Tensor):
            labels = data_dict["segment"].cpu().numpy().astype(np.int64)
        else:
            labels = np.full(feat.shape[0], -1, dtype=np.int64)

        if labels.shape[0] != feat.shape[0]:
            # Voxelization may have shrunk labels; align to feat
            n = min(labels.shape[0], feat.shape[0])
            feat, labels = feat[:n], labels[:n]

        # Random subsample max_pts points
        if feat.shape[0] > max_pts:
            sel = rng.choice(feat.shape[0], max_pts, replace=False)
            feat, labels = feat[sel], labels[sel]

        all_feat.append(feat.astype(np.float32))
        all_label.append(labels)
        all_scene.append(np.full(feat.shape[0], idx, dtype=np.int32))

        if (idx + 1) % log_every == 0 or idx == n_total_scenes - 1:
            elapsed = time.time() - t0
            rate = (idx + 1) / elapsed if elapsed > 0 else 0
            eta = (n_total_scenes - idx - 1) / rate if rate > 0 else 0
            n_acc = sum(f.shape[0] for f in all_feat)
            print(f"  [{idx+1}/{n_total_scenes}] feat-pts={n_acc:,}  ({rate:.2f} scene/s, ETA {eta/60:.1f}m)")

    if not all_feat:
        print("  ERROR: no features extracted")
        sys.exit(2)

    features = np.concatenate(all_feat, axis=0)
    labels = np.concatenate(all_label, axis=0)
    scene_idx = np.concatenate(all_scene, axis=0)

    elapsed = time.time() - t0
    print(f"\nDONE: {features.shape[0]:,} feat-vectors, {features.shape[1]} dim, {elapsed/60:.1f} min total")
    print(f"  labels: range [{labels.min()}, {labels.max()}], unique={np.unique(labels).tolist()[:15]}")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.savez_compressed(out_path, features=features, labels=labels, scene_idx=scene_idx)
    print(f"  wrote: {out_path}  ({os.path.getsize(out_path)/1e9:.2f} GB)")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arch", required=True, choices=["concerto", "utonia"])
    parser.add_argument("--split", required=True, choices=["bm_train", "pl_train", "pl_val"])
    parser.add_argument("--max-pts", type=int, default=2000,
                        help="Random subsample per scene (default 2000)")
    parser.add_argument("--smoke", type=int, default=None,
                        help="Smoke test: only process this many scenes")
    parser.add_argument("--out-dir", default=os.path.join(SCRIPT_DIR, "results"))
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"],
                        help="GPU required for spconv-based PT-v3m2 (CPU not supported by spconv).")
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        print("ERROR: --device cuda requested but no CUDA device visible.")
        print("       Set CUDA_VISIBLE_DEVICES to a free GPU index.")
        sys.exit(1)

    ckpt_path = os.path.join(REPO_ROOT, ARCH_SPEC[args.arch]["ckpt"])
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    out_path = os.path.join(args.out_dir, args.arch, f"features_{args.split}.npz")
    extract_for_split(
        arch=args.arch,
        split=args.split,
        max_pts=args.max_pts,
        smoke_n=args.smoke,
        log_every=args.log_every,
        out_path=out_path,
        ckpt_path=ckpt_path,
        device=args.device,
    )


if __name__ == "__main__":
    main()
