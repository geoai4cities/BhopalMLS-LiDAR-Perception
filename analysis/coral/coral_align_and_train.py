#!/usr/bin/env python3
"""
Phase 4 — CORAL DA: Stage 2 — Compute CORAL transform + train linear seg-head.

Reads cached source/target features from stage 1, computes the closed-form
CORAL alignment matrix (Sun et al. 2016), applies it to source features,
and trains a sklearn LogisticRegression on the aligned features.

Outputs (per arch):
    results/{arch}/coral_transform.npz   ← 64×64 alignment matrix A and metadata
    results/{arch}/coral_head.pkl        ← trained sklearn LogisticRegression (joblib)
    results/{arch}/identity_head.pkl     ← reproducibility check: head trained without CORAL (A=I)
"""
import argparse
import os
import sys
import time

import joblib
import numpy as np
from scipy.linalg import sqrtm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# BM-12 → nc=11 (PL native): drop class 6 (Wire-Structure Connector)
# Source mapping from `paris_lille.py:202` — class 6 maps to -1 (absent in PL).
BM12_TO_NC11 = np.array([0, 1, 2, 3, 4, 5, -1, 6, 7, 8, 9, 10], dtype=np.int64)
NC11_NAMES = ["Ground", "Vegetation", "Building", "Road", "Wire", "Pole&Tower",
              "Vehicle", "Board", "Fencing", "Human&Animal", "Road-Side-Obj"]


def matrix_sqrt_symmetric(M, eps=1e-5):
    """Compute matrix square root of a symmetric PSD matrix using eigendecomposition.
    More numerically stable than scipy.linalg.sqrtm for SPD matrices."""
    M = (M + M.T) / 2.0
    w, V = np.linalg.eigh(M)
    w = np.clip(w, eps, None)
    return V @ np.diag(np.sqrt(w)) @ V.T


def matrix_inv_sqrt_symmetric(M, eps=1e-5):
    """Inverse of matrix_sqrt_symmetric."""
    M = (M + M.T) / 2.0
    w, V = np.linalg.eigh(M)
    w = np.clip(w, eps, None)
    return V @ np.diag(1.0 / np.sqrt(w)) @ V.T


def coral_transform(F_S, F_T, eps=1e-5):
    """Closed-form CORAL alignment matrix A: x' = x @ A whitens source then recolors with target.

    Following Sun et al. 2016 (Eq. 4) but simplified via eigendecomposition for symmetric PSDs.
    """
    C_S = np.cov(F_S, rowvar=False).astype(np.float64) + eps * np.eye(F_S.shape[1])
    C_T = np.cov(F_T, rowvar=False).astype(np.float64) + eps * np.eye(F_T.shape[1])
    inv_sqrt_C_S = matrix_inv_sqrt_symmetric(C_S, eps=eps)
    sqrt_C_T = matrix_sqrt_symmetric(C_T, eps=eps)
    A = inv_sqrt_C_S @ sqrt_C_T
    return A, C_S, C_T


def remap_bm12_to_nc11(labels_bm12):
    """Remap BM-12 labels to nc=11 (drop class 6 → -1)."""
    out = np.full_like(labels_bm12, -1, dtype=np.int64)
    valid = (labels_bm12 >= 0) & (labels_bm12 < len(BM12_TO_NC11))
    out[valid] = BM12_TO_NC11[labels_bm12[valid]]
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arch", required=True, choices=["concerto", "utonia"])
    parser.add_argument("--out-dir", default=os.path.join(SCRIPT_DIR, "results"))
    parser.add_argument("--max-iter", type=int, default=2000)
    parser.add_argument("--solver", default="lbfgs", choices=["lbfgs", "sag", "saga"])
    parser.add_argument("--class-weight", default="balanced",
                        choices=["balanced", "none"])
    parser.add_argument("--max-train-samples", type=int, default=2_000_000,
                        help="Cap source training samples for sklearn (default 2M)")
    parser.add_argument("--skip-identity", action="store_true",
                        help="Skip the A=identity reproducibility check")
    args = parser.parse_args()

    arch_dir = os.path.join(args.out_dir, args.arch)
    bm_path = os.path.join(arch_dir, "features_bm_train.npz")
    pl_path = os.path.join(arch_dir, "features_pl_train.npz")
    for p in [bm_path, pl_path]:
        if not os.path.isfile(p):
            raise FileNotFoundError(f"Missing stage-1 output: {p}")

    print(f"=== Loading features ({args.arch}) ===")
    bm = np.load(bm_path)
    pl = np.load(pl_path)
    F_S = bm["features"].astype(np.float64)  # CORAL needs float64 stability
    F_T = pl["features"].astype(np.float64)
    labels_S_bm12 = bm["labels"]
    print(f"  F_S (BM): {F_S.shape}  labels range [{labels_S_bm12.min()}, {labels_S_bm12.max()}]")
    print(f"  F_T (PL): {F_T.shape}")

    # CORAL transform
    print(f"\n=== Computing CORAL transform ({args.arch}) ===")
    t0 = time.time()
    A, C_S, C_T = coral_transform(F_S, F_T)
    t_coral = time.time() - t0
    print(f"  A.shape = {A.shape}  ({t_coral:.1f}s)")

    # Sanity: cov(F_S @ A) should match cov(F_T)
    F_S_aligned = F_S @ A
    C_aligned = np.cov(F_S_aligned, rowvar=False)
    delta = np.linalg.norm(C_aligned - np.cov(F_T, rowvar=False), ord="fro")
    print(f"  ||cov(F_S @ A) - cov(F_T)||_F = {delta:.4e}  (should be small, ideally O(eps))")

    # Save transform + metadata
    transform_path = os.path.join(arch_dir, "coral_transform.npz")
    np.savez(transform_path,
             A=A.astype(np.float32),
             cov_source=C_S.astype(np.float32),
             cov_target=C_T.astype(np.float32),
             cov_aligned=C_aligned.astype(np.float32))
    print(f"  wrote: {transform_path}")

    # Remap labels to nc=11
    labels_S_nc11 = remap_bm12_to_nc11(labels_S_bm12)
    keep = labels_S_nc11 >= 0
    F_S_aligned_keep = F_S_aligned[keep]
    labels_S_nc11_keep = labels_S_nc11[keep]
    print(f"\n  After nc=11 remap + ignore filter: {F_S_aligned_keep.shape[0]:,} train samples")

    # Optional cap for sklearn fit speed
    if F_S_aligned_keep.shape[0] > args.max_train_samples:
        rng = np.random.default_rng(123)
        sel = rng.choice(F_S_aligned_keep.shape[0], args.max_train_samples, replace=False)
        F_S_aligned_keep = F_S_aligned_keep[sel]
        labels_S_nc11_keep = labels_S_nc11_keep[sel]
        print(f"  Subsampled to {args.max_train_samples:,} for sklearn fit")

    # Train CORAL-aligned linear head
    print(f"\n=== Training linear head on CORAL-aligned features ({args.arch}) ===")
    from sklearn.linear_model import LogisticRegression
    cw = "balanced" if args.class_weight == "balanced" else None
    clf_coral = LogisticRegression(
        solver=args.solver,
        max_iter=args.max_iter, class_weight=cw, C=1.0,
    )
    t0 = time.time()
    clf_coral.fit(F_S_aligned_keep.astype(np.float32), labels_S_nc11_keep)
    print(f"  Fit done in {(time.time()-t0)/60:.1f} min")
    print(f"  classes seen: {clf_coral.classes_.tolist()}")
    coral_head_path = os.path.join(arch_dir, "coral_head.pkl")
    joblib.dump(clf_coral, coral_head_path)
    print(f"  wrote: {coral_head_path}")

    # Reproducibility check: A=identity → should reproduce vanilla baseline
    if not args.skip_identity:
        print(f"\n=== Training identity-head (A=I) for reproducibility ({args.arch}) ===")
        F_S_keep = F_S[keep]
        if F_S_keep.shape[0] > args.max_train_samples:
            rng = np.random.default_rng(123)
            sel = rng.choice(F_S_keep.shape[0], args.max_train_samples, replace=False)
            F_S_keep = F_S_keep[sel]
        clf_id = LogisticRegression(
            solver=args.solver,
            max_iter=args.max_iter, class_weight=cw, C=1.0,
        )
        t0 = time.time()
        clf_id.fit(F_S_keep.astype(np.float32), labels_S_nc11_keep[: F_S_keep.shape[0]]
                   if F_S_keep.shape[0] != labels_S_nc11_keep.shape[0] else labels_S_nc11_keep)
        print(f"  Fit done in {(time.time()-t0)/60:.1f} min")
        joblib.dump(clf_id, os.path.join(arch_dir, "identity_head.pkl"))
        print(f"  wrote: {os.path.join(arch_dir, 'identity_head.pkl')}")

    print(f"\n[{args.arch}] Stage 2 DONE.")


if __name__ == "__main__":
    main()
