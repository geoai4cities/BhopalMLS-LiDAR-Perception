# Phase 4 — CORAL DA baseline (R3-M4)

Closed-form (post-hoc) CORAL Domain Adaptation on **BM → PL**, applied in feature space using a frozen FT-9ch encoder + retrained linear seg-head. CPU-only.

Sun et al. AAAI 2016, *"Return of Frustratingly Easy Domain Adaptation"*.

## Why post-hoc CORAL (not deep CORAL)

Deep CORAL needs gradient descent through PT-v3m2-sonata — infeasible on CPU. Post-hoc CORAL extracts features once with the frozen encoder, computes a 64×64 covariance-alignment transform, and trains a tiny linear head — all CPU-feasible (~5-6 hr total).

## Pipeline (3 stages × 2 archs)

```
extract_features.py    → cache (features, labels) per (arch, split)
       │
       ▼
coral_align_and_train.py → compute 64×64 transform A; train sklearn LogisticRegression head
       │
       ▼
eval_coral.py          → predict on PL val features; produce supp .tex
```

## Files

| File | Purpose |
|---|---|
| `extract_features.py` | Forward frozen encoder on subsampled scenes; cache 64-dim features + labels |
| `coral_align_and_train.py` | Compute CORAL transform; train linear head on transformed source |
| `eval_coral.py` | Evaluate trained head on PL val; emit supp.tex fragment |
| `run_all.sh` | Orchestrate stages 1-3 for both archs end-to-end |

## Usage

```bash
cd analysis/coral

# Smoke test (5 scenes per split)
./run_all.sh --smoke

# Full run (all scenes × 2K points/scene; ~5-6 hr CPU)
./run_all.sh
```

Or run individual stages:
```bash
python extract_features.py --arch concerto --split bm_train --max-pts 2000
python extract_features.py --arch concerto --split pl_train --max-pts 2000
python extract_features.py --arch concerto --split pl_val   --max-pts 2000
python coral_align_and_train.py --arch concerto
python eval_coral.py --arch concerto

# Repeat for utonia
```

## Outputs

```
results/
├── concerto/
│   ├── features_bm_train.npz       # ~2.74M × 64 features
│   ├── features_pl_train.npz       # ~5.67M × 64 features
│   ├── features_pl_val.npz         # PL val features
│   ├── coral_transform.npz         # 64×64 alignment matrix A
│   ├── coral_head.pkl              # sklearn LogisticRegression
│   └── eval_metrics.json           # per-class IoU + summary mIoU
├── utonia/
│   └── (same structure)
└── coral_da_supp.tex               # final supp table fragment
```

## Method (math)

Given source features $F_S \in \mathbb{R}^{N_S \times 64}$ (BM, with labels) and target features $F_T \in \mathbb{R}^{N_T \times 64}$ (PL, no labels needed):

1. Source covariance: $C_S = \mathrm{cov}(F_S) + \epsilon I$
2. Target covariance: $C_T = \mathrm{cov}(F_T) + \epsilon I$
3. Whitening + recoloring: $A = C_S^{-1/2} C_T^{1/2}$  ($\mathbb{R}^{64\times 64}$)
4. Apply: $F_S^{\text{aligned}} = F_S A$
5. Train linear classifier: $W: \mathbb{R}^{64} \to \mathbb{R}^{11}$ on $(F_S^{\text{aligned}}, y_S^{\text{nc11}})$
6. Evaluate: $\hat{y}_T = \mathrm{argmax}(F_T^{\text{val}} W^\top)$

**Note on label space:** BM-12 labels are remapped to PL's nc=11 (drop class 6, Wire-Structure Connector — absent in PL).

## Baselines (vanilla BM→PL — what we beat)

From `RESULTS.md §E.5`:
- **Concerto FT-9ch:** 48.84 (nc=11) / 60.70 (common-7)
- **Utonia FT-9ch:**   55.68 (nc=11) / 65.70 (common-7)

## Constraints

- CPU only (the A100 host's GPU is busy with Phase 3 ablations)
- Frozen FT-9ch encoder; only the seg head retrains
- Subsample 2K points/scene (for stage 1 wall-clock)

## Risks acknowledged in supp

If $\Delta \le 0$ (CORAL doesn't help): report honestly. *"2nd-order alignment alone insufficient; the BM↔PL asymmetry has structural causes beyond covariance shift."* This actually strengthens R1-W2.

If LogisticRegression underperforms: optionally re-train with sklearn's SGDClassifier (streaming, faster) or initialize from existing seg head weights.
