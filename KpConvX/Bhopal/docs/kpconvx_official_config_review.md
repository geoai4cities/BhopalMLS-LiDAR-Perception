# KpConvX Official Configuration Review

**Date**: 2026-02-12
**Purpose**: Audit of our Bhopal Pointcept KpConvX configs against official Apple ml-kpconvx sources and GitHub issues.

---

## Sources Reviewed

### Code (local archive: `ml-kpconvx/`)

| # | Source | Path | Dataset | Notes |
|---|--------|------|---------|-------|
| 1 | Standalone Bhopal | `Standalone/KPConvX/experiments/S3DIS/train_S3DIS.py (adapted)` | Bhopal MLS (outdoor) | Direct reference for our dataset |
| 2 | Standalone S3DIS | `Standalone/KPConvX/experiments/S3DIS/train_S3DIS.py` | S3DIS (indoor) | Most complete official config with comments |
| 3 | Standalone ScanObjectNN | `Standalone/KPConvX/experiments/ScanObjectNN/train_ScanObj.py` | ScanObjectNN (classification) | Object classification variant |
| 4 | Pointcept-wrapper ScanNet | `Pointcept-wrapper/configs/scannet/semseg-kpconvx-base.py` | ScanNet (indoor) | Only official Pointcept config |
| 5 | Master Defaults | `Standalone/KPConvX/utils/config.py` | N/A | All default values |

### GitHub Issues (https://github.com/apple/ml-kpconvx/issues)

| # | Issue | Title | Relevance |
|---|-------|-------|-----------|
| #1 | [SensatUrban](https://github.com/apple/ml-kpconvx/issues/1) | Using large dataset such as SensatUrban | Outdoor dataset config, author response with tile_size recommendations |
| #8 | [Neighbors](https://github.com/apple/ml-kpconvx/issues/8) | neighbors limit | Paper's rationale for (12,16,20,20,20) |
| #9 | [Imbalance](https://github.com/apple/ml-kpconvx/issues/9) | Imbalance datasets tackling | Potential bug in validation metric calculation |
| #14 | [RAM](https://github.com/apple/ml-kpconvx/issues/14) | RAM limits for big datasets | Tiling + custom loader for large datasets |

---

## Complete Official Hyperparameter Reference

### Model Architecture

| Parameter | Default (`config.py`) | S3DIS (standalone) | ScanObjectNN (standalone) | | ScanNet (Pointcept) |
|-----------|----------------------|---------------------|--------------------------|---------------------|---------------------|
| layer_blocks | (2, 1, 1) | **(3, 3, 9, 12, 3)** | (3, 3, 9, 12, 3) | (3, 3, 9, 12, 3) | (3, 3, 9, 12, 3) |
| init_channels | 64 | **64** | 64 | 64 | 64 |
| channel_scaling | sqrt(2) | **1.41** | 1.41 | 1.41 | 1.414 |
| kp_mode | 'kpconv' | **'kpconvx'** | 'kpconvd' | 'kpconvx' | 'kpconvx' |
| shell_sizes | [15] | **[1, 14, 28]** | [1, 14, 28] | [1, 14, 28] | (1, 14, 28) |
| kp_radius | 2.5 | **2.1** | 2.1 | 2.1 | **2.3** |
| kp_sigma | 1.0 | **2.1** (auto=kp_radius) | 2.1 | 2.1 | **2.3** |
| kp_influence | 'linear' | **'linear'** | 'linear' | 'linear' | **'constant'** |
| kp_aggregation | 'sum' | **'nearest'** | 'nearest' | 'nearest' | 'nearest' |
| share_kp | False | **True** | True | True | **False** |
| conv_groups | 1 | **-1** | -1 | -1 | -1 |
| radius_scaling | 2.0 | **2.2** | 2.0 | 2.2 | 2.2 |
| neighbor_limits | [] (calibrated) | **[12, 16, 20, 20, 20]** | [12, 16, 20, 20, 20] | [12, 16, 20, 20, 20] | (12, 16, 20, 20, 20) |
| drop_path_rate | 0.0 | **0.3** | 0.3 | 0.3 | 0.3 |
| grid_pool | False | **True** | True | True | True |
| decoder_layer | False | **True** | True | True | True |
| upsample_n | 1 | **3** | 3 | 3 | 3 |
| first_inv_layer | 0 | **1** | 1 | 1 | 1 |
| inv_groups | 1 | **4** | 8 | **8** | 8 |
| inv_grp_norm | False | **True** | True | True | True |
| inv_act | 'sigmoid' | **'sigmoid'** | 'sigmoid' | 'sigmoid' | 'sigmoid' |
| kpx_upcut | False | **False** | False | False | False |
| norm | 'batch' | **'batch'** | 'batch' | 'batch' | 'batch' |
| bn_momentum | 0.1 | **0.1** | 0.1 | 0.1 | 0.1 |
| smooth_labels | False | **False** | **True** | False | False |
| input_channels | 1 | **5** (XYZ+RGB) | 4 | **2** (XYZ+intensity) | **9** (XYZ+RGB+normal) |

> **Key observation**: `init_channels=64` and `neighbor_limits=(12, 16, 20, 20, 20)` are FIXED across every single official config. The paper states: *"In all our experiments, we use [12, 16, 20, 20, 20] neighbors for each layer."*

### Subsampling / Grid

| Parameter | Default | S3DIS | ScanObjectNN | | ScanNet (Pointcept) |
|-----------|---------|-------|--------------|---------------------|---------------------|
| subsample_size (in_sub_size) | 0.04 | **0.04** | 0.019 | **0.04** | **0.02** |
| init_sub_size | 0.02 | **0.04** | -1 (none) | **0.04** | N/A (GridSample) |
| GridSample grid_size | N/A | N/A | N/A | N/A | **0.02** |

> Note: In the official Pointcept ScanNet config, `GridSample grid_size = 0.02` matches `subsample_size = 0.02`. They are equal.

### Training

| Parameter | Default | S3DIS | ScanObjectNN | | ScanNet (Pointcept) |
|-----------|---------|-------|--------------|---------------------|---------------------|
| optimizer | SGD | **AdamW** | AdamW | **AdamW** | **AdamW** |
| lr (max) | 1e-2 | **5e-3** | 5e-3 | **5e-3** | **0.005** |
| weight_decay | 1e-2 | **0.05** | 0.01 | **0.05** | **0.02** |
| batch_size | 16 | **4** | 32 | **16** | **2** (example) |
| accum_batch | 1 | **6** | 2 | **2** | N/A |
| effective_batch | 16 | **24** | 64 | **32** | N/A |
| max_epoch | 300 | **450** | 250 | **450** | **1000** |
| steps_per_epoch | 1000 | **300** | 1000 | **300** | N/A |

### Learning Rate Schedule

| Parameter | S3DIS | ScanObjectNN | | ScanNet (Pointcept) |
|-----------|-------|--------------|---------------------|---------------------|
| Schedule type | Cyclic | Cyclic | Cyclic | **OneCycleLR** |
| cyc_lr0 (min) | 1e-4 | 5e-4 | 1e-4 | div_factor=100 -> 5e-5 |
| cyc_lr1 (max) | 5e-3 | 5e-3 | 5e-3 | max_lr=0.005 |
| cyc_raise_n | 30 | 1 | 30 | pct_start=0.05 |
| cyc_decrease10 | 120 | 100 | 120 | anneal_strategy='cos' |
| cyc_plateau | 5 | 1 | 5 | final_div_factor=1000 |

> The standalone uses a custom cyclic LR. The Pointcept-wrapper uses PyTorch's OneCycleLR. Both achieve similar shapes.

### Augmentations

| Transform | S3DIS | | ScanNet (Pointcept) |
|-----------|-------|---------------------|---------------------|
| Anisotropic scale | [0.9, 1.1] | [0.9, 1.1] | [0.9, 1.1] |
| Flips | [0.5, 0, 0] | [0.5, 0, 0] | p=0.5 |
| Rotations | 'vertical' | 'vertical' | Z-axis |
| Jitter | 0.005 | 0.005 | sigma=0.005, clip=0.02 |
| Color drop | 0.2 | 0.2 | N/A |
| Chromatic contrast | True | **False** | p=0.2 |
| Chromatic norm | True | **False** | N/A |
| ElasticDistortion | N/A | N/A | [[0.2,0.4],[0.8,1.6]] |
| RandomDropout | N/A | N/A | ratio=0.2, app_ratio=0.2 |
| ShufflePoint | N/A | N/A | Yes |
| mix3D | -1.0 (disabled) | -1.0 (disabled) | mix_prob=0.8 |

### Test-Time Augmentation (TTA)

| Source | Augmentations | Count |
|--------|--------------|-------|
| ScanNet (Pointcept, official) | 4 rotations + 4 rot*scale(0.95) + 4 rot*scale(1.05) + 1 flip | **13** |
| Standalone S3DIS/Bhopal | Vertical rotation + slight scale | Few |

---

## Discrepancy Analysis: Our Config vs Official

### CRITICAL (Must Fix)

#### 1. neighbor_limits: `(16, 20, 24, 24, 24)` -> `(12, 16, 20, 20, 20)`

**Severity**: HIGH

Our config inflated neighbor limits for "140GB GPU", but the official value is **fixed at (12, 16, 20, 20, 20) across ALL datasets and GPU sizes**. The paper explicitly states this. Higher neighbor limits change the effective receptive field and may hurt performance since the model architecture was designed and tested with these specific values.

The S3DIS config even shows commented alternatives that were tried and rejected:
```python
# cfg.model.neighbor_limits = [10, 12, 12, 12, 12]
# cfg.model.neighbor_limits = [12, 14, 16, 16, 16]
# cfg.model.neighbor_limits = [16, 17, 18, 18, 18]
# cfg.model.neighbor_limits = [35, 40, 50, 50, 50]
cfg.model.neighbor_limits = [12, 16, 20, 20, 20]   # <-- chosen value
```

Irony: Our 40GB script overrides to `(12, 16, 20, 20, 20)` which is actually the correct official value.

**Paper rationale (Issue #8)**: "The number of neighbors is lower in the early layers because the point clouds are more sparse before being subsampled."

#### 2. init_channels: `96` -> `64`

**Severity**: HIGH

Our config set init_channels=96 for "140GB GPU", but the official is **64 across ALL datasets**. The S3DIS config explicitly comments `"48, 64, 80, 96"` as options but uses 64. There is NO official KpConvX config anywhere that uses 96.

| init_channels | Approx params | Source |
|--------------|--------------|--------|
| 64 | ~40M | Official (all configs) |
| 96 | ~63M | Our config (non-standard) |

Irony: Our 40GB script overrides to `init_channels=64` which is actually the correct official value.

#### 3. epoch: `100` -> `450`

**Severity**: HIGH

KpConvX trains from scratch with no pre-training. 100 epochs is insufficient:

| Source | Epochs |
|--------|--------|
| Bhopal standalone | **450** |
| S3DIS standalone | **450** |
| ScanObjectNN standalone | 250 |
| ScanNet Pointcept | **1000** |
| Our config | 100 |

The standalone Bhopal config uses exactly 450. Since we train from scratch like the standalone, 450 is the right target.

### MEDIUM (Should Fix)

#### 4. Test TTA: 4 augmentations -> 13 augmentations

**Severity**: MEDIUM (only affects test-time evaluation)

Official Pointcept ScanNet config uses 13 TTA augmentations:
```python
aug_transform=[
    # 4 rotations (0, 90, 180, 270 degrees)
    [dict(type="RandomRotateTargetAngle", angle=[0], ...)],
    [dict(type="RandomRotateTargetAngle", angle=[1/2], ...)],
    [dict(type="RandomRotateTargetAngle", angle=[1], ...)],
    [dict(type="RandomRotateTargetAngle", angle=[3/2], ...)],
    # 4 rotations + scale 0.95
    [dict(type="RandomRotateTargetAngle", angle=[0], ...),
     dict(type="RandomScale", scale=[0.95, 0.95])],
    [dict(type="RandomRotateTargetAngle", angle=[1/2], ...),
     dict(type="RandomScale", scale=[0.95, 0.95])],
    [dict(type="RandomRotateTargetAngle", angle=[1], ...),
     dict(type="RandomScale", scale=[0.95, 0.95])],
    [dict(type="RandomRotateTargetAngle", angle=[3/2], ...),
     dict(type="RandomScale", scale=[0.95, 0.95])],
    # 4 rotations + scale 1.05
    [dict(type="RandomRotateTargetAngle", angle=[0], ...),
     dict(type="RandomScale", scale=[1.05, 1.05])],
    [dict(type="RandomRotateTargetAngle", angle=[1/2], ...),
     dict(type="RandomScale", scale=[1.05, 1.05])],
    [dict(type="RandomRotateTargetAngle", angle=[1], ...),
     dict(type="RandomScale", scale=[1.05, 1.05])],
    [dict(type="RandomRotateTargetAngle", angle=[3/2], ...),
     dict(type="RandomScale", scale=[1.05, 1.05])],
    # 1 flip
    [dict(type="RandomFlip", p=1)],
]
```

Our config only uses 4 rotations. Adding the full 13 TTA improves test accuracy at the cost of 3.25x slower inference.

### LOW (Framework Adaptation Differences)

These differ between the Standalone and Pointcept-wrapper implementations. Since we use the Pointcept framework, our values follow the Pointcept-wrapper convention, which is correct.

#### 5. kp_radius / kp_sigma: 2.1 vs 2.3

| Source | kp_radius | kp_sigma |
|--------|-----------|----------|
| Standalone S3DIS | 2.1 | 2.1 (auto: =kp_radius when aggregation='nearest') |
| Standalone Bhopal | 2.1 | 2.1 |
| Pointcept ScanNet | **2.3** | **2.3** |
| Our config | 2.1 | 2.1 |

The standalone Bhopal uses 2.1 and we follow that. The Pointcept-wrapper uses 2.3. Since our model code comes from the Pointcept-wrapper copy, 2.3 may be more appropriate, but 2.1 is what was used for outdoor Bhopal data specifically.

Auto-calculation rule from standalone code:
```python
if kp_aggregation == 'nearest':
    kp_sigma = kp_radius    # sigma equals radius
else:
    kp_sigma = 0.7 * kp_radius
```

#### 6. kp_influence: 'constant' (Pointcept) vs 'linear' (standalone)

The standalone uses `'linear'` while the Pointcept-wrapper uses `'constant'`. This is a framework adaptation. Our config uses `'constant'` (Pointcept convention), which is fine since we use the Pointcept code path.

#### 7. share_kp: False (Pointcept) vs True (standalone)

Same framework-level difference. Standalone shares kernel points within layers; Pointcept does not.

---

## GridSample grid_size vs Model subsample_size

| Config | GridSample grid_size | Model subsample_size | Match? |
|--------|---------------------|---------------------|--------|
| Official ScanNet (Pointcept) | 0.02 | 0.02 | Yes |
| Official Bhopal (standalone) | N/A (init_sub_size=0.04) | 0.04 | Yes |
| Our Bhopal (Pointcept) | **0.01** | **0.04** | **No** |

In the official configs, the transform voxelization and the model's internal subsampling always match. Our config produces 1cm voxelized data (GridSample) that the model immediately re-subsamples to 4cm (subsample_size), which is extra computation.

However, `grid_size=0.01` was chosen deliberately to match PTv3's Bhopal config for cross-model consistency. This is a valid design choice for benchmarking fairness (all models see the same input resolution).

---

## GitHub Issues: Key Findings for Outdoor Datasets

### Issue #1: SensatUrban (Large Outdoor Dataset)

Author @HuguesTHOMAS (maintainer) responded:
- Increase `tile_size` beyond 20.5 for outdoor scenes (lower point density)
- Original parameters were optimized for **indoor scenes only**

User found working parameters for outdoor:
```
init_sub_size = 0.25    (much coarser than indoor 0.02-0.04)
in_sub_size = 0.25
in_radius = 20
tile_size = 25
```

> Our Bhopal MLS data at `subsample_size=0.04` is appropriate since MLS data is denser than vehicle-mounted LiDAR.

### Issue #8: Neighbor Limits

The paper states: *"Each layer has its own fixed number of neighbors. In all our experiments, we use [12, 16, 20, 20, 20]."*

Rationale: *"The number of neighbors is lower in the early layers because the point clouds are more sparse before being subsampled."*

No author response. The question asked why fewer neighbors at early (sparser) layers. The explanation is that **before** subsampling, points are dense at fine resolution. After grid pooling reduces density, more neighbors are needed in deeper layers. But the maximum is capped at 20.

### Issue #9: Imbalanced Datasets

A potential **bug in validation metric calculation** was identified:
> "This seems to be a bug in the validation metric calculation process. You can see that this outlier value is always the average of other categories."

This is in the standalone code only, not in Pointcept's SemSegEvaluator.

### Issue #14: RAM Limits for Big Datasets

Author recommended:
- Split datasets into smaller tiles
- Smart data loaders reading per-sphere from tile files
- Keep tile sizes slightly larger than sphere diameter
- Use balanced class or spatial sampling

Not directly relevant since we use Pointcept's BhopalSceneHDF5Dataset with HDF5 random access.

---

## Recommended Fixes Summary

| # | Parameter | Current | Fix To | Confidence |
|---|-----------|---------|--------|------------|
| 1 | neighbor_limits | (16, 20, 24, 24, 24) | **(12, 16, 20, 20, 20)** | Very High (paper-stated, all configs) |
| 2 | init_channels | 96 | **64** | Very High (all official configs) |
| 3 | epoch | 100 | **450** | Very High (Bhopal standalone = 450) |
| 4 | eval_epoch | 100 | **450** | Adjust with epoch |
| 5 | Test TTA aug_transform | 4 augmentations | **13 augmentations** | High (official Pointcept config) |
| 6 | 40GB script overrides | init_channels=64, neighbors=(12,16,20,20,20) | **Remove these overrides** (now matching base config) | High |

### Files Affected

- `configs/bhopal/kpconvx_140gb_12class_train_9ch.py`
- `configs/bhopal/kpconvx_140gb_12class_train_4ch.py`
- `scripts/train_bhopal_140gb.sh` (comments, display strings)
- `scripts/train_bhopal_40gb.sh` (remove now-unnecessary overrides)
- `README.md` (model size ~40M not ~63M)

### SemanticKITTI Configs (Same Issues)

- `configs/semantickitti/kpconvx_140gb_12class.py`
- `configs/semantickitti/kpconvx_140gb_19class.py`
- `SemanticKITTI/scripts/train_semantickitti_140gb.sh`
- `SemanticKITTI/scripts/train_semantickitti_40gb.sh`

---

## Appendix: Effective Receptive Field

With `subsample_size=0.04` and `radius_scaling=2.2`:

| Layer | Subsample (m) | Radius (m) with kp_radius=2.1 |
|-------|--------------|-------------------------------|
| 0 | 0.04 | 0.04 * 2.1 = 0.084 |
| 1 | 0.04 * 2.2 = 0.088 | 0.088 * 2.1 = 0.185 |
| 2 | 0.088 * 2.2 = 0.194 | 0.194 * 2.1 = 0.407 |
| 3 | 0.194 * 2.2 = 0.426 | 0.426 * 2.1 = 0.895 |
| 4 | 0.426 * 2.2 = 0.938 | 0.938 * 2.1 = 1.969 |

With kp_radius=2.3 instead:

| Layer | Subsample (m) | Radius (m) with kp_radius=2.3 |
|-------|--------------|-------------------------------|
| 0 | 0.04 | 0.04 * 2.3 = 0.092 |
| 1 | 0.088 | 0.088 * 2.3 = 0.202 |
| 2 | 0.194 | 0.194 * 2.3 = 0.446 |
| 3 | 0.426 | 0.426 * 2.3 = 0.980 |
| 4 | 0.938 | 0.938 * 2.3 = 2.157 |

The difference is small (~10% larger receptive field with 2.3). Both are reasonable for outdoor MLS.
