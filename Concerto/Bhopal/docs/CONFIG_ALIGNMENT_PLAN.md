# Concerto Bhopal Config Alignment Plan

## Context

Bhopal MLS is an **outdoor** dataset. The official Concerto repo provides:
- **Outdoor semantic seg** configs: SemanticKITTI, nuScenes, Waymo (LP only)
- **Indoor instance seg** configs: ScanNet, ScanNet200, ScanNetPP, S3DIS (LP/DP/FT)
- **No outdoor instance seg** configs exist

Our Bhopal **semantic seg** configs were built from the official outdoor (KITTI/nuScenes) configs.
Our Bhopal **instance seg** configs were built from the official indoor (ScanNet/S3DIS) configs,
since no outdoor instance seg reference exists.

This document identifies every difference and recommends whether to align or keep.

---

## Part A: Semantic Segmentation (Bhopal vs Official Outdoor)

Reference: `semseg-ptv3-large-v1m1-kitti-4a-lin.py`, `semseg-ptv3-large-v1m1-nuscenes-5a-lin.py`

### Status: WELL ALIGNED

The Bhopal semseg configs already closely follow the official outdoor pattern:

| Parameter | Official KITTI/nuScenes | Bhopal | Match? |
|-----------|------------------------|--------|--------|
| RandomScale (1st) | [0.2, 0.2] | [0.2, 0.2] | YES |
| RandomRotate z-axis | [-1, 1], p=0.5 | [-1, 1], p=0.5 | YES |
| RandomRotate x/y | Commented out | Not present | YES |
| RandomScale (2nd) | [0.9, 1.1] | [0.9, 1.1] | YES |
| RandomFlip | p=0.5 | p=0.5 | YES |
| ElasticDistortion | Commented out | Not present | YES |
| CenterShift | Commented out | Not present | YES |
| RandomDropout | Commented out | Not present | YES |
| ChromaticAug | Not present | Not present | YES |
| GridSample | grid_size=0.01 | grid_size=0.01 | YES |
| NormalizeColor | Present | Present | YES |
| lr, weight_decay | 0.002, 0.005 | 0.002, 0.005 | YES |
| pct_start | 0.04 | 0.04 | YES |
| final_div_factor | 100.0 | 100.0 | YES |
| Loss: CE + Lovasz | ignore_index=-1 | ignore_index=-1 | YES |
| Test TTA | nuScenes: 5 scales x flip | Same pattern | YES |

### Differences (Semantic Seg)

| # | Parameter | Official Outdoor | Bhopal | Action |
|---|-----------|-----------------|--------|--------|
| S1 | **RandomJitter (9ch)** | sigma=0.0025, clip=0.01 | sigma=0.001, clip=0.004 | See below |
| S2 | **SphereCrop** | KITTI: sample_rate=0.8 + point_max=120K; nuScenes: commented out | Not present | KEEP (see below) |
| S3 | **Epochs** | 50 | 100 | KEEP at 100 |
| S4 | **CE class_weight** | KITTI: per-class weights; nuScenes: none | Bhopal-specific weights | KEEP |
| S5 | **PointClip range** | KITTI-specific bounds | Bhopal-specific bounds | KEEP |
| S6 | **CheckpointLoader** | `"module.student.backbone"` -> `"module.backbone"` | `"module."` -> `"module.backbone."` | See below |

### Semantic Seg Decisions

**S1 - RandomJitter**: The 4ch config already matches KITTI (0.0025/0.01). The 9ch config
uses smaller jitter (0.001/0.004) to be gentler on RGB channels. This is a deliberate choice
since Bhopal has only ~35-40% RGB coverage.
- **Decision: KEEP as-is** - domain-appropriate for partial RGB data

**S2 - SphereCrop**: Official nuScenes/Waymo also comment it out. KITTI uses it because
KITTI scans are 360-degree LiDAR sweeps. Bhopal uses PointClip instead (rectangular crop
matching MLS corridor geometry).
- **Decision: KEEP as-is** - PointClip is more appropriate for MLS corridors

**S3 - Epochs**: Bhopal has fewer scenes (~2340) than KITTI (~19K) or nuScenes (~40K).
More epochs compensates for smaller dataset.
- **Decision: KEEP at 100**

**S6 - CheckpointLoader**: The released `concerto_large_outdoor.pth` has direct keys
(no `student.backbone` prefix), unlike the raw pre-training checkpoints. Both patterns
work with their respective checkpoint formats.
- **Decision: KEEP as-is** - works correctly with released weights

**CONCLUSION: Semantic seg configs need NO changes.**

---

## Part B: Instance Segmentation (Bhopal vs Official Indoor + Outdoor Patterns)

Reference for insseg params: `insseg-ptv3-large-v1m1-0a-scannet-pg-lin.py` (ScanNet LP)
Reference for outdoor augmentations: `semseg-ptv3-large-v1m1-kitti-4a-lin.py` (KITTI semseg)

### Current Differences

| # | Parameter | Official InsSeg (ScanNet) | Official Outdoor SemSeg (KITTI) | Bhopal InsSeg | Source of Truth |
|---|-----------|--------------------------|-------------------------------|--------------|-----------------|
| I1 | **weight_decay** | 0.05 | 0.005 | 0.005 | Outdoor |
| I2 | **final_div_factor** | 1000.0 | 100.0 | 100.0 | Outdoor |
| I3 | **pct_start** | 0.05 | 0.04 | 0.04 | Outdoor |
| I4 | **Epochs** | 800 (ScanNet), 3000 (S3DIS) | 50 | 100 | Custom |
| I5 | **CenterShift** | YES (before+after) | Commented out | Not present | Outdoor |
| I6 | **RandomDropout** | YES (0.2, 0.5) | Commented out | Not present | Outdoor |
| I7 | **RandomRotate x/y** | YES ([-1/64, 1/64]) | Commented out | Not present | Outdoor |
| I8 | **ElasticDistortion** | YES | Commented out | Not present | Outdoor |
| I9 | **ChromaticAug** | YES (3 types) | Not present | Not present | Outdoor |
| I10 | **SphereCrop** | YES (sample_rate=0.8) | YES (KITTI) / Commented (nuScenes) | Not present | Outdoor |
| I11 | **RandomJitter** | sigma=0.005, clip=0.02 | sigma=0.0025, clip=0.01 | sigma=0.001, clip=0.004 | Custom |
| I12 | **GridSample** | grid_size=0.02 | grid_size=0.01 | grid_size=0.01 | Outdoor |
| I13 | **voxel_size** | 0.02 (default) | N/A (no PointGroup) | 0.01 | Matches grid_size |
| I14 | **CE class_weight** | None (uniform) | KITTI: per-class weights | Bhopal weights | Custom |

### Key Insight

For items I1-I10, there's a conflict between:
- The **official insseg configs** (indoor, with heavy augmentation)
- The **official outdoor semseg configs** (simpler augmentation, different hyperparams)

Since Bhopal is outdoor data, we should follow the **outdoor pattern** for augmentations
and hyperparams, and only take the **insseg-specific settings** (PointGroup clustering,
InstanceParser, InsSegEvaluator) from the indoor insseg configs.

---

## Part C: Recommended Changes

### CHANGE 1: Align InsSeg RandomJitter with Outdoor KITTI Pattern
- **Current (insseg)**: sigma=0.001, clip=0.004
- **Official outdoor KITTI**: sigma=0.0025, clip=0.01
- **Official indoor insseg**: sigma=0.005, clip=0.02
- **Proposed**: sigma=0.0025, clip=0.01 (match KITTI outdoor)
- **Reasoning**: The semseg 4ch config already uses KITTI values. The 9ch semseg uses
  smaller values due to RGB sensitivity, but instance seg benefits from stronger
  augmentation for offset prediction robustness. KITTI values are a good middle ground.
- **Affects**: All 6 insseg configs (12c LP/DP/FT, 21c LP/DP/FT)
- **Risk**: Low - slightly stronger augmentation
- **Priority**: LOW

### CHANGE 2: NO CHANGE to weight_decay (already matches outdoor)
- **Current (insseg)**: 0.005
- **Official outdoor**: 0.005
- **Official indoor insseg**: 0.05
- **Decision**: KEEP at 0.005 - already aligned with outdoor pattern
- **Reasoning**: Indoor insseg uses 0.05 with 800-3000 epochs. With only 100 epochs,
  stronger regularization would hurt convergence. 0.005 matches outdoor.

### CHANGE 3: NO CHANGE to final_div_factor (already matches outdoor)
- **Current (insseg)**: 100.0
- **Official outdoor**: 100.0
- **Official indoor insseg**: 1000.0
- **Decision**: KEEP at 100.0 - already aligned with outdoor pattern

### CHANGE 4: NO CHANGE to pct_start (already matches outdoor)
- **Current (insseg)**: 0.04
- **Official outdoor**: 0.04
- **Official indoor insseg**: 0.05
- **Decision**: KEEP at 0.04 - already aligned with outdoor pattern

### CHANGE 5: NO CHANGE to CenterShift
- Not in official outdoor configs (commented out)
- Present in official indoor insseg
- **Decision**: KEEP absent - following outdoor pattern

### CHANGE 6: NO CHANGE to RandomDropout
- Not in official outdoor configs (commented out)
- Present in official indoor insseg
- **Decision**: KEEP absent - following outdoor pattern

### CHANGE 7: NO CHANGE to RandomRotate x/y axes
- Not in official outdoor configs (commented out)
- Present in official indoor insseg
- **Decision**: KEEP absent - following outdoor pattern
- **Note**: For outdoor MLS, small tilts are less relevant than for indoor scans

### CHANGE 8: NO CHANGE to ElasticDistortion
- Not in official outdoor configs (commented out)
- Present in official indoor insseg
- **Decision**: KEEP absent - designed for indoor non-rigid deformation

### CHANGE 9: NO CHANGE to ChromaticAug
- Not in official outdoor configs at all
- Present in official indoor insseg only
- **Decision**: KEEP absent - Bhopal has partial RGB (~35-40%)

### CHANGE 10: NO CHANGE to SphereCrop
- Mixed in official outdoor (KITTI yes, nuScenes/Waymo commented out)
- Present in official indoor insseg
- **Decision**: KEEP absent - using PointClip for MLS corridor geometry instead

### CHANGE 11: NO CHANGE to GridSample/voxel_size
- Bhopal: grid_size=0.01, voxel_size=0.01 (matching)
- Official indoor: grid_size=0.02, voxel_size=0.02 (matching)
- Official outdoor: grid_size=0.01 (no PointGroup)
- **Decision**: KEEP at 0.01 - matches outdoor grid_size, voxel_size correctly aligned

### CHANGE 12: NO CHANGE to Epochs
- Current: 100
- Official indoor insseg: 800-3000 (but with much smaller datasets)
- Official outdoor semseg: 50 (but LP only)
- **Decision**: KEEP at 100 - reasonable for Bhopal dataset size (~2340 scenes)

### CHANGE 13: NO CHANGE to class_weight in CE loss
- Official indoor insseg: uniform (no weights)
- Official outdoor KITTI semseg: per-class weights
- Bhopal: sqrt-scaled inverse frequency
- **Decision**: KEEP - Bhopal has severe class imbalance (Commercial Building ~47%)

---

## Summary

### Changes to Make

| # | Change | Configs Affected | Priority |
|---|--------|-----------------|----------|
| 1 | RandomJitter: sigma 0.001->0.0025, clip 0.004->0.01 | 6 insseg configs | LOW |

### No Changes Needed (Already Correct)

| # | Parameter | Reason |
|---|-----------|--------|
| 2-4 | weight_decay, final_div_factor, pct_start | Already matches outdoor pattern |
| 5-10 | CenterShift, RandomDropout, x/y Rotate, ElasticDistortion, ChromaticAug, SphereCrop | Not used in official outdoor configs either |
| 11 | GridSample/voxel_size | Correctly set to 0.01 matching outdoor |
| 12 | Epochs | Appropriate for dataset size |
| 13 | CE class_weight | Needed for Bhopal's class imbalance |

### Conclusion

The Bhopal configs are **well-aligned with the official outdoor Concerto pattern**.
The initial comparison against indoor insseg configs was misleading because those configs
use heavy augmentation for small indoor scenes (ScanNet: ~1500 scenes, S3DIS: 271 rooms).

The only minor change is aligning RandomJitter in insseg configs with the KITTI outdoor
values (currently slightly more conservative). All other parameters either already match
the outdoor pattern or are domain-appropriate customizations for Bhopal MLS data.

---

## Reference: Official Config Files Analyzed

### Outdoor Semantic Segmentation
- `Pointcept/configs/concerto/semseg-ptv3-large-v1m1-kitti-4a-lin.py`
- `Pointcept/configs/concerto/semseg-ptv3-large-v1m1-nuscenes-5a-lin.py`
- `Pointcept/configs/concerto/semseg-ptv3-large-v1m1-waymo-6a-lin.py`

### Indoor Instance Segmentation
- `Pointcept/configs/concerto/insseg-ptv3-large-v1m1-0a-scannet-pg-lin.py`
- `Pointcept/configs/concerto/insseg-ptv3-large-v1m1-0b-scannet-pg-dec.py`
- `Pointcept/configs/concerto/insseg-ptv3-large-v1m1-0c-scannet-pg-ft.py`
- `Pointcept/configs/concerto/insseg-ptv3-large-v1m1-3a-s3dis-pg-lin.py`
- `Pointcept/configs/concerto/insseg-ptv3-large-v1m1-3b-s3dis-pg-dec.py`
- `Pointcept/configs/concerto/insseg-ptv3-large-v1m1-3c-s3dis-pg-ft.py`

### Indoor Semantic Segmentation (Base variant)
- `Pointcept/configs/concerto/semseg-ptv3-base-v1m1-0a-scannet-lin.py`
- `Pointcept/configs/concerto/semseg-ptv3-base-v1m1-3a-s3dis-lin.py`
