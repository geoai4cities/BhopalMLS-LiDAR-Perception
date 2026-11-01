# Utonia (PT-v3m3) vs Concerto (PT-v3m2) Architecture Comparison

## Overview

Utonia and Concerto are both based on the Point Transformer V3 architecture but differ in key design choices. This document details the architectural differences and their implications for training and deployment.

**Papers:**
- Utonia: [Toward One Encoder for All Point Clouds](https://arxiv.org/abs/2603.03283) (2026)
- Concerto: Part of the Pointcept framework (2025)

---

## Architecture Comparison

### Encoder Parameters

| Parameter | Concerto (PT-v3m2) | Utonia (PT-v3m3) |
|-----------|-------------------|-----------------|
| `enc_channels` | (64, 128, 256, 512, 768) | (54, 108, 216, 432, 576) |
| `enc_num_head` | (4, 8, 16, 32, 48) | (3, 6, 12, 24, 32) |
| `enc_depths` | (3, 3, 3, 12, 3) | (3, 3, 3, 12, 3) |
| `enc_patch_size` | (1024, 1024, 1024, 1024, 1024) | (1024, 1024, 1024, 1024, 1024) |
| Total encoder params | ~207M | **~137.25M** |
| Checkpoint size | 830MB (outdoor-specific) | **524MB (unified)** |

### Decoder Parameters

| Parameter | Concerto (PT-v3m2) | Utonia (PT-v3m3) |
|-----------|-------------------|-----------------|
| `dec_depths` | (2, 2, 2, 2) | **(3, 3, 3, 3)** |
| `dec_channels` | (64, 96, 192, 384) | **(96, 96, 192, 384)** |
| `dec_num_head` | (4, 6, 12, 24) | **(8, 8, 16, 32)** |
| `dec_patch_size` | (1024, 1024, 1024, 1024) | (1024, 1024, 1024, 1024) |
| Total decoder params | ~20.5M | **~20.5M** |

**Note on Utonia `dec_num_head`:** The original PTv3 defaults are `(6, 6, 12, 32)` giving head_dim=16, but Utonia's Point3DRoPE requires `head_dim % 3 == 0`. We use `(8, 8, 16, 32)` giving head_dim=12 for all decoder stages. This is safe because the decoder is randomly initialized (not in the pre-trained checkpoint).

### Output Dimensions

| Mode | Concerto | Utonia | Notes |
|------|----------|--------|-------|
| LP/LoRA (`enc_mode=True`) | 1728 (64+128+256+512+768) | **1386** (54+108+216+432+576) | Multi-scale concat |
| DP/FT (`enc_mode=False`) | 64 (first `dec_channel`) | **96** (first `dec_channel`) | Decoder output |

### Positional Encoding

| Feature | Concerto (PT-v3m2) | Utonia (PT-v3m3) |
|---------|-------------------|-----------------|
| Position encoding | RPE (Relative Position Encoding) | **Point3DRoPE** (3D Rotary Position Embedding) |
| `rope_base` | N/A | **10** |
| Coord augmentations | N/A | `shift_coords`, `jitter_coords`, `rescale_coords` |

---

## Key Differences Explained

### 1. 3D Rotary Position Embedding (RoPE)

Utonia replaces RPE with Point3DRoPE — a 3D extension of Rotary Position Embeddings. This provides:
- Better generalization across point cloud scales and densities
- Continuous position encoding (vs discrete RPE tables)
- Cross-domain robustness (same encoding works for indoor, outdoor, object-level data)

**RoPE Constraint:** Point3DRoPE splits the head dimension into 3 equal chunks (one per spatial axis: X, Y, Z), so `head_dim = channels / num_heads` must be divisible by 3. This constraint applies to ALL attention blocks — both encoder and decoder. The encoder's head_dim=18 (all stages) satisfies this. For the decoder, we adjust `dec_num_head` from PTv3 defaults to ensure head_dim=12.

### 2. Unified vs Domain-Specific Pretraining

```
Concerto:
  concerto_large_outdoor.pth  → Outdoor LiDAR (Waymo, nuScenes, KITTI)
  concerto_large_indoor.pth   → Indoor RGB-D (ScanNet, S3DIS, etc.)

Utonia:
  utonia.pth                  → ALL domains (outdoor, indoor, object, remote sensing, RGB-lifted)
```

Utonia's single checkpoint simplifies deployment — no need to choose indoor vs outdoor variant.

### 3. Smaller but More Efficient Encoder

Utonia's encoder is ~34% smaller (137.25M vs 207M params) but was trained on more diverse data across 5 domain types. The paper (Table 4) shows comparable or better performance on outdoor semantic segmentation despite fewer parameters.

### 4. Deeper Decoder

Utonia uses `dec_depths=(3,3,3,3)` vs Concerto's `(2,2,2,2)` — 50% more decoder blocks. The first decoder stage also widens from 64 to 96 channels, providing richer feature reconstruction.

---

## Parameter Count Summary (from Paper Table 4)

### Outdoor Semantic Segmentation

| Model | Fine-tuning (full) | Decoder Probe | Linear Probe |
|-------|-------------------|---------------|--------------|
| Concerto | 157.7M | 20.5M | <0.2M |
| Utonia | 157.7M | 20.5M | <0.2M |

**Note:** Fine-tuning parameter counts match because the decoder is deeper but narrower in Utonia — these balance out.

---

## Seg Head Dimensions

| Strategy | Concerto | Utonia |
|----------|----------|--------|
| LP | 1728 -> num_classes (~20.7K for 12 classes) | 1386 -> num_classes (~16.6K for 12 classes) |
| DP/FT | 64 -> num_classes (~768 for 12 classes) | 96 -> num_classes (~1.2K for 12 classes) |
| LoRA | 1728 -> num_classes (~20.7K for 12 classes) | 1386 -> num_classes (~16.6K for 12 classes) |

---

## Multi-Scale Feature Concatenation (LP/LoRA)

Both models use `DefaultSegmentorV2` which walks back through `pooling_parent` to concatenate features from all encoder stages:

```
Stage 0: 54ch  ─┐
Stage 1: 108ch ─┤
Stage 2: 216ch ─┼─> Concatenate → 1386ch → seg_head → num_classes
Stage 3: 432ch ─┤
Stage 4: 576ch ─┘
```

This is handled by the `pooling_parent` / `pooling_inverse` mechanism in `GridPooling` (which internally sets `traceable=True` to store these references).

---

## Checkpoint Verification

### Utonia Checkpoint Keys (utonia.pth)

```python
import torch
ckpt = torch.load('utonia.pth', map_location='cpu')
# Keys: 'config', 'state_dict'

config = ckpt['config']
# {
#   'in_channels': 9,
#   'enc_channels': (54, 108, 216, 432, 576),
#   'enc_num_head': (3, 6, 12, 24, 32),
#   'rope_base': 10,
#   'traceable': True,
#   'mask_token': True,    # SSL pretraining flag
#   'enc_mode': True,
# }

state_dict = ckpt['state_dict']
# Encoder-only weights, ~137.25M parameters
# No decoder weights — decoder is randomly initialized for DP/FT
```

### Key Config Notes

- `mask_token=True` in checkpoint: used during self-supervised pretraining (masked point prediction). Our downstream configs set `mask_token=False` — this is correct.
- `traceable=True` in checkpoint: GridPooling stores `pooling_parent`/`pooling_inverse` needed for multi-scale feature concatenation. Our LP configs set `traceable=False` at the top level, but `GridPooling` internally defaults to `traceable=True`, so multi-scale works correctly.

---

## Practical Implications for Training

### What Changes vs Concerto Configs

1. **Model type**: `PT-v3m2` -> `PT-v3m3`
2. **Encoder channels/heads**: updated to Utonia dimensions
3. **Decoder dimensions**: deeper (3,3,3,3) and wider first stage (96)
4. **Decoder num_heads**: `(4,6,12,24)` -> `(8,8,16,32)` to satisfy RoPE head_dim%3==0
5. **backbone_out_channels**: 1728->1386 (LP/LoRA), 64->96 (DP/FT)
6. **Checkpoint path**: single `utonia.pth` instead of `concerto_large_outdoor.pth`
7. **New params**: `rope_base=10` (automatically handled by PT-v3m3)

### What Stays the Same

- Dataset config (Bhopal HDF5, class_groups, transforms)
- Training hyperparameters (LR, epochs, weight_decay, etc.)
- Augmentation pipeline
- CheckpointLoader mechanism (same keywords/replacement pattern)
- DefaultSegmentorV2 / DefaultLORASegmentorV2 usage

---

*Last Updated: 2026-03-11*
*Verified against: utonia.pth checkpoint, Utonia paper Tables 3/4, official demo code*
