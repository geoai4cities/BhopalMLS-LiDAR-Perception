# Utonia Official vs Custom Configurations

## Comprehensive Analysis: Official Utonia Demo vs Our Bhopal/SemanticKITTI Configurations

**Date**: 2026-03-11
**Purpose**: Document differences between official Utonia setup and our LiDAR-only adaptations

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Official Utonia Overview](#2-official-utonia-overview)
3. [Official Demo Configuration](#3-official-demo-configuration)
4. [Our Custom Configurations](#4-our-custom-configurations)
5. [Key Differences Analysis](#5-key-differences-analysis)
6. [Checkpoint Loading Implications](#6-checkpoint-loading-implications)
7. [Recommendations](#7-recommendations)

---

## 1. Executive Summary

### Critical Differences at a Glance

| Aspect | Official Utonia | Our Bhopal | Our SemanticKITTI |
|--------|----------------|------------|-------------------|
| **Input Channels** | **9** (XYZ+RGB+normal) | 9 (XYZ+RGB+normal)* | 9 (XYZ+RGB+normal)* |
| **Grid Size** | 0.5m (demo default) | 0.02m (2cm) | 0.05m (5cm) |
| **Training Mode** | Feature extraction | Multiple (LP/DP/FT/LoRA) | Multiple (LP/DP/FT/LoRA) |
| **Checkpoint** | Single `utonia.pth` | Same `utonia.pth` | Same `utonia.pth` |
| **Backbone** | PT-v3m3 (standalone) | PT-v3m3 (Pointcept) | PT-v3m3 (Pointcept) |

*Our configs use 9 channels (coord+color+normal) where color and normal are zero-filled when not available from data.

### Key Insight

Our configurations use the same unified checkpoint as the official demo. Unlike Concerto (which had separate indoor/outdoor checkpoints), Utonia's single cross-domain model simplifies deployment. The main adaptation is integrating PT-v3m3 into Pointcept's training framework.

---

## 2. Official Utonia Overview

### 2.1 Cross-Domain Pretraining

Utonia was pretrained across **5 domain types**:

| Domain | Examples | Characteristics |
|--------|----------|-----------------|
| Remote Sensing | Aerial LiDAR | Very large scale, sparse |
| Outdoor LiDAR | Waymo, nuScenes, KITTI | Driving scenes, medium density |
| Indoor RGB-D | ScanNet, S3DIS | Dense, small rooms |
| Object-Centric | ShapeNet, ModelNet | Clean CAD models |
| RGB-Lifted | RE10K | Reconstructed from video |

### 2.2 Official Repository Structure

```
Pointcept/Utonia/
├── utonia/
│   ├── model.py           # Standalone PT-v3m3 model
│   ├── data.py            # Data loading utilities
│   ├── transform.py       # Default transforms
│   └── utils.py           # Seed setting, etc.
├── demo/
│   ├── 1_feature_extraction.py   # Extract encoder features
│   ├── 2_sem_seg.py              # Semantic segmentation (ScanNet)
│   └── 3_ins_seg.py              # Instance segmentation
└── utonia_pretrained_weights/
    └── utonia.pth                # Single unified checkpoint
```

---

## 3. Official Demo Configuration

### 3.1 Model Loading (from demo/2_sem_seg.py)

```python
import utonia

# With flash attention
model = utonia.load("utonia", repo_id="Pointcept/Utonia")

# Without flash attention
model = utonia.load("utonia", repo_id="Pointcept/Utonia",
    custom_config=dict(
        enc_patch_size=[1024 for _ in range(5)],
        enable_flash=False,
    ))
```

### 3.2 Default Input Format

```python
# 9-channel input: coord (3) + color (3) + normal (3)
point = {
    "coord": np.array(...),    # (N, 3) XYZ
    "color": np.array(...),    # (N, 3) RGB [0-255]
    "normal": np.array(...),   # (N, 3) surface normals
    "segment": np.array(...),  # (N,) labels
}

# If color/normal unavailable, set to zeros:
point["color"] = np.zeros_like(point["coord"])
point["normal"] = np.zeros_like(point["coord"])
```

### 3.3 Multi-Scale Feature Extraction

The official demo shows how features are extracted through the encoder's pooling hierarchy:

```python
# Forward pass
point = model(point)

# Walk back through pooling parents to concatenate multi-scale features
while "pooling_parent" in point.keys():
    parent = point.pop("pooling_parent")
    inverse = point.pop("pooling_inverse")
    parent.feat = torch.cat([parent.feat, point.feat[inverse]], dim=-1)
    point = parent

feat = point.feat  # Shape: (N, 1386)
```

This is exactly what `DefaultSegmentorV2` does in Pointcept.

### 3.4 Default Transform

```python
transform = utonia.transform.default(0.5)  # Grid size 0.5m for demo
```

---

## 4. Our Custom Configurations

### 4.1 Bhopal Configuration

```python
# Our Bhopal config (utonia_140gb_12class_linear_probe_9ch.py)

# INPUT CONFIGURATION
in_channels = 9  # coord (3) + color (3) + normal (3)
feat_keys = ("coord", "color", "normal")

# VOXELIZATION
grid_size = 0.02  # 2cm voxels

# MODEL (encoder - same for all strategies)
backbone = dict(
    type="PT-v3m3",
    enc_channels=(54, 108, 216, 432, 576),
    enc_num_head=(3, 6, 12, 24, 32),    # head_dim=18 (all %3==0 for RoPE)
    enc_depths=(3, 3, 3, 12, 3),
    rope_base=10,
)

# DECODER (DP/FT only - dec_num_head adjusted for RoPE compatibility)
# Original PTv3 defaults (6,6,12,32) give head_dim=16 which fails RoPE %3 check
dec_num_head=(8, 8, 16, 32),    # head_dim=12 (all %3==0 for RoPE)

# CLASS IMBALANCE HANDLING
ClassBalancedSample = True  # Building -> 40%
class_weight = [4.95, 4.65, 1.0, ...]  # Sqrt-scaled inverse frequency
```

### 4.2 SemanticKITTI Configuration

```python
# Our SemanticKITTI config
in_channels = 9
feat_keys = ("coord", "color", "normal")
grid_size = 0.05  # 5cm voxels
```

---

## 5. Key Differences Analysis

### 5.1 Input Channels: 9ch with Zero-Filling

Unlike our Concerto configs which used 4 channels (XYZ + intensity), our Utonia configs use **9 channels** matching the official format. When actual color/normal data is not available from the Bhopal MLS dataset, these channels are zero-filled.

```
Official Utonia (9 channels):
  Channel 0-2:  XYZ coordinates
  Channel 3-5:  RGB color
  Channel 6-8:  Surface normals

Our Configs (9 channels, partially zero-filled):
  Channel 0-2:  XYZ coordinates
  Channel 3-5:  RGB color (zeros if unavailable)
  Channel 6-8:  Surface normals (zeros if unavailable)

BENEFIT:
  • Embedding layer loads pretrained weights directly (no shape mismatch)
  • Full encoder pretrained weights are utilized
  • Better transfer than 4-channel input with random embedding
```

### 5.2 Pointcept Integration vs Standalone

| Aspect | Official (Standalone) | Our Setup (Pointcept) |
|--------|----------------------|----------------------|
| Model file | `utonia/model.py` | `point_transformer_v3m3_utonia.py` |
| Registration | `PyTorchModelHubMixin` | `@MODELS.register_module("PT-v3m3")` |
| Data loading | `utonia.data.load()` | Pointcept dataset loaders |
| Training | Manual loop (demo) | Pointcept training engine |
| Checkpointing | HuggingFace Hub | `CheckpointLoader` hook |
| Multi-scale | Manual pooling walk-back | `DefaultSegmentorV2` |

### 5.3 Grid Size Comparison

| Config | Grid Size | Avg Voxels | Use Case |
|--------|-----------|------------|----------|
| **Official Demo** | **0.5m** | ~1-5K | Quick demo/inference |
| Our Bhopal | 0.02m | ~350K | Fine infrastructure detail |
| Our SemanticKITTI | 0.05m | ~40K | Standard outdoor |

The official demo uses a very coarse grid (0.5m) for quick demonstration. Our training configs use much finer grids appropriate for each dataset.

---

## 6. Checkpoint Loading Implications

### 6.1 What Gets Loaded

When loading `utonia.pth` with our 9-channel configs:

```
Component              | Checkpoint | Our Config | Status
───────────────────────┼────────────┼────────────┼──────────────────────
backbone.embedding     | [54, 9]    | [54, 9]    | PRETRAINED (match!)
backbone.enc.enc0.*    | 3 blocks   | 3 blocks   | PRETRAINED
backbone.enc.enc1.*    | 3 blocks   | 3 blocks   | PRETRAINED
backbone.enc.enc2.*    | 3 blocks   | 3 blocks   | PRETRAINED
backbone.enc.enc3.*    | 12 blocks  | 12 blocks  | PRETRAINED
backbone.enc.enc4.*    | 3 blocks   | 3 blocks   | PRETRAINED
backbone.dec.*         | Not present| (DP/FT)    | RANDOM (not in ckpt)
seg_head               | Not present| 12 classes | RANDOM (task-specific)

SUMMARY:
  Pretrained: ALL encoder weights including embedding (~137.25M params)
  Random:     Decoder (if used) + Seg Head
```

### 6.2 Advantage Over Concerto

With Concerto, the embedding layer had a shape mismatch (9 vs 4 channels) and was randomly initialized. With Utonia's 9-channel configs, **the entire encoder including embedding loads pretrained weights** — better transfer learning.

---

## 7. Recommendations

### 7.1 For Best Results

```python
# Use 9-channel input (matches official)
in_channels = 9
feat_keys = ("coord", "color", "normal")

# If color/normal unavailable, the dataset loader should zero-fill
# The model was designed to handle this (from README:
# "If colors and normals are not available, please set them to zeros")

# Use grid size appropriate for your data density
grid_size = 0.02  # Bhopal (dense MLS)
grid_size = 0.05  # SemanticKITTI (standard outdoor)
```

### 7.2 Decision Tree

```
Q: Which checkpoint to use?
   └──> utonia.pth (always — single unified model)

Q: Which training strategy?
   │
   ├──> Quick baseline evaluation
   │    └──> Linear Probing (LP)
   │
   ├──> Better performance, frozen encoder
   │    └──> Decoder Probing (DP)
   │
   ├──> Best performance
   │    └──> Full Fine-tuning (FT)
   │
   └──> Limited GPU / parameter efficient
        └──> LoRA Fine-tuning

Q: Input channels?
   └──> Always 9ch (zero-fill missing modalities)
```

---

## Appendix: File Paths Reference

### Official Utonia

```
Utonia_pretrained/
├── utonia/model.py                           # Standalone model
├── demo/2_sem_seg.py                         # Reference inference
└── utonia_pretrained_weights/utonia.pth      # 524MB unified checkpoint
```

### Our Pointcept Integration

```
Pointcept/
└── pointcept/models/point_transformer_v3/
    ├── __init__.py                            # Updated: imports PT-v3m3
    └── point_transformer_v3m3_utonia.py       # PT-v3m3 for Pointcept
```

### Our Custom Configs

```
Utonia/
├── Bhopal/configs/bhopal/
│   ├── utonia_140gb_12class_linear_probe_9ch.py
│   ├── utonia_140gb_12class_decoder_probe_9ch.py
│   ├── utonia_140gb_12class_finetune_9ch.py
│   └── utonia_140gb_12class_lora_9ch.py
│   └── (same 4 config types)
├── Kerala/configs/kerala/
│   └── (same 4 config types)
└── SemanticKITTI/configs/semantickitti/
    └── (same 4 config types)
```

---

*Last Updated: 2026-03-11*
*Covers: Official Utonia demo, checkpoint analysis, and comparison with our Pointcept-integrated configs*
