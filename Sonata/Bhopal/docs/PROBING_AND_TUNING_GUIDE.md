# Concerto Probing and Tuning Guide for Bhopal MLS Dataset

This document explains how to use the official Concerto probing and fine-tuning methods for Bhopal MLS semantic segmentation. It covers the different training strategies, their configurations, and how to adapt them for your specific use case.

## Table of Contents

1. [Overview](#overview)
2. [Pre-trained Model Weights](#pre-trained-model-weights)
3. [Training Strategies Explained](#training-strategies-explained)
4. [Configuration Parameter Reference](#configuration-parameter-reference)
5. [Memory Requirements](#memory-requirements)
6. [Training Commands](#training-commands)
7. [Checkpoint Loading](#checkpoint-loading)
8. [Expected Performance](#expected-performance)
9. [Troubleshooting](#troubleshooting)
10. [Official Reference Configs](#official-reference-configs)
11. [Indoor vs Outdoor Dataset Settings](#indoor-vs-outdoor-dataset-settings)

---

## Overview

Concerto provides several strategies for adapting pre-trained models to downstream tasks like Bhopal MLS semantic segmentation:

| Strategy | Backbone Frozen | Decoder | Trainable Params | Use Case | Expected Performance |
|----------|-----------------|---------|------------------|----------|---------------------|
| **Linear Probing (LP)** | Yes (entire) | Not used | ~20.7K | Quick evaluation | Baseline |
| **Decoder Probing (DP)** | Yes (encoder) | Trainable | ~30-40M | Better feature utilization | Improved |
| **Full Fine-tuning (FT)** | No | Trainable | ~240M | Best performance | Best |
| **LoRA Fine-tuning** | Partially | Not used | ~0.2M | Parameter-efficient | Good |

---

## Pre-trained Model Weights

### Recommended Model for Bhopal (Outdoor MLS)

**Model:** `concerto_large_outdoor.pth` (830MB)

This model is specifically optimized for outdoor LiDAR datasets and should be used for:
- Bhopal MLS
- SemanticKITTI
- nuScenes
- Waymo
- Other outdoor driving/mapping scenarios

**Location:** `Concerto_pretrained/concerto_large_outdoor.pth`

### Model Architecture (PT-v3m2 Large)

```python
# Encoder
enc_depths=(3, 3, 3, 12, 3)
enc_channels=(64, 128, 256, 512, 768)
enc_num_head=(4, 8, 16, 32, 48)
enc_patch_size=(1024, 1024, 1024, 1024, 1024)

# Decoder (when enc_mode=False)
dec_depths=(2, 2, 2, 2)
dec_channels=(64, 96, 192, 384)
dec_num_head=(4, 6, 12, 24)
dec_patch_size=(1024, 1024, 1024, 1024)
```

### Important: Encoder vs Decoder in Pre-trained Weights

**All backbone checkpoints contain only ENCODER weights.** The decoder is NOT pre-trained.

| Component | Pre-trained? | Notes |
|-----------|--------------|-------|
| **Encoder** | Yes | Core pre-trained backbone, loaded from checkpoint |
| **Decoder** | No | Randomly initialized, trained during fine-tuning |
| **Seg Head** | No | Randomly initialized, always trained |

---

## Training Strategies Explained

### 1. Linear Probing (LP) - `lp 9ch`

**Purpose:** Evaluate the quality of pre-trained features without modifying them.

**Configuration:**
```python
model = dict(
    type="DefaultSegmentorV2",
    backbone_out_channels=1728,  # Concatenated encoder features
    backbone=dict(
        enc_mode=True,         # Encoder-only, no decoder
        freeze_encoder=False,  # Handled by freeze_backbone
    ),
    freeze_backbone=True,      # Freeze entire backbone
)
```

**What Trains:**
- Only `seg_head` (1728 → 12 classes = ~20.7K params)

**Use Case:** Quick baseline to measure feature quality.

---

### 2. Decoder Probing (DP) - `dp 9ch`

**Purpose:** Train a decoder on top of frozen encoder features for better multi-scale utilization.

**Configuration:**
```python
model = dict(
    type="DefaultSegmentorV2",
    backbone_out_channels=64,   # Decoder output
    backbone=dict(
        enc_mode=False,         # Use decoder
        freeze_encoder=True,    # Freeze encoder, train decoder
        dec_depths=(2, 2, 2, 2),
        dec_channels=(64, 96, 192, 384),
        dec_num_head=(4, 6, 12, 24),
        dec_patch_size=(1024, 1024, 1024, 1024),
    ),
    freeze_backbone=False,      # Allow decoder training
)
```

**What Trains:**
- Decoder (~30-40M params)
- `seg_head` (64 → 12 classes = 768 params)

**Use Case:** Better utilize spatial hierarchy from pre-trained features.

---

### 3. Full Fine-tuning (FT) - `ft 9ch`

**Purpose:** Adapt the entire model to the target dataset for best performance.

**Configuration:**
```python
model = dict(
    type="DefaultSegmentorV2",
    backbone_out_channels=64,
    backbone=dict(
        enc_mode=False,          # Use decoder
        freeze_encoder=False,    # Train encoder (with lower LR)
        dec_depths=(2, 2, 2, 2),
        dec_channels=(64, 96, 192, 384),
        dec_num_head=(4, 6, 12, 24),
        dec_patch_size=(1024, 1024, 1024, 1024),
    ),
    freeze_backbone=False,       # Train everything
)

# Differential learning rates
scheduler = dict(
    max_lr=[0.002, 0.0002],  # Head LR, Backbone LR (10x lower)
)
param_dicts = [dict(keyword="block", lr=0.0002)]
```

**What Trains:**
- Encoder (~207M params) - with 10x lower learning rate
- Decoder (~30-40M params)
- `seg_head` (64 → 12 classes = 768 params)

**Use Case:** Best performance when you have sufficient data.

---

### 4. LoRA Fine-tuning - `lora 9ch`

**Purpose:** Parameter-efficient fine-tuning using Low-Rank Adaptation.

**Configuration:**
```python
model = dict(
    type="DefaultLORASegmentorV2",  # Special LoRA model
    backbone_out_channels=1728,
    use_lora=True,
    lora_r=8,           # LoRA rank
    lora_alpha=16,      # LoRA scaling
    lora_dropout=0.1,   # LoRA dropout
    backbone_path="path/to/checkpoint.pth",  # Required for LoRA
    backbone=dict(
        enc_mode=True,
        traceable=True,       # Required for LoRA
        freeze_encoder=True,
    ),
    freeze_backbone=False,    # LoRA adapters are trainable
)
```

**What Trains:**
- LoRA adapters (~0.2M params, ~0.1% of encoder)
- `seg_head` (1728 → 12 classes = ~20.7K params)

**Use Case:** When you want fine-tuning quality with limited compute/memory.

---

## Configuration Parameter Reference

### Key Parameters Comparison

| Parameter | Linear Probe | Decoder Probe | Full FT | LoRA |
|-----------|-------------|---------------|---------|------|
| `type` | `DefaultSegmentorV2` | `DefaultSegmentorV2` | `DefaultSegmentorV2` | `DefaultLORASegmentorV2` |
| `enc_mode` | `True` | `False` | `False` | `True` |
| `freeze_encoder` | N/A* | `True` | `False` | `True` |
| `freeze_backbone` | `True` | `False` | `False` | `False` |
| `backbone_out_channels` | 1728 | 64 | 64 | 1728 |
| `traceable` | `False` | `False` | `False` | `True` |
| `use_lora` | N/A | N/A | N/A | `True` |
| `lora_r` | N/A | N/A | N/A | 8 |
| `backbone_path` | N/A | N/A | N/A | Required |

*For LP, `freeze_encoder` is irrelevant because `freeze_backbone=True` freezes everything.

### Understanding freeze_encoder vs freeze_backbone

- **`freeze_backbone`** (model-level): Freezes the entire backbone module. Used in **Linear Probing**.
- **`freeze_encoder`** (backbone-level): Only relevant when `enc_mode=False`. Freezes just the encoder while allowing decoder to train. Used in **Decoder Probing**.

---

## Memory Requirements

### Estimated GPU Memory Usage

#### 40GB GPU (A100/A6000)

| Strategy | batch_size | grad_accum | Effective Batch | Est. Memory |
|----------|------------|------------|-----------------|-------------|
| Linear Probing | 8 | 8 | 64 | ~35-40GB |
| Decoder Probing | 6 | 8 | 48 | ~45-50GB |
| Full Fine-tuning | 4 | 8 | 32 | ~55-60GB |
| LoRA Fine-tuning | 8 | 8 | 64 | ~35-40GB |

#### 140GB GPU (H200)

| Strategy | batch_size | grad_accum | Effective Batch | Est. Memory |
|----------|------------|------------|-----------------|-------------|
| Linear Probing | 8 | 8 | 64 | ~40-50GB |
| Decoder Probing | 8 | 8 | 64 | ~60-70GB |
| Full Fine-tuning | 6 | 8 | 48 | ~80-90GB |
| LoRA Fine-tuning | 8 | 8 | 64 | ~40-50GB |

---

## Training Commands

### Available Commands

```bash
cd Concerto/Bhopal

# Linear Probing
./scripts/train_bhopal_40gb.sh lp 4ch       # Geometry only
./scripts/train_bhopal_40gb.sh lp 9ch       # XYZ + RGB + normals
./scripts/train_bhopal_40gb.sh lp-resume 9ch
./scripts/train_bhopal_40gb.sh lp-stop 9ch

# Decoder Probing
./scripts/train_bhopal_40gb.sh dp 9ch
./scripts/train_bhopal_40gb.sh dp-resume 9ch
./scripts/train_bhopal_40gb.sh dp-stop 9ch

# Full Fine-tuning
./scripts/train_bhopal_40gb.sh ft 9ch
./scripts/train_bhopal_40gb.sh ft-resume 9ch
./scripts/train_bhopal_40gb.sh ft-stop 9ch

# LoRA Fine-tuning
./scripts/train_bhopal_40gb.sh lora 9ch
./scripts/train_bhopal_40gb.sh lora-resume 9ch
./scripts/train_bhopal_40gb.sh lora-stop 9ch
```

### Experiment Matrix

| Command | Strategy | Config File | Encoder | Decoder | Est. Memory |
|---------|----------|-------------|---------|---------|-------------|
| `lp 4ch` | Linear Probing | `concerto_140gb_12class_linear_probe_4ch.py` | Frozen | N/A | ~35-40GB |
| `lp 9ch` | Linear Probing | `concerto_140gb_12class_linear_probe_9ch.py` | Frozen | N/A | ~35-40GB |
| `dp 9ch` | Decoder Probing | `concerto_140gb_12class_decoder_probe_9ch.py` | Frozen | Trainable | ~45-50GB |
| `ft 9ch` | Full Fine-tuning | `concerto_140gb_12class_finetune_9ch.py` | Trainable | Trainable | ~55-60GB |
| `lora 9ch` | LoRA Fine-tuning | `concerto_140gb_12class_lora_9ch.py` | LoRA | N/A | ~35-40GB |

### Recommended Order

1. **`lp 9ch`** - Establish baseline with official Concerto input
2. **`dp 9ch`** - Better feature utilization with trainable decoder
3. **`ft 9ch`** - Best performance with full adaptation
4. **`lora 9ch`** - Parameter-efficient alternative if FT is too expensive

---

## Checkpoint Loading

### For Released Weights (concerto_large_outdoor.pth)

All strategies (except LoRA) use the same CheckpointLoader configuration:

```python
hooks = [
    dict(
        type="CheckpointLoader",
        keywords="module.",           # Released weights have direct keys
        replacement="module.backbone.",  # Map to backbone.*
    ),
    # ... other hooks
]

weight = "/path/to/concerto_large_outdoor.pth"
```

### For LoRA

LoRA uses `backbone_path` in the model config instead of the weight parameter:

```python
model = dict(
    type="DefaultLORASegmentorV2",
    backbone_path="/path/to/concerto_large_outdoor.pth",
    keywords="module.",
    replacements="module.backbone.",
    # ...
)
```

### Expected "Missing Keys" Warning

When loading checkpoints, you will see:
```
Missing keys: ['seg_head.weight', 'seg_head.bias']
```

**This is expected!** The seg_head is task-specific and randomly initialized.

For Decoder Probing and Full Fine-tuning, decoder keys will also be "unexpected" since the checkpoint only contains encoder weights.

---

## Expected Performance

### Estimated mIoU (Bhopal 12-class)

| Strategy | Input | Expected mIoU Range | Training Time (50 epochs) |
|----------|-------|---------------------|---------------------------|
| LP-4ch | Geometry only | ~40-50% | ~30-40 hours |
| LP-9ch | XYZ+RGB+normal | ~50-60% | ~30-40 hours |
| DP-9ch | XYZ+RGB+normal | ~55-65% | ~50-60 hours |
| FT-9ch | XYZ+RGB+normal | ~60-70% | ~70-90 hours |
| LoRA-9ch | XYZ+RGB+normal | ~55-65% | ~30-40 hours |

**Notes:**
- These are rough estimates based on similar datasets
- Actual performance depends on data quality, class distribution, and hyperparameters
- The Bhopal dataset has significant class imbalance (Building: 65%) which affects mIoU

---

## Troubleshooting

### Common Issues

**1. Out of Memory (OOM)**

```bash
# Reduce batch size
--options batch_size=4 gradient_accumulation_steps=16

# Or enable empty_cache
--options empty_cache=True
```

**2. LoRA Model Not Found**

If you see `DefaultLORASegmentorV2 not found`:
- Ensure Pointcept version supports LoRA models
- Check that the LoRA module is properly registered

**3. Checkpoint Loading Errors**

For LoRA, ensure `backbone_path` is set correctly in the config:
```python
model = dict(
    backbone_path="concerto_large_outdoor.pth",
    # ...
)
```

**4. NaN Loss**

- Reduce learning rate
- Enable gradient clipping: `clip_grad=3.0`
- Check for invalid input data

### Verifying Correct Setup

1. **Check trainable parameters:**
   ```
   grep "trainable" in training log
   ```

2. **Monitor GPU memory:**
   ```bash
   nvidia-smi --query-gpu=memory.used --format=csv -l 5
   ```

3. **Verify loss is decreasing:**
   ```bash
   tail -f outputs/*/logs/train_*.log | grep "loss:"
   ```

---

## Summary

### Quick Decision Guide

| Scenario | Recommended Strategy |
|----------|---------------------|
| Quick evaluation of pre-trained features | **Linear Probing (LP)** |
| Better feature utilization with frozen encoder | **Decoder Probing (DP)** |
| Best possible performance | **Full Fine-tuning (FT)** |
| Limited GPU memory or want parameter efficiency | **LoRA Fine-tuning** |
| Fast experimentation | **LP or LoRA** |

### Key Takeaways

1. **Linear Probing**: `enc_mode=True`, `freeze_backbone=True` - only seg_head trains
2. **Decoder Probing**: `enc_mode=False`, `freeze_encoder=True` - decoder + seg_head train
3. **Full Fine-tuning**: `enc_mode=False`, `freeze_encoder=False` - everything trains
4. **LoRA**: `enc_mode=True`, `traceable=True`, `use_lora=True` - LoRA adapters + seg_head train

---

## Official Reference Configs

The Bhopal configs are based on official Pointcept Concerto configurations. Below are the reference files for each strategy:

### Reference Config Files (Pointcept)

**Location:** `Pointcept/configs/concerto/`

| Strategy | Bhopal Config | Official Reference Config |
|----------|---------------|---------------------------|
| **Decoder Probing** | `concerto_140gb_12class_decoder_probe_9ch.py` | `semseg-ptv3-large-v1m1-0b-scannet-dec.py` |
| **Full Fine-tuning** | `concerto_140gb_12class_finetune_9ch.py` | `semseg-ptv3-large-v1m1-0c-scannet-ft.py` |
| **LoRA Fine-tuning** | `concerto_140gb_12class_lora_9ch.py` | `semseg-ptv3-large-v1m1-0f-scannet-ft-lora.py` |

### LoRA Variants (Official)

The official Pointcept repo includes several LoRA variants with different hyperparameters:

| Config | Description |
|--------|-------------|
| `semseg-ptv3-large-v1m1-0f-scannet-ft-lora.py` | Base LoRA config |
| `semseg-ptv3-large-v1m1-0e8-scannet-ft-lora-la20.py` | LoRA alpha=20 |
| `semseg-ptv3-large-v1m1-0e9-scannet-ft-lora-la50.py` | LoRA alpha=50 |
| `semseg-ptv3-large-v1m1-0e10-scannet-ft-lora-la100.py` | LoRA alpha=100 |
| `semseg-ptv3-large-v1m1-0e12-scannet-ft-lora-lr1.py` | LR variant 1 |
| `semseg-ptv3-large-v1m1-0e13-scannet-ft-lora-lr5.py` | LR variant 5 |

### Other Dataset Configs (for Reference)

| Dataset | Decoder Probe | Full FT |
|---------|---------------|---------|
| ScanNet | `semseg-ptv3-large-v1m1-0b-scannet-dec.py` | `semseg-ptv3-large-v1m1-0c-scannet-ft.py` |
| ScanNet200 | `semseg-ptv3-base-v1m1-1c-scannet200-dec.py` | `semseg-ptv3-base-v1m1-1c-scannet200-ft.py` |
| ScanNet++ | `semseg-ptv3-base-v1m1-2c-scannetpp-dec.py` | N/A |
| S3DIS | `semseg-ptv3-base-v1m1-3c-s3dis-dec.py` | N/A |

### Instance Segmentation Configs

For instance segmentation tasks, similar patterns apply:

| Dataset | Decoder Probe | Full FT |
|---------|---------------|---------|
| ScanNet | `insseg-ptv3-large-v1m1-0b-scannet-pg-dec.py` | `insseg-ptv3-large-v1m1-0c-scannet-pg-ft.py` |
| ScanNet200 | `insseg-ptv3-large-v1m1-1b-scannet200-pg-dec.py` | `insseg-ptv3-large-v1m1-1c-scannet200-pg-ft.py` |

---

## Indoor vs Outdoor Dataset Settings

The Bhopal configs use hyperparameters specifically tuned for **outdoor datasets**, which differ significantly from indoor datasets like ScanNet. This section explains these differences.

### Parameter Comparison

| Parameter | Indoor (ScanNet) | Outdoor (Bhopal/KITTI/nuScenes/Waymo) |
|-----------|------------------|---------------------------------------|
| `weight_decay` | 0.02 | **0.005** |
| `pct_start` | 0.05 | **0.04** |
| `final_div_factor` | 1000.0 | **100.0** |
| `epoch` | 800 | **50** |
| `mix_prob` | 0.8 | 0.8 |
| `lr` | 0.006 | 0.002 |

### Why Outdoor Datasets Use Different Settings

**1. Lower Weight Decay (0.005 vs 0.02)**
- Outdoor point clouds are sparser and noisier than indoor scenes
- Less regularization prevents underfitting on challenging outdoor data
- Indoor scenes have denser, cleaner geometry allowing stronger regularization

**2. Faster Warmup (pct_start=0.04 vs 0.05)**
- Outdoor datasets typically have fewer training samples per epoch
- Faster warmup helps reach peak learning rate sooner
- Indoor datasets with more samples benefit from longer warmup

**3. Gentler LR Decay (final_div_factor=100 vs 1000)**
- Outdoor scenes are more variable, requiring continued learning
- Final LR = max_lr / final_div_factor: 0.002/100 = 0.00002 (outdoor) vs 0.006/1000 = 0.000006 (indoor)
- Higher final LR helps adapt to diverse outdoor conditions

**4. Fewer Epochs (50 vs 800)**
- Outdoor LiDAR datasets have larger scenes with more points per sample
- Each epoch covers more data, so fewer epochs are needed
- Indoor datasets like ScanNet have smaller, denser rooms requiring more iterations

### Verification: Bhopal Uses Official Outdoor Settings

Our Bhopal configs match the official Pointcept outdoor configurations:

| Config Source | weight_decay | pct_start | final_div_factor | epoch |
|---------------|--------------|-----------|------------------|-------|
| **Bhopal (ours)** | 0.005 | 0.04 | 100.0 | 50 |
| nuScenes (official) | 0.005 | 0.04 | 100.0 | 50 |
| SemanticKITTI (official) | 0.005 | 0.04 | 100.0 | 50 |
| Waymo (official) | 0.005 | 0.04 | 100.0 | 50 |
| ScanNet (official) | 0.02 | 0.05 | 1000.0 | 800 |

**Conclusion:** The Bhopal configs are **correctly configured** for outdoor MLS data, following the same hyperparameters as nuScenes, SemanticKITTI, and Waymo.

### Official Outdoor Reference Configs

| Dataset | Base Config File | Location |
|---------|-----------------|----------|
| nuScenes | `semseg-pt-v3m1-0-base.py` | `Pointcept/configs/nuscenes/` |
| SemanticKITTI | `semseg-pt-v2m2-0-base.py` | `Pointcept/configs/semantickitti/` |
| Waymo | `semseg-pt-v3m1-0-base.py` | `Pointcept/configs/waymo/` |

---

For more details on the official Concerto configurations, refer to the [Pointcept Concerto documentation](https://github.com/Pointcept/Pointcept).
