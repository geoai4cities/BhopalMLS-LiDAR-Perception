# Utonia Probing and Tuning Guide for Bhopal MLS Dataset

This document explains how to use the Utonia probing and fine-tuning methods for Bhopal MLS semantic segmentation. It covers the different training strategies, their configurations, and how to adapt them for your specific use case.

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

---

## Overview

Utonia provides several strategies for adapting its cross-domain pre-trained encoder to downstream tasks like Bhopal MLS semantic segmentation:

| Strategy | Backbone Frozen | Decoder | Trainable Params | Use Case | Expected Performance |
|----------|-----------------|---------|------------------|----------|---------------------|
| **Linear Probing (LP)** | Yes (entire) | Not used | ~16.6K | Quick evaluation | Baseline |
| **Decoder Probing (DP)** | Yes (encoder) | Trainable | ~20.5M | Better feature utilization | Improved |
| **Full Fine-tuning (FT)** | No | Trainable | ~157.7M | Best performance | Best |
| **LoRA Fine-tuning** | Partially | Not used | ~0.2M | Parameter-efficient | Good |

---

## Pre-trained Model Weights

### Unified Cross-Domain Model

**Model:** `utonia.pth` (524MB, ~137.25M encoder params)

Unlike Concerto which has separate indoor/outdoor checkpoints, Utonia uses a **single unified checkpoint** trained across diverse domains (remote sensing, outdoor LiDAR, indoor RGB-D, object-centric CAD, and RGB-lifted point clouds).

**Location:** `Utonia_pretrained/utonia_pretrained_weights/utonia.pth`

### Model Architecture (PT-v3m3)

```python
# Encoder
enc_depths=(3, 3, 3, 12, 3)
enc_channels=(54, 108, 216, 432, 576)
enc_num_head=(3, 6, 12, 24, 32)
enc_patch_size=(1024, 1024, 1024, 1024, 1024)

# New: 3D Rotary Position Embedding (replaces RPE in PT-v3m2)
# RoPE CONSTRAINT: head_dim (channels/num_heads) must be divisible by 3
rope_base=10

# Decoder (when enc_mode=False)
# NOTE: dec_num_head adjusted from PTv3 defaults (6,6,12,32) to satisfy
# RoPE head_dim%3==0 constraint. Decoder is randomly initialized so this is safe.
dec_depths=(3, 3, 3, 3)
dec_channels=(96, 96, 192, 384)
dec_num_head=(8, 8, 16, 32)   # head_dims: 12, 12, 12, 12 (all %3==0)
dec_patch_size=(1024, 1024, 1024, 1024)
```

### Important: Encoder vs Decoder in Pre-trained Weights

**The checkpoint contains only ENCODER weights.** The decoder is NOT pre-trained.

| Component | Pre-trained? | Notes |
|-----------|--------------|-------|
| **Encoder** | Yes | Core pre-trained backbone (~137.25M params), loaded from checkpoint |
| **Decoder** | No | Randomly initialized, trained during DP/FT |
| **Seg Head** | No | Randomly initialized, always trained |

---

## Training Strategies Explained

### 1. Linear Probing (LP) - `lp 9ch`

**Purpose:** Evaluate the quality of pre-trained features without modifying them.

**Configuration:**
```python
model = dict(
    type="DefaultSegmentorV2",
    backbone_out_channels=1386,  # Concatenated encoder features (54+108+216+432+576)
    backbone=dict(
        type="PT-v3m3",
        enc_mode=True,         # Encoder-only, no decoder
        freeze_encoder=False,  # Handled by freeze_backbone
    ),
    freeze_backbone=True,      # Freeze entire backbone
)
```

**What Trains:**
- Only `seg_head` (1386 -> 12 classes = ~16.6K params)

**Use Case:** Quick baseline to measure feature quality.

---

### 2. Decoder Probing (DP) - `dp 9ch`

**Purpose:** Train a decoder on top of frozen encoder features for better multi-scale utilization.

**Configuration:**
```python
model = dict(
    type="DefaultSegmentorV2",
    backbone_out_channels=96,   # Decoder output (first dec_channel)
    backbone=dict(
        type="PT-v3m3",
        enc_mode=False,         # Use decoder
        freeze_encoder=True,    # Freeze encoder, train decoder
        dec_depths=(3, 3, 3, 3),
        dec_channels=(96, 96, 192, 384),
        dec_num_head=(8, 8, 16, 32),   # head_dim=12 (must be %3==0 for RoPE)
        dec_patch_size=(1024, 1024, 1024, 1024),
    ),
    freeze_backbone=False,      # Allow decoder training
)
```

**What Trains:**
- Decoder (~20.5M params)
- `seg_head` (96 -> 12 classes = ~1.2K params)

**Use Case:** Better utilize spatial hierarchy from pre-trained features.

---

### 3. Full Fine-tuning (FT) - `ft 9ch`

**Purpose:** Adapt the entire model to the target dataset for best performance.

**Configuration:**
```python
model = dict(
    type="DefaultSegmentorV2",
    backbone_out_channels=96,
    backbone=dict(
        type="PT-v3m3",
        enc_mode=False,          # Use decoder
        freeze_encoder=False,    # Train encoder (with lower LR)
        dec_depths=(3, 3, 3, 3),
        dec_channels=(96, 96, 192, 384),
        dec_num_head=(8, 8, 16, 32),   # head_dim=12 (must be %3==0 for RoPE)
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
- Encoder (~137.25M params) - with 10x lower learning rate
- Decoder (~20.5M params)
- `seg_head` (96 -> 12 classes = ~1.2K params)
- **Total: ~157.7M params** (matches Utonia paper Table 4)

**Use Case:** Best performance when you have sufficient data.

---

### 4. LoRA Fine-tuning - `lora 9ch`

**Purpose:** Parameter-efficient fine-tuning using Low-Rank Adaptation.

**Configuration:**
```python
model = dict(
    type="DefaultLORASegmentorV2",  # Special LoRA model
    backbone_out_channels=1386,
    use_lora=True,
    lora_r=8,           # LoRA rank
    lora_alpha=16,      # LoRA scaling
    lora_dropout=0.1,   # LoRA dropout
    backbone_path="path/to/utonia.pth",  # Required for LoRA
    backbone=dict(
        type="PT-v3m3",
        enc_mode=True,
        traceable=True,       # Required for LoRA
        freeze_encoder=True,
    ),
    freeze_backbone=False,    # LoRA adapters are trainable
)
```

**What Trains:**
- LoRA adapters (~0.2M params, ~0.1% of encoder)
- `seg_head` (1386 -> 12 classes = ~16.6K params)

**Use Case:** When you want fine-tuning quality with limited compute/memory.

---

## Configuration Parameter Reference

### Key Parameters Comparison

| Parameter | Linear Probe | Decoder Probe | Full FT | LoRA |
|-----------|-------------|---------------|---------|------|
| `backbone.type` | `PT-v3m3` | `PT-v3m3` | `PT-v3m3` | `PT-v3m3` |
| `enc_mode` | `True` | `False` | `False` | `True` |
| `freeze_encoder` | N/A* | `True` | `False` | `True` |
| `freeze_backbone` | `True` | `False` | `False` | `False` |
| `backbone_out_channels` | 1386 | 96 | 96 | 1386 |
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
| Linear Probing | 8 | 8 | 64 | ~30-35GB |
| Decoder Probing | 6 | 8 | 48 | ~40-45GB |
| Full Fine-tuning | 4 | 8 | 32 | ~50-55GB |
| LoRA Fine-tuning | 8 | 8 | 64 | ~30-35GB |

#### 140GB GPU (H200)

| Strategy | batch_size | grad_accum | Effective Batch | Est. Memory |
|----------|------------|------------|-----------------|-------------|
| Linear Probing | 64 | 1 | 64 | ~35-45GB |
| Decoder Probing | 8 | 8 | 64 | ~55-65GB |
| Full Fine-tuning | 6 | 8 | 48 | ~75-85GB |
| LoRA Fine-tuning | 8 | 8 | 64 | ~35-45GB |

**Note:** Utonia's slightly smaller encoder (137.25M vs Concerto's 207M) means somewhat lower memory usage per batch compared to Concerto.

---

## Training Commands

### Available Commands

```bash
cd Utonia/Bhopal

# Linear Probing
./scripts/train_bhopal_40gb.sh lp 9ch
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
| `lp 9ch` | Linear Probing | `utonia_140gb_12class_linear_probe_9ch.py` | Frozen | N/A | ~30-35GB |
| `dp 9ch` | Decoder Probing | `utonia_140gb_12class_decoder_probe_9ch.py` | Frozen | Trainable | ~40-45GB |
| `ft 9ch` | Full Fine-tuning | `utonia_140gb_12class_finetune_9ch.py` | Trainable | Trainable | ~50-55GB |
| `lora 9ch` | LoRA Fine-tuning | `utonia_140gb_12class_lora_9ch.py` | LoRA | N/A | ~30-35GB |

### Recommended Order

1. **`lp 9ch`** - Establish baseline with frozen Utonia encoder
2. **`dp 9ch`** - Better feature utilization with trainable decoder
3. **`ft 9ch`** - Best performance with full adaptation
4. **`lora 9ch`** - Parameter-efficient alternative if FT is too expensive

---

## Checkpoint Loading

### For Released Weights (utonia.pth)

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

weight = "Utonia_pretrained/utonia_pretrained_weights/utonia.pth"
```

### For LoRA

LoRA uses `backbone_path` in the model config instead of the weight parameter:

```python
model = dict(
    type="DefaultLORASegmentorV2",
    backbone_path=".../utonia.pth",
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

### Checkpoint Config (Stored Inside utonia.pth)

The checkpoint stores its own config for verification:
```python
{
    'in_channels': 9,
    'enc_channels': (54, 108, 216, 432, 576),
    'enc_num_head': (3, 6, 12, 24, 32),
    'rope_base': 10,
    'traceable': True,      # GridPooling default
    'mask_token': True,      # SSL pretraining flag (ignored in downstream)
    'enc_mode': True,
}
```

**Note:** `mask_token=True` in the checkpoint was used during self-supervised pretraining. Our configs set `mask_token=False` for downstream tasks — this is correct.

---

## Expected Performance

### Estimated mIoU (Bhopal 12-class)

| Strategy | Input | Expected mIoU Range | Training Time (50 epochs) |
|----------|-------|---------------------|---------------------------|
| LP-9ch | XYZ+RGB+normal | ~50-60% | ~25-35 hours |
| DP-9ch | XYZ+RGB+normal | ~55-65% | ~45-55 hours |
| FT-9ch | XYZ+RGB+normal | ~60-70% | ~65-85 hours |
| LoRA-9ch | XYZ+RGB+normal | ~55-65% | ~25-35 hours |

**Notes:**
- Utonia's cross-domain pretraining may provide slightly different transfer characteristics than Concerto's outdoor-specific pretraining
- The Bhopal dataset has significant class imbalance (Building: 65%) which affects mIoU
- Actual performance depends on data quality, class distribution, and hyperparameters

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
    backbone_path="utonia.pth",
    # ...
)
```

**4. RoPE Head Dimension Error**

If you see `Head dimension must be divisible by 3 for 3D RoPE`:
- Utonia's Point3DRoPE splits head_dim into 3 equal parts (X, Y, Z)
- Every attention block (encoder AND decoder) must have `channels / num_heads` divisible by 3
- Encoder head_dims are all 18 (OK). Decoder defaults from PTv3 give head_dim=16 (NOT OK)
- Fix: use `dec_num_head=(8, 8, 16, 32)` instead of `(6, 6, 12, 32)` → head_dim=12
- This only affects DP and FT (`enc_mode=False`). LP and LoRA (`enc_mode=True`) skip the decoder

**5. NaN Loss**

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

1. **Linear Probing**: `enc_mode=True`, `freeze_backbone=True` - only seg_head trains (~16.6K params)
2. **Decoder Probing**: `enc_mode=False`, `freeze_encoder=True` - decoder + seg_head train (~20.5M params)
3. **Full Fine-tuning**: `enc_mode=False`, `freeze_encoder=False` - everything trains (~157.7M params)
4. **LoRA**: `enc_mode=True`, `traceable=True`, `use_lora=True` - LoRA adapters + seg_head train (~0.2M params)

---

*Last Updated: 2026-03-11*
*Covers: Utonia (PT-v3m3) probing and fine-tuning for Bhopal MLS dataset*
