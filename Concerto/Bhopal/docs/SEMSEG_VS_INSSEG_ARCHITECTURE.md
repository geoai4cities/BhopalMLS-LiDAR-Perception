# Semantic Segmentation vs Instance Segmentation: Architecture Comparison

## Overview

Both tasks share the same Concerto (PT-v3m2) backbone but differ in the model wrapper,
prediction heads, loss functions, data pipeline, and evaluation.

```
SEMANTIC SEGMENTATION                    INSTANCE SEGMENTATION
(DefaultSegmentorV2)                     (PG-v1m2 / PointGroup)

Input Point Cloud                        Input Point Cloud
       |                                        |
  [ PT-v3m2 Backbone ]                   [ PT-v3m2 Backbone ]
       |                                        |
  [ seg_head ]                           [ seg_head ] + [ bias_head ]
       |                                    |               |
  Per-point class logits              Class logits    Centroid offsets
       |                                    |               |
  CrossEntropy + Lovasz              CrossEntropy     L1 + Cosine loss
                                    + Lovasz                |
                                         |          (inference only)
                                         |        Ball Query Clustering
                                         |               |
                                    Per-point class   Instance proposals
                                                     (mAP@25, mAP@50)
```

---

## 1. Model Wrapper

| Aspect | Semantic Seg | Instance Seg |
|--------|-------------|-------------|
| **Model class** | `DefaultSegmentorV2` | `PG-v1m2` (PointGroup) |
| **Code location** | `pointcept/models/default.py` | `pointcept/models/point_group/point_group_v1m2_custom_criteria.py` |
| **Prediction heads** | `seg_head` only | `seg_head` + `bias_head` |
| **Output (train)** | `{loss}` | `{loss, seg_loss, bias_l1_loss, bias_cosine_loss}` |
| **Output (eval)** | `{loss, seg_logits}` | `{loss, ..., pred_scores, pred_masks, pred_classes}` |

---

## 2. Prediction Heads

### Semantic Segmentation: `seg_head` only

```python
seg_head = nn.Linear(backbone_out_channels, num_classes)
```

Single linear layer mapping backbone features to per-point class logits.

### Instance Segmentation: `seg_head` + `bias_head`

```python
seg_head = nn.Linear(backbone_out_channels, num_classes)

bias_head = nn.Sequential(
    nn.Linear(backbone_out_channels, backbone_out_channels),
    nn.BatchNorm1d(backbone_out_channels),
    nn.ReLU(),
    nn.Linear(backbone_out_channels, 3),   # predicts XYZ offset
)
```

The `bias_head` is a 2-layer MLP that predicts, for each point, the 3D offset vector
from that point to the centroid of the instance it belongs to. This groups points
into instances at inference time.

| Head | Input dim | Output dim | Parameters (LP, 12-class) |
|------|----------|-----------|--------------------------|
| `seg_head` | 1728 | 12 | 20,748 |
| `bias_head` | 1728 | 3 | 2,992,899 + 3,459 (BN) |
| **Total trainable (LP)** | | | ~3.0M |

---

## 3. Backbone Configuration (3 Probing Modes)

The PT-v3m2 backbone is identical in both tasks. The probing mode controls which
parts are frozen and determines `backbone_out_channels`.

### Linear Probe (LP): `enc_mode=True, freeze_backbone=True`

```
Encoder only (no decoder). All encoder weights frozen.
Only seg_head (+ bias_head for insseg) train.

Encoder stages: 64 + 128 + 256 + 512 + 768 = 1728-dim concatenated
backbone_out_channels = 1728
```

### Decoder Probe (DP): `enc_mode=False, freeze_encoder=True`

```
Full encoder + decoder. Encoder frozen, decoder trains.
Decoder + seg_head (+ bias_head for insseg) train.

Decoder output: 64-dim
backbone_out_channels = 64
```

### Full Fine-tune (FT): `enc_mode=False, freeze_encoder=False`

```
Full encoder + decoder. Everything trains.

Decoder output: 64-dim
backbone_out_channels = 64
```

| Mode | backbone_out_channels | Trainable params (approx) |
|------|----------------------|--------------------------|
| LP (semseg) | 1728 | ~21K (seg_head only) |
| LP (insseg) | 1728 | ~3.0M (seg_head + bias_head) |
| DP (semseg) | 64 | ~30M (decoder + seg_head) |
| DP (insseg) | 64 | ~30M (decoder + seg_head + bias_head) |
| FT (semseg) | 64 | ~300M (full model) |
| FT (insseg) | 64 | ~300M (full model) |

---

## 4. Loss Functions

### Semantic Segmentation

```python
loss = CrossEntropyLoss(seg_logits, segment) + LovaszLoss(seg_logits, segment)
```

Single combined loss. Both losses use `ignore_index=-1`.

### Instance Segmentation

```python
# 1. Semantic segmentation loss (same as semseg)
seg_loss = CrossEntropyLoss(logit_pred, segment) + LovaszLoss(logit_pred, segment)

# 2. Bias L1 loss: offset magnitude
bias_gt = instance_centroid - coord              # ground truth offset
bias_dist = |bias_pred - bias_gt|.sum(dim=-1)    # L1 distance
bias_l1_loss = (bias_dist * mask).sum() / mask.sum()

# 3. Bias cosine loss: offset direction
bias_cosine_loss = -(normalize(bias_pred) * normalize(bias_gt)).sum(-1)
bias_cosine_loss = (bias_cosine_loss * mask).sum() / mask.sum()

# Total
loss = seg_loss + bias_l1_loss + bias_cosine_loss
```

The `mask` excludes points with `instance == instance_ignore_index` (stuff classes
and padding). Only "thing" class points contribute to offset losses.

| Loss component | What it learns | Applied to |
|---------------|---------------|-----------|
| `seg_loss` | Per-point class prediction | All valid points |
| `bias_l1_loss` | Distance to instance centroid | Thing points only |
| `bias_cosine_loss` | Direction to instance centroid | Thing points only |

---

## 5. Stuff vs Thing Classes

Instance segmentation distinguishes between:
- **Stuff classes**: Continuous/amorphous regions not decomposed into instances
  (e.g., Ground, Road, Vegetation, Wire). Excluded from clustering and offset loss.
- **Thing classes**: Discrete, countable objects that form instances
  (e.g., Vehicle, Building, Human, Pole).

This is controlled by `segment_ignore_index`:

| Config | segment_ignore_index | Stuff classes | Thing classes |
|--------|---------------------|--------------|--------------|
| 12-class | (-1, 0, 1, 3, 4) | padding, Ground, Vegetation, Road, Wire | 8 things |
| 21-class | (-1, 0, 2, 5, 6) | padding, Ground, Vegetation, Road, Wire | 17 things |

Semantic segmentation has no stuff/thing distinction - all classes are predicted equally.

---

## 6. Inference Pipeline

### Semantic Segmentation

```
1. Forward pass -> seg_logits [N, C]
2. argmax(seg_logits) -> per-point class prediction
3. Compute mIoU, mAcc, OA
```

### Instance Segmentation

```
1. Forward pass -> seg_logits [N, C] + bias_pred [N, 3]
2. center_pred = coord + bias_pred                    # shifted point positions
3. center_pred /= voxel_size                           # normalize to grid space
4. segment_pred = argmax(softmax(seg_logits))          # per-point class
5. Filter out stuff classes (segment_ignore_index)
6. Ball query clustering:
   - ballquery_batch_p(center_pred, cluster_thresh=1.5, closed_points=600)
   - bfs_cluster(segment_pred, min_points=50)
7. For each cluster proposal:
   - Score = mean softmax probability of predicted class
   - Class = majority vote from segment_pred
   - Mask = binary point membership
8. Filter proposals with > cluster_propose_points (200)
9. Evaluate with mAP@25, mAP@50
```

Key clustering parameters (Bhopal):

| Parameter | Value | Meaning |
|-----------|-------|---------|
| `voxel_size` | 0.01 | Grid normalization (must match GridSample grid_size) |
| `cluster_thresh` | 1.5 | Ball query radius (in normalized space) |
| `cluster_closed_points` | 600 | Max neighbors in ball query |
| `cluster_propose_points` | 200 | Min points for a valid proposal |
| `cluster_min_points` | 50 | Min points for BFS cluster seed |

---

## 7. Data Pipeline Differences

### Shared transforms (identical in both)

```
RandomScale([0.2, 0.2])        # 0.2x downscaling for MLS data
RandomRotate(z-axis)            # Full rotation around vertical
RandomScale([0.9, 1.1])         # Scale augmentation
RandomFlip(p=0.5)               # Horizontal flip
RandomJitter(0.0025, 0.01)      # Coordinate noise
GridSample(grid_size=0.01)      # Voxelization
PointClip(...)                  # Spatial bounding box
NormalizeColor()                # RGB /= 255
ToTensor()
```

### Instance seg ADDITIONAL transforms

```
InstanceParser(segment_ignore_index, instance_ignore_index=-1)
```

This transform processes instance labels:
1. Sets instance IDs of stuff-class points to `instance_ignore_index` (-1)
2. Re-indexes remaining instance IDs to consecutive integers (0, 1, 2, ...)
3. Computes per-instance centroids and bounding boxes
4. Adds `instance_centroid` and `bbox` to data dict

### Collect keys

| Task | keys | feat_keys |
|------|------|-----------|
| Semantic seg | coord, grid_coord, segment | coord, color, normal |
| Instance seg | coord, grid_coord, segment, **instance**, **instance_centroid**, **bbox** | coord, color, normal |

### Dataset loading

| Aspect | Semantic seg | Instance seg |
|--------|-------------|-------------|
| `load_instance` | False (default) | **True** |
| HDF5 fields loaded | points, labels, normals, rgb | points, labels, **global_ids**, normals, rgb |

---

## 8. Evaluation Metrics

| Metric | Semantic Seg | Instance Seg |
|--------|-------------|-------------|
| **Primary** | mIoU | mAP (mean Average Precision) |
| **Secondary** | mAcc, OA (Overall Accuracy) | AP@50, AP@25 |
| **Evaluator hook** | `SemSegEvaluator` | `InsSegEvaluator` |
| **Tester class** | `SemSegTester` | `InsSegTester` |
| **Best model saved by** | Highest mIoU | Highest mAP |
| **Per-class metric** | IoU per class | AP per thing class |
| **Stuff classes** | Included in mIoU | Excluded from mAP |

### mIoU (Semantic Seg)

```
IoU(class_c) = TP_c / (TP_c + FP_c + FN_c)
mIoU = mean(IoU) over all classes
```

### mAP (Instance Seg)

```
For each IoU threshold t in {25%, 50%}:
  For each thing class c:
    1. Rank predicted instances by confidence score
    2. Match predictions to ground truth instances (IoU > t)
    3. Compute precision-recall curve
    4. AP(c, t) = area under PR curve
  AP@t = mean(AP(c, t)) over thing classes

mAP = mean(AP@25, AP@50)
```

---

## 9. Config File Mapping

| Probing Mode | Semantic Seg Config | Instance Seg Config |
|-------------|--------------------|--------------------|
| LP 9ch (12c) | `concerto_140gb_12class_linear_probe_9ch.py` | `concerto_140gb_12class_insseg_lp_9ch.py` |
| DP 9ch (12c) | `concerto_140gb_12class_decoder_probe_9ch.py` | `concerto_140gb_12class_insseg_dp_9ch.py` |
| FT 9ch (12c) | `concerto_140gb_12class_finetune_9ch.py` | `concerto_140gb_12class_insseg_ft_9ch.py` |
| LP 9ch (21c) | N/A | `concerto_140gb_21class_insseg_lp_9ch.py` |
| DP 9ch (21c) | N/A | `concerto_140gb_21class_insseg_dp_9ch.py` |
| FT 9ch (21c) | N/A | `concerto_140gb_21class_insseg_ft_9ch.py` |

---

## 10. Training Scripts

| Aspect | Semantic Seg | Instance Seg |
|--------|-------------|-------------|
| **Script** | `train_bhopal_140gb.sh` | `train_bhopal_insseg.sh` |
| **Batch size (LP)** | 32 (grad_accum=2, eff=64) | 4 (grad_accum=6, eff=24) |
| **Batch size (DP)** | 16 (grad_accum=4, eff=64) | 4 (grad_accum=6, eff=24) |
| **Batch size (FT)** | 16 (grad_accum=4, eff=64) | 2 (grad_accum=12, eff=24) |
| **Memory** | ~60-80 GB | ~80-120 GB |

Instance seg uses smaller batch sizes because `bias_head` (2-layer MLP) and
per-point centroid storage (`instance_centroid [N, 3]`) require additional memory.

---

## 11. Summary Table

| Feature | Semantic Segmentation | Instance Segmentation |
|---------|----------------------|----------------------|
| Model | DefaultSegmentorV2 | PG-v1m2 (PointGroup) |
| Backbone | PT-v3m2 (shared) | PT-v3m2 (shared) |
| Heads | seg_head (linear) | seg_head (linear) + bias_head (MLP) |
| Loss | CE + Lovasz | CE + Lovasz + L1 offset + cosine offset |
| Stuff/Thing | No distinction | Stuff excluded from clustering + offset loss |
| Inference | argmax | argmax + ball query + BFS clustering |
| Primary metric | mIoU | mAP |
| Data needed | coord, segment | coord, segment, instance, instance_centroid |
| Memory | Lower | Higher (~1.5x) |
| Effective batch | 64 | 24 |
