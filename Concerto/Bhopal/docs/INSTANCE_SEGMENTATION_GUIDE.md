# Instance Segmentation Guide for Bhopal MLS (PointGroup-based)

This document covers adapting Concerto's PointGroup instance segmentation pipeline from indoor datasets (ScanNet) to the outdoor Bhopal MLS dataset. It details the architecture, data analysis, 21-class taxonomy design, instance-cutting analysis, and configuration.

## Table of Contents

1. [Overview: Semantic vs Instance Segmentation](#overview-semantic-vs-instance-segmentation)
2. [PointGroup Architecture (PG-v1m2)](#pointgroup-architecture-pg-v1m2)
3. [Data Requirements](#data-requirements)
4. [Bhopal HDF5 Data Availability](#bhopal-hdf5-data-availability)
5. [Instance Cutting Analysis](#instance-cutting-analysis)
6. [Do We Need to Remake the Data?](#do-we-need-to-remake-the-data)
7. [Dataset Loader Changes](#dataset-loader-changes)
8. [InstanceParser Transform](#instanceparser-transform)
9. [Thing vs Stuff: Definitions and Decisions](#thing-vs-stuff-definitions-and-decisions)
10. [21-Class Taxonomy](#21-class-taxonomy)
11. [Per-Split Instance Statistics](#per-split-instance-statistics)
12. [Configuration Reference](#configuration-reference)
13. [Training Pipeline](#training-pipeline)
14. [Clustering Parameters for Outdoor MLS](#clustering-parameters-for-outdoor-mls)
15. [Evaluation: InsSegEvaluator](#evaluation-insseg-evaluator)
16. [Implementation Checklist](#implementation-checklist)
17. [Reference: Official ScanNet Instance Seg Configs](#reference-official-scannet-instance-seg-configs)
18. [Testing Pipeline (InsSegTester)](#testing-pipeline-inssegtester)
19. [Known Issues and Lessons from Pointcept GitHub](#known-issues-and-lessons-from-pointcept-github)
20. [Bug Fix: Label Remapping for Unmapped Classes](#bug-fix-label-remapping-for-unmapped-classes)

---

## Overview: Semantic vs Instance Segmentation

| Aspect | Semantic Segmentation (current) | Instance Segmentation (target) |
|--------|--------------------------------|-------------------------------|
| **Task** | Per-point class label | Per-point class + instance grouping |
| **Output** | "This point is a Vehicle" | "This point is Vehicle #7" |
| **Model** | `DefaultSegmentorV2` | `PG-v1m2` (PointGroup) |
| **Heads** | `seg_head` only | `seg_head` + `bias_head` |
| **Losses** | CE + Lovasz | CE + Lovasz + bias_L1 + bias_cosine |
| **Evaluator** | `SemSegEvaluator` (mIoU) | `InsSegEvaluator` (mAP@25, mAP@50) |
| **Tester** | Default | `InsSegTester` |
| **Num Classes** | 12 (semantic seg) | **12** (baseline) or **21** (fine-grained taxonomy) |

**Why instance segmentation matters for mobility studies:** Semantic segmentation tells you "there is a sidewalk here." Instance segmentation tells you "this is sidewalk segment #3, that is sidewalk segment #4" -- enabling measurement of individual obstacle dimensions, connectivity analysis and per-object statistics. The finer 21-class taxonomy distinguishes stairs (mobility-relevant), vehicle types (2/3/4-wheeler), and roadside infrastructure.

---

## PointGroup Architecture (PG-v1m2)

**Source file:** `Pointcept/pointcept/models/point_group/point_group_v1m2_custom_criteria.py`

PointGroup extends the backbone (PT-v3m2) with a bias regression head that predicts the offset from each point to its instance centroid. At inference, these predicted centroids are clustered to form instance proposals.

### Architecture Diagram

```
Input Point Cloud (coord, color, normal)
          |
     [ PT-v3m2 Backbone ]  (encoder + decoder)
          |
     backbone features (64-dim)
          |
    +-----+-----+
    |             |
[ seg_head ]  [ bias_head ]
    |             |
semantic logits   bias (3D offset to centroid)
    |             |
    +------+------+
           |
   [ Ball Query Clustering ]  (inference only)
           |
   Instance proposals (masks, scores, classes)
```

### Model Components

```python
@MODELS.register_module("PG-v1m2")
class PointGroup(nn.Module):
    def __init__(self,
        backbone,                          # PT-v3m2 backbone config
        backbone_out_channels=64,          # Must match decoder output (64 for large)
        semantic_num_classes=20,           # Number of semantic classes
        semantic_ignore_index=-1,          # Ignore index for CE loss
        segment_ignore_index=(-1, 0, 1),  # Classes excluded from clustering
        instance_ignore_index=-1,          # Ignore flag for instance labels
        cluster_thresh=1.5,               # Ball query radius (in voxel units)
        cluster_closed_points=300,        # Mean active points for ball query
        cluster_propose_points=100,       # Min points per proposal
        cluster_min_points=50,            # Min points for BFS cluster
        voxel_size=0.02,                  # For normalizing center predictions
        criteria=None,                    # Loss functions (CE + Lovasz)
        freeze_backbone=False,            # Whether to freeze backbone
    ):
```

### Training Losses

| Loss | Formula | Purpose |
|------|---------|---------|
| `seg_loss` | CrossEntropy + Lovasz on semantic predictions | Semantic segmentation accuracy |
| `bias_l1_loss` | `\|bias_pred - (centroid - coord)\|` for valid instances | Centroid offset magnitude |
| `bias_cosine_loss` | `-cos(bias_pred, centroid - coord)` for valid instances | Centroid offset direction |
| **Total** | `seg_loss + bias_l1_loss + bias_cosine_loss` | |

### Inference Clustering

At test time, PointGroup:

1. Predicts semantic class and centroid offset for each point
2. Shifts points by predicted offset: `center_pred = coord + bias_pred`
3. Normalizes by voxel size: `center_pred /= voxel_size`
4. Excludes points from `segment_ignore_index` classes (stuff classes)
5. Runs `ballquery_batch_p` with `cluster_thresh` radius
6. Runs `bfs_cluster` to form connected components
7. Filters proposals by `cluster_propose_points` minimum
8. Scores each proposal by mean softmax confidence

**Output:** `pred_scores`, `pred_masks` (binary, [num_instances, N]), `pred_classes`

---

## Data Requirements

### Fields Required by PointGroup

| Field | Shape | Source | Description |
|-------|-------|--------|-------------|
| `coord` | [N, 3] | HDF5 `scenes/points` | XYZ coordinates |
| `segment` | [N] | HDF5 `scenes/labels` | Per-point semantic class |
| `instance` | [N] | HDF5 `scenes/global_ids` | Per-point instance ID |
| `instance_centroid` | [N, 3] | Computed by `InstanceParser` | Instance centroid per point |
| `bbox` | [K, 8] | Computed by `InstanceParser` | Per-instance bounding box |
| `color` | [N, 3] | HDF5 `scenes/rgb` | RGB (normalized) |
| `normal` | [N, 3] | HDF5 `scenes/normals` | Surface normals |
| `grid_coord` | [N, 3] | Computed by `GridSample` | Voxelized coordinates |
| `offset` | [B] | Computed by `Collect` | Batch offsets |

### Fields Computed by InstanceParser (Not in HDF5)

The `InstanceParser` transform takes raw `instance` IDs and produces:

- **`instance`** (re-indexed): Consecutive IDs from 0 to K-1, with `instance_ignore_index` for stuff classes
- **`instance_centroid`** [N, 3]: Mean coordinate of all points belonging to the same instance; used as regression target for `bias_head`
- **`bbox`** [K, 8]: Per-instance bounding box = `[center_x, center_y, center_z, size_x, size_y, size_z, theta, class]`

---

## Bhopal HDF5 Data Availability

### HDF5 Structure (V9)

```
File: data/Bhopal/bhopal_scene_v9_normals_rgb.h5
Total scenes: 2410 (train: 1353, val: 568, test: 489)
Scene cutting: 5m base spatial cutting along road axis (adaptive subdivision)
Max points per scene: 1,000,000
Total points: 968,771,926

scenes/points              [2410, 1050000, 3]    float32  - XYZ (scene-centered)
scenes/intensity           [2410, 1050000]       float32  - Raw values (0-65535)
scenes/labels              [2410, 1050000]       int32    - Per-point class (0-61, -1=padding)
scenes/global_ids          [2410, 1050000]       int32    - Per-point instance ID (-1=padding)
scenes/rgb                 [2410, 1050000, 3]    uint8    - RGB values
scenes/normals             [2410, 1050000, 3]    float32  - Surface normals
scenes/num_points          [2410]                uint32   - Valid point count

splits/train_indices       [1353]   - Paths 1,2,3,4
splits/val_indices         [568]    - Paths 5,6
splits/test_indices        [489]    - Paths 7,8
```

**`scenes/global_ids` contains per-point instance IDs and is already in the HDF5 file.** The dataset loader with `load_instance=True` reads this field.

---

## Instance Cutting Analysis

### The Problem

The Bhopal HDF5 was created for **semantic segmentation** using spatial cutting: the continuous MLS survey is sliced into 5m segments along the road axis. Physical objects that span more than one segment get **cut** -- part of the object appears in scene A, the rest in scene B.

### Empirical Analysis

| Metric | Value |
|--------|-------|
| Total unique instances (global_ids) | 19,067 |
| Single-scene instances (not cut) | 11,309 (59.3%) |
| Multi-scene instances (cut) | 7,758 (40.7%) |
| Instances spanning train/val/test | **0** (splits are clean) |
| Multi-scene from different bhopal_paths | **0** (all within same path) |

**How badly are cut instances split?** (largest fragment as fraction of total)

| Severity | Count | % of cut instances |
|----------|-------|--------------------|
| >90% in one scene (trivially cut) | 1,796 | 23.2% |
| >80% in one scene (mostly intact) | 2,638 | 34.0% |
| <50% in one scene (badly split) | 1,989 | 25.6% |

**Cutting rate by class (12-class grouping):**

| Class | Total Inst. | Cut | Cut% | Badly Cut (<50%) | Bad% |
|-------|-------------|-----|------|-------------------|------|
| Ground | 955 | 873 | 91.4% | 470 | 49.2% |
| Vegetation | 365 | 259 | 71.0% | 50 | 13.7% |
| **Building** | 1,322 | 1,289 | **97.5%** | 829 | **62.7%** |
| Road | 289 | 284 | 98.3% | 252 | 87.2% |
| Wire | 518 | 493 | 95.2% | 294 | 56.8% |
| **Pole** | 1,044 | 231 | 22.1% | 1 | **0.1%** |
| Wire-Structure | 78 | 57 | 73.1% | 4 | 5.1% |
| **Vehicle** | 8,017 | 2,693 | 33.6% | 7 | **0.1%** |
| **Board** | 835 | 140 | 16.8% | 2 | **0.2%** |
| Fencing | 124 | 89 | 71.8% | 31 | 25.0% |
| **Human&Animal** | 3,258 | 457 | 14.0% | 0 | **0.0%** |
| Road Side Obj | 2,262 | 893 | 39.5% | 49 | 2.2% |

**Key insight:** The classes that matter most for instance segmentation (discrete, countable thing classes) are barely affected:
- **Vehicle**: 0.1% badly cut (small objects fit in 5m scenes)
- **Human&Animal**: 0.0% badly cut
- **Pole**: 0.1% badly cut (compact vertical objects)
- **Board**: 0.2% badly cut

The heavily cut classes are either **stuff** (Ground, Road, Wire) or **Building** (large structures spanning many scenes).

---

## Do We Need to Remake the Data?

**No. The existing HDF5 data works for instance segmentation.**

### How Standard Datasets Handle This

| Dataset | Domain | Scene creation | Instance cutting? |
|---------|--------|---------------|-------------------|
| **ScanNet** | Indoor | Complete room scans | Minimal (objects at room edges clipped) |
| **S3DIS** | Indoor | Room-level point clouds | Objects at room boundaries clipped |
| **nuScenes** | Outdoor | Single LiDAR sweeps | Objects at range boundary partially observed |
| **SemanticKITTI** | Outdoor | Single LiDAR sweeps | Objects at range boundary partially observed |
| **Bhopal MLS** | Outdoor | 5m spatial cuts | Objects at cut boundaries split |

**All outdoor datasets have boundary clipping.**

### Why the Existing Data Works

1. **PointGroup processes one scene at a time.** Each scene's `global_ids` correctly identifies which points belong to the same object fragment. The `InstanceParser` computes centroids for whatever fragment is in the scene.

2. **The key thing classes are barely cut.** Vehicle (0.1% badly cut), Human (0.0%), Pole (0.1%), Board (0.2%). These compact objects almost always fit entirely within a 5m scene.

3. **No cross-split leakage.** Zero instances span train/val/test splits. The path-based split ensures complete geographic separation.

4. **Building is inherently hard.** Even with perfect data, building instance segmentation with PointGroup would struggle because buildings span 10-30+ scenes.

### Mitigations (Without Remaking Data)

1. **`min_region_sizes` in InsSegEvaluator**: Skip GT instances smaller than N points to ignore tiny boundary fragments.
2. **`cluster_propose_points=200`**: Rejects predicted clusters smaller than 200 points.
3. **Accept lower Building AP**: Document building cutting as a known limitation.
4. **Future: boundary-aware filtering** (optional): Detect instances touching scene boundaries and mark as ignored.

---

## Dataset Loader Changes

The `BhopalSceneHDF5Dataset` at `Concerto/Bhopal/scripts/bhopal_scene_hdf5_dataset.py` already supports `load_instance=True`:
- Loads `scenes/global_ids` as `instance` in `get_data()`
- Applies class remapping to instance labels
- Includes `instance` in the returned `data_dict`

The instance segmentation configs use `load_instance=True` in all splits.

**Important: class remapping and instance labels.** When `remap_classes=True`, the dataset loader remaps `segment` (semantic labels) from original IDs to grouped class IDs using `class_groups`. The `instance` field (`global_ids`) is loaded as-is -- it contains raw instance IDs that are independent of semantic class IDs. The `InstanceParser` transform (applied later in the pipeline) uses the already-remapped `segment` labels to determine which points belong to stuff classes (`segment_ignore_index`), then re-indexes instance IDs to consecutive 0..K-1. So the flow is: dataset loads raw `instance` IDs + remaps `segment` -> `InstanceParser` reads remapped `segment` to mask stuff classes in `instance` -> re-indexes remaining thing instances.

---

## InstanceParser Transform

**Source:** `Pointcept/pointcept/datasets/transform.py:1276`

The `InstanceParser` transform must appear in the data pipeline AFTER `GridSample` (voxelization) and BEFORE `ToTensor` / `Collect`.

### What InstanceParser Does

```python
class InstanceParser(object):
    def __init__(self, segment_ignore_index=(-1, 0, 1), instance_ignore_index=-1):
        ...

    def __call__(self, data_dict):
        coord = data_dict["coord"]
        segment = data_dict["segment"]
        instance = data_dict["instance"]

        # 1. Mask out points from ignored segment classes (stuff)
        mask = ~np.in1d(segment, self.segment_ignore_index)
        instance[~mask] = self.instance_ignore_index

        # 2. Re-index instances to consecutive 0..K-1
        unique, inverse = np.unique(instance[mask], return_inverse=True)
        instance_num = len(unique)
        instance[mask] = inverse

        # 3. Compute per-instance centroid and bbox
        centroid = np.ones((coord.shape[0], 3)) * self.instance_ignore_index
        bbox = np.ones((instance_num, 8)) * self.instance_ignore_index

        for instance_id in range(instance_num):
            mask_ = instance == instance_id
            coord_ = coord[mask_]
            bbox_centroid = coord_.mean(0)
            centroid[mask_] = bbox_centroid
            bbox[instance_id] = [center, size, theta, class]  # 8-dim

        data_dict["instance"] = instance
        data_dict["instance_centroid"] = centroid
        data_dict["bbox"] = bbox
        return data_dict
```

### Critical Parameter: `segment_ignore_index`

This determines which semantic classes are treated as **stuff** (not instanceable). Points from these classes have their instance IDs set to `instance_ignore_index=-1` and are excluded from clustering.

**ScanNet default:** `(-1, 0, 1)` -- ignores padding, wall, floor

**For Bhopal 21-class:** `(-1, 0, 2, 5, 6)` -- ignores padding, Ground, Vegetation, Road, Wire

---

## Thing vs Stuff: Definitions and Decisions

### Definitions

**Thing**: A discrete, countable object with a well-defined boundary. You can answer "how many?" -- *3 cars, 2 poles, 1 building*. The three-question test:

1. **Can you draw a boundary around one instance?** If two adjacent patches exist, is there a clear physical gap between them?
2. **Does counting instances make sense?** Would a human annotator agree on "how many" are in a scene?
3. **Can PointGroup's centroid-offset clustering work?** Does each instance have a meaningful centroid that points can regress toward?

If all three are **yes** -> **Thing**. If any is **no** -> **Stuff**.

**Stuff**: An amorphous, continuous region without natural instance boundaries. You can't meaningfully count it -- *ground, road, sky*.

### Stuff Class Decisions

| Class | Rationale |
|-------|-----------|
| **Ground** (ID 0) | Continuous terrain surface. No countable boundaries. 91.4% cut rate. |
| **Vegetation** (ID 2) | Low/general vegetation is amorphous ground-cover. Only 19 instances in val -- too few to split into subtypes. High vegetation (trees) is countable but inseparable from amorphous subtypes at this grouping level. |
| **Road** (ID 5) | Continuous paved surface. No natural instance boundary. 98.3% cut rate. |
| **Wire** (ID 6) | Continuous linear features spanning across scenes. No natural "where does one wire end?" boundary. 95.2% cut rate, 56.8% badly cut. Centroid regression meaningless for a long wire. |

### Ignored Classes

| Original IDs | Subclass | Reason |
|-------------|----------|--------|
| 47 | Animal | 0 instances in val and test |
| 48 | Dog | 0 instances in val and test |
| 49 | Cow | Only 4 instances in test -- not viable for evaluation |

These 3 original subclasses are mapped to `ignore_index` (not assigned to any of the 21 classes). Total ignored: Train=19, Val=12, Test=4 instances.

---

## 21-Class Taxonomy

### Design Principles

1. **Data-driven thresholds**: Every class must have **>=15 instances in all splits** (train, val, test) for meaningful AP computation
2. **Domain-relevant splits**: Stairs (mobility), 2/3/4-Wheeler (India-specific vehicle taxonomy), roadside infrastructure subtypes
3. **Stuff = amorphous/continuous**: Ground, Vegetation, Road, Wire
4. **Animals ignored**: Too few instances (4 total in test) for reliable evaluation

### Class Table

| ID | Class | Type | Original IDs | Train | Val | Test |
|----|-------|------|-------------|-------|-----|------|
| 0 | Ground | **Stuff** | 0,1,2 | 405 | 141 | 101 |
| 1 | Stairs | Thing | 3 | 156 | 115 | 37 |
| 2 | Vegetation | **Stuff** | 4,5,6,7 | 294 | 19 | 52 |
| 3 | Building (other) | Thing | 8,9,11,12 | 296 | 186 | 117 |
| 4 | Commercial | Thing | 10 | 414 | 137 | 172 |
| 5 | Road | **Stuff** | 13 | 153 | 84 | 52 |
| 6 | Wire | **Stuff** | 14 | 332 | 131 | 55 |
| 7 | Pole (other) | Thing | 15,17,19 | 174 | 29 | 55 |
| 8 | Electric Pole | Thing | 16 | 349 | 114 | 123 |
| 9 | Street Light | Thing | 18 | 139 | 84 | 55 |
| 10 | 2-Wheeler | Thing | 22,23,32,33,34 | 4,198 | 938 | 1,545 |
| 11 | 3-Wheeler | Thing | 25,27,28,30 | 184 | 15 | 69 |
| 12 | 4-Wheeler | Thing | 20,21,26,29,31 | 551 | 46 | 128 |
| 13 | Hand-Cart | Thing | 24 | 164 | 60 | 119 |
| 14 | Board | Thing | 35,36,37 | 554 | 109 | 172 |
| 15 | Fencing | Thing | 38,39,40,41,42 | 78 | 20 | 26 |
| 16 | Human | Thing | 43,44,45,46 | 2,123 | 445 | 655 |
| 17 | Road Side Furniture | Thing | 53 | 737 | 240 | 454 |
| 18 | Shop Shade | Thing | 52 | 130 | 72 | 110 |
| 19 | Mannequin | Thing | 57 | 226 | 40 | 98 |
| 20 | Road Side Obj (other) | Thing | 50,51,54,55,56,58,59,60,61 | 110 | 19 | 26 |

**4 stuff + 17 thing = 21 classes. 17 thing classes evaluated for mAP.**

### What's in Each Class

**Stairs (ID 1):** Steeply sloping ground surfaces, which in the Bhopal urban context are actually stairs/steps. Relevant for mobility studies analysis -- stairs are barriers for wheelchair users. Separated from Ground because they are discrete, countable structures with clear boundaries (156/115/37 instances).

**Building split:**
- **Building (other)** (ID 3): General buildings + Residential + Industrial + Special. Merges low-count subtypes (Industrial=1, Special=56) with general.
- **Commercial** (ID 4): Commercial buildings. Largest building subclass (724 total instances, 52% of all points). Distinguishing commercial vs other buildings is useful for urban planning.

**Pole split:**
- **Pole (other)** (ID 7): General poles + Mobile Tower + Wire-Structure Connector. Wire-Structure (only 8 in val) merged here since they're physically mounted on poles.
- **Electric Pole** (ID 8): Dominant pole type (615 total). Visually distinct.
- **Street Light** (ID 9): 289 total. Different form factor from utility poles.

**Vehicle taxonomy (wheel-based, India-specific):**
- **2-Wheeler** (ID 10): Mobike + Cycle + Mobike Rider + Cycle Rider + Rider on Vehicle. Overwhelmingly Mobike (6,335 instances). Vehicle gen (ID 20) goes to 4-Wheeler as those are cars/trucks. Riders are on 2-wheelers.
- **3-Wheeler** (ID 11): Tuktuk + Rickshaw + E-Rickshaw + Dynamic Tuktuk. Characteristic Indian auto-rickshaws. Tightest class (val=15) but viable.
- **4-Wheeler** (ID 12): Vehicle (gen) + Car + Truck + Dynamic Car + Dynamic Truck. Vehicle (gen) classified here as these are typically cars/trucks.
- **Hand-Cart** (ID 13): Push carts. No engine, not a "wheeler" -- separate class. Common in Indian street scenes (164/60/119).

**Human (ID 16):** Human and Animal (gen) + Male + Female + Child. Animals (Cow/Dog/Animal) excluded -- only 4 test instances, mapped to ignore.

**Road Side Object split:**
- **Road Side Furniture** (ID 17): Benches, bollards, planters. Largest RSO subclass (737/240/454). mobility-relevant -- these objects block or define pedestrian paths.
- **Shop Shade** (ID 18): Awnings/canopies over shops. Relevant for covered walkway detection (130/72/110).
- **Mannequin** (ID 19): Display mannequins outside shops. Unique to Indian street scenes (226/40/98).
- **Road Side Obj (other)** (ID 20): Barricade, Statue, Handpump, Road Side Public Struct, Dustbin, Bus Stand, Overhead Structure, Road Median, Road Side Object (gen). All small subclasses merged (110/19/26).

### Comparison: 12-Class Semantic Seg vs 21-Class Instance Seg

| 12-Class (semantic) | 21-Class (instance) | Splits gained |
|--------------------|--------------------|----|
| Ground | Ground + **Stairs** | Stairs separated (mobility) |
| Vegetation | Vegetation (stuff) | Made stuff (amorphous) |
| Building | Building (other) + **Commercial** | Commercial separated |
| Road | Road (stuff) | Same |
| Wire | Wire (stuff) | Same |
| Pole and Tower | Pole (other) + **Electric Pole** + **Street Light** | 3-way split by type |
| Wire-Structure Connector | (merged into Pole other) | -- |
| Vehicle | **2-Wheeler** + **3-Wheeler** + **4-Wheeler** + **Hand-Cart** | India-specific wheel taxonomy |
| Board | Board | Same |
| Fencing | Fencing | Same |
| Human and Animal | **Human** (animals ignored) | Animals removed |
| Road Side Object | **Road Side Furniture** + **Shop Shade** + **Mannequin** + RSO (other) | 4-way split |

---

## Per-Split Instance Statistics

### Train Split (1,353 scenes, paths 1-4)

| ID | Class | Instances | Points (M) |
|----|-------|-----------|------------|
| 0 | Ground | 405 | 10.81 |
| 1 | Stairs | 156 | 4.31 |
| 2 | Vegetation | 294 | 19.74 |
| 3 | Building (other) | 296 | 53.00 |
| 4 | Commercial | 414 | 257.53 |
| 5 | Road | 153 | 81.85 |
| 6 | Wire | 332 | 19.19 |
| 7 | Pole (other) | 174 | 0.55 |
| 8 | Electric Pole | 349 | 4.99 |
| 9 | Street Light | 139 | 0.17 |
| 10 | 2-Wheeler | 4,198 | 20.52 |
| 11 | 3-Wheeler | 184 | 3.27 |
| 12 | 4-Wheeler | 551 | 9.47 |
| 13 | Hand-Cart | 164 | 1.83 |
| 14 | Board | 554 | 4.47 |
| 15 | Fencing | 78 | 0.94 |
| 16 | Human | 2,123 | 4.68 |
| 17 | Road Side Furniture | 737 | 5.08 |
| 18 | Shop Shade | 130 | 4.83 |
| 19 | Mannequin | 226 | 0.96 |
| 20 | Road Side Obj (other) | 110 | 1.34 |

### Val Split (568 scenes, paths 5-6)

| ID | Class | Instances | Points (M) |
|----|-------|-----------|------------|
| 0 | Ground | 141 | 2.82 |
| 1 | Stairs | 115 | 4.03 |
| 2 | Vegetation | 19 | 1.46 |
| 3 | Building (other) | 186 | 16.03 |
| 4 | Commercial | 137 | 119.17 |
| 5 | Road | 84 | 24.84 |
| 6 | Wire | 131 | 7.15 |
| 7 | Pole (other) | 29 | 0.11 |
| 8 | Electric Pole | 114 | 1.80 |
| 9 | Street Light | 84 | 0.14 |
| 10 | 2-Wheeler | 938 | 5.63 |
| 11 | 3-Wheeler | 15 | 0.42 |
| 12 | 4-Wheeler | 46 | 1.24 |
| 13 | Hand-Cart | 60 | 0.74 |
| 14 | Board | 109 | 1.06 |
| 15 | Fencing | 20 | 0.87 |
| 16 | Human | 445 | 1.27 |
| 17 | Road Side Furniture | 240 | 1.66 |
| 18 | Shop Shade | 72 | 4.11 |
| 19 | Mannequin | 40 | 0.11 |
| 20 | Road Side Obj (other) | 19 | 0.16 |

### Test Split (489 scenes, paths 7-8)

| ID | Class | Instances | Points (M) |
|----|-------|-----------|------------|
| 0 | Ground | 101 | 2.90 |
| 1 | Stairs | 37 | 0.82 |
| 2 | Vegetation | 52 | 8.00 |
| 3 | Building (other) | 117 | 8.50 |
| 4 | Commercial | 172 | 117.55 |
| 5 | Road | 52 | 27.69 |
| 6 | Wire | 55 | 7.52 |
| 7 | Pole (other) | 55 | 0.24 |
| 8 | Electric Pole | 123 | 1.82 |
| 9 | Street Light | 55 | 0.08 |
| 10 | 2-Wheeler | 1,545 | 6.65 |
| 11 | 3-Wheeler | 69 | 1.05 |
| 12 | 4-Wheeler | 128 | 1.87 |
| 13 | Hand-Cart | 119 | 1.14 |
| 14 | Board | 172 | 1.29 |
| 15 | Fencing | 26 | 0.69 |
| 16 | Human | 655 | 1.25 |
| 17 | Road Side Furniture | 454 | 3.81 |
| 18 | Shop Shade | 110 | 4.44 |
| 19 | Mannequin | 98 | 0.43 |
| 20 | Road Side Obj (other) | 26 | 0.18 |

### Split Observations

- **3-Wheeler**: Tightest class with val=15. AP will have some variance but is usable.
- **Vegetation**: Only 19 in val (stuff class, not evaluated for mAP -- no concern).
- **Road Side Obj (other)**: val=19, test=26. Marginal but viable.
- **Fencing**: val=20, test=26. Marginal but viable.
- **All other thing classes**: 29+ instances in every split. Robust.

---

## Configuration Reference

### 12-Class Config Values

```python
num_classes = 12
segment_ignore_index = (-1, 0, 1, 3, 4)  # padding, Ground(0), Vegetation(1), Road(3), Wire(4)

class_names = [
    "Ground",                # 0  - STUFF
    "Vegetation",            # 1  - STUFF
    "Building",              # 2
    "Road",                  # 3  - STUFF
    "Wire",                  # 4  - STUFF
    "Pole and Tower",        # 5
    "Wire-Structure Conn.",  # 6
    "Vehicle",               # 7
    "Board",                 # 8
    "Fencing",               # 9
    "Human and Animal",      # 10
    "Road Side Object",      # 11
]
# 4 stuff + 8 thing = 12 classes. 8 thing classes evaluated for mAP.
```

### 21-Class Config Values

```python
num_classes = 21
segment_ignore_index = (-1, 0, 2, 5, 6)  # padding, Ground(0), Vegetation(2), Road(5), Wire(6)

class_names = [
    "Ground",                # 0  - STUFF
    "Stairs",                # 1
    "Vegetation",            # 2  - STUFF
    "Building (other)",      # 3
    "Commercial",            # 4
    "Road",                  # 5  - STUFF
    "Wire",                  # 6  - STUFF
    "Pole (other)",          # 7
    "Electric Pole",         # 8
    "Street Light",          # 9
    "2-Wheeler",             # 10
    "3-Wheeler",             # 11
    "4-Wheeler",             # 12
    "Hand-Cart",             # 13
    "Board",                 # 14
    "Fencing",               # 15
    "Human",                 # 16
    "Road Side Furniture",   # 17
    "Shop Shade",            # 18
    "Mannequin",             # 19
    "Road Side Obj (other)", # 20
]

class_groups = [
    [0, 1, 2],                          # 0:  Ground (Flat + Gently Sloping + Ground gen)
    [3],                                 # 1:  Stairs (Steeply Sloping Ground)
    [4, 5, 6, 7],                        # 2:  Vegetation (all subtypes)
    [8, 9, 11, 12],                      # 3:  Building (other) (Gen+Residential+Industrial+Special)
    [10],                                # 4:  Commercial
    [13],                                # 5:  Road
    [14],                                # 6:  Wire
    [15, 17, 19],                        # 7:  Pole (other) (Gen+MobileTower+WireStruct)
    [16],                                # 8:  Electric Pole
    [18],                                # 9:  Street Light
    [22, 23, 32, 33, 34],               # 10: 2-Wheeler (Mobike+Cycle+RiderOnVeh+MobikeRider+CycleRider)
    [25, 27, 28, 30],                   # 11: 3-Wheeler (Tuktuk+Rickshaw+E-Rickshaw+DynTuktuk)
    [20, 21, 26, 29, 31],               # 12: 4-Wheeler (VehicleGen+Car+Truck+DynCar+DynTruck)
    [24],                                # 13: Hand-Cart
    [35, 36, 37],                        # 14: Board (Board+SignBoard+AdvtBoard)
    [38, 39, 40, 41, 42],               # 15: Fencing (all subtypes)
    [43, 44, 45, 46],                   # 16: Human (HumanAnimal gen+Male+Female+Child)
    [53],                                # 17: Road Side Furniture
    [52],                                # 18: Shop Shade
    [57],                                # 19: Mannequin
    [50, 51, 54, 55, 56, 58, 59, 60, 61],  # 20: Road Side Obj (other)
]

# Original IDs 47 (Animal), 48 (Dog), 49 (Cow) are NOT in any group -> mapped to ignore_index
```

### Side-by-Side: ScanNet vs Bhopal

| Setting | ScanNet FT (reference) | Bhopal 12-class | Bhopal 21-class |
|---------|----------------------|-----------------|-----------------|
| `semantic_num_classes` | 20 | **12** | **21** |
| `segment_ignore_index` | `(-1, 0, 1)` | `(-1, 0, 1, 3, 4)` | `(-1, 0, 2, 5, 6)` |
| `voxel_size` | 0.02 (default) | **0.01** | **0.01** |
| `grid_size` (GridSample) | 0.02 | 0.01 | 0.01 |
| CE loss `weight` | None (uniform) | `class_weight` | `class_weight` |
| `epoch` | 800 | 100 | 100 |
| `weight_decay` | 0.05 | 0.005 | 0.005 |
| `final_div_factor` | 1000.0 | 100.0 | 100.0 |
| CheckpointLoader keywords | `module.student.backbone` | `module.` | `module.` |
| CheckpointLoader replacement | `module.backbone` | `module.backbone.` | `module.backbone.` |
| Checkpoint | `pretrain-concerto-*.pth` (7.8GB) | `concerto_large_outdoor.pth` | `concerto_large_outdoor.pth` |

### Key Differences That Must NOT Be Missed

- `voxel_size=0.01` -- ScanNet default is 0.02. Bhopal MUST override this to match grid_size.
- `semantic_num_classes`: 12 or 21, not 20
- `segment_ignore_index`: `(-1, 0, 1, 3, 4)` for 12c, `(-1, 0, 2, 5, 6)` for 21c -- 4 stuff classes + padding
- `weight=class_weight` in CrossEntropyLoss for class imbalance
- `weight_decay=0.005` not 0.05
- `epoch=100` not 800
- `class_weight` computed separately for 12 and 21 classes (sqrt-inverse-frequency from point distribution)
- Original IDs 47, 48, 49 (animals) must be excluded from `class_groups` and mapped to ignore (21c only; in 12c they are part of "Human and Animal")

---

## Training Pipeline

### Transform Pipeline: Bhopal 21-Class Instance Seg

```python
train=dict(
    type="BhopalSceneHDF5Dataset",
    h5_path=H5_PATH,
    split="train",
    load_instance=True,
    remap_classes=True,
    class_groups=class_groups,
    class_names=class_names,
    transform=[
        dict(type="RandomScale", scale=[0.2, 0.2]),
        dict(type="RandomRotate", angle=[-1, 1], axis="z", center=[0, 0, 0], p=0.5),
        dict(type="RandomScale", scale=[0.9, 1.1]),
        dict(type="RandomFlip", p=0.5),
        dict(type="RandomJitter", sigma=0.0025, clip=0.01),
        dict(type="GridSample", grid_size=0.01, hash_type="fnv", mode="train", return_grid_coord=True),
        dict(type="PointClip", point_cloud_range=(-149*0.2, -109*0.2, -22*0.2, 213*0.2, 159*0.2, 43*0.2)),
        dict(type="NormalizeColor"),
        dict(type="InstanceParser",
             segment_ignore_index=segment_ignore_index,  # (-1, 0, 2, 5, 6)
             instance_ignore_index=-1),
        dict(type="ToTensor"),
        dict(type="Collect",
             keys=("coord", "grid_coord", "segment",
                   "instance", "instance_centroid", "bbox"),
             feat_keys=("coord", "color", "normal")),
    ],
)
```

### Validation Transform

The validation pipeline needs additional `origin_*` copies for evaluation:

```python
val=dict(
    type="BhopalSceneHDF5Dataset",
    h5_path=H5_PATH,
    split="val",
    load_instance=True,
    remap_classes=True,
    class_groups=class_groups,
    class_names=class_names,
    transform=[
        dict(type="RandomScale", scale=[0.2, 0.2]),
        dict(type="Copy", keys_dict={
            "coord": "origin_coord",
            "segment": "origin_segment",
            "instance": "origin_instance",
        }),
        dict(type="GridSample", grid_size=0.01, hash_type="fnv", mode="train", return_grid_coord=True),
        dict(type="PointClip", point_cloud_range=(-149*0.2, -109*0.2, -22*0.2, 213*0.2, 159*0.2, 43*0.2)),
        dict(type="NormalizeColor"),
        dict(type="InstanceParser",
             segment_ignore_index=segment_ignore_index,
             instance_ignore_index=-1),
        dict(type="ToTensor"),
        dict(type="Collect",
             keys=("coord", "grid_coord", "segment", "instance",
                   "origin_coord", "origin_segment", "origin_instance",
                   "instance_centroid", "bbox", "name"),
             feat_keys=("coord", "color", "normal"),
             offset_keys_dict=dict(offset="coord", origin_offset="origin_coord")),
    ],
)
```

### Hooks Config

```python
hooks = [
    dict(type="CheckpointLoader", keywords="module.", replacement="module.backbone."),
    dict(type="IterationTimer", warmup_iter=2),
    dict(type="InformationWriter"),
    dict(type="InsSegEvaluator",
         segment_ignore_index=segment_ignore_index,  # Must match model config
         instance_ignore_index=-1),
    dict(type="CheckpointSaver", save_freq=None),
    dict(type="PreciseEvaluator", test_last=False),
]

test = dict(
    type="InsSegTester",
    segment_ignore_index=segment_ignore_index,
    instance_ignore_index=-1,
    verbose=False,
)

weight = CONCERTO_CHECKPOINT
```

---

## Clustering Parameters for Outdoor MLS

### The Scale Problem

The ScanNet configs were tuned for indoor scenes with no global scaling. Bhopal MLS data goes through a **0.2x `RandomScale`** at the start of the pipeline. The clustering operates on coordinates **divided by `voxel_size`**.

### How `cluster_thresh` Maps to Physical Distance

```python
center_pred = coord + bias_pred   # in transformed coordinate space
center_pred /= self.voxel_size    # normalize to voxel units
```

| Dataset | voxel_size | Pre-scale | Effective radius (transformed) | Effective radius (original) |
|---------|-----------|-----------|-------------------------------|---------------------------|
| ScanNet | 0.02 (default) | None | 1.5 * 0.02 = **0.03m** | **0.03m** |
| Bhopal | 0.01 | 0.2x | 1.5 * 0.01 = **0.015m** | 0.015 / 0.2 = **0.075m** |

### Parameter Reference

| Parameter | PG-v1m2 Default | ScanNet FT | Bhopal Config |
|-----------|----------------|------------|---------------|
| `cluster_thresh` | 1.5 | 1.5 | 1.5 |
| `cluster_closed_points` | 300 | 600 | 600 |
| `cluster_propose_points` | 100 | 200 | 200 |
| `cluster_min_points` | 50 | 50 | 50 |
| `voxel_size` | 0.02 | 0.02 | **0.01** |

### Tuning Strategy

1. Start with ScanNet FT values and `voxel_size=0.01`
2. If instances are over-segmented: increase `cluster_thresh`
3. If instances are under-segmented: decrease `cluster_thresh`
4. If small objects are missed: decrease `cluster_min_points` and `cluster_propose_points`
5. If large objects (buildings, fences) fragment: increase `cluster_thresh` (e.g., 3.0+)
6. Monitor `bias_l1_loss` -- if it plateaus high, the model struggles with centroid prediction

---

## Evaluation: InsSegEvaluator

**Source:** `Pointcept/pointcept/engines/hooks/evaluator.py:247`

### Metrics

| Metric | Description |
|--------|-------------|
| **mAP** | Mean Average Precision across IoU thresholds 0.50 to 0.90 (step 0.05) + 0.25 |
| **AP@50** | Average Precision at 50% IoU threshold |
| **AP@25** | Average Precision at 25% IoU threshold (lenient) |
| Per-class AP | AP for each thing class |

### Evaluated Thing Classes (17 total)

```python
# With segment_ignore_index=(-1, 0, 2, 5, 6) and 21 Bhopal classes:
# Stuff (excluded from mAP): 0=Ground, 2=Vegetation, 5=Road, 6=Wire
# Thing (evaluated, 17 classes):
#   1: Stairs, 3: Building(other), 4: Commercial,
#   7: Pole(other), 8: Electric Pole, 9: Street Light,
#   10: 2-Wheeler, 11: 3-Wheeler, 12: 4-Wheeler, 13: Hand-Cart,
#   14: Board, 15: Fencing, 16: Human,
#   17: Road Side Furniture, 18: Shop Shade, 19: Mannequin, 20: RSO(other)
```

### `min_region_sizes` Parameter

The evaluator ignores GT instances smaller than `min_region_sizes=100` points. For Bhopal MLS with 1cm voxels after 0.2x scaling, consider:
- Reducing to 50 for small outdoor objects (street lights, small boards)
- Increasing to 200 to filter more boundary fragments

---

## Implementation Checklist

### Done

- [x] `load_instance=True` in `BhopalSceneHDF5Dataset`
- [x] Dataset loader reads `scenes/global_ids`
- [x] 12-class configs: `concerto_140gb_12class_insseg_{lp,dp,ft}_9ch.py`
- [x] 21-class configs: `concerto_140gb_21class_insseg_{lp,dp,ft}_9ch.py`
- [x] `class_weight` computed for 21 classes (sqrt-inverse-frequency)
- [x] `class_groups` correctly maps original IDs 47,48,49 (animals) to ignore
- [x] `segment_ignore_index` set: `(-1, 0, 1, 3, 4)` for 12c, `(-1, 0, 2, 5, 6)` for 21c
- [x] Training script: `train_bhopal_insseg.sh` (supports both 12c and 21c)

### TODO: Training & Evaluation

- [ ] Verify `pointgroup_ops` installed: `python -c "from pointgroup_ops import ballquery_batch_p, bfs_cluster"`
- [ ] Run single training step to verify loss computation
- [ ] Train LP baseline, monitor `seg_loss`, `bias_l1_loss`, `bias_cosine_loss`
- [ ] Evaluate mAP@25 and mAP@50
- [ ] Tune `cluster_thresh` based on qualitative results

### TODO: Epoch Count — Monitor and Decide

Current setting: **100 epochs** for all modes (LP, DP, FT).

**Why 100 (not 800 like ScanNet):**
- Pre-trained Concerto backbone — not training from scratch
- Bhopal scenes are large (up to 1M points) — more data per epoch than ScanNet (~100K points/scene)
- Official outdoor configs (KITTI, nuScenes semseg) use 50-100 epochs
- OneCycleLR scheduler is shaped for the configured epoch count

**What to monitor:**
- [ ] Check if `bias_l1_loss` is still decreasing at epoch 90-100 → if yes, increase to 150-200
- [ ] Check if `mAP` plateaus early (epoch 60-70) → if yes, 100 is sufficient
- [ ] Compare LP vs FT convergence speed — LP (only heads train) may converge faster
- [ ] `bias_cosine_loss` should reach ~ -0.85 to -0.95 (good directional alignment)

**If more epochs needed:** Use `train_bhopal_insseg.sh {lp|dp|ft}-resume` — the resume function will continue from the last checkpoint. Update `epoch` in the config before resuming.

---

## Reference: Official ScanNet Instance Seg Configs

All configs located in `pointcept/configs/concerto/`:

| Config File | Strategy | Key Settings |
|-------------|----------|-------------|
| `insseg-ptv3-large-v1m1-0a-scannet-pg-lin.py` | Linear Probing | `enc_mode=True`, `freeze_backbone=True`, `backbone_out_channels=1728` |
| `insseg-ptv3-large-v1m1-0b-scannet-pg-dec.py` | Decoder Probing | `enc_mode=False`, `freeze_encoder=True`, `backbone_out_channels=64` |
| `insseg-ptv3-large-v1m1-0c-scannet-pg-ft.py` | Full Fine-tuning | `enc_mode=False`, `freeze_encoder=False`, `backbone_out_channels=64` |

| Parameter | Linear Probe | Decoder Probe | Full FT |
|-----------|-------------|---------------|---------|
| `enc_mode` | `True` | `False` | `False` |
| `freeze_encoder` | `False`* | `True` | `False` |
| `freeze_backbone` | `True` | `False` | `False` |
| `backbone_out_channels` | 1728 | 64 | 64 |

*`freeze_encoder` is irrelevant when `freeze_backbone=True`.

**No outdoor instance seg configs exist** in the official Concerto repository. The Bhopal configs (12-class and 21-class) are the first outdoor MLS instance segmentation configs for Concerto.

### Dependencies

```bash
# Required for PointGroup clustering (compiled CUDA ops)
python -c "from pointgroup_ops import ballquery_batch_p, bfs_cluster"

# If not installed:
cd Pointcept/libs/pointgroup_ops && python setup.py install
```

---

## Testing Pipeline (InsSegTester)

**Source:** `Pointcept/pointcept/engines/test.py:897`

### Overview

The `InsSegTester` evaluates trained instance segmentation models on the test split. Unlike semantic segmentation, **no TTA (test-time augmentation) is supported** -- the official code has a `# TODO: TTA` comment. This matches the official configs where `test_mode=False`.

### Requirements

| Requirement | Value | Why |
|------------|-------|-----|
| `batch_size_test` | **1** | `InsSegTester` asserts `batch_size == 1` (line 918) |
| `pointops` | installed | Used for `knn_query` reverse mapping |
| `pointgroup_ops` | installed | Used for `ballquery_batch_p` and `bfs_cluster` |
| `test_mode` | `False` | No TTA support; same as official configs |

### Data Flow

```
Test data (BhopalSceneHDF5Dataset, load_instance=True)
    |
    v
Transforms:
  RandomScale(0.2) -> Copy(origin_*) -> GridSample -> PointClip
  -> NormalizeColor -> InstanceParser -> ToTensor -> Collect
    |
    v
PG-v1m2 forward():
  1. data_dict["instance_centroid"] accessed (computed by InstanceParser)
  2. Backbone (PT-v3m2) extracts features
  3. seg_head -> semantic logits
  4. bias_head -> 3D centroid offsets
  5. Loss computed if "segment" and "instance" in data_dict (yes for test)
  6. Ball query clustering -> instance proposals
    |
    v
InsSegTester.test():
  1. Reverse mapping: pointops.knn_query maps predictions from
     grid-subsampled coords back to origin_coord (full resolution)
  2. pred_masks[:, reverse] applied to match original point count
  3. segment = origin_segment, instance = origin_instance for GT
  4. associate_instances() matches predictions to GT
  5. Compute mAP/AP50/AP25 per thing class
```

### Collect Keys Required for Test/Val

The test and val transforms must include these keys in `Collect`:

```python
keys=("coord", "grid_coord", "segment", "instance",
      "origin_coord", "origin_segment", "origin_instance",
      "instance_centroid", "bbox", "name"),
feat_keys=("coord", "color", "normal"),
offset_keys_dict=dict(offset="coord", origin_offset="origin_coord"),
```

- **`origin_coord`/`origin_segment`/`origin_instance`**: Copied BEFORE GridSample via `Copy` transform. Used by InsSegTester for full-resolution evaluation.
- **`origin_offset`**: Computed by `Collect` from `origin_coord`. Required by `pointops.knn_query`.
- **`name`**: Scene name from dataset, used as dict key in InsSegTester results.
- **`instance_centroid`**: Accessed unconditionally in PG-v1m2 forward (line 72). Must be present.

### Running Tests

```bash
# Via training script
bash train_bhopal_insseg.sh test 9ch 21c outdoor

# Direct command
python tools/test.py \
    --config-file configs/bhopal/concerto_140gb_21class_insseg_ft_9ch.py \
    --options save_path=outputs/insseg_21c_ft_9ch/test \
    weight=outputs/insseg_21c_ft_9ch/model/model_best.pth
```

### Output Format

```
Val result: mAP/AP50/AP25 0.XXXX/0.XXXX/0.XXXX.
Class_0-Stairs Result: AP/AP50/AP25 ...
Class_1-Building (other) Result: AP/AP50/AP25 ...
...
Class_16-Road Side Obj (other) Result: AP/AP50/AP25 ...
```

Note: Class indices in the output are **thing-class indices** (0 to 16 for 21-class), not the original class IDs. The mapping is:

| Output Index | Class Name | Original ID |
|-------------|-----------|------------|
| 0 | Stairs | 1 |
| 1 | Building (other) | 3 |
| 2 | Commercial | 4 |
| 3 | Pole (other) | 7 |
| 4 | Electric Pole | 8 |
| 5 | Street Light | 9 |
| 6 | 2-Wheeler | 10 |
| 7 | 3-Wheeler | 11 |
| 8 | 4-Wheeler | 12 |
| 9 | Hand-Cart | 13 |
| 10 | Board | 14 |
| 11 | Fencing | 15 |
| 12 | Human | 16 |
| 13 | Road Side Furniture | 17 |
| 14 | Shop Shade | 18 |
| 15 | Mannequin | 19 |
| 16 | Road Side Obj (other) | 20 |

---

## Known Issues and Lessons from Pointcept GitHub

Key findings from open/closed issues on the [Pointcept GitHub](https://github.com/Pointcept/Pointcept/issues) relevant to our instance segmentation setup.

### 1. No Official Outdoor Instance Segmentation Configs

**Issues:** [#158](https://github.com/Pointcept/Pointcept/issues/158), [#573](https://github.com/Pointcept/Pointcept/issues/573)

The maintainer (Xiaoyang Wu) confirmed:
- "I haven't written a test script for instance segmentation" (#158, later addressed by adding InsSegTester)
- "Sonata and Concerto are pretrained backbones, instead of an instance segmentation architecture. We use PointGroup as our instance segmentation architecture, which is a bit out of date." (#573)
- Concurrent work is underway to redesign instance seg architectures for SSL backbones

**Implication for Bhopal:** Our configs are the **first outdoor MLS instance segmentation configs** for Concerto. There are no official outdoor baselines to compare against directly. Indoor ScanNet/S3DIS configs are the closest reference.

### 2. Clustering Parameters Are Critical for Reproducibility

**Issue:** [#499](https://github.com/Pointcept/Pointcept/issues/499) (Sonata instance seg on ScanNet)

A user couldn't reproduce paper results. The maintainer's fix was updating clustering parameters:
- `cluster_closed_points`: 300 → **600**
- `cluster_propose_points`: 100 → **200**
- `batch_size`: reduced to 12
- `mix_prob`: 0.8

After these changes, results matched the paper. **Our Bhopal configs already use the correct values** (`cluster_closed_points=600`, `cluster_propose_points=200`, `mix_prob=0.8`).

### 3. PTv3 + PointGroup Compatibility Fix

**Issue:** [#335](https://github.com/Pointcept/Pointcept/issues/335)

PTv3 returns a `Point` object (not raw features). The original PG-v1m2 code passed this directly to `seg_head`/`bias_head`, causing crashes. Fixed in commit `58e101f`.

**Our status:** The PG-v1m2 code we use already has the fix (lines 76-85 with `isinstance(point, Point)` check).

### 4. First Category AP = 0 Can Indicate Index Misconfiguration

**Issue:** [#295](https://github.com/Pointcept/Pointcept/issues/295)

A user reported the first category always gets AP=0 despite good semantic segmentation (mIoU=80). The maintainer asked for `segment_ignore_index` and `instance_ignore_index` — suggesting misconfiguration of these parameters is a common root cause.

**Implication:** If we see AP=0 for any class, first check:
1. Is the class accidentally in `segment_ignore_index`?
2. Is `instance_ignore_index` correct (-1)?
3. Does the class have enough instances in the evaluation split?

### 5. `ignore_index` and `num_classes` Must Be Consistent

**Issue:** [#437](https://github.com/Pointcept/Pointcept/issues/437)

Maintainer clarified: `num_classes` should never be 1 (single-class). For binary problems, use `num_classes=2` with background as class 0 and `instance_ignore_index=0`.

**Our status:** Correct — we use `num_classes=12` or `21` with `ignore_index=-1`.

### 6. Validation Results Are Reliable (No Separate Test Script Needed)

**Issue:** [#158](https://github.com/Pointcept/Pointcept/issues/158)

Maintainer: "The validation result is already precise and can be used as final results."

The `PreciseEvaluator` hook runs full validation without augmentation at the end of training. For papers, validation mAP from training is acceptable. The `InsSegTester` exists for explicit test-split evaluation.

### 7. voxel_size Affects Clustering Behavior

**Issue:** [#568](https://github.com/Pointcept/Pointcept/issues/568)

A user with `voxel_size=0.1` (5x larger than ScanNet's 0.02) got near-zero AP for wall instances. No maintainer response, but the issue highlights that clustering parameters are tuned relative to `voxel_size`.

**Implication for Bhopal:** Our `voxel_size=0.01` (half of ScanNet's 0.02) means the clustering operates at finer resolution. With the 0.2x RandomScale, the effective physical clustering radius is 0.075m (see [Clustering Parameters section](#clustering-parameters-for-outdoor-mls)). If AP is low:
- **Over-segmentation** (too many small instances): increase `cluster_thresh` from 1.5 to 2.0-3.0
- **Under-segmentation** (instances merged): decrease `cluster_thresh`
- **Small objects missed**: decrease `cluster_min_points` from 50 to 30

### 8. Outdoor Data Augmentation is Minimal

**Issue:** [#500](https://github.com/Pointcept/Pointcept/issues/500)

For outdoor datasets (Sensat, Toronto3D-like), the maintainer recommended:
1. Use SSL pre-trained model (Sonata/Concerto) and freeze the encoder
2. Ensure LovaszLoss is enabled
3. Manually reduce data redundancy

Several augmentations common in indoor configs (ElasticDistortion, ChromaticAug, CenterShift) are intentionally commented out in official outdoor configs. **Our Bhopal configs follow this pattern** — we use the outdoor augmentation pipeline (RandomScale, RandomRotate, RandomFlip, RandomJitter, GridSample) without indoor-specific augmentations.

### 9. Attention Layer Learning Rate

**Issue:** [#203](https://github.com/Pointcept/Pointcept/issues/203)

For PTv3-based instance segmentation: "Setting the learning rate for the attention layer to 1/10 of the usual learning rate is important."

**Our status:** Already configured via `param_dicts = [dict(keyword="block", lr=0.0002)]` which sets attention block LR to 1/10 of the base LR (0.002).

### 10. Negative bias_cosine_loss is Normal

**Issue:** [#203](https://github.com/Pointcept/Pointcept/issues/203)

Users reported negative `bias_cosine_loss` values (e.g., -0.8968). This is expected — the cosine loss is defined as `-cos(pred, gt)`, so values range from -1 (perfect alignment) to +1 (opposite direction). A value of -0.9 means the predicted offset direction is very close to the ground truth direction.

---

## Bug Fix: Label Remapping for Unmapped Classes

### The Bug

**File:** `Pointcept/pointcept/datasets/bhopal.py:674-688`

The `_remap_labels()` method in `BhopalSceneHDF5Dataset` had a critical bug that only affected the **21-class** taxonomy. It caused an NLL loss assertion failure (`t >= 0 && t < n_classes`) at random iterations.

### Root Cause

The 21-class taxonomy intentionally excludes 3 original subclasses:
- ID 47: Animal
- ID 48: Dog
- ID 49: Cow

These are NOT assigned to any `class_groups` entry, so they should be mapped to `ignore_index=-1`.

**Buggy code:**
```python
def _remap_labels(self, labels):
    remapped = labels.copy()  # BUG: starts with original labels
    for old_label, new_label in self.class_mapping.items():
        remapped[labels == old_label] = new_label
    return remapped
```

Using `labels.copy()` as the starting point meant any original label NOT in `class_mapping` (i.e., 47, 48, 49) retained its raw value. When a scene contained an animal point with label 47, this passed through to the loss function which expected labels in range [0, 20] or -1.

### Why It Was Intermittent

Animal/Dog/Cow are extremely rare (19 instances in train, 12 in val, 4 in test). Most scenes don't contain them, so the crash only occurred when the dataloader happened to sample a scene with animal points — in one case, at iteration 44.

### Fix

```python
def _remap_labels(self, labels):
    remapped = np.full_like(labels, self.ignore_index)  # FIX: default to ignore
    for old_label, new_label in self.class_mapping.items():
        remapped[labels == old_label] = new_label
    remapped[labels == self.ignore_index] = self.ignore_index
    return remapped
```

Starting with `ignore_index` everywhere ensures any unmapped class automatically becomes ignored.

### Affected File

- **Active file (used by Pointcept):** `Pointcept/pointcept/datasets/bhopal.py` — **FIXED**
- **Reference copy:** `Concerto/Bhopal/scripts/bhopal_scene_hdf5_dataset.py` — already had correct implementation

### Impact

- **12-class configs:** Not affected (all 62 subclasses map to one of 12 groups, including animals → "Human and Animal")
- **21-class configs:** Affected (animals excluded from class_groups → crash when encountered)
