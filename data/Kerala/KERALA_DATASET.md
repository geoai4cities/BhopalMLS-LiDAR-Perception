# Ke-MLS (Kerala) scene file

Class-wise and scene-wise information on the Ke-MLS scene file of the cross-region study, built from the Ke-MLS release (Kumar et al. 2026) with the BhopalMLS corridor-cut recipe. The file is not part of the BhopalMLS data release (see `data/README.md`). Configs: `<Model>/Kerala/configs/kerala/`; wrappers: `<Model>/Kerala/scripts/{train,test}.sh`.

**Primary Dataset File**: `data/Kerala/Kerala_scene_v1_corridor.h5`

---

## Dataset Overview

| Property | Value |
|----------|-------|
| **Version** | 1.0 (Corridor Subdivision) |
| **Total Points** | 1,381,494,597 (~1.38B) |
| **Total Scenes** | 2,199 |
| **File Size** | 25 GB |
| **Fine-Grained Classes** | 48 |
| **Parent Classes** | 12 |
| **Max Points/Scene** | 1,000,000 (capped) |
| **Min Points/Scene** | 3,105 |
| **Source Paths** | 1-71 |
| **RGB Data** | None (all zeros) |
| **Normals** | Pre-computed (Open3D KNN-PCA, k=30) |

---

## Data Splits

| Split | Valid Scenes | Percentage |
|-------|-------------|------------|
| **Train** | 1,197 | 54.4% |
| **Val** | 401 | 18.2% |
| **Test** | 601 | 27.3% |
| **Total** | 2,199 | 100% |

---

## Point Cloud Bounds

Scene-centered coordinates (before 0.2 scaling):

| Axis | Min | Max | Range |
|------|-----|-----|-------|
| **X** | -78.27 | 116.93 | 195.20 |
| **Y** | -143.83 | 90.79 | 234.62 |
| **Z** | -27.94 | 46.75 | 74.69 |

**PointClip config** (with 10% margin, after 0.2 scaling):

```python
_point_clip_range = (-87*0.2, -159*0.2, -31*0.2, 129*0.2, 100*0.2, 52*0.2)
```

---

## 12 Parent Classes (Training Labels)

### Class Distribution

| ID | Class Name | Points | Percentage | Weight |
|----|------------|--------|------------|--------|
| 0 | Ground | 190,786,671 | 13.81% | 2.03 |
| 1 | **Vegetation** | **783,859,601** | **56.74%** | 1.00 |
| 2 | Building | 76,752,303 | 5.56% | 3.20 |
| 3 | Road | 268,211,819 | 19.41% | 1.71 |
| 4 | Wire | 6,511,514 | 0.47% | 10.97 |
| 5 | Pole and Tower | 6,096,327 | 0.44% | 11.34 |
| 6 | Wire-Structure Connector | 77,695 | 0.01% | 100.44 |
| 7 | Vehicle | 7,238,628 | 0.52% | 10.41 |
| 8 | Board | 5,048,037 | 0.37% | 12.46 |
| 9 | Fencing | 34,822,217 | 2.52% | 4.74 |
| 10 | Human and Animal | 1,955,286 | 0.14% | 20.02 |
| 11 | Road Side Object | 134,499 | 0.01% | 76.34 |

### Class Imbalance Analysis

```
Vegetation ████████████████████████████████████████████████████  56.74%
Road       ██████████████████                                   19.41%
Ground     █████████████                                        13.81%
Building   █████                                                 5.56%
Fencing    ██                                                    2.52%
Vehicle    ▌                                                     0.52%
Wire       ▌                                                     0.47%
Pole       ▌                                                     0.44%
Board      ▌                                                     0.37%
Human      ▌                                                     0.14%
RoadObj    ▌                                                     0.01%
WireCon    ▌                                                     0.01%
```

**Key Observations**:
- **Vegetation dominates** with 56.7% of all points (highest single-class ratio across all cities)
- **Vegetation + Road + Ground** together = 89.96% of all points
- **Rarest classes**: Wire-Structure Connector (0.01%, 78K pts), Road Side Object (0.01%, 134K pts)
- **Class imbalance ratio**: ~10,089:1 (Vegetation vs Wire-Structure Connector)
- **1 empty fine-grained class**: Road Side Object generic (tc=44) has 0 points

### Class Weights

Sqrt-scaled inverse frequency (min=1.0 for Vegetation):

```python
class_weight = [
    2.03,     # 0: Ground (13.81%)
    1.00,     # 1: Vegetation (56.74%) — dominant class
    3.20,     # 2: Building (5.56%)
    1.71,     # 3: Road (19.41%)
    10.97,    # 4: Wire (0.47%)
    11.34,    # 5: Pole and Tower (0.44%)
    100.44,   # 6: Wire-Structure Connector (0.01%) — rarest class
    10.41,    # 7: Vehicle (0.52%)
    12.46,    # 8: Board (0.37%)
    4.74,     # 9: Fencing (2.52%)
    20.02,    # 10: Human and Animal (0.14%)
    76.34,    # 11: Road Side Object (0.01%)
]
```

---

## 48 Fine-Grained Classes

### Parent 0: Ground (4 classes)

| TC | Fine Class | Points | % of Total | % of Parent | Scenes |
|----|------------|--------|------------|-------------|--------|
| 0 | Ground | 164,729,758 | 11.92% | 86.3% | 1,875 |
| 1 | Flat Ground | 4,772,620 | 0.35% | 2.5% | 82 |
| 2 | Gently Sloping Ground | 2,181,415 | 0.16% | 1.1% | 43 |
| 3 | Steeply Sloping Ground | 19,102,878 | 1.38% | 10.0% | 261 |

### Parent 1: Vegetation (4 classes) — **Dominant**

| TC | Fine Class | Points | % of Total | % of Parent | Scenes |
|----|------------|--------|------------|-------------|--------|
| 4 | Vegetation | 4,919,125 | 0.36% | 0.6% | 556 |
| 5 | **Low Vegetation** | **269,403,563** | **19.50%** | **34.4%** | 2,074 |
| 6 | Medium Vegetation | 139,763,171 | 10.12% | 17.8% | 1,974 |
| 7 | **High Vegetation** | **369,773,742** | **26.77%** | **47.2%** | 2,061 |

**Note**: High Vegetation alone represents 26.8% and Low Vegetation 19.5% of all points. Together the Vegetation parent class is 56.7% of the dataset.

### Parent 2: Building (4 classes)

| TC | Fine Class | Points | % of Total | % of Parent | Scenes |
|----|------------|--------|------------|-------------|--------|
| 8 | Building | 41,031,952 | 2.97% | 53.5% | 873 |
| 9 | Residential | 19,582,925 | 1.42% | 25.5% | 579 |
| 10 | Commercial | 15,138,631 | 1.10% | 19.7% | 225 |
| 11 | Special | 998,795 | 0.07% | 1.3% | 25 |

### Parent 3: Road (1 class)

| TC | Fine Class | Points | % of Total | Scenes |
|----|------------|--------|------------|--------|
| 12 | Road | 268,211,819 | 19.41% | 1,898 |

### Parent 4: Wire (1 class)

| TC | Fine Class | Points | % of Total | Scenes |
|----|------------|--------|------------|--------|
| 13 | Wire | 6,511,514 | 0.47% | 1,933 |

### Parent 5: Pole and Tower (3 classes)

| TC | Fine Class | Points | % of Total | % of Parent | Scenes |
|----|------------|--------|------------|-------------|--------|
| 14 | Pole and Tower | 74,337 | 0.01% | 1.2% | 12 |
| 15 | Electric Pole | 5,912,782 | 0.43% | 97.0% | 468 |
| 16 | Telephone Pole | 109,208 | 0.01% | 1.8% | 5 |

**Note**: Electric Pole dominates this parent at 97%.

### Parent 6: Wire-Structure Connector (2 classes)

| TC | Fine Class | Points | % of Total | % of Parent | Scenes |
|----|------------|--------|------------|-------------|--------|
| 17 | Wire-Structure Connector | 29,855 | 0.00% | 38.4% | 3 |
| 18 | Wire-Guard (Shield) | 47,840 | 0.00% | 61.6% | 3 |

### Parent 7: Vehicle (10 classes)

| TC | Fine Class | Points | % of Total | % of Parent | Scenes |
|----|------------|--------|------------|-------------|--------|
| 19 | Vehicle | 612,540 | 0.04% | 8.5% | 23 |
| 20 | Car | 1,496,482 | 0.11% | 20.7% | 129 |
| 21 | Mobike (Two-Wheeler) | 537,288 | 0.04% | 7.4% | 146 |
| 22 | Cycle | 9,539 | 0.00% | 0.1% | 5 |
| 23 | Tuktuk (3 Wheeler) | 1,136,061 | 0.08% | 15.7% | 63 |
| 24 | Bus | 1,306,057 | 0.09% | 18.0% | 52 |
| 25 | Truck | 1,160,950 | 0.08% | 16.0% | 32 |
| 26 | Rider on Vehicle | 189,307 | 0.01% | 2.6% | 7 |
| 27 | Mobike Rider (2 Wheeler) | 784,565 | 0.06% | 10.8% | 119 |
| 28 | Cycle Rider | 5,839 | 0.00% | 0.1% | 4 |


### Parent 8: Board (3 classes)

| TC | Fine Class | Points | % of Total | % of Parent | Scenes |
|----|------------|--------|------------|-------------|--------|
| 29 | Board | 121,057 | 0.01% | 2.4% | 20 |
| 30 | Sign Board | 56,497 | 0.00% | 1.1% | 13 |
| 31 | **Advt Board** | **4,870,483** | **0.35%** | **96.5%** | 207 |

**Note**: Advt Board dominates this parent at 96.5%.

### Parent 9: Fencing (5 classes)

| TC | Fine Class | Points | % of Total | % of Parent | Scenes |
|----|------------|--------|------------|-------------|--------|
| 32 | Fencing | 3,922,249 | 0.28% | 11.3% | 226 |
| 33 | **Masonry Wall** | **28,199,009** | **2.04%** | **81.0%** | 842 |
| 34 | Wired Boundary Fencing | 121,281 | 0.01% | 0.3% | 30 |
| 35 | Gate of Boundary Wall | 2,453,272 | 0.18% | 7.0% | 270 |
| 36 | Road Fencing | 126,406 | 0.01% | 0.4% | 21 |

### Parent 10: Human and Animal (7 classes)

| TC | Fine Class | Points | % of Total | % of Parent | Scenes |
|----|------------|--------|------------|-------------|--------|
| 37 | Human and Animal | 184,669 | 0.01% | 9.4% | 56 |
| 38 | **Male** | **1,280,473** | **0.09%** | **65.5%** | 166 |
| 39 | Female | 386,479 | 0.03% | 19.8% | 76 |
| 40 | Child | 78,397 | 0.01% | 4.0% | 18 |
| 41 | Animal | 504 | 0.00% | 0.0% | 1 |
| 42 | Dog | 24,156 | 0.00% | 1.2% | 20 |
| 43 | Cow | 608 | 0.00% | 0.0% | 1 |

### Parent 11: Road Side Object (4 classes)

| TC | Fine Class | Points | % of Total | % of Parent | Scenes |
|----|------------|--------|------------|-------------|--------|
| 44 | Road Side Object | 0 | 0.00% | 0.0% | 0 |
| 45 | Statue | 3,459 | 0.00% | 2.6% | 1 |
| 46 | Dustbin | 41,481 | 0.00% | 30.8% | 5 |
| 47 | Bus Stand | 89,559 | 0.01% | 66.6% | 5 |

**Note**: Road Side Object generic (tc=44) has 0 points (empty class). This parent has only 134K total points — the rarest parent alongside Wire-Structure Connector (78K).

---

## HDF5 Structure

```
Kerala_scene_v1_corridor.h5
├── scenes/
│   ├── points          [2199, 1050000, 3]  float32  — XYZ (scene-centered)
│   ├── normals         [2199, 1050000, 3]  float32  — Unit normals (KNN-PCA k=30)
│   ├── rgb             [2199, 1050000, 3]  uint8    — All zeros (no RGB)
│   ├── intensity       [2199, 1050000]     float32  — Raw 0-65535
│   ├── gps_time        [2199, 1050000]     float32  — GPS timestamps
│   ├── labels          [2199, 1050000]     int32    — Per-point class (-1=padding)
│   ├── global_ids      [2199, 1050000]     int64    — Per-point instance ID (-1=padding)
│   ├── num_points      [2199]              uint32   — Valid points per scene
│   └── original_num_points [2199]          uint32   — Before sampling
├── metadata/
│   ├── scene_names     [2199]              str      — Scene identifiers
│   ├── parent_scene    [2199]              str      — Source NPZ stem
│   ├── source_paths    [2199]              uint8    — Path numbers (1-71)
│   └── segment_size_m  [2199]              float32  — Corridor segment length (m)
└── splits/
    ├── train_indices   [1197]              — Training split indices
    ├── val_indices     [401]               — Validation split indices
    └── test_indices    [601]               — Test split indices
```

---

## Training Recommendations

### Class Weighting

Due to extreme class imbalance (10,089:1 ratio), use sqrt-inverse-frequency weighting:

```python
class_weight = [
    2.03,     # 0: Ground (13.81%)
    1.00,     # 1: Vegetation (56.74%)
    3.20,     # 2: Building (5.56%)
    1.71,     # 3: Road (19.41%)
    10.97,    # 4: Wire (0.47%)
    11.34,    # 5: Pole and Tower (0.44%)
    100.44,   # 6: Wire-Structure Connector (0.01%)
    10.41,    # 7: Vehicle (0.52%)
    12.46,    # 8: Board (0.37%)
    4.74,     # 9: Fencing (2.52%)
    20.02,    # 10: Human and Animal (0.14%)
    76.34,    # 11: Road Side Object (0.01%)
]
```

**Note**: Wire-Structure Connector weight (100.44) and Road Side Object weight (76.34) are extremely high due to very sparse data (78K and 134K points respectively). These are far above Bhopal's max weight of 15.85. Monitor training for loss instability; consider capping at ~25-30 if needed.

### Loss Functions

```python
criteria=[
    dict(type="CrossEntropyLoss", weight=class_weight, loss_weight=1.0, ignore_index=-1),
    dict(type="LovaszLoss", mode="multiclass", loss_weight=1.0, ignore_index=-1),
]
```

### Input Channels

- **9ch (recommended)**: XYZ(3) + zero-RGB(3) + normals(3) — compatible with Concerto outdoor checkpoint
- **4ch**: XYZ(3) + intensity(1) — trains from scratch (no checkpoint compatibility)

---

## Related files

| File | Location |
|------|----------|
| Scene file | `data/Kerala/Kerala_scene_v1_corridor.h5` (2,199 scenes, 48 classes, 25.9 GB) |
| Class tables | `data/Kerala/class_mapping/Kerala_class_mapping_V13.csv`, `Kerala_class_mapping_V13_TRAINING.csv` |
| Dataset class | `Pointcept/pointcept/datasets/kerala.py` (`KeralaSceneHDF5Dataset`) |
| Configs | `Utonia/Kerala/configs/kerala/`, `Concerto/Kerala/configs/kerala/`, `Sonata/Kerala/configs/kerala/` |
