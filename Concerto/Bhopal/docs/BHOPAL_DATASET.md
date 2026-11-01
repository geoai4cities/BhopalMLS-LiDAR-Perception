# Bhopal MLS Dataset Documentation

Detailed class-wise and scene-wise information for the Bhopal Mobile Laser Scanning dataset.

**Primary Dataset File**: `data/Bhopal/bhopal_scene_v7_train_1234_val_56_test_78.h5`

**Data Creation Scripts**: `data/Bhopal/tools/reference/19b_create_scene_V9_dataset.py`

---

## Dataset Overview

| Property | Value |
|----------|-------|
| **Version** | 7.0 (Adaptive Subdivision) |
| **Total Points** | 968,771,926 (~969M) |
| **Total Scenes** | 2,410 (2,340 valid) |
| **File Size** | 11 GB |
| **Fine-Grained Classes** | 62 |
| **Parent Classes** | 12 |
| **Coordinate System** | UTM Zone 43N (EPSG:32643) |

---

## Data Splits

| Split | Bhopal Paths | Valid Scenes | Points | Percentage |
|-------|--------------|--------------|--------|------------|
| **Train** | 1, 2, 3, 4 | 1,297 | 540,421,158 | 55.8% |
| **Val** | 5, 6 | 557 | 226,568,231 | 23.4% |
| **Test** | 7, 8 | 486 | 201,782,537 | 20.8% |
| **Total** | 1-8 | 2,340 | 968,771,926 | 100% |

---

## Scene Statistics

### Points per Scene

| Metric | Value |
|--------|-------|
| Maximum | 1,000,000 (capped) |
| Mean | 414,005 |
| Median | 380,494 |
| Minimum | 3,006 |

### Segment Size Distribution

V7 uses adaptive subdivision to keep all scenes under 1M points:

| Segment Size | Scenes | Percentage | Description |
|--------------|--------|------------|-------------|
| 5.00m | 1,855 | 79.3% | Original size (most scenes) |
| 2.50m | 317 | 13.5% | First subdivision level |
| 1.25m | 136 | 5.8% | Second subdivision level |
| 0.62m | 32 | 1.4% | Third subdivision level |

---

## 12 Parent Classes (Training Labels)

### Class Distribution

| ID | Class Name | Points | Percentage | Scenes Present |
|----|------------|--------|------------|----------------|
| 0 | Ground | 25,695,841 | 2.65% | 1,035+ |
| 1 | Vegetation | 29,196,896 | 3.01% | 371+ |
| 2 | **Building** | **630,861,910** | **65.12%** | 1,968+ |
| 3 | Road | 134,379,630 | 13.87% | 2,083 |
| 4 | Wire | 33,851,809 | 3.49% | 2,143 |
| 5 | Pole and Tower | 9,632,443 | 0.99% | 583+ |
| 6 | Wire-Structure Connector | 5,709,145 | 0.59% | 138 |
| 7 | Vehicle | 54,976,239 | 5.67% | 1,736+ |
| 8 | Board | 6,820,773 | 0.70% | 404+ |
| 9 | Fencing | 2,510,722 | 0.26% | 195+ |
| 10 | Human and Animal | 7,954,358 | 0.82% | 859+ |
| 11 | Road Side Object | 27,182,160 | 2.81% | 951+ |

### Class Imbalance Analysis

```
Building ████████████████████████████████████████████████ 65.12%
Road     ████████ 13.87%
Vehicle  ███ 5.67%
Wire     ██ 3.49%
Veget.   ██ 3.01%
RoadObj  ██ 2.81%
Ground   ██ 2.65%
Pole     ▌ 0.99%
Human    ▌ 0.82%
Board    ▌ 0.70%
WireCon  ▌ 0.59%
Fencing  ▌ 0.26%
```

**Key Observations**:
- **Building dominates** with 65% of all points
- **Commercial buildings** (class 10) alone = 51% of total points
- **Rare classes**: Fencing (0.26%), Wire-Connector (0.59%)
- **Class imbalance ratio**: ~250:1 (Building vs Fencing)

---

## 62 Fine-Grained Classes

### Parent 0: Ground (4 classes)

| ID | Fine Class | Points | % of Total | % of Parent | Scenes |
|----|------------|--------|------------|-------------|--------|
| 0 | Ground | 7,353,216 | 0.76% | 28.6% | 998 |
| 1 | Flat Ground | 8,529,133 | 0.88% | 33.2% | 1,035 |
| 2 | Gently Sloping Ground | 647,735 | 0.07% | 2.5% | 86 |
| 3 | Steeply Sloping Ground | 9,165,757 | 0.95% | 35.7% | 921 |

### Parent 1: Vegetation (4 classes)

| ID | Fine Class | Points | % of Total | % of Parent | Scenes |
|----|------------|--------|------------|-------------|--------|
| 4 | Vegetation | 1,753,331 | 0.18% | 6.0% | 167 |
| 5 | Low Vegetation | 695,488 | 0.07% | 2.4% | 133 |
| 6 | Medium Vegetation | 1,426,177 | 0.15% | 4.9% | 71 |
| 7 | High Vegetation | 25,321,900 | 2.61% | 86.7% | 371 |

### Parent 2: Building (5 classes) - **Dominant**

| ID | Fine Class | Points | % of Total | % of Parent | Scenes |
|----|------------|--------|------------|-------------|--------|
| 8 | Building | 41,204,846 | 4.25% | 6.5% | 1,192 |
| 9 | Residential | 59,913,053 | 6.18% | 9.5% | 414 |
| 10 | **Commercial** | **494,258,635** | **51.02%** | **78.3%** | 1,968 |
| 11 | Industrial | 401,587 | 0.04% | 0.1% | 5 |
| 12 | Special | 35,083,789 | 3.62% | 5.6% | 326 |

**Note**: Commercial buildings alone represent 51% of all points in the dataset!

### Parent 3: Road (1 class)

| ID | Fine Class | Points | % of Total | Scenes |
|----|------------|--------|------------|--------|
| 13 | Road | 134,379,630 | 13.87% | 2,083 |

### Parent 4: Wire (1 class)

| ID | Fine Class | Points | % of Total | Scenes |
|----|------------|--------|------------|--------|
| 14 | Wire | 33,851,809 | 3.49% | 2,143 |

### Parent 5: Pole and Tower (4 classes)

| ID | Fine Class | Points | % of Total | % of Parent | Scenes |
|----|------------|--------|------------|-------------|--------|
| 15 | Pole and Tower | 503,550 | 0.05% | 5.2% | 140 |
| 16 | Electric Pole | 8,603,598 | 0.89% | 89.3% | 583 |
| 17 | Mobile Tower | 135,700 | 0.01% | 1.4% | 29 |
| 18 | Street Light | 389,595 | 0.04% | 4.0% | 272 |

### Parent 6: Wire-Structure Connector (1 class)

| ID | Fine Class | Points | % of Total | Scenes |
|----|------------|--------|------------|--------|
| 19 | Wire-Structure Connector | 5,709,145 | 0.59% | 138 |

### Parent 7: Vehicle (15 classes)

| ID | Fine Class | Points | % of Total | % of Parent | Scenes |
|----|------------|--------|------------|-------------|--------|
| 20 | Vehicle | 1,618,833 | 0.17% | 2.9% | 153 |
| 21 | Car | 9,874,490 | 1.02% | 18.0% | 440 |
| 22 | **Mobike (Two-Wheeler)** | **29,892,095** | **3.09%** | **54.4%** | 1,736 |
| 23 | Cycle | 194,950 | 0.02% | 0.4% | 103 |
| 24 | Hand-Cart | 3,712,860 | 0.38% | 6.8% | 311 |
| 25 | Tuktuk (3 Wheeler) | 3,373,676 | 0.35% | 6.1% | 208 |
| 26 | Truck | 1,295,783 | 0.13% | 2.4% | 38 |
| 27 | Rickshaw | 278,136 | 0.03% | 0.5% | 10 |
| 28 | E-Rickshaw | 30,370 | 0.00% | 0.1% | 3 |
| 29 | Dynamic Car | 249,547 | 0.03% | 0.5% | 20 |
| 30 | Dynamic Tuktuk | 1,067,709 | 0.11% | 1.9% | 55 |
| 31 | Dynamic Truck | 65,533 | 0.01% | 0.1% | 2 |
| 32 | Rider on Vehicle | 259,383 | 0.03% | 0.5% | 22 |
| 33 | Mobike Rider | 3,028,992 | 0.31% | 5.5% | 360 |
| 34 | Cycle Rider | 33,882 | 0.00% | 0.1% | 13 |

**India-Specific Vehicles**: Mobike (54%), Tuktuk, Hand-Cart, Rickshaw, E-Rickshaw

### Parent 8: Board (3 classes)

| ID | Fine Class | Points | % of Total | % of Parent | Scenes |
|----|------------|--------|------------|-------------|--------|
| 35 | Board | 2,038,181 | 0.21% | 29.9% | 327 |
| 36 | Sign Board | 65,012 | 0.01% | 1.0% | 15 |
| 37 | Advt Board | 4,717,580 | 0.49% | 69.2% | 404 |

### Parent 9: Fencing (5 classes)

| ID | Fine Class | Points | % of Total | % of Parent | Scenes |
|----|------------|--------|------------|-------------|--------|
| 38 | Fencing | 1,052,468 | 0.11% | 41.9% | 195 |
| 39 | Masonry Wall | 903,542 | 0.09% | 36.0% | 91 |
| 40 | Wired Boundary Fencing | 13,054 | 0.00% | 0.5% | 4 |
| 41 | Gate of Boundary Wall | 541,122 | 0.06% | 21.6% | 50 |
| 42 | Road Fencing | 536 | 0.00% | 0.0% | 1 |

### Parent 10: Human and Animal (7 classes)

| ID | Fine Class | Points | % of Total | % of Parent | Scenes |
|----|------------|--------|------------|-------------|--------|
| 43 | Human and Animal | 3,836,175 | 0.40% | 48.2% | 859 |
| 44 | Male | 2,787,509 | 0.29% | 35.0% | 585 |
| 45 | Female | 1,044,422 | 0.11% | 13.1% | 161 |
| 46 | Child | 79,918 | 0.01% | 1.0% | 38 |
| 47 | Animal | 2,039 | 0.00% | 0.0% | 2 |
| 48 | Dog | 2,264 | 0.00% | 0.0% | 3 |
| 49 | Cow | 202,031 | 0.02% | 2.5% | 22 |

### Parent 11: Road Side Object (12 classes)

| ID | Fine Class | Points | % of Total | % of Parent | Scenes |
|----|------------|--------|------------|-------------|--------|
| 50 | Road Side Object | 316,979 | 0.03% | 1.2% | 62 |
| 51 | Barricade | 325,100 | 0.03% | 1.2% | 70 |
| 52 | Shop Shade | 13,390,425 | 1.38% | 49.3% | 454 |
| 53 | Road Side Furniture | 10,553,793 | 1.09% | 38.8% | 951 |
| 54 | Statue | 9,582 | 0.00% | 0.0% | 2 |
| 55 | Handpump | 34,655 | 0.00% | 0.1% | 20 |
| 56 | Road Side Public Structure | 131,563 | 0.01% | 0.5% | 6 |
| 57 | Mannequin | 1,492,048 | 0.15% | 5.5% | 158 |
| 58 | Dustbin | 11,120 | 0.00% | 0.0% | 5 |
| 59 | Bus Stand | 5,420 | 0.00% | 0.0% | 1 |
| 60 | Overhead Structure | 759,427 | 0.08% | 2.8% | 20 |
| 61 | Road Median | 152,048 | 0.02% | 0.6% | 40 |

---

## V7 Processing Strategy

### Why V7?

V7 addresses the challenge of highly variable scene sizes in the original data:

1. **Problem**: Some 5m road segments had >5M points (buildings), unusable for training
2. **Solution**: Adaptive subdivision + building class sampling

### Processing Steps

1. **Load V5 scene definitions** (5m segments)
2. **Estimate points after sampling** using pre-computed class distribution
3. **Compute subdivisions** if estimated > 1M points:
   - 5m → 2×2.5m
   - 2.5m → 2×1.25m
   - 1.25m → 2×0.625m
4. **Apply building class sampling** (90% retention for classes 8-12)
5. **Safety downsampling** to 1M if still exceeds after subdivision
6. **Filter scenes** < 3,000 points

### Building Classes Sampled

| Class ID | Class Name | Original | After 90% Sampling |
|----------|------------|----------|-------------------|
| 8 | Building | 100% | 90% |
| 9 | Residential | 100% | 90% |
| 10 | Commercial | 100% | 90% |
| 11 | Industrial | 100% | 90% |
| 12 | Special | 100% | 90% |

**Result**: ~10% reduction in building points, all scenes capped at 1M points.

---

## Training Recommendations

### Class Weighting

Due to severe class imbalance, use inverse frequency weighting:

```python
# Approximate class weights (inverse sqrt frequency)
class_weights = [
    2.5,   # Ground (2.65%)
    2.3,   # Vegetation (3.01%)
    0.5,   # Building (65.12%) - downweight!
    1.0,   # Road (13.87%)
    2.0,   # Wire (3.49%)
    3.5,   # Pole and Tower (0.99%)
    4.5,   # Wire-Structure Connector (0.59%)
    1.5,   # Vehicle (5.67%)
    4.0,   # Board (0.70%)
    6.0,   # Fencing (0.26%)
    3.8,   # Human and Animal (0.82%)
    2.4,   # Road Side Object (2.81%)
]
```

### Loss Functions

Recommended combination for imbalanced data:

```python
criteria=[
    dict(type="CrossEntropyLoss", loss_weight=1.0, ignore_index=-1),
    dict(type="LovaszLoss", mode="multiclass", loss_weight=1.0, ignore_index=-1),
]
```

### Data Augmentation

Strong augmentation helps with imbalanced classes:

```python
transform=[
    dict(type="RandomDropout", dropout_ratio=0.2),
    dict(type="RandomRotate", angle=[-1, 1], axis="z"),
    dict(type="RandomScale", scale=[0.9, 1.1]),
    dict(type="RandomFlip", p=0.5),
    dict(type="RandomJitter", sigma=0.005, clip=0.02),
    dict(type="ElasticDistortion", ...),
    dict(type="GridSample", grid_size=0.02, mode="train"),
]
```

---

## Cross-Dataset Compatibility (SemanticKITTI)

The 12-class schema enables cross-dataset training with SemanticKITTI:

| Bhopal Class | SemanticKITTI Equivalent |
|--------------|-------------------------|
| 0: Ground | sidewalk, terrain, other-ground |
| 1: Vegetation | vegetation, trunk |
| 2: Building | building |
| 3: Road | road, parking |
| 4: Wire | ❌ Not in KITTI |
| 5: Pole and Tower | pole |
| 6: Wire-Structure Connector | ❌ Not in KITTI |
| 7: Vehicle | car, truck, motorcycle, etc. |
| 8: Board | traffic-sign |
| 9: Fencing | fence |
| 10: Human and Animal | person, bicyclist, motorcyclist |
| 11: Road Side Object | ❌ Not in KITTI |

**Common classes for cross-dataset mIoU**: 0, 1, 2, 3, 5, 7, 8, 9, 10 (9 classes)

---

## Related Files

| File | Location | Description |
|------|----------|-------------|
| Dataset HDF5 | `data/Bhopal/bhopal_scene_v7_*.h5` | Main dataset |
| Class Distribution | `data/Bhopal/v7_class_distribution.h5` | Per-scene stats |
| Creation Script | `data/Bhopal/tools/reference/19b_create_scene_V9_dataset.py` | V7 creator |
| DataLoader | `data/Bhopal/19c_dataloader_test_vis_v5_v6_v7.py` | PyTorch Dataset |
| Data README | `data/Bhopal/README.md` | Data directory docs |
| Training Config | `../configs/bhopal/concerto_140gb_12class.py` | H200 training config |

---

*Last Updated: January 24, 2026*
