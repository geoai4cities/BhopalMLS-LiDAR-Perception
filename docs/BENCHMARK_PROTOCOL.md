# Benchmark protocol

Everything a number in the paper depends on, in one place. The configs inside the model folders are the executable form of this page (index: `docs/CONFIGS.md`).

## 1. Data

- **Scenes.** Each survey path is cut into 5 m corridor segments along the road axis; segments above one million points are subdivided to 2.5, 1.25 and 0.62 m; segments below 3,000 points are dropped; every scene is centred on its own centre. Normals: Open3D PCA with k = 30, horizontal surfaces pointing up, vertical surfaces towards the road. Building-class points (label62 8 to 12) are kept at 90 % (random, seed 42) as part of the class-imbalance recipe.
- **Files.** `bhopal_scene_v9_normals_rgb.h5`: 2,410 scenes, 968.8 M points, 62 static classes (label62). `bhopal_scene_v10_train_1234_val_56_test_78.h5`: 2,434 scenes, 985.4 M points, 68 classes (label68 = label62 + the six moving classes 62 to 67). Per point: XYZ (float32, scene-local metres), RGB (uint8, 0 = no colour), intensity, normal, GPS time, label, 9-digit instance id.
- **Splits by survey path.** Train = paths 1 to 4 (1,353 scenes), validation = paths 5 and 6 (568), test = paths 7 and 8 (489; 3 path-7 scenes are empty). `data/Bhopal/splits/*.txt` lists the scenes. Models are selected on validation and reported once on test.
- **Leakage-free subset.** 180 test scenes (75.7 M points, all 12 classes present) whose bounding box does not intersect the union of training-scene boxes; the remaining 306 overlap. `data/Bhopal/splits/leakage_free_test_partition.csv`. The paper re-scores 36 cached evaluations on both subsets; the method ranking does not change and the mean difference is below a quarter of a mIoU point.

## 2. Inputs

| name | features | used by |
|---|---|---|
| 4ch | XYZ + intensity | supervised baselines (PTv3, LitePT-L, KPConvX-L) |
| 9ch | XYZ + partial RGB (0 where absent) + normals | everything else |
| 3ch | XYZ only: the 9ch pipeline with RGB and normals zeroed (`ZeroColor`, `ZeroNormal`) | geometry-only linear probes (config suffix `linear_probe_4ch`) |

Colour is divided by 255 (`NormalizeColor`); the `RandomScale 0.2` at the head of every pipeline is the Pointcept convention that brings metres into the scale the pretrained models were trained at.

## 3. Taxonomies

- **12 classes**: Ground (label62 0-3), Vegetation (4-7), Building (8-12), Road (13), Wire (14), Pole and Tower (15-18), Wire-Structure Connector (19), Vehicle (20-34), Board (35-37), Fencing (38-42), Human and Animal (43-49), Road Side Object (50-61).
- **21 classes**: Ground (0,1,2), Stairs (3), Vegetation (4-7), Building other (8,9,11,12), Commercial (10), Road (13), Wire (14), Pole other (15,17,19), Electric Pole (16), Street Light (18), 2-Wheeler (22,23,32,33,34), 3-Wheeler (25,27,28,30), 4-Wheeler (20,21,26,29,31), Hand-Cart (24), Board (35-37), Fencing (38-42), Human (43-46), Road Side Furniture (53), Shop Shade (52), Mannequin (57), Road Side Object other (50,51,54,55,56,58,59,60,61). Animal, Dog and Cow (47-49) are ignored.
- **Moving classes († rows).** With the v10 file, Moving Vehicle (62) and Mobike Rider Moving (67) join Vehicle, Moving Humans/Female/Male/Animal (63-66) join Human and Animal; at evaluation they are therefore folded into their static parents and the 12 (or 21) reported classes are unchanged. Configs under `configs/bhopal_v10/`.
- Fine labels that fall in no group map to `ignore_index = -1` in the dataset class (`_remap_labels`).

## 4. Metrics

Semantic segmentation: per-class IoU, mean IoU over the N classes (primary), mean class accuracy, overall accuracy; ignore index -1. Instance segmentation: mAP (IoU thresholds 0.50:0.05:0.95), mAP50 and mAP25 over the thing classes only, Pointcept's `InsSegEvaluator` protocol (minimum region 100 points).

## 5. Semantic segmentation recipes

Common to every run: 1 cm voxels (`GridSample 0.01`, fnv hash); point-cloud range clip at 1.1 times the data extent; augmentation = random z-rotation (p 0.5), random scale 0.9 to 1.1, random flip (p 0.5), coordinate jitter (sigma 0.001, clip 0.004 for the 9ch pretrained models; 0.005 / 0.02 for LitePT), CopyPaste mixing with probability 0.8; loss = cross-entropy with square-root inverse-frequency class weights + Lovasz (multiclass), both weight 1; AdamW, weight decay 0.005; OneCycle schedule, maximum lr 0.002 for heads and 0.0002 for backbone blocks (`param_dicts keyword="block"`), warm-up 4 %, cosine annealing, div factor 10, final div factor 100; mixed precision; 100 epochs; validation every epoch, best checkpoint kept; final test with 10-pass test-time augmentation (scales 0.9, 0.95, 1.0, 1.05, 1.1 with and without flip) at 1 cm voxels and nearest-neighbour upsampling to the full 0.5 cm point set.

Class weights (12-class, sqrt inverse frequency on the training split): Ground 4.9549, Vegetation 4.6483, Building 1.0, Road 2.1667, Wire 4.3169, Pole and Tower 8.0928, Wire-Structure Connector 10.5119, Vehicle 3.3875, Board 9.6172, Fencing 15.8514, Human and Animal 8.9056, Road Side Object 4.8175. The 21-class weights are in the configs (Street Light capped at 20).

| model | setting | effective batch | notes |
|---|---|---|---|
| PTv3 (PT-v3m1, 22 M) | from scratch, 4ch or 9ch | 12 | gradient clip 1.0 |
| LitePT-L (86 M) | from scratch, 4ch or 9ch | 12 | SphereCrop 0.8 / 120 k points as in the official recipe; `grid_size` carried in the batch so CopyPaste recomputes grid coordinates |
| KPConvX-L (40 M) | from scratch, 4ch or 9ch | 12 | official KPConvX-L hyper-parameters; AdamW lr 0.005, wd 0.05; 450 epochs, validation every 50; 13-pass TTA (4 rotations x 3 scales + flip) |
| Sonata(L), Concerto(L/O), Utonia(L) | LP | 64 | frozen backbone, linear head on the concatenated encoder features (1232 / 1728 / 1386 channels) |
| | DP | 64 | frozen encoder, randomly initialised decoder + head trained |
| | LoRA | 64 | frozen encoder with rank-8 adapters (alpha 16, dropout 0.1) + linear head (`DefaultLORASegmentorV2`) |
| | FT | 64 | everything trainable; backbone at 1/10 lr; gradient clip 3.0 |

Checkpoints: Utonia `utonia.pth` (PT-v3m3, channels 54/108/216/432/576, RoPE base 10; decoder heads 8/8/16/32 so that head_dim is divisible by 3); Sonata `pretrain-sonata-v1m1-0-base.pth` (PT-v3m2, 48/96/192/384/512; the student backbone of this file is tensor-for-tensor identical to the released `sonata.pth`); Concerto `concerto_large.pth` = Concerto(L), `concerto_large_outdoor.pth` = Concerto(O) (PT-v3m2, 64/128/256/512/768). Decoders are never pretrained.

## 6. Instance segmentation

PointGroup head (`PG-v1m2`) on the same backbones, 9ch input, voxel size 0.01 (matches the grid), cluster threshold 1.5, closed points 600, propose points 200, minimum 50 points per proposal. Stuff classes excluded from clustering and from mAP: Ground, Vegetation, Road, Wire (12-class indices 0, 1, 3, 4; 21-class indices 0, 2, 5, 6), which leaves 8 and 17 thing classes. Effective batch 24 for the pretrained models (LP/DP 4 x 6; FT 2 x 12 Utonia, 8 x 3 Sonata and Concerto), 12 for PTv3 and LitePT-L. Evaluation with the `InsSegTester`, batch 1, no augmentation.

## 7. Cross-region transfer

Models fine-tuned (FT-9ch, same recipe) on one dataset in the unified 12-class taxonomy are evaluated on another dataset's test split at native resolution with the same test-time augmentation. mIoU and mAcc are computed over the classes present in both datasets: 12 for Ke-MLS, 11 for Paris-Lille-3D, 9 for SemanticKITTI, 7 for Toronto-3D (bounded by the source when it has fewer). Point-cloud range clips are the Bhopal-wide ones on every dataset so no target point is dropped. `<Model>/cross_data_testing/run_cross_test.sh`, configs under `<Model>/cross_data_testing/configs/`. Source models: `<Model>/<Dataset>/scripts/train.sh` with the FT-9ch config of that folder; `<Model>/Kerala/` additionally carries the probe, LoRA and instance configs on Ke-MLS (Utonia, Concerto, Sonata; not paper rows).

## 8. Factor attribution

Four training subsets with the same labelled-point budget (about 135 M points, 25 % of the training split): B1 random scenes (control), B2 only the dominant fine subclass of each parent keeps its label, B3 the most regular corridors (wide carriageway, low curvature), B4 like B2 but Vehicle keeps Car instead of the dominant two-wheeler. Each is fine-tuned with the full recipe and evaluated on the four other datasets. Subset files and definition in `data/Bhopal/` (`bhopal_scene_v9_ablB*.h5`, `ablation_subsets.json`), configs `<Model>/Bhopal/configs/bhopal/*_ablB*`, analysis in `analysis/factor_attribution/`.

## 9. Class-imbalance ablation

C0 full recipe, C1 no class weights, C2 no Lovasz, C3 cross-entropy only, C4 no Building downsampling (needs the full-Building scene file; `analysis/class_imbalance/`). Run for Concerto(L) and Utonia(L).

## 10. CORAL baseline

Post-hoc CORAL on frozen FT-9ch encoders, BhopalMLS -> Paris-Lille-3D: 64-dimensional features extracted on 2,000 random points per scene, covariance alignment, a logistic-regression head trained on aligned source features and evaluated on the Paris-Lille-3D validation file. `analysis/coral/`.
