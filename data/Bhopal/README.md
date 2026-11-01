# data/Bhopal/

Everything the code needs around the BhopalMLS files, laid out as the configs expect it. The two benchmark files are downloaded here (📥 [`../DOWNLOAD.md`](../DOWNLOAD.md)); the rest ships with the repository.

```
data/Bhopal/
├── bhopal_scene_v9_normals_rgb.h5                   <- download (23.0 GB): 62 static classes, every main result
├── bhopal_scene_v10_train_1234_val_56_test_78.h5    <- download (23.4 GB): + 6 moving classes, the paper's dagger rows
├── bhopal_scene_v9_ablB{1_size,2_variety,3_regular,4_subtype}.h5   16 KB each: factor-attribution training subsets
│                                                    (external links into the v9 file by relative name, hence this folder)
├── ablation_subsets.json                            their definition: scene lists, dominant subclass per parent, budgets
├── bhopalmls_download.py                            the download helper (LiDAVerse login, streaming, checksums, resume)
├── class_mapping/                                   V13 code tables, label62 / label68 tables, label_codes.json (all label
│                                                    spaces, 12- and 21-class groupings, palettes), colour tables,
│                                                    class_mapping_utils.py and taxonomy.py (Python helpers)
├── class_colors/                                    CloudCompare colour scales (62 or 71 classes, one per parent class)
├── splits/                                          train.txt, val.txt, test.txt (one scene per line), splits.json,
│                                                    leakage_free_test_partition.csv (+ summary)
└── tools/                                           bhopal_h5_dataset.py (loader, PyTorch Dataset, inspector, renders),
                                                     convert_npz.py (NPZ -> LAS / PLY / CSV / BIN), reference/ (the scripts
                                                     that built the benchmark files)
```

## Label spaces in short

- **V13 codes 0-70**: annotation codes. 0-2 unlabelled and noise, 3-8 the six moving classes, 9-70 the 62 static classes. The `XXZZ` part of every 9-digit instance id is this code (`class_mapping/Bhopal_class_mapping_V13.csv`).
- **label62 = code - 9** (0-61): the v9 benchmark file and the per-path `npz/Bhopal_N.npz` files (`class_mapping/Bhopal_class_mapping_V13_TRAINING.csv`).
- **label68** (0-67): the v10 benchmark file; 62-67 are codes 3-8 (`class_mapping/Bhopal_class_mapping_V13_TRAINING_v10h5.csv`).
- **12-class benchmark**: Ground, Vegetation, Building, Road, Wire, Pole and Tower, Wire-Structure Connector, Vehicle, Board, Fencing, Human and Animal, Road Side Object. Groupings in `class_mapping/label_codes.json["benchmark_12"]` and in every config (`class_groups`).
- **21-class benchmark**: `label_codes.json["benchmark_21"]`; Animal, Dog and Cow (label62 47-49) are ignored.

```python
from data.Bhopal.class_mapping.taxonomy import BENCHMARK_12, groups_to_learning_map   # run from the repository root
lm = groups_to_learning_map(BENCHMARK_12["groups_label62"]); BENCHMARK_12["names"][lm[13]]   # 'Road'
```

## Splits

By survey path: train = paths 1-4 (1,353 scenes), validation = paths 5-6 (568), test = paths 7-8 (489; 3 path-7 scenes are empty). `splits/leakage_free_test_partition.csv` gives, per test scene, the overlap of its bounding box with training coverage and its distance to the nearest training trajectory; `disjoint` marks the 180-scene leakage-free subset the paper re-scores on.

## Colour and privacy

About 35 % of the points on the six paths with imagery carry colour; `0,0,0` means "no colour", not black; paths 2 and 8 have none. The colour of every point labelled as a person (Human and Animal family, riders, the moving human classes) is zero in both releases; geometry and labels are untouched.

## Trajectories

`analysis/leakage/leakage_analysis.py` and `partition_test_scenes.py` read the survey-path GeoPackages from `data/Bhopal/Bhopal_Path_vector_data/Bhopal_{1..8}.gpkg`, which are not part of the v1.0 data release. Their outputs are shipped (`splits/leakage_free_test_partition.csv`, `analysis/leakage/results/`), so re-scoring predictions on the leakage-free subset needs no trajectory data.
