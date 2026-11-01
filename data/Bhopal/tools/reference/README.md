# Reference: how the benchmark files were built

These are the scripts that produced the two files in `benchmark/`, kept as they were run so the
protocol is inspectable. They are not needed to use the dataset.

| script | wrote | inputs |
|---|---|---|
| `19b_create_scene_V9_dataset.py` (run as `19b_create_scene_V8_dataset.py`, `--version 9.0`) | `bhopal_scene_v9_normals_rgb.h5`: 62 classes | `npz/Bhopal_N.npz`, `metadata/scene_division/v5`, `metadata/scene_division/v1` |
| `19b_create_scene_V10_dataset.py` | `bhopal_scene_v10_train_1234_val_56_test_78.h5`: 68 classes | the same plus `npz/Bhopal_N_Unlabelled.npz` for the six moving classes |

What they do, in order: read the 5 m corridor cuts of `scene_division/v5` (each JSON gives the scene's
origin, road direction and the projected start and end along it; `v1` lists the instances of the coarse
scenes the cuts were derived from, with extra instances assigned per scene); gather the points of each
cut from the NPZ; keep 90 % of the Building-class points (random, seed 42); subdivide cuts above
1,000,000 points to 2.5, 1.25 and 0.62 m; drop cuts below 3,000 points; centre each scene on its own
centre; estimate normals with Open3D (k = 30, horizontal surfaces Z-up, vertical surfaces toward the
road); split by path (1-4 train, 5-6 val, 7-8 test); write the padded HDF5 layout described in the
data card. The V10 script also assigns the moving-class points of `Bhopal_N_Unlabelled.npz` to scenes
by XY bounding box with 5 m padding and relabels codes 3-8 to 62-67.

To re-run: set `BASE_DIR` near the top of the script to a folder holding `Combined_NPZ_Files_V14_Training_And_Unlabeled/`
(the `npz/` files), `Bhopal_scene_division_v5/` and `Bhopal_scene_division_v1/` (unzip `metadata/scene_division.zip`: it holds `v1/`, `v5/` and a README)
and `class_mapping/` (from `metadata/`). Needs numpy, h5py, pandas, tqdm and open3d; a full run takes
hours and tens of GB of RAM. The output is not guaranteed bit-identical to the shipped files (Open3D
normal estimation and library versions), which is why the files themselves are released.
