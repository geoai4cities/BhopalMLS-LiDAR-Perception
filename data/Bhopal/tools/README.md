# BhopalMLS tools

Two scripts, both MIT-licensed and dependent only on numpy and h5py unless noted:

| script | what it does |
|---|---|
| `bhopal_h5_dataset.py` | reads the `benchmark/` H5 files: numpy loader (`load_scene`), PyTorch `BhopalSceneDataset` (torch optional), `--info` summary and class table, `--samples`, `--dataloader` smoke test, `--render` PNGs with Open3D |
| `convert_npz.py` | exports the `npz/` files to LAS, PLY, CSV or BIN |
| `reference/` | the two scripts that built the `benchmark/` files from `npz/` and `metadata/scene_division.zip`, kept for provenance (see `reference/README.md`) |

## bhopal_h5_dataset.py

```bash
python tools/bhopal_h5_dataset.py benchmark/bhopal_scene_v9_normals_rgb.h5 --info
python tools/bhopal_h5_dataset.py benchmark/bhopal_scene_v9_normals_rgb.h5 --samples 3 --split test
python tools/bhopal_h5_dataset.py benchmark/bhopal_scene_v9_normals_rgb.h5 --dataloader --num-points 65536
xvfb-run -a python tools/bhopal_h5_dataset.py benchmark/bhopal_scene_v10_train_1234_val_56_test_78.h5 \
    --render 2 --color-by labels --out renders/        # also: parent, rgb, normals
```

```python
from bhopal_h5_dataset import BhopalSceneDataset, load_scene, parent_map
ds = BhopalSceneDataset("benchmark/bhopal_scene_v9_normals_rgb.h5", split="train", num_points=65536)
item = ds[0]      # points [N,3], rgb [N,3] in [0,1], normals [N,3], intensity [N], labels [N] (-1 = pad), global_ids
```

Both benchmark files work (v9: 62 labels; v10: 68, with 62-67 the moving classes). Names, colours and
the 12-class parent of each label come from the file's `class_info` group; `parent_map(f)[labels]`
gives 12-class labels. Scenes are scene-local; add `metadata/center[i]` to return to the file's frame.

## convert_npz.py

`convert_npz.py` exports a per-path NPZ file of this release to LAS, PLY, CSV or BIN. It needs
Python 3.9+ and numpy; LAS export needs `laspy` (`pip install laspy`), and the EPSG tag on the India
release needs `pyproj` (optional). MIT licence.

```bash
python tools/convert_npz.py npz/Bhopal_6.npz --format las --out export/
python tools/convert_npz.py npz/Bhopal_6.npz npz/Bhopal_6_Unlabelled.npz --format ply csv --out export/
python tools/convert_npz.py npz/Bhopal_6.npz --format bin --out export/ --per-instance
```

By default each NPZ becomes one file (`export/Bhopal_6.las`). With `--per-instance` every instance
becomes one file named by its 9-digit global id under `export/Bhopal_6/<parent class>/<class>/`, the
layout of the original per-instance release. Coordinates, labels and ids are written exactly as stored
in the NPZ: absolute UTM 43N in the India release, offset coordinates in the international release.

| format | contents |
|---|---|
| `las` | LAS 1.4, point format 7. `x y z` at 0.1 mm resolution; `intensity`; `gps_time` (the sensor time base of the NPZ); `red green blue` as 16-bit values (8-bit value x 257; 0 = no colour); `classification` = label; extra bytes `label` (uint16) and `global_id` (uint32). India-release files carry EPSG:32643 when pyproj is installed. |
| `ply` | binary little-endian: `x y z` (double), `red green blue` (uchar), `intensity` (ushort), `gps_time` (float), `label` (ushort), `global_id` (uint). Opens in CloudCompare, MeshLab, Open3D. |
| `csv` | header `x,y,z,intensity,gps_time,label,global_id,red,green,blue`; `x y z` and `gps_time` with 4 decimals. |
| `bin` | BPLC v3, below. |

A whole path is held in memory while writing (up to about 10 GB for the largest path, Bhopal_2). Use
`--per-instance` on small machines.

## BPLC v3 (bin)

Little-endian. Header of 48 bytes: magic `BPLC` (4), version uint16 = 3, reserved uint16 = 0,
num_points uint32, instance_id uint32 (0 for a whole path), instance_label uint16 (65535 for a whole
path), xxzz uint32 (the V13 class code of the instance; 0 for a whole path), 26 bytes of zeros.
Then num_points records of 39 bytes each: `x y z` float64, `intensity` uint16, `gps_time` float32,
`label` uint16, `global_id` uint32, `red green blue` uint8.

`read_bin(path)` in `convert_npz.py` reads v3 and the earlier v2 (36-byte records without colour):

```python
from convert_npz import read_bin
header, points = read_bin("export/Bhopal_6.bin")
print(header, points["x"][:3], points["label"][:3])
```
