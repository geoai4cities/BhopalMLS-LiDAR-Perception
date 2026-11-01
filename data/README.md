# data/

The data lives on LiDAVerse, not in git; this folder holds what the code needs around it. 📥 Getting the BhopalMLS files: [`DOWNLOAD.md`](DOWNLOAD.md).

```
data/
├── DOWNLOAD.md               account, India vs international release, every download command
├── Bhopal/                   BhopalMLS: class tables, colours, splits, tools, the 4 ablation subset files
│                             + the benchmark H5 files you download (gitignored)        -> Bhopal/README.md
├── Kerala/                   Ke-MLS class tables (class_mapping/) and KERALA_DATASET.md (+ Kerala_scene_v1_corridor.h5, gitignored)
├── Toronto_3D/               preprocess_toronto_to_hdf5.py, class names (+ L00x.ply and the H5, gitignored)
├── Paris-Lille-3D/           preprocess_pl3d_to_hdf5.py (+ training_50_classes/ and the H5, gitignored)
└── SemanticKITTI/            prepare_semantickitti_9ch.sh (+ official/ and dataset/, gitignored)
```

Paths are relative to the repository root, exactly as the configs name them (`data/Bhopal/bhopal_scene_v9_normals_rgb.h5`, `data/Kerala/Kerala_scene_v1_corridor.h5`, `data/Toronto_3D/toronto_scene_v1_normals_rgb.h5`, `data/Paris-Lille-3D/paris_lille_scene_v1_normals.h5`, `data/SemanticKITTI/dataset/sequences/`).

## Comparison datasets of the cross-region study

The cross-region experiments fine-tune Concerto(L), Concerto(O) and Utonia(L) on each dataset in the unified 12-class BhopalMLS taxonomy and evaluate every model on every other dataset. The three public datasets are converted into the same scene-HDF5 layout as BhopalMLS (5 m XY tiles, adaptive subdivision, scene-local coordinates, Open3D normals with k = 30) so that one dataset class and one set of transforms serve all of them.

| dataset | where to get it | preprocessing here | expected file |
|---|---|---|---|
| Toronto-3D (Tan et al. 2020) | https://github.com/WeikaiTan/Toronto-3D (`L001.ply` to `L004.ply`) | `python data/Toronto_3D/preprocess_toronto_to_hdf5.py --output data/Toronto_3D/toronto_scene_v1_normals_rgb.h5` with the PLY files in `data/Toronto_3D/` | 1.6 GB; L001, L003, L004 train, L002 val and test |
| Paris-Lille-3D (Roynard et al. 2018) | https://npm3d.fr/paris-lille-3d (50-class training set: `Lille1.ply`, `Paris.ply`, `Lille2.ply`) | `python data/Paris-Lille-3D/preprocess_pl3d_to_hdf5.py --output data/Paris-Lille-3D/paris_lille_scene_v1_normals.h5` with the PLY files in `data/Paris-Lille-3D/training_50_classes/` | 3.2 GB; Lille1 + Paris train, Lille2 val and test |
| SemanticKITTI (Behley et al. 2019) | http://semantic-kitti.org (velodyne, labels, calibration) and the KITTI odometry colour images | `data/SemanticKITTI/prepare_semantickitti_9ch.sh` arranges the official downloads under `data/SemanticKITTI/dataset/sequences/`; the configs use Pointcept's `SemanticKITTIImagePointDataset`, which colours the points from the stereo images at load time | folder, no H5 |
| Ke-MLS (Kumar et al. 2026, *Environment and Planning B* 53(8)) | the Ke-MLS release; its scene file `Kerala_scene_v1_corridor.h5` (2,199 scenes, 48 classes, 25.9 GB) was built with the same corridor-cut recipe as BhopalMLS | none; place the file at `data/Kerala/Kerala_scene_v1_corridor.h5` | class tables in `Kerala/class_mapping/`, file description in `Kerala/KERALA_DATASET.md`; besides the FT-9ch source config, `<Model>/Kerala/configs/kerala/` holds probe, LoRA and instance configs on Ke-MLS for Utonia, Concerto and Sonata |

Label mappings into the 12-class taxonomy live in the dataset classes (`Pointcept/pointcept/datasets/{toronto,paris_lille,kerala}.py`) and, for SemanticKITTI, in the `learning_map` of the SemanticKITTI configs. Classes a dataset does not label are excluded from its metrics; the common-class counts the paper uses are 12 for Ke-MLS, 11 for Paris-Lille-3D (no Wire-Structure Connector), 9 for SemanticKITTI (no Wire, Wire-Structure Connector, Road Side Object) and 7 for Toronto-3D (Vegetation, Building, Road, Wire, Pole and Tower, Vehicle, Fencing).

Both preprocessing scripts take `--normal-k`, `--seed` and `--limit-tiles-per-file` (for a smoke test). Normals are oriented towards the sensor: Paris-Lille-3D carries the sensor origin per point; Toronto-3D has no origin, so the GPS time of each point is binned to one rotation and the sensor is placed at that bin's XY centroid above the scene. Toronto-3D coordinates are shifted by a fixed UTM offset before the float32 cast (see the script header). Running the scripts does not give bit-identical files to the authors' copies (Open3D versions), which is why the paper quotes the tolerance of its re-evaluation.
