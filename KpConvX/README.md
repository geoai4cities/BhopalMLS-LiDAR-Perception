# KpConvX

**KpConvX-L** in the paper: KPConvX-L (Apple), 13.5 M parameters, trained from scratch; needs the C++ wrappers (`setup_venv.sh --with-kpconvx`) (parameters counted on the model built from the config).

Rows reproduced from this folder: semantic segmentation from scratch with 4ch and 9ch inputs for 12 and 21 classes.

```
KpConvX/
├── README.md
└── Bhopal/
    ├── configs/_base_/default_runtime.py
    ├── configs/bhopal/            one config per paper row (see docs/CONFIGS.md)
    ├── scripts/train_bhopal.sh    train
    ├── scripts/test_bhopal.sh     evaluate on the test split
    └── outputs/<run>/             created by the launcher (gitignored)
```

## Commands

```bash
Bhopal/scripts/train_bhopal.sh train 4ch              # XYZ + intensity
Bhopal/scripts/train_bhopal.sh train 9ch 21c          # XYZ + partial RGB + normals, 21 classes
Bhopal/scripts/test_bhopal.sh  train 9ch
```

Run from this folder or any other; the wrappers resolve the repository root themselves (paths are relative to the repository root; a relative path that exists from the current folder is accepted too). Batch settings of the paper are applied by `scripts/launch.sh`; override with `--batch B --accum A`. Results land in `Bhopal/outputs/<run>/` (comparison datasets: `<Dataset>/outputs/<run>/`) with the run names of `MODEL_ZOO.md`.
