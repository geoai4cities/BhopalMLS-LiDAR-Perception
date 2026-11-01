# Sonata

**Sonata(L)** in the paper: PT-v3m2 backbone (48/96/192/384/512); 125 M parameters with decoder and head (parameters counted on the model built from the config).

Rows reproduced from this folder: linear probe (3ch and 9ch), decoder probe, LoRA and fine-tuning for 12 and 21 classes; instance segmentation with the PointGroup head under LP, DP and FT.

```
Sonata/
├── README.md
└── Bhopal/
    ├── configs/_base_/default_runtime.py
    ├── configs/bhopal/            one config per paper row (see docs/CONFIGS.md)
    ├── scripts/train_bhopal.sh    train
    ├── scripts/test_bhopal.sh     evaluate on the test split
    └── outputs/<run>/             created by the launcher (gitignored)
Sonata/Kerala/                 configs/kerala/ (lp, dp, lora, ft, insseg_lp, insseg_dp, insseg_ft; 4ch and 9ch) + scripts/{train,test}.sh on Ke-MLS
```

Pretrained checkpoint: `sonata_pretrained/pretrain-sonata-v1m1-0-base.pth` from https://huggingface.co/facebook/sonata (CC BY-NC 4.0). `setup_venv.sh` does not download it; see `docs/REPRODUCING.md`, step 3.

## Commands

```bash
Bhopal/scripts/train_bhopal.sh lp 4ch            # geometry-only linear probe (the paper's 3ch row)
Bhopal/scripts/train_bhopal.sh lp 9ch 21c        # 21-class linear probe
Bhopal/scripts/train_bhopal.sh dp 9ch
Bhopal/scripts/train_bhopal.sh lora 9ch
Bhopal/scripts/train_bhopal.sh ft 9ch            # fine-tuning
Bhopal/scripts/train_bhopal.sh insseg_ft 9ch 21c # instance segmentation, 17 thing classes
Bhopal/scripts/test_bhopal.sh  ft 9ch            # test-split number with test-time augmentation
Kerala/scripts/train.sh                           # Ke-MLS source model (ft 9ch); Kerala/ also has lp, dp, lora and insseg_* configs
Kerala/scripts/train.sh lp 9ch
```

Run from this folder or any other; the wrappers resolve the repository root themselves (paths are relative to the repository root; a relative path that exists from the current folder is accepted too). Batch settings of the paper are applied by `scripts/launch.sh`; override with `--batch B --accum A`. Results land in `Bhopal/outputs/<run>/` (comparison datasets: `<Dataset>/outputs/<run>/`) with the run names of `MODEL_ZOO.md`.
