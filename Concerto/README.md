# Concerto

**Concerto(L) and Concerto(O)** in the paper: PT-v3m2 backbone (64/128/256/512/768); 224 M parameters with decoder and head; `concerto_large.pth` = Concerto(L), general-purpose; `concerto_large_outdoor.pth` = Concerto(O), outdoor-pretrained (parameters counted on the model built from the config).

Rows reproduced from this folder: linear probe (3ch and 9ch), decoder probe, LoRA and fine-tuning for 12 and 21 classes; instance segmentation with the PointGroup head under LP, DP and FT.

```
Concerto/
├── README.md
└── Bhopal/
    ├── configs/_base_/default_runtime.py
    ├── configs/bhopal/            one config per paper row (see docs/CONFIGS.md)
    ├── configs/bhopal_v10/        trained with the 6 moving classes (the paper's dagger rows)
    ├── configs/bhopal_ablations/  class-imbalance cells C1 to C4 (B1 to B4 are in configs/bhopal/)
    ├── scripts/train_bhopal.sh    train
    ├── scripts/test_bhopal.sh     evaluate on the test split
    └── outputs/<run>/             created by the launcher (gitignored)
Concerto/Kerala/                 configs/kerala/ (lp, dp, lora, ft, insseg_lp, insseg_dp, insseg_ft; 4ch and 9ch) + scripts/{train,test}.sh on Ke-MLS
Concerto/{Toronto-3D,Paris-Lille-3D,SemanticKITTI}/   FT-9ch configs + scripts/{train,test}.sh for the cross-region sources
Concerto/cross_data_testing/     transfer-evaluation configs + run_cross_test.sh
```

Pretrained checkpoints: `Concerto_pretrained/concerto_large.pth` and `Concerto_pretrained/concerto_large_outdoor.pth` from https://huggingface.co/Pointcept/Concerto (CC BY-NC 4.0). `setup_venv.sh` does not download them; see `docs/REPRODUCING.md`, step 3.

## Commands

```bash
Bhopal/scripts/train_bhopal.sh lp 4ch outdoor            # geometry-only linear probe (the paper's 3ch row)
Bhopal/scripts/train_bhopal.sh lp 9ch 21c outdoor        # 21-class linear probe
Bhopal/scripts/train_bhopal.sh dp 9ch outdoor
Bhopal/scripts/train_bhopal.sh lora 9ch outdoor
Bhopal/scripts/train_bhopal.sh ft 9ch outdoor            # fine-tuning
Bhopal/scripts/train_bhopal.sh insseg_ft 9ch 21c outdoor # instance segmentation, 17 thing classes
Bhopal/scripts/test_bhopal.sh  ft 9ch outdoor            # test-split number with test-time augmentation
Bhopal/scripts/train_bhopal.sh ft 9ch v10 outdoor        # dagger row (68-class file)
Bhopal/scripts/train_bhopal.sh ft 9ch --config Concerto/Bhopal/configs/bhopal_ablations/concerto_ablC2_nolovasz_finetune_9ch.py
Kerala/scripts/train.sh indoor                           # Ke-MLS source model (ft 9ch); Kerala/ also has lp, dp, lora and insseg_* configs
Kerala/scripts/train.sh lp 9ch indoor
Toronto-3D/scripts/train.sh indoor                       # Toronto-3D source model; likewise Paris-Lille-3D/ and SemanticKITTI/
cross_data_testing/run_cross_test.sh bhopal_on_tr --weight Concerto/Bhopal/outputs/concerto_bhopal_finetune_9ch_indoor/model/model_best.pth
cross_data_testing/run_cross_test.sh tr_on_bhopal --weight Concerto/Toronto-3D/outputs/concerto_toronto_finetune_9ch_indoor/model/model_best.pth
# `outdoor` = concerto_large_outdoor.pth = Concerto(O); `indoor` = concerto_large.pth = Concerto(L) (run folders get the _indoor suffix)
Bhopal/scripts/train_bhopal.sh ft 9ch indoor
```

Run from this folder or any other; the wrappers resolve the repository root themselves (paths are relative to the repository root; a relative path that exists from the current folder is accepted too). Batch settings of the paper are applied by `scripts/launch.sh`; override with `--batch B --accum A`. Results land in `Bhopal/outputs/<run>/` (comparison datasets: `<Dataset>/outputs/<run>/`) with the run names of `MODEL_ZOO.md`.
