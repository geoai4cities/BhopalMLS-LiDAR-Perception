# Reproducing the paper

## 1. Clone and build the environment

```bash
git clone https://github.com/geoai4cities/BhopalMLS-LiDAR-Perception.git && cd BhopalMLS-LiDAR-Perception
./setup_venv.sh --with-kpconvx --with-pointrope
source activate.sh
```

`setup_venv.sh` needs a CUDA toolkit (11.8 or 12.x) and a C++ compiler; the flash-attention build is the slow part (details and troubleshooting: `docs/ENVIRONMENT.md`). The frameworks are vendored: `Pointcept/` (shared by PTv3, KPConvX, Sonata, Concerto, Utonia) and `LitePT/litept/`, at the commits the runs used; `framework_changes/` lists every change against upstream.

## 2. Get the data

```bash
python3 data/Bhopal/bhopalmls_download.py --release international --files benchmark --out ~/BhopalMLS
ln -s ~/BhopalMLS/benchmark/*.h5 data/Bhopal/
python data/Bhopal/tools/bhopal_h5_dataset.py data/Bhopal/bhopal_scene_v9_normals_rgb.h5 --info
```

Use `--release india` from an Indian institution. Both releases give identical numbers. The configs read `data/Bhopal/bhopal_scene_v9_normals_rgb.h5` and `data/Bhopal/bhopal_scene_v10_train_1234_val_56_test_78.h5`; the four small subset files for the factor-attribution ablation already sit in `data/Bhopal/`. Everything about the data: `data/DOWNLOAD.md` and `data/Bhopal/README.md`.

## 3. Get the pretrained checkpoints

| file | source | used for |
|---|---|---|
| `Utonia_pretrained/utonia.pth` (549 MB) | https://huggingface.co/Pointcept/Utonia | Utonia(L) |
| `sonata_pretrained/pretrain-sonata-v1m1-0-base.pth` (1.95 GB) | https://huggingface.co/facebook/sonata | Sonata(L); `sonata.pth` from the same place holds the identical backbone, but the configs' key remapping is written for the pretraining file |
| `Concerto_pretrained/concerto_large.pth` (831 MB) | https://huggingface.co/Pointcept/Concerto | Concerto(L), launcher keyword `indoor` |
| `Concerto_pretrained/concerto_large_outdoor.pth` (831 MB) | https://huggingface.co/Pointcept/Concerto | Concerto(O), the default (`outdoor`) |

```bash
huggingface-cli download Pointcept/Utonia utonia.pth --local-dir Utonia_pretrained
huggingface-cli download facebook/sonata pretrain-sonata-v1m1-0-base.pth --local-dir sonata_pretrained
huggingface-cli download Pointcept/Concerto concerto_large.pth concerto_large_outdoor.pth --local-dir Concerto_pretrained
```

PTv3, LitePT-L and KPConvX-L train from scratch. The pretrained checkpoints are CC BY-NC 4.0.

## 4. Train and evaluate one row

Every row of `RESULTS.md` names its config; the wrappers in each model folder take the same words:

```bash
Utonia/Bhopal/scripts/train_bhopal.sh ft 9ch                 # Table 4, Utonia(L) FT 9ch, 12 classes
Utonia/Bhopal/scripts/train_bhopal.sh ft 9ch v10             # the dagger row (68-class file)
Concerto/Bhopal/scripts/train_bhopal.sh dp 9ch 21c indoor    # supplementary 21-class table, Concerto(L) DP
Sonata/Bhopal/scripts/train_bhopal.sh lp 4ch                 # geometry-only linear probe (3ch row)
KpConvX/Bhopal/scripts/train_bhopal.sh train 9ch
LitePT/Bhopal/scripts/train_bhopal.sh insseg 9ch 21c         # instance segmentation, 17 thing classes
Utonia/Bhopal/scripts/test_bhopal.sh  ft 9ch                 # the test-split number with test-time augmentation
```

Outputs go to `<Model>/Bhopal/outputs/<run>/` (`model/model_best.pth`, `model/model_last.pth`, `logs/`, `result/`). The final `PreciseEvaluator` block of the training log and the output of `test_bhopal.sh` are the reported numbers; per-epoch validation mIoU is not. Run times in our logs on one H200 (143 GB, the card the configs are written for): fine-tuning a large model 10 to 19 hours, linear and decoder probes 7 to 18 hours (the frozen backbone still runs every forward pass), LoRA about 7 hours, PTv3 from scratch about 8 hours, LitePT-L about 6 hours, KPConvX-L (450 epochs) about 18 hours. Expect run-to-run differences of a few tenths of a mIoU point (seeds are not fixed).

## 5. Cross-region study

1. Build the comparison datasets (`data/README.md`).
2. Fine-tune on each source: `Utonia/Toronto-3D/scripts/train.sh` (and `Paris-Lille-3D`, `SemanticKITTI`, `Kerala`; for Concerto(L) add `indoor`). Outputs go to `<Model>/<Dataset>/outputs/<run>/` (`Utonia/Toronto-3D/outputs/utonia_toronto_finetune_9ch/`, `Concerto/Kerala/outputs/concerto_kerala_finetune_9ch_indoor/`, ...). The BhopalMLS source is the in-domain FT run of step 4. `<Model>/Kerala/` also holds the probe, LoRA and instance configs on Ke-MLS (`Utonia/Kerala/scripts/train.sh lp 9ch`).
3. Evaluate every source on every target: `Utonia/cross_data_testing/run_cross_test.sh bhopal_on_tr --weight Utonia/Bhopal/outputs/utonia_bhopal_finetune_9ch/model/model_best.pth`, `Utonia/cross_data_testing/run_cross_test.sh tr_on_bhopal --weight Utonia/Toronto-3D/outputs/utonia_toronto_finetune_9ch/model/model_best.pth`, and so on for the eight pairs per model.
4. `analysis/factor_attribution/collect_perclass_transfer.py` turns the logs into the common-class table.

## 6. Ablations

- Factor attribution: `Utonia/Bhopal/scripts/train_bhopal.sh ft 9ch --config Utonia/Bhopal/configs/bhopal/utonia_140gb_12class_finetune_9ch_ablB1_size.py` (B1 to B4, Utonia and Concerto), evaluate each arm on the four targets with `run_cross_test.sh --weight ...`, collect with `analysis/factor_attribution/collect_ablB_crosseval.py`, read with `analyze_ablB_attribution.py`.
- Class imbalance: `analysis/class_imbalance/README.md`.
- Leakage-free re-scoring of any run: `python analysis/leakage/rescore_predictions.py --exp-dir Utonia/Bhopal/outputs/utonia_bhopal_finetune_9ch --label utonia_ft`.
- CORAL: `analysis/coral/run_all.sh`.

## 7. Compare

`RESULTS.md` holds every reported value. `analysis/leakage/results/summary_table.md` and `analysis/factor_attribution/results/*.md` hold the supplementary tables as produced.
