# Launchers

`launch.sh` is the single engine; every model folder has thin wrappers that call it with the model filled in, so the commands look like the ones the paper's runs used:

| wrapper | does |
|---|---|
| `<Model>/Bhopal/scripts/train_bhopal.sh <protocol> [4ch|9ch] [12c|21c] [v10] [outdoor|indoor]` | train one benchmark row |
| `<Model>/Bhopal/scripts/test_bhopal.sh <protocol> [...] [--weight PATH]` | evaluate a run on the test split with the paper's test-time augmentation |
| `<Model>/<Dataset>/scripts/train.sh [protocol] [4ch|9ch] [outdoor|indoor]` and `test.sh` | fine-tune on Ke-MLS, Toronto-3D, Paris-Lille-3D or SemanticKITTI in the unified 12-class taxonomy (no protocol = `ft 9ch`, the cross-region source model; `Kerala/` has the full protocol set, the other three FT-9ch only) |
| `<Model>/cross_data_testing/run_cross_test.sh <source>_on_<target> --weight PATH` | evaluate a model trained on one dataset on another |

Protocols: supervised models `train`, `insseg`; foundation models `lp`, `dp`, `lora`, `ft`, `insseg_lp`, `insseg_dp`, `insseg_ft`. Everything the launcher does is printed first; `--dry-run` shows the exact `tools/train.py` or `tools/test.py` command for a manual launch. Any `key=value` argument is forwarded to Pointcept's `--options`.

Run names and output folders are the ones of the paper's runs (`MODEL_ZOO.md`): `Utonia/Bhopal/outputs/utonia_bhopal_finetune_9ch/`, `Concerto/Bhopal/outputs/concerto_bhopal_linear_probe_4ch_21c_indoor/`, `PTv3/Bhopal/outputs/ptv3_bhopal_train_9ch/`, `Utonia/Bhopal/outputs/utonia_bhopal_decoder_probe_9ch_v10/`, ... Cross-region source runs go to `<Model>/<Dataset>/outputs/` (`Utonia/Kerala/outputs/utonia_kerala_finetune_9ch/`, `Concerto/Toronto-3D/outputs/concerto_toronto_finetune_9ch_indoor/`, `Utonia/SemanticKITTI/outputs/utonia_kitti_finetune_9ch/`). A relative `--config` or `--weight` may be given from the repository root or from the folder the command is typed in.

## Reading the right number

- Training evaluates on the validation split after every epoch and keeps `model/model_best.pth`; at the end `PreciseEvaluator` runs the full test-time-augmented pass on the test split. That final block, and the identical number printed by `test_bhopal.sh`, is what the paper reports. Per-epoch validation mIoU is not comparable.
- Instance runs report mAP, mAP50 and mAP25 over the thing classes only (8 in the 12-class setting, 17 in the 21-class one), without augmentation.
- Pointcept's tester silently reuses cached per-scene predictions in `<run>/result/*_pred.npy`; the launcher deletes the cache before each evaluation so a new weight is really scored.

## GPU memory

The defaults are the 143 GB settings of the paper. On a 40 GB card use roughly half the batch and twice the accumulation (`--batch 4 --accum 16` for a fine-tune); the effective batch is what reproduces the result. Probes should fit on a 24 GB card with `--batch 4 --accum 16` (not tested by us); full fine-tuning of the large models does not.

## Notes

- Seeds are not fixed in the configs (Pointcept draws one and records it in the log); runs differ by a few tenths of mIoU.
- `num_worker=16` reads the HDF5 in parallel; lower it on machines with little RAM.
- `--resume` continues the schedule from `model_last.pth`; the checkpoint loader skips its key remapping on resume (one of the framework changes).
