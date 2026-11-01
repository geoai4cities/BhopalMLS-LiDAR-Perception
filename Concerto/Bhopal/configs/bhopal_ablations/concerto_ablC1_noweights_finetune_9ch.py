"""
Class-imbalance ablation cell C1 — drop sqrt-inverse-frequency class weights.

Loss = vanilla CE + Lovász (no class_weight). Building 10% downsample retained
(H5-build-time, V9 H5). All other settings inherit from the base FT-9ch config.

Usage:
  ./scripts/train_bhopal_40gb.sh ft 9ch indoor   # NB: pass abl-config via env
  Or invoke pointcept tools/train.py directly with --config-file pointing here.
"""
_base_ = ["../bhopal/concerto_140gb_12class_finetune_9ch.py"]

# Override criteria: drop the `weight=` kwarg from CE (uniform weighting).
# Lovász is kept.
model = dict(
    criteria=[
        dict(type="CrossEntropyLoss", loss_weight=1.0, ignore_index=-1),
        dict(type="LovaszLoss", mode="multiclass", loss_weight=1.0, ignore_index=-1),
    ],
)
