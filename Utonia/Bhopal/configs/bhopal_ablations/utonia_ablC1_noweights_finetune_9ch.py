"""
Class-imbalance ablation cell C1 (Utonia variant) — drop sqrt-inverse-frequency
class weights.

Loss = vanilla CE + Lovász (no class_weight). Building 10% downsample retained
(H5-build-time, V9 H5). All other settings inherit from the base FT-9ch config.

Mirrors Concerto/Bhopal/configs/bhopal_ablations/concerto_ablC1_noweights_finetune_9ch.py.
"""
_base_ = ["../bhopal/utonia_140gb_12class_finetune_9ch.py"]

# Override criteria: drop the `weight=` kwarg from CE (uniform weighting).
# Lovász is kept.
model = dict(
    criteria=[
        dict(type="CrossEntropyLoss", loss_weight=1.0, ignore_index=-1),
        dict(type="LovaszLoss", mode="multiclass", loss_weight=1.0, ignore_index=-1),
    ],
)
