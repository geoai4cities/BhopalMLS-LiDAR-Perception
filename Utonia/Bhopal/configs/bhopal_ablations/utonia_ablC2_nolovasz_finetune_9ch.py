"""
Class-imbalance ablation cell C2 (Utonia variant) — drop Lovász loss.

Loss = weighted CE only (no Lovász). Building 10% downsample retained.
Class weights inherit from the base config (sqrt-inverse-frequency).

Mirrors Concerto/Bhopal/configs/bhopal_ablations/concerto_ablC2_nolovasz_finetune_9ch.py.
"""
_base_ = ["../bhopal/utonia_140gb_12class_finetune_9ch.py"]

# Re-import class_weight from base for explicit visibility (it is also defined
# in the _base_; we restate it here so the criteria dict can reference it).
class_weight = [
    4.9549, 4.6483, 1.0000, 2.1667, 4.3169, 8.0928,
    10.5119, 3.3875, 9.6172, 15.8514, 8.9056, 4.8175,
]

model = dict(
    criteria=[
        dict(type="CrossEntropyLoss", weight=class_weight, loss_weight=1.0, ignore_index=-1),
    ],
)
