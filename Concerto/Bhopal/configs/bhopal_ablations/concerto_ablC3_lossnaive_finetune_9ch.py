"""
Class-imbalance ablation cell C3 — vanilla CE only (loss-naïve).

Loss = vanilla CE (no class_weight, no Lovász). Building 10% downsample
retained. Most aggressive ablation: tests how much of the mIoU gain comes from
loss-side mitigations combined.
"""
_base_ = ["../bhopal/concerto_140gb_12class_finetune_9ch.py"]

model = dict(
    criteria=[
        dict(type="CrossEntropyLoss", loss_weight=1.0, ignore_index=-1),
    ],
)
