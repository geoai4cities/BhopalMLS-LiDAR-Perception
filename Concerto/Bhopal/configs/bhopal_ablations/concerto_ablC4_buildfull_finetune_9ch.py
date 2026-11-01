"""
Class-imbalance ablation cell C4 — drop the H5-build-time Building 10% downsample.

Uses a re-built V9 H5 with `--building-sample-rate 1.0` (all Building points
retained). Loss = full mitigation set (CE + class_weight + Lovász), same as
the C0 baseline. This isolates the contribution of the data-level mitigation.

Pre-requisite: run `phase3/build_v9_buildfull_h5.sh` to produce
`data/Bhopal/bhopal_scene_v9_normals_rgb_buildfull.h5`.
"""
_base_ = ["../bhopal/concerto_140gb_12class_finetune_9ch.py"]

# Override H5 path only; loss + everything else inherit from base.
H5_PATH = "data/Bhopal/bhopal_scene_v9_normals_rgb_buildfull.h5"

data = dict(
    train=dict(h5_path=H5_PATH),
    val=dict(h5_path=H5_PATH),
    test=dict(h5_path=H5_PATH),
)
