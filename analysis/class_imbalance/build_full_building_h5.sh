#!/usr/bin/env bash
# Rebuild the v9 scene file WITHOUT the 90 % Building downsampling (ablation cell C4).
#
# Inputs, under data/Bhopal/source/ (see data/Bhopal/tools/reference/README.md):
#   Combined_NPZ_Files_V14_Training_And_Unlabeled/Bhopal_{1..8}.npz   the npz/ files of the data release
#   Bhopal_scene_division_v5/, Bhopal_scene_division_v1/               from metadata/scene_division.zip
#   class_mapping/Bhopal_class_mapping_V13_TRAINING.csv                from metadata/
# Output: data/Bhopal/bhopal_scene_v9_normals_rgb_buildfull.h5 (about 24 GB; 2-4 h CPU, tens of GB RAM).
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO"
OUT="${1:-data/Bhopal/bhopal_scene_v9_normals_rgb_buildfull.h5}"
[[ -e "$OUT" ]] && { echo "$OUT exists; remove it first"; exit 1; }
for d in Combined_NPZ_Files_V14_Training_And_Unlabeled Bhopal_scene_division_v5 Bhopal_scene_division_v1 class_mapping; do
  [[ -d "data/Bhopal/source/$d" ]] || { echo "missing data/Bhopal/source/$d"; exit 1; }
done
python data/Bhopal/tools/reference/19b_create_scene_V9_dataset.py \
  --output "$OUT" --building-sample-rate 1.0 \
  --train-paths 1 2 3 4 --val-paths 5 6 --test-paths 7 8 --random-seed 42 --version 9.0
