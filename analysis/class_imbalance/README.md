# Class-imbalance ablation (C0 to C4)

Five settings of the training recipe, each run for Concerto(L) and Utonia(L) with the FT-9ch configuration (supplementary table "Class-Imbalance Ablation"):

| cell | class weights | Lovasz | Building downsampling (90 %) | config |
|---|---|---|---|---|
| C0 full recipe | yes | yes | yes | `<Model>/Bhopal/configs/bhopal/<model>_140gb_12class_finetune_9ch.py` |
| C1 no class weights | no | yes | yes | `<Model>/Bhopal/configs/bhopal_ablations/<model>_ablC1_noweights_finetune_9ch.py` |
| C2 no Lovasz | yes | no | yes | `..._ablC2_nolovasz_finetune_9ch.py` |
| C3 cross-entropy only | no | no | yes | `..._ablC3_lossnaive_finetune_9ch.py` |
| C4 no Building downsampling | yes | yes | no | `..._ablC4_buildfull_finetune_9ch.py` |

C1 to C3 only change the loss. C4 trains on a scene file built with every Building point kept (`bhopal_scene_v9_normals_rgb_buildfull.h5`, about 24 GB), which is not part of the data release; `build_full_building_h5.sh` rebuilds it from the per-path NPZ files and the scene cuts with the reference builder (`data/Bhopal/tools/reference/19b_create_scene_V9_dataset.py`, `--building-sample-rate 1.0`). Expect two to four hours of CPU time and tens of GB of RAM.

```bash
Utonia/Bhopal/scripts/train_bhopal.sh ft 9ch --config Utonia/Bhopal/configs/bhopal_ablations/utonia_ablC2_nolovasz_finetune_9ch.py
Concerto/Bhopal/scripts/train_bhopal.sh ft 9ch indoor --config Concerto/Bhopal/configs/bhopal_ablations/concerto_ablC1_noweights_finetune_9ch.py
```
