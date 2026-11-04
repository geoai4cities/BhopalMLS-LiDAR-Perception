# Model zoo

Trained weights of the paper's runs. Each row is the `model_best.pth` the paper's number was produced from (selected on the validation split, evaluated once on the test split). The 12-class fine-tuned models ([Sonata(L)](https://huggingface.co/geoai4cities/BhopalMLS-Sonata), [Concerto(L)](https://huggingface.co/geoai4cities/BhopalMLS-Concerto), [Utonia(L)](https://huggingface.co/geoai4cities/BhopalMLS-Utonia)) are on Hugging Face as `model.safetensors` (model weights only; the size column is that file); `python scripts/download_weights.py <run>` writes one where the test launcher reads it, e.g. `python scripts/download_weights.py utonia_bhopal_finetune_9ch && Utonia/Bhopal/scripts/test_bhopal.sh ft 9ch`. The other rows are reproducible from their configs and the released data (`docs/REPRODUCING.md`).

Weights derived from Sonata, Concerto and Utonia inherit the CC BY-NC 4.0 terms of those checkpoints; the from-scratch PTv3, LitePT-L and KPConvX-L weights follow this repository's licence.

Run names are the output folders the launcher creates under `<Model>/Bhopal/outputs/`. Concerto runs without a suffix used `concerto_large_outdoor.pth` (Concerto(O)); `_indoor` marks the general-purpose `concerto_large.pth` (Concerto(L)). `_21c` = 21-class taxonomy, `_v10` = trained with the moving classes († rows).

| Run | Model | Config | Size | Link |
|---|---|---|---|---|
| `ptv3_bhopal_train_4ch` | PTv3 | `PTv3/Bhopal/configs/bhopal/ptv3_140gb_12class_train_4ch.py` | 0.55 GB | to be added |
| `ptv3_bhopal_insseg_9ch` | PTv3 | `PTv3/Bhopal/configs/bhopal/ptv3_140gb_12class_insseg_9ch.py` | 0.55 GB | to be added |
| `ptv3_bhopal_train_4ch_21c` | PTv3 | `PTv3/Bhopal/configs/bhopal/ptv3_140gb_21class_train_4ch.py` | 0.55 GB | to be added |
| `ptv3_bhopal_train_9ch_21c` | PTv3 | `PTv3/Bhopal/configs/bhopal/ptv3_140gb_21class_train_9ch.py` | 0.55 GB | to be added |
| `ptv3_bhopal_insseg_9ch_21c` | PTv3 | `PTv3/Bhopal/configs/bhopal/ptv3_140gb_21class_insseg_9ch.py` | 0.55 GB | to be added |
| `kpconvx_bhopal_train_4ch` | KpConvX-L | `KpConvX/Bhopal/configs/bhopal/kpconvx_140gb_12class_train_4ch.py` | 0.16 GB | to be added |
| `kpconvx_bhopal_train_4ch_21c` | KpConvX-L | `KpConvX/Bhopal/configs/bhopal/kpconvx_140gb_21class_train_4ch.py` | 0.16 GB | to be added |
| `kpconvx_bhopal_train_9ch_21c` | KpConvX-L | `KpConvX/Bhopal/configs/bhopal/kpconvx_140gb_21class_train_9ch.py` | 0.16 GB | to be added |
| `litept_L_bhopal_4ch` | LitePT-L | `LitePT/Bhopal/configs/bhopal/litept_L_12class_4ch.py` | 1.03 GB | to be added |
| `litept_L_bhopal_insseg_9ch` | LitePT-L | `LitePT/Bhopal/configs/bhopal/litept_L_12class_insseg_9ch.py` | 1.20 GB | to be added |
| `litept_L_bhopal_4ch_21c` | LitePT-L | `LitePT/Bhopal/configs/bhopal/litept_L_21class_4ch.py` | 1.03 GB | to be added |
| `litept_L_bhopal_9ch_21c` | LitePT-L | `LitePT/Bhopal/configs/bhopal/litept_L_21class_9ch.py` | 1.03 GB | to be added |
| `litept_L_bhopal_insseg_9ch_21c` | LitePT-L | `LitePT/Bhopal/configs/bhopal/litept_L_21class_insseg_9ch.py` | 1.20 GB | to be added |
| `sonata_bhopal_linear_probe_4ch_21c` | Sonata(L) | `Sonata/Bhopal/configs/bhopal/sonata_140gb_21class_linear_probe_4ch.py` | 0.43 GB | to be added |
| `sonata_bhopal_linear_probe_9ch_21c` | Sonata(L) | `Sonata/Bhopal/configs/bhopal/sonata_140gb_21class_linear_probe_9ch.py` | 0.43 GB | to be added |
| `sonata_bhopal_decoder_probe_9ch_21c` | Sonata(L) | `Sonata/Bhopal/configs/bhopal/sonata_140gb_21class_decoder_probe_9ch.py` | 0.63 GB | to be added |
| `sonata_bhopal_finetune_9ch` | Sonata(L) | `Sonata/Bhopal/configs/bhopal/sonata_140gb_12class_finetune_9ch.py` | 0.50 GB | [🤗 BhopalMLS-Sonata](https://huggingface.co/geoai4cities/BhopalMLS-Sonata/tree/main/Bhopal/sonata_bhopal_finetune_9ch) |
| `sonata_bhopal_finetune_9ch_21c` | Sonata(L) | `Sonata/Bhopal/configs/bhopal/sonata_140gb_21class_finetune_9ch.py` | 1.50 GB | to be added |
| `sonata_bhopal_lora_9ch_21c` | Sonata(L) | `Sonata/Bhopal/configs/bhopal/sonata_140gb_21class_lora_9ch.py` | 0.44 GB | to be added |
| `sonata_bhopal_insseg_lp_9ch` | Sonata(L) | `Sonata/Bhopal/configs/bhopal/sonata_140gb_12class_insseg_lp_9ch.py` | 0.45 GB | to be added |
| `sonata_bhopal_insseg_lp_9ch_21c` | Sonata(L) | `Sonata/Bhopal/configs/bhopal/sonata_140gb_21class_insseg_lp_9ch.py` | 0.45 GB | to be added |
| `sonata_bhopal_insseg_dp_9ch` | Sonata(L) | `Sonata/Bhopal/configs/bhopal/sonata_140gb_12class_insseg_dp_9ch.py` | 0.63 GB | to be added |
| `sonata_bhopal_insseg_dp_9ch_21c` | Sonata(L) | `Sonata/Bhopal/configs/bhopal/sonata_140gb_21class_insseg_dp_9ch.py` | 0.63 GB | to be added |
| `sonata_bhopal_insseg_ft_9ch` | Sonata(L) | `Sonata/Bhopal/configs/bhopal/sonata_140gb_12class_insseg_ft_9ch.py` | 1.50 GB | to be added |
| `sonata_bhopal_insseg_ft_9ch_21c` | Sonata(L) | `Sonata/Bhopal/configs/bhopal/sonata_140gb_21class_insseg_ft_9ch.py` | 1.50 GB | to be added |
| `concerto_bhopal_linear_probe_4ch_21c` | Concerto(O) | `Concerto/Bhopal/configs/bhopal/concerto_140gb_21class_linear_probe_4ch.py` | 0.83 GB | to be added |
| `concerto_bhopal_linear_probe_4ch_21c_indoor` | Concerto(L) | `Concerto/Bhopal/configs/bhopal/concerto_140gb_21class_linear_probe_4ch.py (indoor)` | 0.83 GB | to be added |
| `concerto_bhopal_linear_probe_9ch` | Concerto(O) | `Concerto/Bhopal/configs/bhopal/concerto_140gb_12class_linear_probe_9ch.py` | 0.83 GB | to be added |
| `concerto_bhopal_linear_probe_9ch_indoor` | Concerto(L) | `Concerto/Bhopal/configs/bhopal/concerto_140gb_12class_linear_probe_9ch.py (indoor)` | 0.83 GB | to be added |
| `concerto_bhopal_linear_probe_9ch_21c` | Concerto(O) | `Concerto/Bhopal/configs/bhopal/concerto_140gb_21class_linear_probe_9ch.py` | 0.83 GB | to be added |
| `concerto_bhopal_linear_probe_9ch_21c_indoor` | Concerto(L) | `Concerto/Bhopal/configs/bhopal/concerto_140gb_21class_linear_probe_9ch.py (indoor)` | 0.83 GB | to be added |
| `concerto_bhopal_decoder_probe_9ch` | Concerto(O) | `Concerto/Bhopal/configs/bhopal/concerto_140gb_12class_decoder_probe_9ch.py` | 1.03 GB | to be added |
| `concerto_bhopal_decoder_probe_9ch_21c` | Concerto(O) | `Concerto/Bhopal/configs/bhopal/concerto_140gb_21class_decoder_probe_9ch.py` | 1.03 GB | to be added |
| `concerto_bhopal_decoder_probe_9ch_21c_indoor` | Concerto(L) | `Concerto/Bhopal/configs/bhopal/concerto_140gb_21class_decoder_probe_9ch.py (indoor)` | 1.03 GB | to be added |
| `concerto_bhopal_finetune_9ch` | Concerto(O) | `Concerto/Bhopal/configs/bhopal/concerto_140gb_12class_finetune_9ch.py` | 2.69 GB | to be added |
| `concerto_bhopal_finetune_9ch_indoor` | Concerto(L) | `Concerto/Bhopal/configs/bhopal/concerto_140gb_12class_finetune_9ch.py (indoor)` | 0.90 GB | [🤗 BhopalMLS-Concerto](https://huggingface.co/geoai4cities/BhopalMLS-Concerto/tree/main/Bhopal/concerto_bhopal_finetune_9ch_indoor) |
| `concerto_bhopal_finetune_9ch_21c` | Concerto(O) | `Concerto/Bhopal/configs/bhopal/concerto_140gb_21class_finetune_9ch.py` | 2.69 GB | to be added |
| `concerto_bhopal_finetune_9ch_21c_indoor` | Concerto(L) | `Concerto/Bhopal/configs/bhopal/concerto_140gb_21class_finetune_9ch.py (indoor)` | 2.69 GB | to be added |
| `concerto_bhopal_lora_9ch_21c` | Concerto(O) | `Concerto/Bhopal/configs/bhopal/concerto_140gb_21class_lora_9ch.py` | 0.84 GB | to be added |
| `concerto_bhopal_lora_9ch_21c_indoor` | Concerto(L) | `Concerto/Bhopal/configs/bhopal/concerto_140gb_21class_lora_9ch.py (indoor)` | 0.84 GB | to be added |
| `concerto_bhopal_insseg_lp_9ch` | Concerto(O) | `Concerto/Bhopal/configs/bhopal/concerto_140gb_12class_insseg_lp_9ch.py` | 0.87 GB | to be added |
| `concerto_bhopal_insseg_lp_9ch_indoor` | Concerto(L) | `Concerto/Bhopal/configs/bhopal/concerto_140gb_12class_insseg_lp_9ch.py (indoor)` | 0.87 GB | to be added |
| `concerto_bhopal_insseg_lp_9ch_21c` | Concerto(O) | `Concerto/Bhopal/configs/bhopal/concerto_140gb_21class_insseg_lp_9ch.py` | 0.87 GB | to be added |
| `concerto_bhopal_insseg_lp_9ch_21c_indoor` | Concerto(L) | `Concerto/Bhopal/configs/bhopal/concerto_140gb_21class_insseg_lp_9ch.py (indoor)` | 0.87 GB | to be added |
| `concerto_bhopal_insseg_dp_9ch` | Concerto(O) | `Concerto/Bhopal/configs/bhopal/concerto_140gb_12class_insseg_dp_9ch.py` | 1.03 GB | to be added |
| `concerto_bhopal_insseg_dp_9ch_indoor` | Concerto(L) | `Concerto/Bhopal/configs/bhopal/concerto_140gb_12class_insseg_dp_9ch.py (indoor)` | 1.03 GB | to be added |
| `concerto_bhopal_insseg_dp_9ch_21c` | Concerto(O) | `Concerto/Bhopal/configs/bhopal/concerto_140gb_21class_insseg_dp_9ch.py` | 1.03 GB | to be added |
| `concerto_bhopal_insseg_dp_9ch_21c_indoor` | Concerto(L) | `Concerto/Bhopal/configs/bhopal/concerto_140gb_21class_insseg_dp_9ch.py (indoor)` | 1.03 GB | to be added |
| `concerto_bhopal_insseg_ft_9ch` | Concerto(O) | `Concerto/Bhopal/configs/bhopal/concerto_140gb_12class_insseg_ft_9ch.py` | 2.69 GB | to be added |
| `concerto_bhopal_insseg_ft_9ch_indoor` | Concerto(L) | `Concerto/Bhopal/configs/bhopal/concerto_140gb_12class_insseg_ft_9ch.py (indoor)` | 2.69 GB | to be added |
| `concerto_bhopal_insseg_ft_9ch_21c` | Concerto(O) | `Concerto/Bhopal/configs/bhopal/concerto_140gb_21class_insseg_ft_9ch.py` | 2.69 GB | to be added |
| `concerto_bhopal_insseg_ft_9ch_21c_indoor` | Concerto(L) | `Concerto/Bhopal/configs/bhopal/concerto_140gb_21class_insseg_ft_9ch.py (indoor)` | 2.69 GB | to be added |
| `utonia_bhopal_linear_probe_4ch` | Utonia(L) | `Utonia/Bhopal/configs/bhopal/utonia_140gb_12class_linear_probe_4ch.py` | 0.55 GB | to be added |
| `utonia_bhopal_linear_probe_4ch_21c` | Utonia(L) | `Utonia/Bhopal/configs/bhopal/utonia_140gb_21class_linear_probe_4ch.py` | 0.55 GB | to be added |
| `utonia_bhopal_linear_probe_9ch` | Utonia(L) | `Utonia/Bhopal/configs/bhopal/utonia_140gb_12class_linear_probe_9ch.py` | 0.55 GB | to be added |
| `utonia_bhopal_linear_probe_9ch_21c` | Utonia(L) | `Utonia/Bhopal/configs/bhopal/utonia_140gb_21class_linear_probe_9ch.py` | 0.55 GB | to be added |
| `utonia_bhopal_decoder_probe_9ch` | Utonia(L) | `Utonia/Bhopal/configs/bhopal/utonia_140gb_12class_decoder_probe_9ch.py` | 0.85 GB | to be added |
| `utonia_bhopal_decoder_probe_9ch_21c` | Utonia(L) | `Utonia/Bhopal/configs/bhopal/utonia_140gb_21class_decoder_probe_9ch.py` | 0.85 GB | to be added |
| `utonia_bhopal_finetune_9ch` | Utonia(L) | `Utonia/Bhopal/configs/bhopal/utonia_140gb_12class_finetune_9ch.py` | 0.65 GB | [🤗 BhopalMLS-Utonia](https://huggingface.co/geoai4cities/BhopalMLS-Utonia/tree/main/Bhopal/utonia_bhopal_finetune_9ch) |
| `utonia_bhopal_finetune_9ch_21c` | Utonia(L) | `Utonia/Bhopal/configs/bhopal/utonia_140gb_21class_finetune_9ch.py` | 1.95 GB | to be added |
| `utonia_bhopal_lora_9ch` | Utonia(L) | `Utonia/Bhopal/configs/bhopal/utonia_140gb_12class_lora_9ch.py` | 0.55 GB | to be added |
| `utonia_bhopal_lora_9ch_21c` | Utonia(L) | `Utonia/Bhopal/configs/bhopal/utonia_140gb_21class_lora_9ch.py` | 0.55 GB | to be added |
| `utonia_bhopal_insseg_lp_9ch` | Utonia(L) | `Utonia/Bhopal/configs/bhopal/utonia_140gb_12class_insseg_lp_9ch.py` | 0.57 GB | to be added |
| `utonia_bhopal_insseg_lp_9ch_21c` | Utonia(L) | `Utonia/Bhopal/configs/bhopal/utonia_140gb_21class_insseg_lp_9ch.py` | 0.57 GB | to be added |
| `utonia_bhopal_insseg_dp_9ch` | Utonia(L) | `Utonia/Bhopal/configs/bhopal/utonia_140gb_12class_insseg_dp_9ch.py` | 0.85 GB | to be added |
| `utonia_bhopal_insseg_dp_9ch_21c` | Utonia(L) | `Utonia/Bhopal/configs/bhopal/utonia_140gb_21class_insseg_dp_9ch.py` | 0.85 GB | to be added |
| `utonia_bhopal_insseg_ft_9ch` | Utonia(L) | `Utonia/Bhopal/configs/bhopal/utonia_140gb_12class_insseg_ft_9ch.py` | 1.95 GB | to be added |
| `utonia_bhopal_insseg_ft_9ch_21c` | Utonia(L) | `Utonia/Bhopal/configs/bhopal/utonia_140gb_21class_insseg_ft_9ch.py` | 1.95 GB | to be added |
| `utonia_bhopal_linear_probe_9ch_v10` | Utonia(L)† | `Utonia/Bhopal/configs/bhopal_v10/utonia_140gb_12class_linear_probe_9ch.py` | 0.55 GB | to be added |
| `utonia_bhopal_linear_probe_9ch_21c_v10` | Utonia(L)† | `Utonia/Bhopal/configs/bhopal_v10/utonia_140gb_21class_linear_probe_9ch.py` | 0.55 GB | to be added |
| `utonia_bhopal_decoder_probe_9ch_v10` | Utonia(L)† | `Utonia/Bhopal/configs/bhopal_v10/utonia_140gb_12class_decoder_probe_9ch.py` | 0.85 GB | to be added |
| `utonia_bhopal_decoder_probe_9ch_21c_v10` | Utonia(L)† | `Utonia/Bhopal/configs/bhopal_v10/utonia_140gb_21class_decoder_probe_9ch.py` | 0.85 GB | to be added |
| `utonia_bhopal_finetune_9ch_v10` | Utonia(L)† | `Utonia/Bhopal/configs/bhopal_v10/utonia_140gb_12class_finetune_9ch.py` | 1.95 GB | to be added |
| `utonia_bhopal_finetune_9ch_21c_v10` | Utonia(L)† | `Utonia/Bhopal/configs/bhopal_v10/utonia_140gb_21class_finetune_9ch.py` | 1.95 GB | to be added |
| `concerto_bhopal_linear_probe_9ch_v10` | Concerto(O)† | `Concerto/Bhopal/configs/bhopal_v10/concerto_140gb_12class_linear_probe_9ch.py` | 0.83 GB | to be added |
| `concerto_bhopal_linear_probe_9ch_21c_v10` | Concerto(O)† | `Concerto/Bhopal/configs/bhopal_v10/concerto_140gb_21class_linear_probe_9ch.py` | 0.83 GB | to be added |
| `concerto_bhopal_decoder_probe_9ch_21c_v10` | Concerto(O)† | `Concerto/Bhopal/configs/bhopal_v10/concerto_140gb_21class_decoder_probe_9ch.py` | 1.03 GB | to be added |
| `concerto_bhopal_finetune_9ch_v10` | Concerto(O)† | `Concerto/Bhopal/configs/bhopal_v10/concerto_140gb_12class_finetune_9ch.py` | 2.69 GB | to be added |
| `concerto_bhopal_finetune_9ch_21c_v10` | Concerto(O)† | `Concerto/Bhopal/configs/bhopal_v10/concerto_140gb_21class_finetune_9ch.py` | 2.69 GB | to be added |

## Cross-region source models

The FT-9ch models of the cross-region study (main paper Table 5 and the supplementary cross-region table), fine-tuned on each comparison dataset in the unified 12-class taxonomy; the BhopalMLS source is `utonia_bhopal_finetune_9ch` / `concerto_bhopal_finetune_9ch_indoor` above. Run folders are `<Model>/<Dataset>/outputs/<run>`, as `<Model>/<Dataset>/scripts/train.sh` creates them.

| Run | Model | Config | Size | Link |
|---|---|---|---|---|
| `Utonia/Kerala/outputs/utonia_kerala_finetune_9ch` | Utonia(L) | `Utonia/Kerala/configs/kerala/utonia_140gb_12class_finetune_9ch.py` | 1.95 GB | to be added |
| `Utonia/Toronto-3D/outputs/utonia_toronto_finetune_9ch` | Utonia(L) | `Utonia/Toronto-3D/configs/toronto/utonia_140gb_12class_finetune_9ch.py` | 1.95 GB | to be added |
| `Utonia/Paris-Lille-3D/outputs/utonia_paris_lille_finetune_9ch` | Utonia(L) | `Utonia/Paris-Lille-3D/configs/paris_lille/utonia_140gb_12class_finetune_9ch.py` | 1.95 GB | to be added |
| `Utonia/SemanticKITTI/outputs/utonia_kitti_finetune_9ch` | Utonia(L) | `Utonia/SemanticKITTI/configs/semantickitti/utonia_12class_finetune_9ch.py` | 1.95 GB | to be added |
| `Concerto/Kerala/outputs/concerto_kerala_finetune_9ch_indoor` | Concerto(L) | `Concerto/Kerala/configs/kerala/concerto_140gb_12class_finetune_9ch.py (indoor)` | 2.69 GB | to be added |
| `Concerto/Toronto-3D/outputs/concerto_toronto_finetune_9ch_indoor` | Concerto(L) | `Concerto/Toronto-3D/configs/toronto/concerto_140gb_12class_finetune_9ch.py (indoor)` | 2.69 GB | to be added |
| `Concerto/Paris-Lille-3D/outputs/concerto_paris_lille_finetune_9ch_indoor` | Concerto(L) | `Concerto/Paris-Lille-3D/configs/paris_lille/concerto_140gb_12class_finetune_9ch.py (indoor)` | 2.69 GB | to be added |
| `Concerto/SemanticKITTI/outputs/concerto_kitti_finetune_9ch_indoor` | Concerto(L) | `Concerto/SemanticKITTI/configs/semantickitti/concerto_12class_finetune_9ch.py (indoor)` | 2.69 GB | to be added |
| `Concerto/SemanticKITTI/outputs/concerto_kitti_finetune_9ch` | Concerto(O) | `Concerto/SemanticKITTI/configs/semantickitti/concerto_12class_finetune_9ch.py` | 2.69 GB | to be added |

Rows of the paper whose weight file is not retained on the authors' disk (the result stands; the run is reproducible from its config):

- `ptv3_bhopal_train_9ch` (PTv3): `PTv3/Bhopal/configs/bhopal/ptv3_140gb_12class_train_9ch.py`
- `kpconvx_bhopal_train_9ch` (KpConvX-L): `KpConvX/Bhopal/configs/bhopal/kpconvx_140gb_12class_train_9ch.py`
- `litept_L_bhopal_9ch` (LitePT-L): `LitePT/Bhopal/configs/bhopal/litept_L_12class_9ch.py`
- `sonata_bhopal_linear_probe_4ch` (Sonata(L)): `Sonata/Bhopal/configs/bhopal/sonata_140gb_12class_linear_probe_4ch.py`
- `sonata_bhopal_linear_probe_9ch` (Sonata(L)): `Sonata/Bhopal/configs/bhopal/sonata_140gb_12class_linear_probe_9ch.py`
- `sonata_bhopal_decoder_probe_9ch` (Sonata(L)): `Sonata/Bhopal/configs/bhopal/sonata_140gb_12class_decoder_probe_9ch.py`
- `sonata_bhopal_lora_9ch` (Sonata(L)): `Sonata/Bhopal/configs/bhopal/sonata_140gb_12class_lora_9ch.py`
- `concerto_bhopal_linear_probe_4ch` (Concerto(O)): `Concerto/Bhopal/configs/bhopal/concerto_140gb_12class_linear_probe_4ch.py`
- `concerto_bhopal_linear_probe_4ch_indoor` (Concerto(L)): `Concerto/Bhopal/configs/bhopal/concerto_140gb_12class_linear_probe_4ch.py (indoor)`
- `concerto_bhopal_decoder_probe_9ch_indoor` (Concerto(L)): `Concerto/Bhopal/configs/bhopal/concerto_140gb_12class_decoder_probe_9ch.py (indoor)`
- `concerto_bhopal_lora_9ch` (Concerto(O)): `Concerto/Bhopal/configs/bhopal/concerto_140gb_12class_lora_9ch.py`
- `concerto_bhopal_lora_9ch_indoor` (Concerto(L)): `Concerto/Bhopal/configs/bhopal/concerto_140gb_12class_lora_9ch.py (indoor)`
- `concerto_bhopal_decoder_probe_9ch_v10` (Concerto(O)†): `Concerto/Bhopal/configs/bhopal_v10/concerto_140gb_12class_decoder_probe_9ch.py`
