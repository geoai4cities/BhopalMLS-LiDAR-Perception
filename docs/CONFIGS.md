# Configs

Pointcept-style Python configs, one per paper row, inside each model folder. File names are the ones used in the authors' runs (`140gb` = written for a 143 GB H200; the launcher applies the per-GPU batch settings, `scripts/README.md`).

Name grammar: `<model>_140gb_<12class|21class>_<protocol>_<channels>.py`

| protocol | meaning |
|---|---|
| `linear_probe_4ch` | frozen backbone, linear head; RGB and normals zeroed, so the input is geometry only (the paper's **3ch** linear-probe rows) |
| `linear_probe_9ch` | frozen backbone, linear head on XYZ + partial RGB + normals |
| `decoder_probe_9ch` | frozen encoder, trainable decoder and head |
| `lora_9ch` | frozen encoder with LoRA adapters (rank 8, alpha 16) and a linear head |
| `finetune_9ch` | whole network trainable, backbone at 1/10 of the head learning rate |
| `insseg_{lp,dp,ft}_9ch` | the same three regimes with the PointGroup instance head |
| `train_4ch`, `train_9ch`, `_4ch`, `_9ch` | supervised training from scratch (PTv3, KPConvX, LitePT) |

Folders: `configs/bhopal/` = the benchmark rows (plus the factor-attribution arms `*_ablB{1..4}_*`); `configs/bhopal_v10/` = trained with the 68-class v10 file (moving classes folded into Vehicle and Human and Animal; the † rows); `configs/bhopal_ablations/` = class-imbalance cells C1 to C4; `<Model>/<Dataset>/configs/` = fine-tuning on the comparison datasets in the unified 12-class taxonomy (`Kerala/` holds the whole probe, LoRA, fine-tune and instance set on Ke-MLS for Utonia, Concerto and Sonata; the other three datasets the FT-9ch source config); `<Model>/cross_data_testing/configs/` = evaluating a model trained on one dataset on another.

Concerto configs load `concerto_large_outdoor.pth` (Concerto(O)) by default; the launcher keyword `indoor` switches to `concerto_large.pth` (Concerto(L)) by overriding `weight`.

| config | what it is |
|---|---|
| `Utonia/Bhopal/configs/bhopal/utonia_140gb_12class_linear_probe_4ch.py` | Utonia 12class linear_probe_4ch |
| `Utonia/Bhopal/configs/bhopal/utonia_140gb_12class_linear_probe_9ch.py` | Utonia 12class linear_probe_9ch |
| `Utonia/Bhopal/configs/bhopal/utonia_140gb_12class_decoder_probe_9ch.py` | Utonia 12class decoder_probe_9ch |
| `Utonia/Bhopal/configs/bhopal/utonia_140gb_12class_finetune_9ch.py` | Utonia 12class finetune_9ch |
| `Utonia/Bhopal/configs/bhopal/utonia_140gb_12class_lora_9ch.py` | Utonia 12class lora_9ch |
| `Utonia/Bhopal/configs/bhopal/utonia_140gb_12class_insseg_lp_9ch.py` | Utonia 12class insseg_lp_9ch |
| `Utonia/Bhopal/configs/bhopal/utonia_140gb_12class_insseg_dp_9ch.py` | Utonia 12class insseg_dp_9ch |
| `Utonia/Bhopal/configs/bhopal/utonia_140gb_12class_insseg_ft_9ch.py` | Utonia 12class insseg_ft_9ch |
| `Utonia/Bhopal/configs/bhopal/utonia_140gb_21class_linear_probe_4ch.py` | Utonia 21class linear_probe_4ch |
| `Utonia/Bhopal/configs/bhopal/utonia_140gb_21class_linear_probe_9ch.py` | Utonia 21class linear_probe_9ch |
| `Utonia/Bhopal/configs/bhopal/utonia_140gb_21class_decoder_probe_9ch.py` | Utonia 21class decoder_probe_9ch |
| `Utonia/Bhopal/configs/bhopal/utonia_140gb_21class_finetune_9ch.py` | Utonia 21class finetune_9ch |
| `Utonia/Bhopal/configs/bhopal/utonia_140gb_21class_lora_9ch.py` | Utonia 21class lora_9ch |
| `Utonia/Bhopal/configs/bhopal/utonia_140gb_21class_insseg_lp_9ch.py` | Utonia 21class insseg_lp_9ch |
| `Utonia/Bhopal/configs/bhopal/utonia_140gb_21class_insseg_dp_9ch.py` | Utonia 21class insseg_dp_9ch |
| `Utonia/Bhopal/configs/bhopal/utonia_140gb_21class_insseg_ft_9ch.py` | Utonia 21class insseg_ft_9ch |
| `Utonia/Bhopal/configs/bhopal_v10/utonia_140gb_12class_decoder_probe_9ch.py` | Utonia trained with the 6 moving classes (dagger rows) |
| `Utonia/Bhopal/configs/bhopal_v10/utonia_140gb_12class_finetune_9ch.py` | Utonia trained with the 6 moving classes (dagger rows) |
| `Utonia/Bhopal/configs/bhopal_v10/utonia_140gb_12class_linear_probe_9ch.py` | Utonia trained with the 6 moving classes (dagger rows) |
| `Utonia/Bhopal/configs/bhopal_v10/utonia_140gb_12class_lora_9ch.py` | Utonia trained with the 6 moving classes (dagger rows) |
| `Utonia/Bhopal/configs/bhopal_v10/utonia_140gb_21class_decoder_probe_9ch.py` | Utonia trained with the 6 moving classes (dagger rows) |
| `Utonia/Bhopal/configs/bhopal_v10/utonia_140gb_21class_finetune_9ch.py` | Utonia trained with the 6 moving classes (dagger rows) |
| `Utonia/Bhopal/configs/bhopal_v10/utonia_140gb_21class_linear_probe_9ch.py` | Utonia trained with the 6 moving classes (dagger rows) |
| `Utonia/Bhopal/configs/bhopal_v10/utonia_140gb_21class_lora_9ch.py` | Utonia trained with the 6 moving classes (dagger rows) |
| `Utonia/Bhopal/configs/bhopal/utonia_140gb_12class_finetune_9ch_ablB1_size.py` | Utonia factor-attribution arm |
| `Utonia/Bhopal/configs/bhopal/utonia_140gb_12class_finetune_9ch_ablB2_variety.py` | Utonia factor-attribution arm |
| `Utonia/Bhopal/configs/bhopal/utonia_140gb_12class_finetune_9ch_ablB3_regular.py` | Utonia factor-attribution arm |
| `Utonia/Bhopal/configs/bhopal/utonia_140gb_12class_finetune_9ch_ablB4_subtype.py` | Utonia factor-attribution arm |
| `Utonia/Bhopal/configs/bhopal_ablations/utonia_ablC1_noweights_finetune_9ch.py` | Utonia class-imbalance cell |
| `Utonia/Bhopal/configs/bhopal_ablations/utonia_ablC2_nolovasz_finetune_9ch.py` | Utonia class-imbalance cell |
| `Utonia/Bhopal/configs/bhopal_ablations/utonia_ablC3_lossnaive_finetune_9ch.py` | Utonia class-imbalance cell |
| `Utonia/Bhopal/configs/bhopal_ablations/utonia_ablC4_buildfull_finetune_9ch.py` | Utonia class-imbalance cell |
| `Utonia/Kerala/configs/kerala/utonia_140gb_12class_decoder_probe_4ch.py` | Utonia decoder_probe_4ch on Ke-MLS (12-class unified taxonomy) |
| `Utonia/Kerala/configs/kerala/utonia_140gb_12class_decoder_probe_9ch.py` | Utonia decoder_probe_9ch on Ke-MLS (12-class unified taxonomy) |
| `Utonia/Kerala/configs/kerala/utonia_140gb_12class_finetune_4ch.py` | Utonia finetune_4ch on Ke-MLS (12-class unified taxonomy) |
| `Utonia/Kerala/configs/kerala/utonia_140gb_12class_finetune_9ch.py` | Utonia finetune_9ch on Ke-MLS (12-class unified taxonomy) |
| `Utonia/Kerala/configs/kerala/utonia_140gb_12class_insseg_dp_4ch.py` | Utonia insseg_dp_4ch on Ke-MLS (12-class unified taxonomy) |
| `Utonia/Kerala/configs/kerala/utonia_140gb_12class_insseg_dp_9ch.py` | Utonia insseg_dp_9ch on Ke-MLS (12-class unified taxonomy) |
| `Utonia/Kerala/configs/kerala/utonia_140gb_12class_insseg_ft_4ch.py` | Utonia insseg_ft_4ch on Ke-MLS (12-class unified taxonomy) |
| `Utonia/Kerala/configs/kerala/utonia_140gb_12class_insseg_ft_9ch.py` | Utonia insseg_ft_9ch on Ke-MLS (12-class unified taxonomy) |
| `Utonia/Kerala/configs/kerala/utonia_140gb_12class_insseg_lp_4ch.py` | Utonia insseg_lp_4ch on Ke-MLS (12-class unified taxonomy) |
| `Utonia/Kerala/configs/kerala/utonia_140gb_12class_insseg_lp_9ch.py` | Utonia insseg_lp_9ch on Ke-MLS (12-class unified taxonomy) |
| `Utonia/Kerala/configs/kerala/utonia_140gb_12class_linear_probe_4ch.py` | Utonia linear_probe_4ch on Ke-MLS (12-class unified taxonomy) |
| `Utonia/Kerala/configs/kerala/utonia_140gb_12class_linear_probe_9ch.py` | Utonia linear_probe_9ch on Ke-MLS (12-class unified taxonomy) |
| `Utonia/Kerala/configs/kerala/utonia_140gb_12class_lora_9ch.py` | Utonia lora_9ch on Ke-MLS (12-class unified taxonomy) |
| `Utonia/Toronto-3D/configs/toronto/utonia_140gb_12class_finetune_9ch.py` | Utonia FT-9ch on Toronto-3D (12-class unified taxonomy) |
| `Utonia/Paris-Lille-3D/configs/paris_lille/utonia_140gb_12class_finetune_9ch.py` | Utonia FT-9ch on Paris-Lille-3D (12-class unified taxonomy) |
| `Utonia/SemanticKITTI/configs/semantickitti/utonia_12class_finetune_9ch.py` | Utonia FT-9ch on SemanticKITTI (12-class unified taxonomy) |
| `Utonia/cross_data_testing/configs/utonia_crosscity_bhopal_on_kerala_9ch.py` | Utonia transfer evaluation bhopal_on_kerala |
| `Utonia/cross_data_testing/configs/utonia_crosscity_kerala_on_bhopal_9ch.py` | Utonia transfer evaluation kerala_on_bhopal |
| `Utonia/cross_data_testing/configs/utonia_crosscity_bhopal_on_tr_9ch.py` | Utonia transfer evaluation bhopal_on_tr |
| `Utonia/cross_data_testing/configs/utonia_crosscity_tr_on_bhopal_9ch.py` | Utonia transfer evaluation tr_on_bhopal |
| `Utonia/cross_data_testing/configs/utonia_crosscity_bhopal_on_pl_9ch.py` | Utonia transfer evaluation bhopal_on_pl |
| `Utonia/cross_data_testing/configs/utonia_crosscity_pl_on_bhopal_9ch.py` | Utonia transfer evaluation pl_on_bhopal |
| `Utonia/cross_data_testing/configs/utonia_crosscity_bhopal_on_sk_9ch.py` | Utonia transfer evaluation bhopal_on_sk |
| `Utonia/cross_data_testing/configs/utonia_crosscity_sk_on_bhopal_9ch.py` | Utonia transfer evaluation sk_on_bhopal |
| `Sonata/Bhopal/configs/bhopal/sonata_140gb_12class_linear_probe_4ch.py` | Sonata 12class linear_probe_4ch |
| `Sonata/Bhopal/configs/bhopal/sonata_140gb_12class_linear_probe_9ch.py` | Sonata 12class linear_probe_9ch |
| `Sonata/Bhopal/configs/bhopal/sonata_140gb_12class_decoder_probe_9ch.py` | Sonata 12class decoder_probe_9ch |
| `Sonata/Bhopal/configs/bhopal/sonata_140gb_12class_finetune_9ch.py` | Sonata 12class finetune_9ch |
| `Sonata/Bhopal/configs/bhopal/sonata_140gb_12class_lora_9ch.py` | Sonata 12class lora_9ch |
| `Sonata/Bhopal/configs/bhopal/sonata_140gb_12class_insseg_lp_9ch.py` | Sonata 12class insseg_lp_9ch |
| `Sonata/Bhopal/configs/bhopal/sonata_140gb_12class_insseg_dp_9ch.py` | Sonata 12class insseg_dp_9ch |
| `Sonata/Bhopal/configs/bhopal/sonata_140gb_12class_insseg_ft_9ch.py` | Sonata 12class insseg_ft_9ch |
| `Sonata/Bhopal/configs/bhopal/sonata_140gb_21class_linear_probe_4ch.py` | Sonata 21class linear_probe_4ch |
| `Sonata/Bhopal/configs/bhopal/sonata_140gb_21class_linear_probe_9ch.py` | Sonata 21class linear_probe_9ch |
| `Sonata/Bhopal/configs/bhopal/sonata_140gb_21class_decoder_probe_9ch.py` | Sonata 21class decoder_probe_9ch |
| `Sonata/Bhopal/configs/bhopal/sonata_140gb_21class_finetune_9ch.py` | Sonata 21class finetune_9ch |
| `Sonata/Bhopal/configs/bhopal/sonata_140gb_21class_lora_9ch.py` | Sonata 21class lora_9ch |
| `Sonata/Bhopal/configs/bhopal/sonata_140gb_21class_insseg_lp_9ch.py` | Sonata 21class insseg_lp_9ch |
| `Sonata/Bhopal/configs/bhopal/sonata_140gb_21class_insseg_dp_9ch.py` | Sonata 21class insseg_dp_9ch |
| `Sonata/Bhopal/configs/bhopal/sonata_140gb_21class_insseg_ft_9ch.py` | Sonata 21class insseg_ft_9ch |
| `Sonata/Kerala/configs/kerala/sonata_140gb_12class_decoder_probe_4ch.py` | Sonata decoder_probe_4ch on Ke-MLS (12-class unified taxonomy) |
| `Sonata/Kerala/configs/kerala/sonata_140gb_12class_decoder_probe_9ch.py` | Sonata decoder_probe_9ch on Ke-MLS (12-class unified taxonomy) |
| `Sonata/Kerala/configs/kerala/sonata_140gb_12class_finetune_4ch.py` | Sonata finetune_4ch on Ke-MLS (12-class unified taxonomy) |
| `Sonata/Kerala/configs/kerala/sonata_140gb_12class_finetune_9ch.py` | Sonata finetune_9ch on Ke-MLS (12-class unified taxonomy) |
| `Sonata/Kerala/configs/kerala/sonata_140gb_12class_insseg_dp_4ch.py` | Sonata insseg_dp_4ch on Ke-MLS (12-class unified taxonomy) |
| `Sonata/Kerala/configs/kerala/sonata_140gb_12class_insseg_dp_9ch.py` | Sonata insseg_dp_9ch on Ke-MLS (12-class unified taxonomy) |
| `Sonata/Kerala/configs/kerala/sonata_140gb_12class_insseg_ft_4ch.py` | Sonata insseg_ft_4ch on Ke-MLS (12-class unified taxonomy) |
| `Sonata/Kerala/configs/kerala/sonata_140gb_12class_insseg_ft_9ch.py` | Sonata insseg_ft_9ch on Ke-MLS (12-class unified taxonomy) |
| `Sonata/Kerala/configs/kerala/sonata_140gb_12class_insseg_lp_4ch.py` | Sonata insseg_lp_4ch on Ke-MLS (12-class unified taxonomy) |
| `Sonata/Kerala/configs/kerala/sonata_140gb_12class_insseg_lp_9ch.py` | Sonata insseg_lp_9ch on Ke-MLS (12-class unified taxonomy) |
| `Sonata/Kerala/configs/kerala/sonata_140gb_12class_linear_probe_4ch.py` | Sonata linear_probe_4ch on Ke-MLS (12-class unified taxonomy) |
| `Sonata/Kerala/configs/kerala/sonata_140gb_12class_linear_probe_9ch.py` | Sonata linear_probe_9ch on Ke-MLS (12-class unified taxonomy) |
| `Sonata/Kerala/configs/kerala/sonata_140gb_12class_lora_9ch.py` | Sonata lora_9ch on Ke-MLS (12-class unified taxonomy) |
| `Concerto/Bhopal/configs/bhopal/concerto_140gb_12class_linear_probe_4ch.py` | Concerto 12class linear_probe_4ch |
| `Concerto/Bhopal/configs/bhopal/concerto_140gb_12class_linear_probe_9ch.py` | Concerto 12class linear_probe_9ch |
| `Concerto/Bhopal/configs/bhopal/concerto_140gb_12class_decoder_probe_9ch.py` | Concerto 12class decoder_probe_9ch |
| `Concerto/Bhopal/configs/bhopal/concerto_140gb_12class_finetune_9ch.py` | Concerto 12class finetune_9ch |
| `Concerto/Bhopal/configs/bhopal/concerto_140gb_12class_lora_9ch.py` | Concerto 12class lora_9ch |
| `Concerto/Bhopal/configs/bhopal/concerto_140gb_12class_insseg_lp_9ch.py` | Concerto 12class insseg_lp_9ch |
| `Concerto/Bhopal/configs/bhopal/concerto_140gb_12class_insseg_dp_9ch.py` | Concerto 12class insseg_dp_9ch |
| `Concerto/Bhopal/configs/bhopal/concerto_140gb_12class_insseg_ft_9ch.py` | Concerto 12class insseg_ft_9ch |
| `Concerto/Bhopal/configs/bhopal/concerto_140gb_21class_linear_probe_4ch.py` | Concerto 21class linear_probe_4ch |
| `Concerto/Bhopal/configs/bhopal/concerto_140gb_21class_linear_probe_9ch.py` | Concerto 21class linear_probe_9ch |
| `Concerto/Bhopal/configs/bhopal/concerto_140gb_21class_decoder_probe_9ch.py` | Concerto 21class decoder_probe_9ch |
| `Concerto/Bhopal/configs/bhopal/concerto_140gb_21class_finetune_9ch.py` | Concerto 21class finetune_9ch |
| `Concerto/Bhopal/configs/bhopal/concerto_140gb_21class_lora_9ch.py` | Concerto 21class lora_9ch |
| `Concerto/Bhopal/configs/bhopal/concerto_140gb_21class_insseg_lp_9ch.py` | Concerto 21class insseg_lp_9ch |
| `Concerto/Bhopal/configs/bhopal/concerto_140gb_21class_insseg_dp_9ch.py` | Concerto 21class insseg_dp_9ch |
| `Concerto/Bhopal/configs/bhopal/concerto_140gb_21class_insseg_ft_9ch.py` | Concerto 21class insseg_ft_9ch |
| `Concerto/Bhopal/configs/bhopal_v10/concerto_140gb_12class_decoder_probe_9ch.py` | Concerto trained with the 6 moving classes (dagger rows) |
| `Concerto/Bhopal/configs/bhopal_v10/concerto_140gb_12class_finetune_9ch.py` | Concerto trained with the 6 moving classes (dagger rows) |
| `Concerto/Bhopal/configs/bhopal_v10/concerto_140gb_12class_linear_probe_9ch.py` | Concerto trained with the 6 moving classes (dagger rows) |
| `Concerto/Bhopal/configs/bhopal_v10/concerto_140gb_12class_lora_9ch.py` | Concerto trained with the 6 moving classes (dagger rows) |
| `Concerto/Bhopal/configs/bhopal_v10/concerto_140gb_21class_decoder_probe_9ch.py` | Concerto trained with the 6 moving classes (dagger rows) |
| `Concerto/Bhopal/configs/bhopal_v10/concerto_140gb_21class_finetune_9ch.py` | Concerto trained with the 6 moving classes (dagger rows) |
| `Concerto/Bhopal/configs/bhopal_v10/concerto_140gb_21class_linear_probe_9ch.py` | Concerto trained with the 6 moving classes (dagger rows) |
| `Concerto/Bhopal/configs/bhopal_v10/concerto_140gb_21class_lora_9ch.py` | Concerto trained with the 6 moving classes (dagger rows) |
| `Concerto/Bhopal/configs/bhopal/concerto_140gb_12class_finetune_9ch_ablB1_size.py` | Concerto factor-attribution arm |
| `Concerto/Bhopal/configs/bhopal/concerto_140gb_12class_finetune_9ch_ablB2_variety.py` | Concerto factor-attribution arm |
| `Concerto/Bhopal/configs/bhopal/concerto_140gb_12class_finetune_9ch_ablB3_regular.py` | Concerto factor-attribution arm |
| `Concerto/Bhopal/configs/bhopal/concerto_140gb_12class_finetune_9ch_ablB4_subtype.py` | Concerto factor-attribution arm |
| `Concerto/Bhopal/configs/bhopal_ablations/concerto_ablC1_noweights_finetune_9ch.py` | Concerto class-imbalance cell |
| `Concerto/Bhopal/configs/bhopal_ablations/concerto_ablC2_nolovasz_finetune_9ch.py` | Concerto class-imbalance cell |
| `Concerto/Bhopal/configs/bhopal_ablations/concerto_ablC3_lossnaive_finetune_9ch.py` | Concerto class-imbalance cell |
| `Concerto/Bhopal/configs/bhopal_ablations/concerto_ablC4_buildfull_finetune_9ch.py` | Concerto class-imbalance cell |
| `Concerto/Kerala/configs/kerala/concerto_140gb_12class_decoder_probe_4ch.py` | Concerto decoder_probe_4ch on Ke-MLS (12-class unified taxonomy) |
| `Concerto/Kerala/configs/kerala/concerto_140gb_12class_decoder_probe_9ch.py` | Concerto decoder_probe_9ch on Ke-MLS (12-class unified taxonomy) |
| `Concerto/Kerala/configs/kerala/concerto_140gb_12class_finetune_4ch.py` | Concerto finetune_4ch on Ke-MLS (12-class unified taxonomy) |
| `Concerto/Kerala/configs/kerala/concerto_140gb_12class_finetune_9ch.py` | Concerto finetune_9ch on Ke-MLS (12-class unified taxonomy) |
| `Concerto/Kerala/configs/kerala/concerto_140gb_12class_insseg_dp_4ch.py` | Concerto insseg_dp_4ch on Ke-MLS (12-class unified taxonomy) |
| `Concerto/Kerala/configs/kerala/concerto_140gb_12class_insseg_dp_9ch.py` | Concerto insseg_dp_9ch on Ke-MLS (12-class unified taxonomy) |
| `Concerto/Kerala/configs/kerala/concerto_140gb_12class_insseg_ft_4ch.py` | Concerto insseg_ft_4ch on Ke-MLS (12-class unified taxonomy) |
| `Concerto/Kerala/configs/kerala/concerto_140gb_12class_insseg_ft_9ch.py` | Concerto insseg_ft_9ch on Ke-MLS (12-class unified taxonomy) |
| `Concerto/Kerala/configs/kerala/concerto_140gb_12class_insseg_lp_4ch.py` | Concerto insseg_lp_4ch on Ke-MLS (12-class unified taxonomy) |
| `Concerto/Kerala/configs/kerala/concerto_140gb_12class_insseg_lp_9ch.py` | Concerto insseg_lp_9ch on Ke-MLS (12-class unified taxonomy) |
| `Concerto/Kerala/configs/kerala/concerto_140gb_12class_linear_probe_4ch.py` | Concerto linear_probe_4ch on Ke-MLS (12-class unified taxonomy) |
| `Concerto/Kerala/configs/kerala/concerto_140gb_12class_linear_probe_9ch.py` | Concerto linear_probe_9ch on Ke-MLS (12-class unified taxonomy) |
| `Concerto/Kerala/configs/kerala/concerto_140gb_12class_lora_9ch.py` | Concerto lora_9ch on Ke-MLS (12-class unified taxonomy) |
| `Concerto/Toronto-3D/configs/toronto/concerto_140gb_12class_finetune_9ch.py` | Concerto FT-9ch on Toronto-3D (12-class unified taxonomy) |
| `Concerto/Paris-Lille-3D/configs/paris_lille/concerto_140gb_12class_finetune_9ch.py` | Concerto FT-9ch on Paris-Lille-3D (12-class unified taxonomy) |
| `Concerto/SemanticKITTI/configs/semantickitti/concerto_12class_finetune_9ch.py` | Concerto FT-9ch on SemanticKITTI (12-class unified taxonomy) |
| `Concerto/cross_data_testing/configs/concerto_crosscity_bhopal_on_kerala_9ch.py` | Concerto transfer evaluation bhopal_on_kerala |
| `Concerto/cross_data_testing/configs/concerto_crosscity_kerala_on_bhopal_9ch.py` | Concerto transfer evaluation kerala_on_bhopal |
| `Concerto/cross_data_testing/configs/concerto_crosscity_bhopal_on_tr_9ch.py` | Concerto transfer evaluation bhopal_on_tr |
| `Concerto/cross_data_testing/configs/concerto_crosscity_tr_on_bhopal_9ch.py` | Concerto transfer evaluation tr_on_bhopal |
| `Concerto/cross_data_testing/configs/concerto_crosscity_bhopal_on_pl_9ch.py` | Concerto transfer evaluation bhopal_on_pl |
| `Concerto/cross_data_testing/configs/concerto_crosscity_pl_on_bhopal_9ch.py` | Concerto transfer evaluation pl_on_bhopal |
| `Concerto/cross_data_testing/configs/concerto_crosscity_bhopal_on_sk_9ch.py` | Concerto transfer evaluation bhopal_on_sk |
| `Concerto/cross_data_testing/configs/concerto_crosscity_sk_on_bhopal_9ch.py` | Concerto transfer evaluation sk_on_bhopal |
| `PTv3/Bhopal/configs/bhopal/ptv3_140gb_12class_train_4ch.py` | PTv3 ptv3_140gb_12class_train_4ch.py |
| `PTv3/Bhopal/configs/bhopal/ptv3_140gb_12class_train_9ch.py` | PTv3 ptv3_140gb_12class_train_9ch.py |
| `PTv3/Bhopal/configs/bhopal/ptv3_140gb_12class_insseg_9ch.py` | PTv3 ptv3_140gb_12class_insseg_9ch.py |
| `PTv3/Bhopal/configs/bhopal/ptv3_140gb_21class_train_4ch.py` | PTv3 ptv3_140gb_21class_train_4ch.py |
| `PTv3/Bhopal/configs/bhopal/ptv3_140gb_21class_train_9ch.py` | PTv3 ptv3_140gb_21class_train_9ch.py |
| `PTv3/Bhopal/configs/bhopal/ptv3_140gb_21class_insseg_9ch.py` | PTv3 ptv3_140gb_21class_insseg_9ch.py |
| `KpConvX/Bhopal/configs/bhopal/kpconvx_140gb_12class_train_4ch.py` | KpConvX kpconvx_140gb_12class_train_4ch.py |
| `KpConvX/Bhopal/configs/bhopal/kpconvx_140gb_12class_train_9ch.py` | KpConvX kpconvx_140gb_12class_train_9ch.py |
| `KpConvX/Bhopal/configs/bhopal/kpconvx_140gb_21class_train_4ch.py` | KpConvX kpconvx_140gb_21class_train_4ch.py |
| `KpConvX/Bhopal/configs/bhopal/kpconvx_140gb_21class_train_9ch.py` | KpConvX kpconvx_140gb_21class_train_9ch.py |
| `LitePT/Bhopal/configs/bhopal/litept_L_12class_4ch.py` | LitePT litept_L_12class_4ch.py |
| `LitePT/Bhopal/configs/bhopal/litept_L_12class_9ch.py` | LitePT litept_L_12class_9ch.py |
| `LitePT/Bhopal/configs/bhopal/litept_L_12class_insseg_9ch.py` | LitePT litept_L_12class_insseg_9ch.py |
| `LitePT/Bhopal/configs/bhopal/litept_L_21class_4ch.py` | LitePT litept_L_21class_4ch.py |
| `LitePT/Bhopal/configs/bhopal/litept_L_21class_9ch.py` | LitePT litept_L_21class_9ch.py |
| `LitePT/Bhopal/configs/bhopal/litept_L_21class_insseg_9ch.py` | LitePT litept_L_21class_insseg_9ch.py |
