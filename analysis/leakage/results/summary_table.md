# exp_02 summary — leakage-subset re-scoring, full zoo

## 12-class (11 models)

| model | full | disjoint | overlapping | Δ (disj−ovl) |
|---|---|---|---|---|
| utonia_ft | 71.28 | 71.29 | 69.64 | +1.65 |
| utonia_lora | 69.77 | 69.85 | 68.21 | +1.64 |
| concerto_ft | 68.10 | 66.59 | 69.21 | -2.62 |
| ptv3_sup | 67.88 | 66.81 | 68.52 | -1.71 |
| litept_L_sup | 67.53 | 66.96 | 67.51 | -0.55 |
| utonia_dp | 67.03 | 66.55 | 67.13 | -0.58 |
| kpconvx_sup | 66.78 | 66.67 | 65.62 | +1.05 |
| utonia_lp | 64.18 | 64.84 | 62.18 | +2.66 |
| utonia_lp_4ch | 63.16 | 62.12 | 62.76 | -0.64 |
| concerto_lp_indoor | 53.43 | 52.23 | 53.13 | -0.90 |
| concerto_lp | 52.41 | 53.07 | 50.63 | +2.44 |

- mean Δ (disjoint − overlapping): **+0.22 pp** (range -2.62 … +2.66)
- Spearman rank corr, full vs disjoint: **0.909**; full vs overlapping: 0.964
- mean |full − disjoint|: 0.67 pp

## 21-class (26 models)

| model | full | disjoint | overlapping | Δ (disj−ovl) |
|---|---|---|---|---|
| utonia_ft | 61.73 | 60.96 | 62.76 | -1.80 |
| concerto_ft_indoor | 60.83 | 60.37 | 61.48 | -1.11 |
| sonata_ft | 59.68 | 58.63 | 61.07 | -2.44 |
| utonia_lora | 58.97 | 58.16 | 59.30 | -1.15 |
| concerto_ft | 58.97 | 58.10 | 60.19 | -2.10 |
| ptv3_9ch | 58.51 | 57.28 | 60.27 | -2.99 |
| ptv3_4ch | 58.35 | 56.87 | 60.03 | -3.16 |
| utonia_dp | 57.77 | 56.84 | 58.99 | -2.15 |
| concerto_dp_indoor | 56.40 | 56.29 | 55.09 | +1.20 |
| concerto_lora_indoor | 56.01 | 55.87 | 54.82 | +1.05 |
| sonata_dp | 55.90 | 56.23 | 54.75 | +1.48 |
| litept_L_9ch | 55.87 | 55.09 | 56.95 | -1.86 |
| concerto_lora | 55.16 | 54.82 | 55.54 | -0.71 |
| litept_L_4ch | 55.09 | 54.08 | 55.99 | -1.91 |
| kpconvx_4ch | 54.47 | 54.14 | 53.55 | +0.59 |
| kpconvx_9ch | 53.96 | 53.63 | 53.94 | -0.31 |
| utonia_lp | 53.72 | 53.95 | 53.56 | +0.39 |
| utonia_lp_4ch | 53.09 | 52.57 | 52.69 | -0.13 |
| concerto_dp | 52.30 | 52.05 | 52.56 | -0.51 |
| sonata_lora | 52.13 | 52.55 | 50.47 | +2.08 |
| concerto_lp_4ch_indoor | 50.23 | 49.81 | 49.75 | +0.06 |
| concerto_lp_indoor | 50.23 | 50.44 | 49.41 | +1.02 |
| sonata_lp | 46.39 | 47.64 | 43.51 | +4.12 |
| concerto_lp | 44.61 | 45.32 | 42.84 | +2.48 |
| concerto_lp_4ch | 43.20 | 44.02 | 41.22 | +2.80 |
| sonata_lp_4ch | 34.30 | 35.26 | 32.23 | +3.03 |

- mean Δ (disjoint − overlapping): **-0.08 pp** (range -3.16 … +4.12)
- Spearman rank corr, full vs disjoint: **0.997**; full vs overlapping: 0.974
- mean |full − disjoint|: 0.65 pp
