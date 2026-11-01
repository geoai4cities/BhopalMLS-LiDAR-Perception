# exp_03b — factor attribution from cross-region evals (concerto)

mIoU on each pair's intersection class set (same masks as exp_03a / paper Table 5).
`closed %` = share of the full model's BM->X advantage that the ablation removes.

## BM -> KM  (n_c=12)

- paper full model  BM->KM = **49.23**, KM->BM = **32.70**, asymmetry gap = **16.52** pp

| arm | BM->X mIoU | removed (pp) | closed % | source log |
|---|---|---|---|---|
| B1_size | 50.50 | -1.28 | -7.7% | test_20260819_061815.log |
| B2_variety | 39.74 | +9.48 | 57.4% | test_20260819_173616.log |
| B3_regular | 50.44 | -1.22 | -7.4% | test_20260820_050512.log |
| B4_subtype | 40.32 | +8.91 | 53.9% | test_20260820_210339.log |

## BM -> TR  (n_c=7)

- paper full model  BM->TR = **75.20**, TR->BM = **39.11**, asymmetry gap = **36.09** pp

| arm | BM->X mIoU | removed (pp) | closed % | source log |
|---|---|---|---|---|
| B1_size | 72.34 | +2.86 | 7.9% | test_20260819_094550.log |
| B2_variety | 60.10 | +15.10 | 41.8% | test_20260819_210332.log |
| B3_regular | 72.63 | +2.57 | 7.1% | test_20260820_083410.log |
| B4_subtype | 67.83 | +7.37 | 20.4% | test_20260821_004324.log |

## BM -> PL  (n_c=11)

- paper full model  BM->PL = **48.84**, PL->BM = **31.07**, asymmetry gap = **17.76** pp

| arm | BM->X mIoU | removed (pp) | closed % | source log |
|---|---|---|---|---|
| B1_size | 49.75 | -0.92 | -5.2% | test_20260819_101529.log |
| B2_variety | 32.71 | +16.13 | 90.8% | test_20260819_213314.log |
| B3_regular | 49.48 | -0.65 | -3.6% | test_20260820_090353.log |
| B4_subtype | 42.13 | +6.71 | 37.8% | test_20260821_012955.log |

## BM -> SK  (n_c=9)

- paper full model  BM->SK = **32.07**, SK->BM = **12.29**, asymmetry gap = **19.78** pp

| arm | BM->X mIoU | removed (pp) | closed % | source log |
|---|---|---|---|---|
| B1_size | 30.45 | +1.63 | 8.2% | test_20260821_021116.log |
| B2_variety | 16.82 | +15.25 | 77.1% | test_20260822_045554.log |
| B3_regular | 33.07 | -1.00 | -5.1% | test_20260825_093219.log |
| B4_subtype | 25.28 | +6.80 | 34.4% | test_20260825_180733.log |

## Per-class BM->X IoU by arm

### KM

| class | B1_size | B2_variety | B3_regular | B4_subtype |
|---|---|---|---|---|
| Ground | 12.79 | 7.63 | 12.52 | 10.41 |
| Vegetation | 76.18 | 79.16 | 75.82 | 72.04 |
| Building | 68.28 | 75.85 | 63.68 | 66.06 |
| Road | 70.83 | 66.58 | 71.44 | 67.97 |
| Wire | 73.70 | 63.10 | 77.38 | 61.01 |
| Pole and Tower | 60.90 | 36.91 | 54.16 | 35.47 |
| Wire-Structure Connector | 38.95 | 7.70 | 45.24 | 14.43 |
| Vehicle | 82.81 | 17.28 | 76.55 | 57.85 |
| Board | 6.07 | 7.77 | 9.28 | 8.96 |
| Fencing | 36.98 | 57.74 | 40.50 | 42.61 |
| Human and Animal | 75.19 | 56.74 | 73.55 | 46.04 |
| Road Side Object | 3.37 | 0.47 | 5.19 | 0.94 |

### TR

| class | B1_size | B2_variety | B3_regular | B4_subtype |
|---|---|---|---|---|
| Vegetation | 91.78 | 93.33 | 93.33 | 90.22 |
| Building | 92.20 | 87.62 | 88.31 | 82.58 |
| Road | 90.38 | 97.62 | 87.70 | 92.49 |
| Wire | 77.53 | 70.10 | 80.15 | 72.75 |
| Pole and Tower | 57.81 | 59.31 | 55.23 | 61.27 |
| Vehicle | 80.04 | 7.44 | 76.60 | 69.38 |
| Fencing | 16.67 | 5.30 | 27.11 | 6.14 |

### PL

| class | B1_size | B2_variety | B3_regular | B4_subtype |
|---|---|---|---|---|
| Ground | 13.44 | 10.61 | 11.40 | 13.29 |
| Vegetation | 64.94 | 60.47 | 67.92 | 58.47 |
| Building | 97.26 | 95.91 | 97.03 | 94.26 |
| Road | 74.87 | 74.11 | 74.81 | 74.36 |
| Wire | 45.73 | 23.70 | 48.15 | 30.13 |
| Pole and Tower | 21.78 | 21.58 | 19.29 | 14.27 |
| Vehicle | 96.71 | 11.00 | 94.58 | 93.45 |
| Board | 17.39 | 24.99 | 27.08 | 22.48 |
| Fencing | 48.62 | 20.63 | 31.46 | 29.17 |
| Human and Animal | 21.99 | 6.69 | 19.31 | 5.19 |
| Road Side Object | 44.57 | 10.12 | 53.29 | 28.36 |

### SK

| class | B1_size | B2_variety | B3_regular | B4_subtype |
|---|---|---|---|---|
| Ground | 49.67 | 24.61 | 43.34 | 25.04 |
| Vegetation | 32.39 | 23.98 | 34.45 | 42.59 |
| Building | 57.16 | 32.71 | 54.10 | 48.88 |
| Road | 22.86 | 30.55 | 33.24 | 29.93 |
| Pole and Tower | 18.18 | 8.66 | 29.33 | 9.98 |
| Vehicle | 63.10 | 8.03 | 67.84 | 36.42 |
| Board | 0.29 | 2.04 | 8.72 | 2.43 |
| Fencing | 15.13 | 13.13 | 11.03 | 22.89 |
| Human and Animal | 15.24 | 7.68 | 15.60 | 9.33 |

