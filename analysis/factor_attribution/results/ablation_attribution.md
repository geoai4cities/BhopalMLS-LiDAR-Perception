# exp_03b — factor attribution from cross-region evals (utonia)

mIoU on each pair's intersection class set (same masks as exp_03a / paper Table 5).
`closed %` = share of the full model's BM->X advantage that the ablation removes.

## BM -> KM  (n_c=12)

- paper full model  BM->KM = **56.78**, KM->BM = **33.96**, asymmetry gap = **22.82** pp

| arm | BM->X mIoU | removed (pp) | closed % | source log |
|---|---|---|---|---|
| FULL | 56.71 | +0.08 | 0.3% | test_20260817_091349.log |
| B1_size | 52.56 | +4.23 | 18.5% | test_20260812_025325.log |
| B2_variety | 42.22 | +14.57 | 63.8% | test_20260812_065518.log |
| B3_regular | 49.30 | +7.49 | 32.8% | test_20260812_104527.log |
| B4_subtype | 44.12 | +12.67 | 55.5% | test_20260818_183252.log |

## BM -> TR  (n_c=7)

- paper full model  BM->TR = **77.27**, TR->BM = **51.65**, asymmetry gap = **25.62** pp

| arm | BM->X mIoU | removed (pp) | closed % | source log |
|---|---|---|---|---|
| FULL | 77.25 | +0.02 | 0.1% | test_20260817_200319.log |
| B1_size | 76.17 | +1.10 | 4.3% | test_20260812_104547.log |
| B2_variety | 61.63 | +15.64 | 61.1% | test_20260812_104608.log |
| B3_regular | 78.70 | -1.43 | -5.6% | test_20260812_170648.log |
| B4_subtype | 71.66 | +5.61 | 21.9% | test_20260818_223059.log |

## BM -> PL  (n_c=11)

- paper full model  BM->PL = **55.68**, PL->BM = **27.06**, asymmetry gap = **28.62** pp

| arm | BM->X mIoU | removed (pp) | closed % | source log |
|---|---|---|---|---|
| FULL | 55.63 | +0.05 | 0.2% | test_20260817_223623.log |
| B1_size | 48.87 | +6.81 | 23.8% | test_20260812_174626.log |
| B2_variety | 34.90 | +20.78 | 72.6% | test_20260812_183508.log |
| B3_regular | 52.56 | +3.12 | 10.9% | test_20260812_193639.log |
| B4_subtype | 45.37 | +10.31 | 36.0% | test_20260818_231013.log |

## BM -> SK  (n_c=9)

- paper full model  BM->SK = **40.05**, SK->BM = **35.54**, asymmetry gap = **4.50** pp

| arm | BM->X mIoU | removed (pp) | closed % | source log |
|---|---|---|---|---|
| B1_size | 47.94 | -7.89 | -175.1% | test_20260812_202820.log |
| B2_variety | 26.28 | +13.77 | 305.7% | test_20260815_005723.log |
| B3_regular | 46.01 | -5.96 | -132.3% | test_20260815_220547.log |
| B4_subtype | 37.88 | +2.17 | 48.2% | test_20260827_054415.log |

## Per-class BM->X IoU by arm

### KM

| class | FULL | B1_size | B2_variety | B3_regular | B4_subtype |
|---|---|---|---|---|---|
| Ground | 33.47 | 25.49 | 13.19 | 21.72 | 9.81 |
| Vegetation | 75.17 | 72.34 | 78.33 | 71.64 | 75.15 |
| Building | 76.67 | 67.95 | 76.51 | 64.86 | 70.54 |
| Road | 82.08 | 77.88 | 66.62 | 75.65 | 66.53 |
| Wire | 79.06 | 72.51 | 70.41 | 72.87 | 46.69 |
| Pole and Tower | 60.99 | 59.12 | 48.21 | 40.87 | 26.66 |
| Wire-Structure Connector | 40.39 | 44.69 | 17.32 | 12.58 | 17.16 |
| Vehicle | 84.30 | 84.45 | 13.49 | 88.14 | 78.01 |
| Board | 13.46 | 9.86 | 12.01 | 15.95 | 12.07 |
| Fencing | 48.25 | 36.69 | 51.43 | 48.40 | 63.91 |
| Human and Animal | 80.05 | 76.97 | 58.18 | 76.92 | 60.85 |
| Road Side Object | 6.58 | 2.72 | 0.89 | 1.95 | 2.01 |

### TR

| class | FULL | B1_size | B2_variety | B3_regular | B4_subtype |
|---|---|---|---|---|---|
| Vegetation | 94.42 | 92.05 | 94.99 | 94.04 | 94.13 |
| Building | 91.99 | 94.71 | 80.53 | 93.42 | 89.53 |
| Road | 92.28 | 95.63 | 98.26 | 93.21 | 98.49 |
| Wire | 82.66 | 70.52 | 74.89 | 77.88 | 68.62 |
| Pole and Tower | 74.06 | 69.05 | 68.96 | 69.45 | 66.12 |
| Vehicle | 94.15 | 87.36 | 4.37 | 91.39 | 70.25 |
| Fencing | 11.21 | 23.89 | 9.40 | 31.53 | 14.50 |

### PL

| class | FULL | B1_size | B2_variety | B3_regular | B4_subtype |
|---|---|---|---|---|---|
| Ground | 28.00 | 20.09 | 10.73 | 6.70 | 7.20 |
| Vegetation | 65.48 | 64.57 | 65.21 | 68.74 | 61.84 |
| Building | 98.06 | 98.27 | 91.32 | 98.40 | 97.64 |
| Road | 78.24 | 76.59 | 73.62 | 73.55 | 73.56 |
| Wire | 47.73 | 38.99 | 30.14 | 51.90 | 15.74 |
| Pole and Tower | 40.96 | 22.38 | 15.24 | 40.75 | 19.93 |
| Vehicle | 97.78 | 96.45 | 9.71 | 98.48 | 95.05 |
| Board | 24.44 | 26.75 | 20.13 | 27.81 | 25.01 |
| Fencing | 30.71 | 29.86 | 21.69 | 31.00 | 31.23 |
| Human and Animal | 53.28 | 23.02 | 21.28 | 33.58 | 42.09 |
| Road Side Object | 47.26 | 40.60 | 24.85 | 47.22 | 29.77 |

### SK

| class | B1_size | B2_variety | B3_regular | B4_subtype |
|---|---|---|---|---|
| Ground | 70.51 | 36.96 | 63.71 | 36.50 |
| Vegetation | 36.49 | 35.90 | 52.11 | 41.61 |
| Building | 65.49 | 27.99 | 59.07 | 56.22 |
| Road | 78.38 | 50.37 | 67.22 | 48.16 |
| Pole and Tower | 27.43 | 9.06 | 14.43 | 4.80 |
| Vehicle | 79.53 | 3.40 | 81.57 | 66.97 |
| Board | 19.44 | 11.78 | 19.77 | 7.43 |
| Fencing | 19.73 | 13.18 | 17.77 | 21.38 |
| Human and Animal | 34.42 | 47.85 | 38.43 | 57.83 |

