# exp_03b — attribution with controls

Built by `analyze_ablB_attribution.py` from `ablation_attribution.json`
(the H200 host cross-evals) and `perclass_transfer.json` (exp_03a reference).
All three arms carry the same ~135M labelled-point budget, so they differ
only in *what kind* of data is kept.

## 1. BM->X transfer, full class intersection

| pair | n_c | full BM->X | X->BM | gap | B1 size | B2 variety | B3 geometry |
|---|---|---|---|---|---|---|---|
| BM$\to$KM | 12 | 56.78 | 33.96 | 22.82 | 52.56 (-4.23) | 42.22 (-14.57) | 49.30 (-7.49) | 44.12 (-12.67) |
| BM$\to$TR | 7 | 77.27 | 51.65 | 25.62 | 76.17 (-1.10) | 61.63 (-15.64) | 78.70 (+1.43) | 71.66 (-5.61) |
| BM$\to$PL | 11 | 55.68 | 27.06 | 28.62 | 48.87 (-6.81) | 34.90 (-20.78) | 52.56 (-3.12) | 45.37 (-10.31) |
| BM$\to$SK | 9 | 40.05 | 35.54 | 4.50 | 47.94 (+7.89) | 26.28 (-13.77) | 46.01 (+5.96) | — |

Parentheses: change vs the full model. `gap` is what needs explaining.

## 2. Read against B1, the size control (same budget)

| pair | B2 − B1 | B3 − B1 |
|---|---|---|
| BM$\to$KM | -10.34 | -3.26 |
| BM$\to$TR | -14.54 | +2.53 |
| BM$\to$PL | -13.97 | +3.69 |
| BM$\to$SK | -21.66 | -1.93 |
| **mean** | **-15.13** | **+0.26** |

## 3. Transfer penalty net of in-domain penalty

An arm that is simply a weaker model loses transfer for free. Column 2 is
the arm's own in-domain loss on the BhopalMLS test split; column 4 is the
part of the transfer loss that in-domain capability does *not* explain.
Reported over all four pairs and over KM/TR/PL alone, because SK's own
asymmetry gap is 4.50 pp and both control arms *beat* the full model there
(see the caveat in NOTES.md).

**all four**

| arm | in-domain (Δ vs full) | mean transfer Δ | net transfer penalty | ratio |
|---|---|---|---|---|
| B1 size | 67.99 (-3.29) | -1.06 | +2.23 | 0.32× |
| B2 variety | 61.18 (-10.10) | -16.19 | -6.09 | 1.60× |
| B3 geometry | 68.61 (-2.67) | -0.80 | +1.87 | 0.30× |
| B4 subtype | 56.34 (-14.94) | -9.53 | +5.41 | 0.64× |

**KM/TR/PL**

| arm | in-domain (Δ vs full) | mean transfer Δ | net transfer penalty | ratio |
|---|---|---|---|---|
| B1 size | 67.99 (-3.29) | -4.05 | -0.76 | 1.23× |
| B2 variety | 61.18 (-10.10) | -17.00 | -6.90 | 1.68× |
| B3 geometry | 68.61 (-2.67) | -3.06 | -0.39 | 1.15× |
| B4 subtype | 56.34 (-14.94) | -9.53 | +5.41 | 0.64× |

In-domain values are precise (TTA) 12-class test mIoU; the full model is
71.28 (exp_02 `summary_table.md`).

## 4. Same table with *Vehicle* removed from every mask

B2's loss concentrates on Vehicle, whose dominant BhopalMLS subtype is
*Mobike (Two-Wheeler)* (fine class 22) — not the car that dominates every
target dataset. If the variety effect were only that one class, the
contrast would vanish here.

| pair | full | B1 size | B2 variety | B3 geometry | B2 − B1 |
|---|---|---|---|---|---|
| BM$\to$KM (11c) | 54.29 | 49.66 (-4.63) | 44.83 (-9.46) | 45.76 (-8.52) | 41.03 (-13.25) | -4.83 |
| BM$\to$TR (6c) | 74.45 | 74.31 (-0.15) | 71.17 (-3.28) | 76.59 (+2.13) | 71.90 (-2.56) | -3.14 |
| BM$\to$PL (10c) | 51.47 | 44.11 (-7.36) | 37.42 (-14.05) | 47.96 (-3.51) | 40.40 (-11.07) | -6.69 |
| BM$\to$SK (8c) | 36.10 | 43.99 (+7.89) | 29.14 (-6.96) | 41.56 (+5.46) | — | -14.85 |
| **mean B2 − B1** | | | | | **-7.38** |

## 5. B2 − B1 per class (negative = variety collapse costs transfer)

| class | KM | TR | PL | SK |
|---|---|---|---|---|
| Ground | -12.3 | — | -9.4 | -33.5 |
| Vegetation | +6.0 | +2.9 | +0.6 | -0.6 |
| Building | +8.6 | -14.2 | -6.9 | -37.5 |
| Road | -11.3 | +2.6 | -3.0 | -28.0 |
| Wire | -2.1 | +4.4 | -8.9 | — |
| Pole and Tower | -10.9 | -0.1 | -7.1 | -18.4 |
| Wire-Structure Connector | -27.4 | — | — | — |
| Vehicle | -71.0 | -83.0 | -86.7 | -76.1 |
| Board | +2.2 | — | -6.6 | -7.7 |
| Fencing | +14.7 | -14.5 | -8.2 | -6.6 |
| Human and Animal | -18.8 | — | -1.7 | +13.4 |
| Road Side Object | -1.8 | — | -15.8 | — |

