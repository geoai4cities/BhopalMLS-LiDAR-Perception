# exp_03a — per-class cross-region transfer decomposition

Intersection masks (inferred: classes structurally zero in every run of a pair are excluded):

- **CH** (n_c=12): Ground, Vegetation, Building, Road, Wire, Pole and Tower, Wire-Structure Connector, Vehicle, Board, Fencing, Human and Animal, Road Side Object
- **KM** (n_c=12): Ground, Vegetation, Building, Road, Wire, Pole and Tower, Wire-Structure Connector, Vehicle, Board, Fencing, Human and Animal, Road Side Object
- **PL** (n_c=11): Ground, Vegetation, Building, Road, Wire, Pole and Tower, Vehicle, Board, Fencing, Human and Animal, Road Side Object
- **SK** (n_c=9): Ground, Vegetation, Building, Road, Pole and Tower, Vehicle, Board, Fencing, Human and Animal
- **TR** (n_c=7): Vegetation, Building, Road, Wire, Pole and Tower, Vehicle, Fencing

## Validation vs paper Table (Utonia ft_9ch, Concerto ft_9ch_indoor)

| model | pair | paper mIoU | recomputed | match |
|---|---|---|---|---|
| Utonia | BM->KM | 56.78 | 56.78 | OK |
| Utonia | KM->BM | 33.96 | 33.96 | OK |
| Utonia | BM->TR | 77.27 | 77.27 | OK |
| Utonia | TR->BM | 51.65 | 51.65 | OK |
| Utonia | BM->PL | 55.68 | 55.68 | OK |
| Utonia | PL->BM | 27.06 | 27.06 | OK |
| Utonia | BM->SK | 40.05 | 40.05 | OK |
| Utonia | SK->BM | 35.54 | 35.54 | OK |
| Concerto | BM->KM | 49.23 | 49.23 | OK |
| Concerto | KM->BM | 32.70 | 32.70 | OK |
| Concerto | BM->TR | 75.20 | 75.20 | OK |
| Concerto | TR->BM | 39.11 | 39.11 | OK |
| Concerto | BM->PL | 48.84 | 48.84 | OK |
| Concerto | PL->BM | 31.07 | 31.07 | OK |
| Concerto | BM->SK | 32.07 | 32.07 | OK |
| Concerto | SK->BM | 12.29 | 12.29 | OK |

## Per-class asymmetry Δ_c = IoU(BM→X) − IoU(X→BM)


### Utonia ft_9ch

| class | CH: out / in / Δ | KM: out / in / Δ | PL: out / in / Δ | SK: out / in / Δ | TR: out / in / Δ |
|---|---|---|---|---|---|
| Ground | 39.1 / 12.3 / +26.8 | 33.5 / 12.8 / +20.7 | 28.1 / 9.1 / +19.0 | 60.9 / 8.7 / +52.2 | · |
| Vegetation | 95.6 / 88.8 / +6.8 | 75.2 / 58.7 / +16.5 | 65.5 / 65.2 / +0.3 | 37.5 / 51.4 / -13.9 | 94.4 / 80.9 / +13.5 |
| Building | 91.8 / 89.5 / +2.3 | 76.6 / 87.2 / -10.6 | 98.0 / 81.0 / +17.1 | 48.7 / 81.5 / -32.8 | 92.0 / 82.9 / +9.0 |
| Road | 65.5 / 47.6 / +18.0 | 82.1 / 40.6 / +41.5 | 78.3 / 23.2 / +55.0 | 50.4 / 59.3 / -8.9 | 92.2 / 78.1 / +14.1 |
| Wire | 78.6 / 74.9 / +3.8 | 79.1 / 55.2 / +23.9 | 48.5 / 5.7 / +42.9 | · | 82.7 / 66.7 / +16.0 |
| Pole and Tower | 69.2 / 44.6 / +24.7 | 61.1 / 18.8 / +42.3 | 41.0 / 10.7 / +30.3 | 21.1 / 1.8 / +19.3 | 74.2 / 22.5 / +51.7 |
| Wire-Structure Connector | 0.0 / 24.9 / -24.9 | 41.1 / 13.5 / +27.6 | · | · | · |
| Vehicle | 81.8 / 84.6 / -2.9 | 84.2 / 73.1 / +11.1 | 97.8 / 46.3 / +51.4 | 71.6 / 70.6 / +1.1 | 94.2 / 30.4 / +63.8 |
| Board | 22.5 / 1.4 / +21.0 | 13.4 / 3.4 / +10.0 | 24.4 / 1.3 / +23.0 | 16.5 / 0.2 / +16.3 | · |
| Fencing | 55.8 / 21.0 / +34.8 | 48.3 / 21.2 / +27.1 | 30.8 / 15.9 / +14.9 | 12.7 / 23.9 / -11.2 | 11.2 / 0.0 / +11.2 |
| Human and Animal | 43.8 / 31.3 / +12.5 | 80.1 / 21.9 / +58.2 | 52.9 / 31.2 / +21.6 | 40.9 / 22.4 / +18.5 | · |
| Road Side Object | 0.1 / 0.4 / -0.4 | 6.7 / 1.1 / +5.6 | 47.3 / 8.0 / +39.3 | · | · |

### Concerto ft_9ch_indoor

| class | CH: out / in / Δ | KM: out / in / Δ | PL: out / in / Δ | SK: out / in / Δ | TR: out / in / Δ |
|---|---|---|---|---|---|
| Ground | 12.1 / 15.0 / -2.9 | 12.1 / 11.8 / +0.2 | 12.4 / 9.1 / +3.3 | 45.1 / 4.4 / +40.6 | · |
| Vegetation | 96.0 / 84.1 / +11.9 | 75.2 / 57.0 / +18.3 | 60.3 / 58.1 / +2.2 | 25.6 / 10.0 / +15.6 | 94.0 / 61.7 / +32.2 |
| Building | 83.7 / 88.5 / -4.8 | 65.9 / 84.4 / -18.5 | 95.8 / 84.7 / +11.1 | 37.1 / 64.3 / -27.2 | 89.6 / 76.2 / +13.4 |
| Road | 54.2 / 63.9 / -9.7 | 71.8 / 34.9 / +36.9 | 75.0 / 17.6 / +57.4 | 35.1 / 14.5 / +20.6 | 95.4 / 52.3 / +43.1 |
| Wire | 75.6 / 74.6 / +1.0 | 78.6 / 47.1 / +31.5 | 50.7 / 18.8 / +32.0 | · | 79.4 / 46.0 / +33.4 |
| Pole and Tower | 71.1 / 45.4 / +25.7 | 60.9 / 28.4 / +32.5 | 25.2 / 14.5 / +10.8 | 30.8 / 2.3 / +28.5 | 66.0 / 18.8 / +47.2 |
| Wire-Structure Connector | 0.0 / 34.8 / -34.8 | 28.7 / 17.8 / +10.9 | · | · | · |
| Vehicle | 80.7 / 84.0 / -3.2 | 85.2 / 62.0 / +23.2 | 96.4 / 67.2 / +29.1 | 72.2 / 9.2 / +63.1 | 85.9 / 17.8 / +68.1 |
| Board | 21.0 / 6.4 / +14.6 | 7.7 / 5.2 / +2.5 | 20.3 / 0.4 / +19.9 | 0.7 / 1.1 / -0.4 | · |
| Fencing | 31.1 / 16.9 / +14.3 | 24.9 / 23.5 / +1.4 | 21.4 / 20.7 / +0.7 | 14.1 / 4.1 / +10.0 | 16.1 / 0.8 / +15.2 |
| Human and Animal | 56.4 / 26.3 / +30.0 | 76.0 / 20.3 / +55.7 | 19.9 / 38.2 / -18.4 | 28.0 / 0.8 / +27.2 | · |
| Road Side Object | 0.0 / 1.4 / -1.4 | 3.8 / 0.1 / +3.7 | 59.8 / 12.5 / +47.3 | · | · |

### Concerto ft_9ch_outdoor

| class | SK: out / in / Δ |
|---|---|
| Ground | 53.3 / 6.0 / +47.3 |
| Vegetation | 44.5 / 6.8 / +37.7 |
| Building | 52.0 / 17.2 / +34.8 |
| Road | 32.6 / 0.1 / +32.4 |
| Wire | · |
| Pole and Tower | 35.6 / 4.2 / +31.4 |
| Wire-Structure Connector | · |
| Vehicle | 81.2 / 0.2 / +81.0 |
| Board | 1.9 / 0.1 / +1.7 |
| Fencing | 8.7 / 0.3 / +8.4 |
| Human and Animal | 38.0 / 0.0 / +38.0 |
| Road Side Object | · |

### Sonata ft_9ch

| class | CH: out / in / Δ | PL: out / in / Δ | SK: out / in / Δ | TR: out / in / Δ |
|---|---|---|---|---|
| Ground | 23.3 / 9.9 / +13.4 | 23.3 / 7.9 / +15.4 | 53.5 / 7.1 / +46.4 | · |
| Vegetation | 95.2 / 67.8 / +27.3 | 65.0 / 48.5 / +16.6 | 31.8 / 13.6 / +18.2 | 92.2 / 65.5 / +26.7 |
| Building | 90.9 / 88.6 / +2.3 | 98.3 / 80.5 / +17.9 | 47.6 / 55.0 / -7.4 | 91.3 / 75.2 / +16.1 |
| Road | 58.0 / 21.4 / +36.7 | 77.3 / 3.2 / +74.1 | 21.3 / 8.4 / +12.9 | 91.8 / 55.1 / +36.7 |
| Wire | 70.7 / 69.3 / +1.4 | 50.2 / 21.5 / +28.7 | · | 69.8 / 30.1 / +39.8 |
| Pole and Tower | 65.8 / 38.3 / +27.5 | 36.1 / 11.6 / +24.6 | 33.6 / 4.6 / +29.0 | 50.7 / 14.6 / +36.1 |
| Wire-Structure Connector | 0.0 / 28.7 / -28.7 | · | · | · |
| Vehicle | 83.9 / 81.1 / +2.8 | 95.9 / 54.2 / +41.6 | 49.9 / 7.3 / +42.6 | 82.4 / 18.3 / +64.1 |
| Board | 15.0 / 5.2 / +9.8 | 24.5 / 1.4 / +23.1 | 5.7 / 0.5 / +5.2 | · |
| Fencing | 57.1 / 11.7 / +45.5 | 29.7 / 10.3 / +19.4 | 9.2 / 4.6 / +4.6 | 14.4 / 0.6 / +13.8 |
| Human and Animal | 40.9 / 27.1 / +13.7 | 9.0 / 38.0 / -29.0 | 13.1 / 1.5 / +11.6 | · |
| Road Side Object | 0.2 / 1.7 / -1.5 | 41.8 / 12.1 / +29.7 | · | · |
