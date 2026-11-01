# 📥 Downloading the BhopalMLS data

Everything you need to get the data onto your disk, from making an account to checking the files. The data itself is not in this repository; it is hosted on [LiDAVerse](https://lidaverse.com) under CC BY-NC-SA 4.0 with terms of access.

## 1️⃣ Pick your release

There are two copies of the dataset. They are identical in geometry, intensity, colour, labels, instance ids, normals and scene cuts; they differ only in the coordinate frame. Every number in the paper is reproducible from either one.

| | 🇮🇳 India release | 🌍 International release |
|---|---|---|
| LiDAVerse page | https://lidaverse.com/datasets/?id=13 | https://lidaverse.com/datasets/?id=14 |
| coordinates | absolute UTM Zone 43N (EPSG:32643) | UTM 43N minus one constant XY offset that is withheld; Z unchanged |
| who may download | users whose institution is in India (India's National Geospatial Policy, 2022) | everyone, including users in India |
| helper flag | `--release india` | `--release international` |

If you are outside India, use the international release. If you are in India and want the georeferenced copy, use the India release; LiDAVerse checks the account's institution and will refuse the India files otherwise.

## 2️⃣ Make a LiDAVerse account

1. Go to https://lidaverse.com and sign up (free). Use your institutional email and fill in your institution and country: the country decides which release you may download.
2. Open the dataset page of your release (links above) and accept the licence and terms of access shown there.

That is the whole registration. There is no approval step; downloads open as soon as the account exists.

## 3️⃣ Download

### 🖱️ From the browser

The dataset page lists every file. Browser downloads work for the small files (`README.md`, `LICENSE.md`, `SHA256SUMS`, `metadata/`, `tools/`, the 0.5 GB sample). They **fail for the large files** (the two 23 GB benchmark files and the bigger `npz/` files), because the browser downloads into memory first. For those use the helper.

### 🐍 With the helper script (recommended)

`data/Bhopal/bhopalmls_download.py` logs in with your LiDAVerse account, streams each file to disk, verifies its SHA-256 and resumes interrupted transfers when you run the same command again. Python 3.8 or newer, standard library only, no extra packages. The password is asked interactively and never stored.

```bash
# what is there, with sizes
python3 data/Bhopal/bhopalmls_download.py --list

# docs + the 24-scene sample (0.5 GB): a first look
python3 data/Bhopal/bhopalmls_download.py --release international --out ~/BhopalMLS

# the two benchmark files the paper was trained and evaluated on (46 GB)
python3 data/Bhopal/bhopalmls_download.py --release international --files benchmark --out ~/BhopalMLS

# the full dataset as per-path NPZ files (49 GB)
python3 data/Bhopal/bhopalmls_download.py --release international --files npz --out ~/BhopalMLS

# everything (96 GB)
python3 data/Bhopal/bhopalmls_download.py --release international --files all --out ~/BhopalMLS
```

Users in India replace `--release international` with `--release india` in every command.

`--files` takes groups (`docs`, `sample`, `benchmark`, `npz`, `metadata`, `tools`, `all`) or exact release paths, so single files work too:

```bash
# one benchmark file
python3 data/Bhopal/bhopalmls_download.py --release international --files benchmark/bhopal_scene_v9_normals_rgb.h5 --out ~/BhopalMLS
# one survey path, labelled + unlabelled points
python3 data/Bhopal/bhopalmls_download.py --release international --files npz/Bhopal_6.npz npz/Bhopal_6_Unlabelled.npz --out ~/BhopalMLS
# class tables, splits and scene cuts only (2 MB)
python3 data/Bhopal/bhopalmls_download.py --release international --files metadata --out ~/BhopalMLS
```

A partial download leaves `*.part` files; run the same command again and it continues. A file whose checksum does not match is set aside as `*.bad` and never put in place.

## 4️⃣ What each file is for

| file | size | use it for |
|---|---|---|
| `benchmark/bhopal_scene_v9_normals_rgb.h5` | 23.0 GB | **the benchmark**: 2,410 scenes, 62 static classes (label62). Every main-paper table, the 21-class tables, the cross-region study and the ablations were trained and evaluated on this file |
| `benchmark/bhopal_scene_v10_train_1234_val_56_test_78.h5` | 23.4 GB | the same scenes plus the 6 moving classes as labels 62 to 67 (68 classes). Only the paper's † rows (`configs/bhopal_v10/`) use it |
| `npz/Bhopal_{1..8}.npz` | 3.6 to 9.8 GB each, 47 GB | **the full dataset**, every labelled point of each survey path stored instance by instance; the raw material the benchmark files were cut from |
| `npz/Bhopal_{1..8}_Unlabelled.npz` | 0.1 to 0.6 GB each, 2.3 GB | the remaining points of each path: unlabelled and noise returns (codes 0 to 2) and the six moving classes (codes 3 to 8) |
| `metadata/` | 2 MB | class tables, colour maps, `label_codes.json`, `splits.json`, `scene_division.zip` (the 5 m scene cuts) |
| `sample/bhopalmls_sample_v2.tar.gz` | 0.5 GB | 24 complete scenes as PLY + NPZ for a quick look in CloudCompare or Open3D |
| `tools/` | 0.2 MB | the loader, converter and reference scripts, also shipped in `data/Bhopal/tools/` here (the copies here point the reference scripts' input folder at `data/Bhopal/source/`) |
| `README.md`, `LICENSE.md`, `SHA256SUMS` | | data card, licence and terms of access, checksums |

Sizes are those of the international release; the India files differ by up to about 2 %.

**Benchmarking**: download `benchmark/` (and `metadata/` if you want the tables locally). The HDF5 scene files exist because training reads them fast: every scene is a padded, contiguous block, so a DataLoader with many workers streams scenes without parsing per-instance files.

**Everything else**: `npz/` holds the complete point set in survey-path order with per-point XYZ (float64, metres), intensity (uint16), GPS time, label, 9-digit instance id and RGB (`0,0,0` = no colour), plus per-instance index ranges. Convert any path to a GIS or viewer format with the shipped converter:

```bash
python data/Bhopal/tools/convert_npz.py ~/BhopalMLS/npz/Bhopal_6.npz --format las --out export/              # LAS 1.4 (needs laspy)
python data/Bhopal/tools/convert_npz.py ~/BhopalMLS/npz/Bhopal_6.npz --format ply csv --out export/          # PLY + CSV
python data/Bhopal/tools/convert_npz.py ~/BhopalMLS/npz/Bhopal_6.npz --format bin --out export/ --per-instance   # one file per object
```

`data/Bhopal/tools/README.md` documents the formats; `data/Bhopal/tools/reference/` holds the scripts that built the benchmark files from `npz/` and `metadata/scene_division.zip`, so the cut protocol is inspectable and re-runnable.

## 5️⃣ Check the files and wire them into the code

```bash
cd ~/BhopalMLS && sha256sum -c --ignore-missing SHA256SUMS            # every line should say OK
cd <this repository>
ln -s ~/BhopalMLS/benchmark/*.h5 data/Bhopal/                         # configs read data/Bhopal/<file>.h5 (or download straight into data/Bhopal/)
python data/Bhopal/tools/bhopal_h5_dataset.py data/Bhopal/bhopal_scene_v9_normals_rgb.h5 --info
```

The `--info` call prints the splits (1,353 / 568 / 489 scenes), the class table and the colour coverage. `data/Bhopal/README.md` explains the folder further (class tables, colours, splits, the small subset files of the factor-attribution ablation that already sit next to the benchmark files); `data/README.md` covers the comparison datasets of the cross-region study.

## 🔒 Colour and privacy

About 35 % of the points on the six paths with imagery carry colour; `0,0,0` means "no colour", not black. The colour of every point labelled as a person (pedestrians, riders, the moving human classes) is set to zero in both releases; geometry and labels are untouched, so results are unaffected. Details in the data card (`README.md` of the release).

## 🐛 Problems?

If a download fails, a checksum does not match, or a file does not open, please open an issue in this repository with the command you ran, the output, your operating system and Python version. We will try to solve it as soon as possible.

## ⭐ Using the data?

Please cite the paper (BibTeX in the top-level `README.md`) and give the repository a star so others find it. Thank you!
