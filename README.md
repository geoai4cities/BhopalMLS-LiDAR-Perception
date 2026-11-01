# BhopalMLS-LiDAR-Perception

Training configs, frameworks, launchers, data tools and analysis scripts behind

> 📄 **BhopalMLS: A LiDAR Dataset for Semantic Scene Understanding in Heterogeneous Urban Environments.**
> Pyare Lal Chauhan, Vaibhav Kumar, Bharat Lohani. WACV 2027.
> 🌐 Project page, 3D viewer and video: https://geoai4cities.github.io/bhopalmls

BhopalMLS is a handheld mobile laser scanning dataset of dense, unplanned streets in Bhopal, India: eight survey paths, about 10 km of road corridor, more than 1.2 billion points, 19,518 annotated instances, 62 fine-grained static classes plus 6 optional moving classes, and 12-class and 21-class benchmark taxonomies. This repository reproduces every number in the paper: the semantic and instance segmentation benchmarks (supervised PTv3, LitePT-L and KPConvX-L; Sonata, Concerto and Utonia under linear probing, decoder probing, LoRA and fine-tuning), the cross-region transfer study against Ke-MLS, Toronto-3D, Paris-Lille-3D and SemanticKITTI, the factor-attribution and class-imbalance ablations, the spatial-leakage analysis and the CORAL domain-adaptation baseline.

If this code or the dataset helps your work, please ⭐ star the repository and 📚 cite the paper (BibTeX below). Problems or questions: 🐛 open an issue, we will try to solve it as soon as possible.

## 🧭 Where to look

| | |
|---|---|
| 📥 Getting the data | [`data/DOWNLOAD.md`](data/DOWNLOAD.md): account, India vs international release, every download command, which file is for what |
| 🗂️ Data layout and label spaces | [`data/Bhopal/README.md`](data/Bhopal/README.md), comparison datasets in [`data/README.md`](data/README.md) |
| 📊 Every reported number with its config | [`RESULTS.md`](RESULTS.md) |
| 💾 Trained weights | [`MODEL_ZOO.md`](MODEL_ZOO.md): the paper's runs; download links follow, every row is reproducible from its config meanwhile |
| 📐 Benchmark protocol (splits, taxonomies, metrics, recipes) | [`docs/BENCHMARK_PROTOCOL.md`](docs/BENCHMARK_PROTOCOL.md) |
| 🧪 Step-by-step reproduction | [`docs/REPRODUCING.md`](docs/REPRODUCING.md) |
| ⚙️ Environment | [`docs/ENVIRONMENT.md`](docs/ENVIRONMENT.md) |
| 🚀 Launchers | [`scripts/README.md`](scripts/README.md) |
| 🧩 Framework changes against upstream | [`framework_changes/README.md`](framework_changes/README.md) |
| 📋 Config index | [`docs/CONFIGS.md`](docs/CONFIGS.md) |

## 🚀 Quick start

```bash
git clone https://github.com/geoai4cities/BhopalMLS-LiDAR-Perception.git && cd BhopalMLS-LiDAR-Perception
./setup_venv.sh --with-kpconvx --with-pointrope      # Python 3.11 venv, torch 2.5.0, flash-attn, CUDA extensions (30 to 60 min)
source activate.sh

# data: a free LiDAVerse account, then the two benchmark files (46 GB) into data/Bhopal/; see data/DOWNLOAD.md
python3 data/Bhopal/bhopalmls_download.py --release international --files benchmark --out ~/BhopalMLS
ln -s ~/BhopalMLS/benchmark/*.h5 data/Bhopal/
python data/Bhopal/tools/bhopal_h5_dataset.py data/Bhopal/bhopal_scene_v9_normals_rgb.h5 --info

# pretrained checkpoints from Hugging Face into Utonia_pretrained/, sonata_pretrained/, Concerto_pretrained/ (docs/REPRODUCING.md, step 3)

PTv3/Bhopal/scripts/train_bhopal.sh train 9ch       # supervised baseline, 12 classes
Utonia/Bhopal/scripts/train_bhopal.sh ft 9ch        # Utonia(L) fine-tuning, the paper's 71.28 mIoU row
Utonia/Bhopal/scripts/test_bhopal.sh  ft 9ch        # test split, precise evaluation with test-time augmentation
```

## 🗂️ Repository layout

One folder per model, the shared framework next to them, the data folder as the configs name it:

```
Utonia/   Sonata/   Concerto/        foundation models: Bhopal/{configs,scripts,docs}, outputs/ (gitignored)
                                     Kerala/: probe, LoRA, fine-tuning and instance configs on Ke-MLS (all three models)
                                     Utonia and Concerto also: Toronto-3D/, Paris-Lille-3D/, SemanticKITTI/ (cross-region
                                     source configs) and cross_data_testing/ (transfer evaluation)
PTv3/     KpConvX/  LitePT/          supervised baselines: Bhopal/{configs,scripts}; LitePT/litept/ = the LitePT fork
Pointcept/                           the Pointcept framework, vendored at the pinned commit with our additions
framework_changes/                   exactly what differs from upstream Pointcept and LitePT (patches, file lists)
Utonia_pretrained/  sonata_pretrained/  Concerto_pretrained/    where the pretrained checkpoints go (gitignored)
data/Bhopal/                         class tables, colours, splits, tools, ablation subset files + the benchmark H5 you download
data/{Kerala,Toronto_3D,Paris-Lille-3D,SemanticKITTI}/          comparison datasets: class tables, preprocessing
analysis/                            spatial leakage, factor attribution, CORAL, class imbalance (scripts + their results)
scripts/launch.sh                    the engine every per-model wrapper calls
setup_venv.sh  activate.sh  requirements-lock.txt   environment
docs/  RESULTS.md  MODEL_ZOO.md  LICENSE  CITATION.cff
```

Each model folder has a README with its commands. Every path in the configs is relative to the repository root, so `cd` here (the wrappers do it for you).

## 📐 The benchmark in one paragraph

Scenes are 5 m corridor cuts of the survey paths (subdivided where they exceed one million points), stored in one HDF5 file per taxonomy version with per-point XYZ, intensity, RGB where the imagery allowed it, surface normals, label and instance id. Paths 1 to 4 train, 5 and 6 validate, 7 and 8 test. Models see 4 channels (XYZ + intensity), 9 channels (XYZ + partial RGB + normals) or, for the geometry-only probes, XYZ alone; all use 1 cm voxels, square-root inverse-frequency class weights with cross-entropy plus Lovasz loss, AdamW with a one-cycle schedule, 100 epochs (KPConvX 450) and Pointcept's 10-pass test-time augmentation. Instance segmentation uses the PointGroup head on the same backbones over the thing classes. Every hyper-parameter: [`docs/BENCHMARK_PROTOCOL.md`](docs/BENCHMARK_PROTOCOL.md).

## 🧩 Frameworks

Pointcept (MIT) and LitePT (MIT) are vendored in full at the commits the runs used, so a clone of this repository is self-contained. Our additions (the BhopalMLS, Ke-MLS, Toronto-3D and Paris-Lille-3D dataset classes, the Utonia backbone port, the KPConvX port from Apple's `ml-kpconvx`, instance-segmentation testers, gradient accumulation for LitePT and a few fixes) are listed in [`framework_changes/`](framework_changes/README.md) as file lists plus two small patches against upstream, so the exact difference is inspectable.

## 🐛 Issues

If something does not download, build or run, please open an issue with the command, its output, your OS, GPU, CUDA and Python versions. We will try to solve it as soon as possible.

## 📜 Licence

Code: MIT (see `LICENSE`; the KPConvX files are Apple's, also MIT). Dataset: CC BY-NC-SA 4.0 with terms of access, released separately on LiDAVerse. Sonata, Concerto and Utonia checkpoints: CC BY-NC 4.0 from their authors.

## 📚 Citation

```bibtex
@inproceedings{chauhan2027bhopalmls,
  title     = {{BhopalMLS}: A {LiDAR} Dataset for Semantic Scene Understanding in Heterogeneous Urban Environments},
  author    = {Chauhan, Pyare Lal and Kumar, Vaibhav and Lohani, Bharat},
  booktitle = {Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)},
  year      = {2027}
}
```

## 🙏 Acknowledgements

The benchmark builds on [Pointcept](https://github.com/Pointcept/Pointcept), [LitePT](https://github.com/prs-eth/LitePT), [KPConvX](https://github.com/apple/ml-kpconvx), [Sonata](https://github.com/facebookresearch/sonata), [Concerto](https://github.com/Pointcept/Concerto) and [Utonia](https://github.com/Pointcept/Utonia). Funding: iHub-Data (IIIT Hyderabad) and MPSeDC.
