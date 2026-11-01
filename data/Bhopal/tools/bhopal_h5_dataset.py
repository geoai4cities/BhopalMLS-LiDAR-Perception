#!/usr/bin/env python3
# MIT License
#
# Copyright (c) 2026 the BhopalMLS authors, IISER Bhopal
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and
# associated documentation files (the "Software"), to deal in the Software without restriction,
# including without limitation the rights to use, copy, modify, merge, publish, distribute,
# sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions: the above copyright notice and this
# permission notice shall be included in all copies or substantial portions of the Software.
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND.
"""Read the BhopalMLS benchmark HDF5 files: a numpy loader, a PyTorch Dataset and a quick inspector.

    bhopal_h5_dataset.py benchmark/bhopal_scene_v9_normals_rgb.h5 --info
    bhopal_h5_dataset.py benchmark/bhopal_scene_v9_normals_rgb.h5 --samples 3 --split test
    bhopal_h5_dataset.py benchmark/bhopal_scene_v9_normals_rgb.h5 --dataloader --num-points 65536
    xvfb-run -a bhopal_h5_dataset.py benchmark/bhopal_scene_v10_train_1234_val_56_test_78.h5 \\
        --render 2 --color-by labels --out renders/

In code:
    from bhopal_h5_dataset import BhopalSceneDataset, load_scene, parent_map
    ds = BhopalSceneDataset("benchmark/bhopal_scene_v9_normals_rgb.h5", split="train", num_points=65536)
    item = ds[0]              # torch tensors: points, rgb, normals, intensity, labels, global_ids, ...
    with h5py.File(path) as f:
        scene = load_scene(f, int(f["splits/test_indices"][0]))     # numpy, no torch needed
        p12 = parent_map(f)[scene["labels"]]                           # 12-class labels

Works for both benchmark files: the 62-class v9 (labels 0-61) and the 68-class v10 (62-67 = moving
classes). Class names, colours and the 12-class parent of every label are read from the file's
class_info group, so nothing else is needed. Labels -1 mean ignore / padding. Needs numpy and h5py;
torch only for the Dataset, open3d only for --render (run under xvfb-run on a headless machine).
"""

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np

try:
    import torch
    from torch.utils.data import Dataset
    HAS_TORCH = True
except ImportError:                       # the numpy functions work without torch
    HAS_TORCH = False

    class Dataset:                        # placeholder so the class below can be defined
        pass


# ---------------------------------------------------------------- numpy helpers
def _str(x):
    return x.decode() if isinstance(x, bytes) else str(x)


def class_names(f):
    return [_str(x) for x in f["class_info/names"][:]]


def parent_names(f):
    return [_str(x) for x in f["class_info/parent_names"][:]]


def parent_map(f):
    """label -> 12-class parent id, as an int array indexed by label (label -1 is not covered)."""
    return f["class_info/parent_ids"][:].astype(np.int64)


def scene_indices(f, split="train", min_points=1):
    """Scene indices of a split ('train', 'val', 'test' or 'all'), empty scenes dropped."""
    if split == "all":
        idx = np.concatenate([f[f"splits/{k}_indices"][:] for k in ("train", "val", "test")])
    else:
        idx = f[f"splits/{split}_indices"][:]
    n = f["scenes/num_points"][:]
    return idx[n[idx] >= max(1, min_points)]


def load_scene(f, i, num_points=None, rng=None):
    """One scene as numpy arrays (valid rows only; optional random subsample)."""
    n = int(f["scenes/num_points"][i])
    sel = slice(0, n)
    if num_points and n > num_points:
        rng = rng or np.random.default_rng()
        sel = np.sort(rng.choice(n, num_points, replace=False))
    out = {k: f[f"scenes/{k}"][i, :n][sel] if isinstance(sel, np.ndarray) else f[f"scenes/{k}"][i, sel]
           for k in ("points", "rgb", "normals", "intensity", "labels", "global_ids")}
    out.update(num_points_original=n, scene_name=_str(f["metadata/scene_names"][i]),
               path=int(f["metadata/bhopal_paths"][i]), center=f["metadata/center"][i],
               num_instances=int(f["metadata/num_instances"][i]), index=int(i))
    return out


# ---------------------------------------------------------------- PyTorch Dataset
class BhopalSceneDataset(Dataset):
    """Scene-level dataset over one benchmark H5.

    Each item: points [N,3] float32 (metres, scene-local), rgb [N,3] float32 in [0,1] (0 = no colour),
    normals [N,3], intensity [N] in [0,1], labels [N] long (-1 = padding), global_ids [N] long,
    plus num_points, original_num_points, scene_name, path. With num_points set, scenes are randomly
    subsampled or zero-padded to that size so they can be batched. The file is opened lazily in each
    worker, so num_workers > 0 is safe.
    """

    def __init__(self, h5_path, split="train", num_points=None, min_points=1, transform=None):
        if not HAS_TORCH:
            raise ImportError("BhopalSceneDataset needs torch; use load_scene() for numpy access")
        self.h5_path, self.split, self.num_points, self.transform = str(h5_path), split, num_points, transform
        with h5py.File(self.h5_path, "r") as f:
            self.indices = scene_indices(f, split, min_points)
            self.num_classes = int(f.attrs["num_classes"])
            self.class_names = class_names(f)
            self.parent_map = parent_map(f)
        self._f = None

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, k):
        if self._f is None:
            self._f = h5py.File(self.h5_path, "r")
        s = load_scene(self._f, int(self.indices[k]), self.num_points)
        n = s["points"].shape[0]
        pad = (self.num_points - n) if self.num_points and n < self.num_points else 0
        if pad:
            z3 = np.zeros((pad, 3), np.float32)
            s["points"] = np.vstack([s["points"], z3])
            s["normals"] = np.vstack([s["normals"], z3])
            s["rgb"] = np.vstack([s["rgb"], np.zeros((pad, 3), np.uint8)])
            s["intensity"] = np.concatenate([s["intensity"], np.zeros(pad, np.float32)])
            s["labels"] = np.concatenate([s["labels"], np.full(pad, -1, np.int32)])
            s["global_ids"] = np.concatenate([s["global_ids"], np.full(pad, -1, np.int32)])
        pts = torch.from_numpy(np.ascontiguousarray(s["points"], dtype=np.float32))
        if self.transform:
            pts = self.transform(pts)
        return {
            "points": pts,
            "rgb": torch.from_numpy(s["rgb"].astype(np.float32) / 255.0),
            "normals": torch.from_numpy(np.ascontiguousarray(s["normals"], dtype=np.float32)),
            "intensity": torch.from_numpy(s["intensity"].astype(np.float32) / 65535.0),
            "labels": torch.from_numpy(s["labels"].astype(np.int64)),
            "global_ids": torch.from_numpy(s["global_ids"].astype(np.int64)),
            "num_points": n + pad, "original_num_points": s["num_points_original"],
            "scene_name": s["scene_name"], "path": s["path"],
        }


# ---------------------------------------------------------------- inspector
def info(f):
    a = f.attrs
    g = lambda k, d="": _str(a[k]) if k in a else d
    n = f["scenes/num_points"][:]
    print(f"file            {Path(f.filename).name}")
    print(f"dataset         {g('dataset_name')}  version {g('version')}  built {g('creation_date')[:10]}")
    print(f"release         {g('release_name', '-')} v{g('release_version', '-')} tier={g('release_tier', '-')}")
    print(f"coordinates     {g('coordinate_system')}")
    print(f"classes         {g('num_classes')} ({g('num_parent_classes')} parents); moving classes: {g('has_moving_classes', 'False')}")
    print(f"scenes          {n.shape[0]:,} total, {int((n > 0).sum()):,} non-empty, {int(n.sum()):,} points; "
          f"max {int(n.max()):,} per scene")
    for k in ("train", "val", "test"):
        idx = f[f"splits/{k}_indices"][:]
        paths = sorted(set(int(p) for p in f["metadata/bhopal_paths"][:][idx]))
        print(f"  {k:5s}  {len(idx):5d} scenes  {int(n[idx].sum()):>13,} points  paths {paths}")
    print(f"rgb             {g('rgb_coverage_note')}; person RGB zeroed: {g('rgb_zeroed_points', '-')} points")
    print(f"normals         k={g('normal_k_neighbors')}, {g('normal_orientation')}")
    names, pn, pm = class_names(f), parent_names(f), parent_map(f)
    print("\nlabel  class                           parent (12-class)")
    for i, nm in enumerate(names):
        print(f"{i:5d}  {nm:30s}  {pm[i]:2d} {pn[pm[i]]}")


def samples(f, split, k, seed=0):
    idx = scene_indices(f, split)
    rng = np.random.default_rng(seed)
    pick = rng.choice(idx, min(k, len(idx)), replace=False)
    names = class_names(f)
    for i in pick:
        s = load_scene(f, int(i))
        lab = s["labels"]
        u, c = np.unique(lab[lab >= 0], return_counts=True)
        top = ", ".join(f"{names[a]} {100*b/lab.size:.0f}%" for a, b in sorted(zip(u, c), key=lambda t: -t[1])[:5])
        rgb_pct = 100 * s["rgb"].any(axis=1).mean()
        ext = s["points"].max(0) - s["points"].min(0)
        print(f"scene {s['scene_name']:10s} path {s['path']}  {s['num_points_original']:>9,} pts  "
              f"{s['num_instances']:4d} inst  extent {ext[0]:.0f}x{ext[1]:.0f}x{ext[2]:.0f} m  rgb {rgb_pct:.0f}%  | {top}")


def test_dataloader(h5_path, split, num_points, batch_size, num_batches, workers=0):
    if not HAS_TORCH:
        sys.exit("--dataloader needs torch")
    from torch.utils.data import DataLoader
    ds = BhopalSceneDataset(h5_path, split=split, num_points=num_points)
    print(f"{len(ds)} scenes in '{split}', {ds.num_classes} classes, num_points={num_points}")
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=workers)
    for b, batch in enumerate(dl):
        lab = batch["labels"]
        print(f"batch {b}: points {tuple(batch['points'].shape)} rgb {tuple(batch['rgb'].shape)} "
              f"normals {tuple(batch['normals'].shape)} labels {tuple(lab.shape)} "
              f"range [{int(lab.min())}, {int(lab.max())}]  scenes {list(batch['scene_name'])}")
        if b + 1 >= num_batches:
            break


def render(f, split, k, out, color_by="labels", max_points=300_000, elevation=30, azimuth=45, zoom=0.3,
           point_size=2.0, width=1600, height=900, seed=0):
    try:
        import open3d as o3d
    except ImportError:
        sys.exit("--render needs open3d (pip install open3d); on a headless machine run under xvfb-run -a")
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    colors = f["class_info/colors"][:].astype(np.float64) / 255.0
    pcol = f["class_info/parent_colors"][:].astype(np.float64) / 255.0
    pm = parent_map(f)
    idx = scene_indices(f, split)
    rng = np.random.default_rng(seed)
    for i in rng.choice(idx, min(k, len(idx)), replace=False):
        s = load_scene(f, int(i), max_points, rng)
        lab = s["labels"]
        if color_by == "rgb":
            c = s["rgb"].astype(np.float64) / 255.0
            c[~s["rgb"].any(axis=1)] = 0.5
        elif color_by == "normals":
            c = np.clip((s["normals"] + 1.0) / 2.0, 0, 1)
        elif color_by == "parent":
            c = np.where(lab[:, None] >= 0, pcol[pm[np.clip(lab, 0, None)]], 0.5)
        else:
            c = np.where(lab[:, None] >= 0, colors[np.clip(lab, 0, None)], 0.5)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(s["points"].astype(np.float64) - s["points"].mean(0))
        pcd.colors = o3d.utility.Vector3dVector(c)
        vis = o3d.visualization.Visualizer()
        vis.create_window(visible=False, width=width, height=height)
        vis.add_geometry(pcd)
        opt = vis.get_render_option()
        opt.point_size, opt.background_color = point_size, np.array([1.0, 1.0, 1.0])
        ctr = vis.get_view_control()
        el, az = np.radians(elevation), np.radians(azimuth)
        ctr.set_front([np.cos(el) * np.sin(az), np.cos(el) * np.cos(az), np.sin(el)])
        ctr.set_up([0, 0, 1])
        ctr.set_lookat([0, 0, 0])
        ctr.set_zoom(zoom)
        vis.poll_events()
        vis.update_renderer()
        png = out / f"{s['scene_name']}_path{s['path']}_{color_by}.png"
        vis.capture_screen_image(str(png), do_render=True)
        vis.destroy_window()
        print(f"{png}  ({s['points'].shape[0]:,} of {s['num_points_original']:,} points)")


def main():
    here = Path(__file__).resolve().parent
    default = sorted((here.parent / "benchmark").glob("bhopal_scene_v9*.h5")) + \
        sorted((here.parent / "benchmark").glob("*.h5"))
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("h5", nargs="?", type=Path, default=default[0] if default else None, help="benchmark H5 file")
    ap.add_argument("--split", default="train", choices=["train", "val", "test", "all"])
    ap.add_argument("--info", action="store_true", help="file summary, splits and the class table")
    ap.add_argument("--samples", type=int, default=0, metavar="N", help="describe N random scenes of the split")
    ap.add_argument("--dataloader", action="store_true", help="iterate a few PyTorch DataLoader batches")
    ap.add_argument("--num-points", type=int, default=65536)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--num-batches", type=int, default=3)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--render", type=int, default=0, metavar="N", help="render N random scenes to PNG (open3d)")
    ap.add_argument("--color-by", default="labels", choices=["labels", "parent", "rgb", "normals"])
    ap.add_argument("--out", type=Path, default=Path("renders"))
    ap.add_argument("--max-render-points", type=int, default=300_000)
    ap.add_argument("--elevation", type=float, default=30)
    ap.add_argument("--azimuth", type=float, default=45)
    ap.add_argument("--zoom", type=float, default=0.3)
    ap.add_argument("--point-size", type=float, default=2.0)
    a = ap.parse_args()
    if a.h5 is None or not a.h5.exists():
        sys.exit(f"H5 file not found: {a.h5}")
    if not (a.info or a.samples or a.dataloader or a.render):
        a.info = True
    with h5py.File(a.h5, "r") as f:
        if a.info:
            info(f)
        if a.samples:
            print()
            samples(f, a.split, a.samples)
        if a.render:
            render(f, a.split, a.render, a.out, a.color_by, a.max_render_points,
                   a.elevation, a.azimuth, a.zoom, a.point_size)
    if a.dataloader:
        print()
        test_dataloader(a.h5, a.split, a.num_points, a.batch_size, a.num_batches, a.workers)


if __name__ == "__main__":
    main()
