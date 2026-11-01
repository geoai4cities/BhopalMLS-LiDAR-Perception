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
"""Export a BhopalMLS per-path NPZ file to LAS, PLY, CSV or BIN.

    convert_npz.py npz/Bhopal_6.npz --format las --out export/
    convert_npz.py npz/Bhopal_6.npz npz/Bhopal_6_Unlabelled.npz --format ply csv --out export/
    convert_npz.py npz/Bhopal_6.npz --format bin --out export/ --per-instance

Whole-path mode (default) writes one file per NPZ: export/Bhopal_6.las.
--per-instance writes one file per instance, named by its 9-digit global id, under
export/Bhopal_6/<parent class>/<class>/NXXZZYYYY.<ext> (the layout of the original per-instance
release). The class names come from metadata/Bhopal_class_mapping_V13.csv (--mapping).

Formats
  las  LAS 1.4, point format 7. x y z at 0.1 mm resolution; intensity; gps_time; red green blue
       (8-bit values scaled to 16-bit; 0 = no colour); classification = label; extra bytes
       `label` (uint16) and `global_id` (uint32). Needs laspy. Files of the India release carry
       EPSG:32643 when pyproj is installed.
  ply  binary little-endian: x y z (double), red green blue (uchar), intensity (ushort),
       gps_time (float), label (ushort), global_id (uint).
  csv  header x,y,z,intensity,gps_time,label,global_id,red,green,blue; x y z with 4 decimals.
  bin  BPLC v3: 48-byte header + 39-byte little-endian records (see tools/README.md and read_bin()).

Coordinates, labels and ids are written exactly as stored in the NPZ. Only numpy is required
(laspy for LAS). A whole path is held in memory while writing; use --per-instance on small machines.
"""

import argparse
import csv
import json
import struct
import sys
from pathlib import Path

import numpy as np

FIELDS = ("points", "intensity", "gps_time", "labels", "global_ids", "rgb",
          "instance_indices", "instance_global_ids", "instance_labels", "instance_xxzz")

BIN_MAGIC = b"BPLC"
BIN_HEADER = struct.Struct("<4sHHIIHI26x")          # magic, version, reserved, n, instance_id, label, xxzz, pad = 48 B
BIN_V2 = np.dtype([("x", "<f8"), ("y", "<f8"), ("z", "<f8"), ("intensity", "<u2"), ("gps_time", "<f4"),
                   ("label", "<u2"), ("global_id", "<u4")])                                          # 36 B
BIN_V3 = np.dtype(BIN_V2.descr + [("red", "u1"), ("green", "u1"), ("blue", "u1")])                 # 39 B
PLY_DT = np.dtype([("x", "<f8"), ("y", "<f8"), ("z", "<f8"), ("red", "u1"), ("green", "u1"), ("blue", "u1"),
                   ("intensity", "<u2"), ("gps_time", "<f4"), ("label", "<u2"), ("global_id", "<u4")])
PLY_TYPE = {("f", 8): "double", ("f", 4): "float", ("u", 1): "uchar", ("u", 2): "ushort", ("u", 4): "uint"}
CSV_FMT = ["%.4f", "%.4f", "%.4f", "%d", "%.4f", "%d", "%d", "%d", "%d", "%d"]


def load_npz(path):
    z = np.load(path, allow_pickle=False)
    d = {k: z[k] for k in FIELDS}
    d["metadata"] = json.loads(str(z["metadata"]))
    d["names"] = json.loads(str(z["class_code_to_name"]))
    return d


def load_mapping(path):
    """V13 class mapping CSV -> ({XX: parent name}, {'XXZZ': class name})."""
    parent, child = {}, {}
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh):
            xx, zz = int(r["XX"]), int(r["ZZ"])
            if zz == 0:
                parent[xx] = r["class_name"]
            child[f"{xx:02d}{zz:02d}"] = r["class_name"]
    return parent, child


def folder_name(s):
    return s.replace("/", "-").strip()


# ---------------------------------------------------------------- writers
def write_las(path, d, sl, crs_epsg=None, **_):
    try:
        import laspy
    except ImportError:
        sys.exit("LAS export needs laspy: pip install laspy")
    pts = d["points"][sl]
    hdr = laspy.LasHeader(point_format=7, version="1.4")
    hdr.offsets = np.floor(pts.min(axis=0))
    hdr.scales = np.array([0.0001, 0.0001, 0.0001])
    hdr.add_extra_dim(laspy.ExtraBytesParams(name="label", type="uint16", description="BhopalMLS label"))
    hdr.add_extra_dim(laspy.ExtraBytesParams(name="global_id", type="uint32", description="instance id NXXZZYYYY"))
    if crs_epsg:
        try:
            import pyproj
            hdr.add_crs(pyproj.CRS.from_epsg(crs_epsg))
        except Exception:
            pass
    las = laspy.LasData(hdr)
    las.x, las.y, las.z = pts[:, 0], pts[:, 1], pts[:, 2]
    las.intensity = d["intensity"][sl].astype(np.uint16)
    las.gps_time = d["gps_time"][sl].astype(np.float64)
    rgb = d["rgb"][sl].astype(np.uint16) * 257
    las.red, las.green, las.blue = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    las.classification = np.minimum(d["labels"][sl], 255).astype(np.uint8)
    las.label = d["labels"][sl].astype(np.uint16)
    las.global_id = d["global_ids"][sl].astype(np.uint32)
    las.write(str(path))


def fill_records(arr, d, sl):
    pts = d["points"][sl]
    arr["x"], arr["y"], arr["z"] = pts[:, 0], pts[:, 1], pts[:, 2]
    rgb = d["rgb"][sl]
    arr["red"], arr["green"], arr["blue"] = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    arr["intensity"] = d["intensity"][sl]
    arr["gps_time"] = d["gps_time"][sl]
    arr["label"] = d["labels"][sl]
    arr["global_id"] = d["global_ids"][sl]
    return arr


def write_ply(path, d, sl, **_):
    n = d["points"][sl].shape[0]
    arr = fill_records(np.empty(n, dtype=PLY_DT), d, sl)
    header = ["ply", "format binary_little_endian 1.0",
              "comment BhopalMLS; coordinate frame and label space as in the source NPZ metadata",
              "comment red green blue 0,0,0 = no colour", f"element vertex {n}"]
    for name in PLY_DT.names:
        t = PLY_DT[name]
        header.append(f"property {PLY_TYPE[(t.kind, t.itemsize)]} {name}")
    with open(path, "wb") as fh:
        fh.write(("\n".join(header) + "\nend_header\n").encode("ascii"))
        arr.tofile(fh)


def write_csv(path, d, sl, chunk=2_000_000, **_):
    pts = d["points"][sl]
    inten, gps, lab, gid, rgb = (d[k][sl] for k in ("intensity", "gps_time", "labels", "global_ids", "rgb"))
    n = pts.shape[0]
    with open(path, "w", newline="") as fh:
        fh.write("x,y,z,intensity,gps_time,label,global_id,red,green,blue\n")
        for s in range(0, n, chunk):
            e = min(n, s + chunk)
            block = np.column_stack([pts[s:e], inten[s:e], gps[s:e], lab[s:e], gid[s:e], rgb[s:e]]).astype(np.float64)
            np.savetxt(fh, block, fmt=CSV_FMT, delimiter=",")


def write_bin(path, d, sl, instance_id=0, label=0xFFFF, xxzz=0, **_):
    """BPLC v3. Whole-path files use instance_id 0, label 65535 and xxzz 0."""
    n = d["points"][sl].shape[0]
    arr = fill_records(np.empty(n, dtype=BIN_V3), d, sl)
    with open(path, "wb") as fh:
        fh.write(BIN_HEADER.pack(BIN_MAGIC, 3, 0, n, instance_id, label, xxzz))
        arr.tofile(fh)


def read_bin(path):
    """Read a BPLC v2 (no colour) or v3 file -> (header dict, structured array)."""
    raw = Path(path).read_bytes()
    magic, ver, _, n, iid, lab, xxzz = BIN_HEADER.unpack_from(raw, 0)
    if magic != BIN_MAGIC:
        raise ValueError(f"{path}: not a BPLC file")
    dt = {2: BIN_V2, 3: BIN_V3}[ver]
    rec = np.frombuffer(raw, dtype=dt, count=n, offset=BIN_HEADER.size)
    return {"version": ver, "num_points": n, "instance_id": iid, "instance_label": lab, "xxzz": xxzz}, rec


WRITERS = {"las": write_las, "ply": write_ply, "csv": write_csv, "bin": write_bin}


# ---------------------------------------------------------------- driver
def convert(npz_path, formats, out, per_instance, mapping_csv, max_instances=None):
    d = load_npz(npz_path)
    stem = npz_path.stem
    crs = 32643 if d["metadata"].get("coordinate_system", "").startswith("UTM Zone 43N (EPSG:32643), absolute") else None
    if not per_instance:
        for fmt in formats:
            dst = out / f"{stem}.{fmt}"
            WRITERS[fmt](dst, d, slice(None), crs_epsg=crs)
            print(f"{dst}  {d['points'].shape[0]:,} points")
        return
    parent, child = load_mapping(mapping_csv)
    idx, gids, labs, xz = (d[k] for k in ("instance_indices", "instance_global_ids", "instance_labels", "instance_xxzz"))
    n = len(idx) if max_instances is None else min(len(idx), max_instances)
    written = 0
    for i in range(n):
        s, e = int(idx[i, 0]), int(idx[i, 1])
        xxzz = str(xz[i])
        xx = int(xxzz[:2])
        folder = out / stem / folder_name(parent.get(xx, f"XX{xx:02d}")) / folder_name(child.get(xxzz, xxzz))
        folder.mkdir(parents=True, exist_ok=True)
        for fmt in formats:
            WRITERS[fmt](folder / f"{int(gids[i])}.{fmt}", d, slice(s, e), crs_epsg=crs,
                         instance_id=int(gids[i]), label=int(labs[i]), xxzz=int(xxzz))
            written += 1
    print(f"{out / stem}: {n:,} instances, {written:,} files")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("npz", nargs="+", type=Path, help="per-path NPZ file(s) of the release")
    ap.add_argument("--format", nargs="+", required=True, choices=sorted(WRITERS), help="one or more output formats")
    ap.add_argument("--out", type=Path, required=True, help="output directory")
    ap.add_argument("--per-instance", action="store_true", help="one file per instance in class folders")
    ap.add_argument("--mapping", type=Path,
                    default=Path(__file__).resolve().parent.parent / "metadata" / "Bhopal_class_mapping_V13.csv",
                    help="V13 class mapping CSV for the per-instance folder names")
    ap.add_argument("--max-instances", type=int, help="stop after this many instances (testing)")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    for p in a.npz:
        convert(p, a.format, a.out, a.per_instance, a.mapping, a.max_instances)


if __name__ == "__main__":
    main()
