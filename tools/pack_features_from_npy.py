#!/usr/bin/env python3
"""
Pack per-video .npy CLIP features into pickled shards that NitzcheCLIP can load.

Inputs:
  --features-dir  Directory with per-video .npy files (e.g., example_dataset/clip-features-32)
  --map-dir       Directory with per-video CSVs mapping keyframe order 'n' -> 'frame_idx'
  --frames-prefix Path prefix to prepend to frame paths in pickles (default: './data/video_frames')
  --outdir        Output directory for shards (default: aic-24-BE/data/clip_features)
  --ext           Image extension for frame files (default: .jpg)

Behavior:
  - For each {video_id}.npy, load features [N, D].
  - Load {map-dir}/{video_id}.csv; read ordered list of 'n' mapping to 'frame_idx'.
  - Build file_path_list as '{frames-prefix}/{video_id}/{frame_idx}{ext}' in ascending n.
  - Save a pickle shard '{outdir}/{video_id}.pkl' containing (file_path_list, image_feature).

Note:
  - Ensure your frames at '{frames-prefix}/{video_id}/{frame_idx}{ext}' exist (use the renaming tool first).
  - Make sure your runtime text encoder backbone matches these features.
"""
from __future__ import annotations

import argparse
import csv
import os
import pickle
from pathlib import Path
from typing import List, Tuple

import numpy as np


def load_map(path: Path) -> List[Tuple[int, int]]:
    """Return ordered list of (n, frame_idx)."""
    out: List[Tuple[int, int]] = []
    with path.open('r', encoding='utf-8') as f:
        rdr = csv.reader(f)
        rows = list(rdr)
        if not rows:
            return out
        header = rows[0]
        has_header = any(not c.isdigit() for c in header)
        start = 1 if has_header else 0
        # detect columns
        n_idx = 0
        fi_idx = 1
        if has_header:
            lower = [h.strip().lower() for h in header]
            for i, h in enumerate(lower):
                if h in {"n", "keyframe", "idx", "index"}:
                    n_idx = i
                if h in {"frame_idx", "frame_index", "frameid", "frame"}:
                    fi_idx = i
        for row in rows[start:]:
            if not row:
                continue
            try:
                n = int(float(row[n_idx]))
                fi = int(float(row[fi_idx]))
            except Exception:
                continue
            out.append((n, fi))
    out.sort(key=lambda x: x[0])
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Pack .npy features into NitzcheCLIP shards")
    ap.add_argument('--features-dir', required=True)
    ap.add_argument('--map-dir', required=True)
    ap.add_argument('--frames-prefix', default='./data/video_frames')
    ap.add_argument('--outdir', default='aic-24-BE/data/clip_features')
    ap.add_argument('--ext', default='.jpg')
    args = ap.parse_args()

    fdir = Path(args.features_dir)
    mdir = Path(args.map_dir)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    npy_files = sorted(fdir.glob('*.npy'))
    if not npy_files:
        print(f"No .npy files in {fdir}")
        return 2

    total = 0
    for npy in npy_files:
        video_id = npy.stem
        map_csv = mdir / f"{video_id}.csv"
        if not map_csv.exists():
            print(f"[SKIP] map not found for {video_id}: {map_csv}")
            continue
        order = load_map(map_csv)
        if not order:
            print(f"[SKIP] empty map for {video_id}")
            continue
        feats = np.load(npy)
        if feats.ndim == 1:
            feats = feats.reshape(1, -1)
        if feats.shape[0] != len(order):
            print(f"[WARN] feature count {feats.shape[0]} != mapped frames {len(order)} for {video_id}; will align to min")
        n = min(feats.shape[0], len(order))
        feats = feats[:n]
        file_paths = [
            os.path.join(args.frames_prefix, video_id, f"{fi}{args.ext}")
            for _, fi in order[:n]
        ]

        shard_path = outdir / f"{video_id}.pkl"
        with shard_path.open('wb') as f:
            pickle.dump((file_paths, feats.astype(np.float32)), f)
        total += 1
        print(f"Wrote shard: {shard_path} with {n} items")

    print(f"Done. Shards written: {total}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

