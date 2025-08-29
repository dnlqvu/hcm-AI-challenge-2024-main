#!/usr/bin/env python3
"""
Convert HERO CLIP feature outputs (.npz per video) into NitzcheCLIP shards and
optionally emit the exact frame indices needed for image extraction.

Inputs:
  --hero-clip-dir  Folder with HERO CLIP outputs (.npz) per video
  --media-info     Folder with media-info/{video_id}.json to read fps (fallback 25)
  --clip-len       Seconds per feature used by HERO (e.g., 1.5 or 2.0)
  --frames-prefix  Prefix for frame image paths in shards (default './data/video_frames')
  --outdir         Output folder for shards (default aic-24-BE/data/clip_features)
  --emit-frame-list  Path to write CSV of 'video_id,frame_idx' (for crop_frame.py --frame-list)

Behavior:
  - For each {video_id}.npz, load array 'features' of shape [N, D].
  - Compute timestamps t_i = (i + 0.5) * clip_len.
  - Map to original frame indices: frame_idx = int(round(t_i * fps)).
  - Build file paths: '{frames-prefix}/{video_id}/{frame_idx}.jpg'.
  - Write shard '{outdir}/{video_id}.pkl' containing (file_path_list, features_float32).
  - Optionally append 'video_id,frame_idx' lines to --emit-frame-list to pre-extract frames.

Usage:
  python tools/convert_hero_clip_to_shards.py \
    --hero-clip-dir /output/clip-vit_features \
    --media-info hcm-AI-challenge-2024-main/aic-24-BE/data/media-info \
    --clip-len 1.5 \
    --outdir hcm-AI-challenge-2024-main/aic-24-BE/data/clip_features \
    --emit-frame-list selected_frames_from_hero.csv

Then extract frames:
  python hcm-AI-challenge-2024-main/aic-24-BE/data_processing/crop_frame.py \
    --input-dir <videos_dir> --output-dir hcm-AI-challenge-2024-main/aic-24-BE/data/video_frames \
    --frame-list selected_frames_from_hero.csv

Finally, load shards with NitzcheCLIP or rebuild model pickle via setup.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import List, Optional

import numpy as np


def load_fps(media_info_dir: Path, video_id: str) -> float:
    ji = media_info_dir / f"{video_id}.json"
    try:
        with ji.open("r", encoding="utf-8") as f:
            data = json.load(f)
        fps = float(data.get("fps", 25))
        if fps <= 0:
            return 25.0
        return fps
    except Exception:
        return 25.0


def main() -> int:
    ap = argparse.ArgumentParser(description="Convert HERO CLIP .npz to NitzcheCLIP shards and optional frame list")
    ap.add_argument("--hero-clip-dir", required=True)
    ap.add_argument("--media-info", required=True)
    ap.add_argument("--clip-len", type=float, required=True, help="Seconds per feature used by HERO CLIP extractor")
    ap.add_argument("--frames-prefix", default="./data/video_frames")
    ap.add_argument("--outdir", default="aic-24-BE/data/clip_features")
    ap.add_argument("--emit-frame-list", default=None, help="CSV path to write 'video_id,frame_idx' lines")
    args = ap.parse_args()

    clip_dir = Path(args.hero_clip_dir)
    media_info_dir = Path(args.media_info)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    npz_files = sorted([p for p in clip_dir.glob("*.npz")])
    if not npz_files:
        print(f"[ERROR] No .npz files found in {clip_dir}")
        return 2

    frame_list_writer: Optional[csv.writer] = None
    f_csv = None
    if args.emit_frame_list:
        f_csv = open(args.emit_frame_list, "w", encoding="utf-8", newline="")
        frame_list_writer = csv.writer(f_csv)

    converted = 0
    for npz_path in npz_files:
        vid = npz_path.stem
        try:
            with np.load(npz_path) as z:
                feats = z.get("features")
                if feats is None:
                    # Some variants might store arrays under different keys
                    # Try the first array
                    if len(z.files) == 0:
                        print(f"[SKIP] Empty npz: {npz_path}")
                        continue
                    feats = z[z.files[0]]
        except Exception as e:
            print(f"[SKIP] Failed to read {npz_path}: {e}")
            continue

        if feats.ndim == 1:
            feats = feats.reshape(1, -1)
        n = feats.shape[0]
        if n == 0:
            print(f"[SKIP] No features in {npz_path}")
            continue

        fps = load_fps(media_info_dir, vid)
        # Midpoints of each clip
        times = (np.arange(n, dtype=np.float64) + 0.5) * float(args.clip_len)
        frame_ids = np.rint(times * fps).astype(int)
        # Clamp to non-negative
        frame_ids = np.maximum(frame_ids, 0)

        file_paths: List[str] = [
            os.path.join(args.frames_prefix, vid, f"{fi}.jpg") for fi in frame_ids
        ]
        out_pkl = outdir / f"{vid}.pkl"
        try:
            import pickle

            with out_pkl.open("wb") as f:
                pickle.dump((file_paths, feats.astype(np.float32)), f)
            converted += 1
            print(f"Wrote shard: {out_pkl} with {n} items (fps={fps})")
        except Exception as e:
            print(f"[SKIP] Failed to write {out_pkl}: {e}")
            continue

        if frame_list_writer is not None:
            for fi in frame_ids:
                frame_list_writer.writerow([vid, int(fi)])

    if f_csv is not None:
        f_csv.close()
        print(f"Frame list written: {args.emit_frame_list}")

    print(f"Done. Converted videos: {converted}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

