#!/usr/bin/env python3
"""
Rename keyframe images to original frame indices using map-keyframes CSVs.

Expected layouts:
  - Frames: aic-24-BE/data/video_frames/{video_id}/{filename}
  - Maps:   aic-24-BE/data/map-keyframes/{video_id}.csv  (or a single aggregated CSV)

CSV columns (flexible):
  - Must contain an integer column for keyframe order ("n") and one for original frame index ("frame_idx").
  - If names differ (e.g., "frame_index"), the script tries common variants.

Filename parsing:
  - By default, extracts the last integer from the filename stem as keyframe order n.
  - You can provide a custom regex via --pattern; the first captured integer is used.

Usage:
  python tools/rename_keyframes_from_map.py \
    --frames-dir hcm-AI-challenge-2024-main/aic-24-BE/data/video_frames \
    --map-dir    hcm-AI-challenge-2024-main/aic-24-BE/data/map-keyframes \
    [--pattern '(?P<n>\d+)$'] [--dry-run] [--allow-overwrite]
"""
from __future__ import annotations

import argparse
import csv
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def find_columns(header: List[str]) -> Tuple[Optional[int], Optional[int]]:
    lower = [h.strip().lower() for h in header]
    n_aliases = {"n", "keyframe", "kf", "idx", "index", "k"}
    f_aliases = {"frame_idx", "frameindex", "frame_index", "frameid", "frame", "original_frame", "orig_frame"}

    n_idx = None
    f_idx = None
    for i, h in enumerate(lower):
        if h in n_aliases and n_idx is None:
            n_idx = i
        if h in f_aliases and f_idx is None:
            f_idx = i
    # Fallbacks: first and last numeric-looking columns if ambiguous
    return n_idx, f_idx


def load_map_csv(path: Path) -> Dict[int, int]:
    mapping: Dict[int, int] = {}
    with path.open("r", encoding="utf-8") as f:
        rdr = csv.reader(f)
        rows = list(rdr)
        if not rows:
            return mapping
        header = rows[0]
        # Detect header if first row is non-numeric
        has_header = any(not c.isdigit() for c in header)
        start_idx = 1 if has_header else 0
        n_col = None
        fi_col = None
        if has_header:
            n_col, fi_col = find_columns(header)
        for row in rows[start_idx:]:
            if not row:
                continue
            try:
                if has_header and n_col is not None and fi_col is not None:
                    n = int(float(row[n_col]))
                    fi = int(float(row[fi_col]))
                else:
                    # assume first two columns are n, frame_idx
                    n = int(float(row[0]))
                    fi = int(float(row[1]))
            except Exception:
                continue
            mapping[n] = fi
    return mapping


def load_aggregated_map(path: Path, video_id: str) -> Dict[int, int]:
    mapping: Dict[int, int] = {}
    with path.open("r", encoding="utf-8") as f:
        rdr = csv.DictReader(f)
        # Require a column indicating video id
        vid_key = None
        for cand in ("video_id", "video", "vid", "name"):
            if cand in (rdr.fieldnames or []):
                vid_key = cand
                break
        if vid_key is None:
            return mapping
        # Flexible key names for n and frame_idx
        n_key = None
        fi_key = None
        for cand in ("n", "keyframe", "idx", "index", "k"):
            if cand in (rdr.fieldnames or []):
                n_key = cand
                break
        for cand in ("frame_idx", "frame_index", "frameid", "frame", "original_frame", "orig_frame"):
            if cand in (rdr.fieldnames or []):
                fi_key = cand
                break
        if n_key is None or fi_key is None:
            return mapping
        for row in rdr:
            if (row.get(vid_key) or "").split(".")[0] != video_id:
                continue
            try:
                n = int(float(row[n_key]))
                fi = int(float(row[fi_key]))
            except Exception:
                continue
            mapping[n] = fi
    return mapping


def extract_n_from_name(name: str, pattern: Optional[re.Pattern]) -> Optional[int]:
    stem = Path(name).stem
    if pattern is not None:
        m = pattern.search(stem)
        if m:
            # Prefer named group 'n', else first captured group; else last digit run
            if "n" in m.groupdict() and m.group("n"):
                return int(m.group("n"))
            for g in m.groups():
                if g and g.isdigit():
                    return int(g)
    # default: last run of digits in stem
    m = re.search(r"(\d+)(?!.*\d)", stem)
    if m:
        return int(m.group(1))
    # if the whole stem is a number
    if stem.isdigit():
        return int(stem)
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description="Rename keyframe images to original frame_idx using map-keyframes CSVs.")
    ap.add_argument("--frames-dir", required=True, help="Root directory of video frames (video_id subfolders)")
    ap.add_argument("--map-dir", required=True, help="Directory containing map CSVs (one per video or an aggregated CSV)")
    ap.add_argument("--pattern", default=None, help="Optional regex to extract keyframe order n from filename (e.g., '(?P<n>\\d+)$')")
    ap.add_argument("--dry-run", action="store_true", help="Print actions without renaming files")
    ap.add_argument("--allow-overwrite", action="store_true", help="Overwrite existing target filenames if they exist")
    ap.add_argument("--videos", nargs="*", default=None, help="Optional list of video_ids to process (defaults to all folders)")
    args = ap.parse_args()

    frames_root = Path(args.frames_dir)
    map_root = Path(args.map_dir)
    pattern = re.compile(args.pattern) if args.pattern else None

    if not frames_root.is_dir():
        print(f"[ERROR] frames dir not found: {frames_root}")
        return 2
    if not map_root.is_dir():
        print(f"[ERROR] map dir not found: {map_root}")
        return 2

    # Candidate aggregated CSV (single file in map dir)
    aggregated_csv: Optional[Path] = None
    csvs = list(map_root.glob("*.csv"))
    if len(csvs) == 1:
        aggregated_csv = csvs[0]

    videos: List[Path] = []
    if args.videos:
        for vid in args.videos:
            p = frames_root / vid
            if p.is_dir():
                videos.append(p)
            else:
                print(f"[WARN] video folder not found: {p}")
    else:
        videos = [p for p in frames_root.iterdir() if p.is_dir()]

    total_renamed = 0
    total_skipped = 0
    total_missing_map = 0

    for vdir in videos:
        video_id = vdir.name
        mapping: Dict[int, int] = {}
        map_csv = map_root / f"{video_id}.csv"
        if map_csv.exists():
            mapping = load_map_csv(map_csv)
        elif aggregated_csv:
            mapping = load_aggregated_map(aggregated_csv, video_id)
        else:
            # try any csv that contains video_id in filename
            candidates = [c for c in csvs if video_id in c.name]
            if len(candidates) == 1:
                mapping = load_map_csv(candidates[0])
        if not mapping:
            print(f"[WARN] No mapping found for {video_id}")
            total_missing_map += 1
            continue

        # Iterate files
        for file in sorted(vdir.iterdir()):
            if not file.is_file():
                continue
            n = extract_n_from_name(file.name, pattern)
            if n is None:
                total_skipped += 1
                continue
            if n not in mapping:
                total_skipped += 1
                continue
            frame_idx = mapping[n]
            target = file.with_name(f"{frame_idx}{file.suffix}")
            if target == file:
                continue
            if target.exists() and not args.allow_overwrite:
                print(f"[SKIP] target exists: {target}")
                total_skipped += 1
                continue
            print(f"{'[DRY] ' if args.dry_run else ''}rename {file} -> {target}")
            if not args.dry_run:
                try:
                    os.replace(str(file), str(target))
                    total_renamed += 1
                except Exception as e:
                    print(f"[ERROR] {e}")
                    total_skipped += 1

    print(f"Done. Renamed: {total_renamed}, Skipped: {total_skipped}, Videos without map: {total_missing_map}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

