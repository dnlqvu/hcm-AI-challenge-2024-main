#!/usr/bin/env python3
"""
Automate wiring the example_dataset into the AIC‑25 backend.

Steps:
 1) Copy media-info → aic-24-BE/data/media-info
 2) Copy keyframes → aic-24-BE/data/video_frames
 3) Rename keyframes from keyframe order (n) to original frame_idx using map-keyframes CSVs
 4) Pack per-video .npy features into NitzcheCLIP shards (pickles)
 5) Build the model pickle and update .env to point to it

Assumes extracted example_dataset with subfolders:
  - media-info, keyframes, map-keyframes, clip-features-32

Usage:
  python tools/setup_example_dataset.py \
    --example-dir example_dataset \
    --be-dir hcm-AI-challenge-2024-main/aic-24-BE \
    [--model-name clip_vit_b32_nitzche.pkl]
"""
from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path


def copy_tree(src: Path, dst: Path):
    if not src.exists():
        print(f"[SKIP] Missing source: {src}")
        return
    dst.mkdir(parents=True, exist_ok=True)
    print(f"Copying {src} -> {dst}")
    # dirs_exist_ok requires py3.8+
    shutil.copytree(src, dst, dirs_exist_ok=True)


def patch_env(env_path: Path, model_name: str):
    lines = []
    if env_path.exists():
        lines = env_path.read_text(encoding='utf-8').splitlines()
    updated = []
    saw_model_path = False
    saw_model_16 = False
    for line in lines:
        if line.strip().startswith('MODEL_PATH='):
            updated.append('MODEL_PATH="./models/"')
            saw_model_path = True
        elif line.strip().startswith('MODEL_16='):
            updated.append(f'MODEL_16="{model_name}"')
            saw_model_16 = True
        else:
            updated.append(line)
    if not saw_model_path:
        updated.append('MODEL_PATH="./models/"')
    if not saw_model_16:
        updated.append(f'MODEL_16="{model_name}"')
    env_path.write_text("\n".join(updated) + "\n", encoding='utf-8')
    print(f"Updated {env_path} with MODEL_16={model_name}")


def build_model_pickle(be_dir: Path, model_name: str):
    # Import NitzcheCLIP and save the model using feature shards
    sys.path.insert(0, str(be_dir))
    from nitzche_clip import NitzcheCLIP  # type: ignore
    feature_dir = be_dir / 'data' / 'clip_features'
    models_dir = be_dir / 'models'
    models_dir.mkdir(parents=True, exist_ok=True)
    print(f"Building model from {feature_dir} ...")
    m = NitzcheCLIP(str(feature_dir))
    out_path = models_dir / model_name
    m.save(str(out_path))
    print(f"Saved model: {out_path}")


def main() -> int:
    ap = argparse.ArgumentParser(description='Setup example_dataset into aic-24-BE')
    ap.add_argument('--example-dir', default='example_dataset')
    ap.add_argument('--be-dir', default='hcm-AI-challenge-2024-main/aic-24-BE')
    ap.add_argument('--model-name', default='clip_vit_b32_nitzche.pkl')
    ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args()

    exdir = Path(args.example_dir)
    bedir = Path(args.be_dir)
    assert exdir.exists(), f"example dir missing: {exdir}"
    assert bedir.exists(), f"backend dir missing: {bedir}"

    media_src = exdir / 'media-info'
    frames_src = exdir / 'keyframes'
    maps_src = exdir / 'map-keyframes'
    feats_src = exdir / 'clip-features-32'

    media_dst = bedir / 'data' / 'media-info'
    frames_dst = bedir / 'data' / 'video_frames'
    feats_dst = bedir / 'data' / 'clip_features'

    if not args.dry_run:
        copy_tree(media_src, media_dst)
        copy_tree(frames_src, frames_dst)

        # Rename frames using maps
        cmd = [
            sys.executable,
            str(Path(__file__).with_name('rename_keyframes_from_map.py')),
            '--frames-dir', str(frames_dst),
            '--map-dir', str(maps_src),
            '--pattern', r'(?P<n>\d+)$'
        ]
        print('Renaming keyframes...', ' '.join(cmd))
        subprocess.check_call(cmd)

        # Pack features from .npy
        feats_dst.mkdir(parents=True, exist_ok=True)
        cmd = [
            sys.executable,
            str(Path(__file__).with_name('pack_features_from_npy.py')),
            '--features-dir', str(feats_src),
            '--map-dir', str(maps_src),
            '--frames-prefix', './data/video_frames',
            '--outdir', str(feats_dst),
            '--ext', '.jpg'
        ]
        print('Packing .npy features...', ' '.join(cmd))
        subprocess.check_call(cmd)

        # Build model pickle and update .env
        build_model_pickle(bedir, args.model_name)
        patch_env(bedir / '.env', args.model_name)

    print('Done. Next: start the API with `uvicorn main:app --reload --host 0.0.0.0 --port 8000` in aic-24-BE')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
