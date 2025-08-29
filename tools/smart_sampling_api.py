#!/usr/bin/env python3
"""
Notebook-friendly API to run smart video sampling and extract frames.

Example (in a Jupyter notebook):

from tools.smart_sampling_api import smart_sample_and_extract
csv_path = smart_sample_and_extract(
    videos_dir="example_dataset/Videos_L21_a",
    frames_dir="hcm-AI-challenge-2024-main/aic-24-BE/data/video_frames",
    strategy="clip-delta",  # or "shots"
    decode_fps=2.0,
    target_fps=1.0,
)
print("Selected indices at:", csv_path)

This wraps the existing CLI tools so it works without importing heavy
dependencies into your notebook kernel.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional


ROOT = Path(__file__).resolve().parents[1]


def _auto_device() -> str:
    # Prefer CUDA if available (best-effort, avoids importing torch in the kernel)
    if shutil.which("nvidia-smi") or os.environ.get("CUDA_VISIBLE_DEVICES"):
        return "cuda"
    return "cpu"


def smart_sample_and_extract(
    *,
    videos_dir: str,
    frames_dir: str,
    strategy: str = "clip-delta",
    # clip-delta params
    decode_fps: float = 2.0,
    target_fps: float = 1.0,
    min_gap_sec: float = 0.5,
    model: str = "ViT-B-32",
    # shots params
    shot_decode_fps: float = 10.0,
    shot_long_sec: float = 4.0,
    shot_per_long: int = 3,
    # shared
    exts: str = ".mp4,.avi,.mov,.mkv,.webm",
    out_csv: Optional[str] = None,
    device: Optional[str] = None,
) -> Path:
    """
    Run AI-driven sampling (clip-delta or shots) and extract only those frames
    into `frames_dir`, preserving original frame indices.

    Returns the path to the CSV listing selected indices (video_id,frame_idx).
    """
    videos_dir = str(videos_dir)
    frames_dir = str(frames_dir)
    device = device or _auto_device()
    out_csv = out_csv or "selected_frames.csv"
    out_csv_path = Path(out_csv).resolve()

    smart = ROOT / "tools" / "smart_sampling.py"
    crop = ROOT / "aic-24-BE" / "data_processing" / "crop_frame.py"

    cmd = [
        sys.executable, str(smart),
        "--videos-dir", videos_dir,
        "--strategy", strategy,
        "--device", device,
        "--out-csv", str(out_csv_path),
        "--exts", exts,
    ]
    if strategy == "clip-delta":
        cmd += [
            "--decode-fps", str(decode_fps),
            "--target-fps", str(target_fps),
            "--min-gap-sec", str(min_gap_sec),
            "--model", model,
        ]
    elif strategy == "shots":
        cmd += [
            "--shot-decode-fps", str(shot_decode_fps),
            "--shot-long-sec", str(shot_long_sec),
            "--shot-per-long", str(shot_per_long),
        ]
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    print("$", " ".join(cmd))
    subprocess.check_call(cmd, cwd=str(ROOT))

    # Extract exact frames by list
    extract_cmd = [
        sys.executable, str(crop),
        "--input-dir", videos_dir,
        "--output-dir", frames_dir,
        "--frame-list", str(out_csv_path),
    ]
    print("$", " ".join(extract_cmd))
    subprocess.check_call(extract_cmd, cwd=str(ROOT))

    return out_csv_path


__all__ = ["smart_sample_and_extract"]

