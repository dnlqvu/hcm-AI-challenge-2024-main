#!/usr/bin/env python3
"""
Colab-friendly CLIP feature extractor (no Docker).

Two backends:
  - lighthouse: uses vendor/lighthouse CLIPLoader (ffmpeg-based decode, resize,
    centercrop) at target fps = 1/clip_len, then encodes with open_clip.
  - fallback: simple midpoint sampling via OpenCV, then encode with open_clip.

Outputs per-video .npz files with array `features`, compatible with the
converter `tools/convert_hero_clip_to_shards.py`.

Usage:
  python tools/extract_clip_features_colab.py \
    --videos-dir example_dataset/Videos_L21_a \
    --outdir hero_colab_out/clip-vit_features \
    --clip-len 1.5

Then convert to shards and extract frames:
  python tools/convert_hero_clip_to_shards.py \
    --hero-clip-dir hero_colab_out/clip-vit_features \
    --media-info hcm-AI-challenge-2024-main/aic-24-BE/data/media-info \
    --clip-len 1.5 \
    --outdir hcm-AI-challenge-2024-main/aic-24-BE/data/clip_features \
    --emit-frame-list selected_frames_from_clip.csv

Finally, use crop_frame.py --frame-list and build-model-from-shards.
"""
from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
import sys
from typing import List, Tuple, Optional

import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm


def list_videos(input_dir: Path, exts: Tuple[str, ...]) -> List[Path]:
    return [p for p in sorted(input_dir.iterdir()) if p.is_file() and p.suffix.lower() in exts]


def probe_video(path: Path) -> Tuple[int, float, float]:
    cap = cv2.VideoCapture(path.as_posix())
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    if fps <= 0 or math.isnan(fps):
        fps = 25.0
    duration = float(total) / max(1e-6, fps)
    cap.release()
    return total, fps, duration


def read_frame(cap: cv2.VideoCapture, idx: int):
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ret, frame = cap.read()
    if not ret:
        return None
    return frame


def load_clip(model_name: str = "ViT-B-32", device: str = "cpu", pretrained: str = "laion2b_s34b_b79k"):
    import open_clip  # type: ignore

    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name, pretrained=pretrained, device=device
    )
    model.eval()
    return model, preprocess


def try_lighthouse_loader(target_fps: float) -> Optional[object]:
    """Return a configured CLIPLoader if lighthouse is available; else None."""
    try:
        from lighthouse.frame_loaders.clip_loader import CLIPLoader  # type: ignore
    except Exception:
        return None
    # For CLIPLoader: assert clip_len == 1/framerate
    clip_len = 1.0 / max(1e-6, target_fps)
    loader = CLIPLoader(
        clip_len=clip_len,
        framerate=target_fps,
        size=224,
        device="cpu",
        centercrop=True,
    )
    return loader


def encode_frames(model, preprocess, frames_bgr: List[np.ndarray], device: str) -> np.ndarray:
    imgs = []
    for fr in frames_bgr:
        img = Image.fromarray(fr[:, :, ::-1])  # BGR -> RGB
        imgs.append(preprocess(img))
    with torch.no_grad():
        batch = torch.stack(imgs).to(device)
        feats = model.encode_image(batch)
        feats = feats / feats.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    return feats.cpu().float().numpy()


def main() -> int:
    p = argparse.ArgumentParser(description="Extract CLIP (ViT-B/32) features per clip_len seconds without Docker")
    p.add_argument("--videos-dir", required=True)
    p.add_argument("--outdir", default="hero_colab_out/clip-vit_features")
    p.add_argument("--clip-len", type=float, default=1.5, help="Seconds per feature")
    p.add_argument("--model", default="ViT-B-32")
    p.add_argument("--pretrained", default="laion2b_s34b_b79k", help="open_clip pretrained tag or hf-hub:<repo>")
    p.add_argument("--device", default=("cuda" if torch.cuda.is_available() else "cpu"))
    p.add_argument("--exts", default=".mp4,.avi,.mov,.mkv,.webm")
    p.add_argument("--use-lighthouse", action="store_true", help="Deprecated: use --backend lighthouse-clip")
    p.add_argument("--backend", choices=["auto", "clip", "lighthouse-clip", "slowfast"], default="auto",
                   help="Decoding backend: OpenCV midpoints, Lighthouse CLIPLoader, or Lighthouse SlowFastLoader")
    p.add_argument("--frames-per-clip", type=int, default=8, help="When using slowfast: number of frames to encode per clip (averaged)")
    args = p.parse_args()

    in_dir = Path(args.videos_dir)
    out_dir = Path(args.outdir)
    out_dir.mkdir(parents=True, exist_ok=True)
    exts = tuple(e.strip().lower() for e in args.exts.split(",") if e.strip())

    # Make vendor/lighthouse importable if present
    repo_root = Path(__file__).resolve().parents[1]
    lh_path = repo_root / "vendor" / "lighthouse-main"
    if lh_path.exists() and lh_path.as_posix() not in sys.path:
        sys.path.insert(0, lh_path.as_posix())

    model, preprocess = load_clip(args.model, args.device, args.pretrained)
    target_fps = 1.0 / max(1e-6, args.clip_len)
    # Select backend
    loader = None
    backend = args.backend
    if args.use_lighthouse and backend == "auto":
        backend = "lighthouse-clip"
    if backend == "lighthouse-clip":
        loader = try_lighthouse_loader(target_fps)

    vids = list_videos(in_dir, exts)
    if not vids:
        print(f"No videos found in {in_dir}")
        return 2

    for vp in vids:
        vid = vp.stem
        if backend == "slowfast":
            # Use Lighthouse SlowFastLoader and aggregate CLIP over sampled frames per clip
            try:
                from lighthouse.frame_loaders.slowfast_loader import SlowFastLoader  # type: ignore
            except Exception:
                print(f"[WARN] lighthouse slowfast not available; falling back to OpenCV midpoints for {vid}")
                backend = "clip"
            else:
                sf_loader = SlowFastLoader(
                    clip_len=float(args.clip_len),
                    framerate=30,
                    size=224,
                    device="cpu",
                    centercrop=True,
                )
                video_tensor = sf_loader(vp.as_posix())
                if video_tensor is None or len(video_tensor) == 0:
                    print(f"[SKIP] {vid}: slowfast decode failed")
                    continue
                # video_tensor: [num_clips, T, H, W, C] uint8
                num_clips = int(video_tensor.shape[0])
                feats_all = []
                for ci in range(num_clips):
                    clip_tensor = video_tensor[ci]  # [T, H, W, C]
                    T = int(clip_tensor.shape[0])
                    if T <= 0:
                        continue
                    # Sample frames_per_clip evenly
                    k = max(1, int(args.frames_per_clip))
                    idxs = np.linspace(0, T - 1, k).round().astype(int)
                    frames_bgr = [
                        cv2.cvtColor(clip_tensor[i].numpy(), cv2.COLOR_RGB2BGR) for i in idxs
                    ]
                    clip_feats = encode_frames(model, preprocess, frames_bgr, args.device)
                    # Average pool frame features
                    clip_feat = clip_feats.mean(axis=0, keepdims=True)
                    feats_all.append(clip_feat)
                if not feats_all:
                    print(f"[SKIP] {vid}: no clips after sampling")
                    continue
                feats = np.vstack(feats_all)
                np.savez(out_dir / f"{vid}.npz", features=feats)
                print(f"Wrote {out_dir / (vid + '.npz')} with {feats.shape[0]} clip features [slowfast]")
                continue

        if loader is not None and backend == "lighthouse-clip":
            # Use lighthouse CLIPLoader: returns tensor [N, C, H, W]
            video_tensor = loader(vp.as_posix())
            if video_tensor is None or len(video_tensor) == 0:
                print(f"[SKIP] {vid}: lighthouse decode failed")
                continue
            # Convert to numpy HWC images for preprocess
            video_np = (video_tensor.float().clamp(0, 255).permute(0, 2, 3, 1).cpu().numpy()).astype(np.uint8)
            frames_bgr = [cv2.cvtColor(fr, cv2.COLOR_RGB2BGR) for fr in video_np]
            feats = encode_frames(model, preprocess, frames_bgr, args.device)
            np.savez(out_dir / f"{vid}.npz", features=feats)
            print(f"Wrote {out_dir / (vid + '.npz')} with {feats.shape[0]} features [lighthouse]")
        else:
            # Fallback midpoint sampling
            total, fps, duration = probe_video(vp)
            cap = cv2.VideoCapture(vp.as_posix())
            times = []
            t = 0.5 * args.clip_len
            while t < duration:
                times.append(t)
                t += args.clip_len
            if not times:
                cap.release()
                print(f"[SKIP] {vid}: too short for clip_len={args.clip_len}")
                continue
            frames_bgr: List[np.ndarray] = []
            for tsec in times:
                idx = int(round(tsec * fps))
                idx = max(0, min(total - 1, idx))
                fr = read_frame(cap, idx)
                if fr is None:
                    continue
                frames_bgr.append(fr)
            cap.release()
            if not frames_bgr:
                print(f"[SKIP] {vid}: decode failed")
                continue
            feats = encode_frames(model, preprocess, frames_bgr, args.device)
            np.savez(out_dir / f"{vid}.npz", features=feats)
            print(f"Wrote {out_dir / (vid + '.npz')} with {feats.shape[0]} features (fps≈{fps:.2f})")

    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
