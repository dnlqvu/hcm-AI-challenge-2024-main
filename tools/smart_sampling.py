#!/usr/bin/env python3
"""
Smart video frame sampling using AI signals.

Strategies implemented:
  - clip-delta: decode frames at a low FPS (e.g., 2 fps), compute CLIP ViT-B/32
    image embeddings, and keep frames where semantic change exceeds an adaptive
    threshold to meet a target budget (target fps). A minimum time gap (NMS)
    enforces temporal diversity. Outputs selected original frame indices.
  - shots: detect shot boundaries (TransNetV2 if available; fallback to a
    content-based heuristic) and keep representative keyframes (center per
    shot, plus extra samples for long shots).

Outputs:
  - CSV lines of: video_id,frame_idx (one per line)

Example:
  python tools/smart_sampling.py \
    --videos-dir example_dataset/Videos_L21_a \
    --strategy clip-delta \
    --decode-fps 2.0 \
    --target-fps 1.0 \
    --out-csv selected_frames.csv

Shot-aware example:
  python tools/smart_sampling.py \
    --videos-dir example_dataset/Videos_L21_a \
    --strategy shots \
    --shot-decode-fps 10.0 \
    --shot-long-sec 4.0 \
    --out-csv selected_frames.csv

Then extract those exact indices into frame images:
  python hcm-AI-challenge-2024-main/aic-24-BE/data_processing/crop_frame.py \
    --input-dir example_dataset/Videos_L21_a \
    --output-dir hcm-AI-challenge-2024-main/aic-24-BE/data/video_frames \
    --frame-list selected_frames.csv

Dependencies:
  pip install open_clip_torch pillow tqdm opencv-python torch numpy

Note: If open_clip is unavailable, attempts to fallback to the 'clip' package.
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import sys
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
import cv2
import torch


# Try open_clip first, fall back to clip
def load_clip(model_name: str = "ViT-B-32", device: str = "cpu", pretrained: str = "laion2b_s34b_b79k"):
    try:
        import open_clip  # type: ignore

        model, _, preprocess = open_clip.create_model_and_transforms(
            model_name,
            pretrained=pretrained,
            device=device,
        )
        model.eval()
        use_open_clip = True
        return model, preprocess, use_open_clip
    except Exception:
        try:
            import clip  # type: ignore
            model, preprocess = clip.load("ViT-B/32", device=device, jit=False)
            model.eval()
            use_open_clip = False
            return model, preprocess, use_open_clip
        except Exception as e:
            raise RuntimeError(
                "Failed to import open_clip or clip. Install one: \n"
                "  pip install open_clip_torch  OR  pip install git+https://github.com/openai/CLIP.git\n"
                f"Original error: {e}"
            )


def ensure_dir(path: str) -> None:
    if not os.path.exists(path):
        os.makedirs(path)


def list_videos(input_dir: str, exts: Tuple[str, ...], recursive: bool = False) -> List[str]:
    out: List[str] = []
    if not recursive:
        for name in sorted(os.listdir(input_dir)):
            p = os.path.join(input_dir, name)
            if os.path.isfile(p) and any(name.lower().endswith(e) for e in exts):
                out.append(p)
        return out
    # recursive walk
    for root, _, files in os.walk(input_dir):
        for name in sorted(files):
            if any(name.lower().endswith(e) for e in exts):
                out.append(os.path.join(root, name))
    return out


def basename_no_ext(path: str) -> str:
    base = os.path.basename(path)
    return os.path.splitext(base)[0]


@dataclass
class VideoInfo:
    fps: float
    total_frames: int
    duration_sec: float


def probe_video(path: str) -> VideoInfo:
    cap = cv2.VideoCapture(path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    if fps <= 0 or math.isnan(fps):
        fps = 25.0
    duration = float(total_frames) / max(1e-6, fps)
    cap.release()
    return VideoInfo(fps=fps, total_frames=total_frames, duration_sec=duration)


def sample_decode_indices(info: VideoInfo, decode_fps: float) -> List[int]:
    step = int(round(info.fps / max(1e-6, decode_fps)))
    step = max(1, step)
    return list(range(0, info.total_frames, step))


def read_frame_at(cap: cv2.VideoCapture, index: int) -> Optional[np.ndarray]:
    cap.set(cv2.CAP_PROP_POS_FRAMES, index)
    ret, frame = cap.read()
    if not ret:
        return None
    return frame


def to_pil_image(arr: np.ndarray):
    from PIL import Image

    if arr.ndim == 3 and arr.shape[2] == 3:
        return Image.fromarray(arr[:, :, ::-1])  # BGR -> RGB
    raise ValueError("Expected HxWx3 BGR frame array")


def encode_images_clip(
    model,
    preprocess,
    frames_bgr: Sequence[np.ndarray],
    device: str,
    use_open_clip: bool,
) -> np.ndarray:
    imgs = [to_pil_image(fr) for fr in frames_bgr]
    with torch.no_grad():
        if use_open_clip:
            import open_clip  # type: ignore

            batch = torch.stack([preprocess(img) for img in imgs]).to(device)
            feats = model.encode_image(batch)
        else:
            import clip  # type: ignore

            batch = torch.stack([preprocess(img) for img in imgs]).to(device)
            feats = model.encode_image(batch)
        feats = feats.float()
        feats = feats / feats.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    return feats.cpu().numpy()


def select_by_delta(
    embs: np.ndarray,
    decode_indices: List[int],
    info: VideoInfo,
    target_fps: float,
    min_gap_sec: float,
) -> List[int]:
    if len(embs) == 0:
        return []
    if len(embs) == 1:
        return [decode_indices[0]]

    # Cosine similarity deltas vs previous embedding
    dots = np.sum(embs[1:] * embs[:-1], axis=1)
    deltas = 1.0 - np.clip(dots, -1.0, 1.0)

    # Estimate budget in kept frames
    target_count = max(1, int(round(info.duration_sec * target_fps)))

    # Rank candidates by delta descending (skip the first frame which we always keep)
    cand = list(range(1, len(decode_indices)))
    cand.sort(key=lambda i: float(deltas[i - 1]), reverse=True)

    # Greedy pick with temporal NMS
    kept = [0]  # always keep first decoded frame
    kept_times = [decode_indices[0] / max(1e-6, info.fps)]
    min_gap = float(min_gap_sec)
    for i in cand:
        if len(kept) >= target_count:
            break
        t = decode_indices[i] / max(1e-6, info.fps)
        if all(abs(t - kt) >= min_gap for kt in kept_times):
            kept.append(i)
            kept_times.append(t)

    kept.sort()
    # Map decode positions back to original frame indices
    return [decode_indices[i] for i in kept]


def write_selected(out_csv: str, video_id: str, frames: Iterable[int]) -> None:
    with open(out_csv, "a", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        for fr in frames:
            w.writerow([video_id, int(fr)])


def detect_shots_transnet(frames_bgr: List[np.ndarray]) -> Optional[List[Tuple[int, int]]]:
    """Detect shots using TransNetV2 if installed. Returns list of (start_idx, end_idx)
    in the frame sequence provided. Returns None if unavailable or failed.
    """
    try:
        from transnetv2 import TransNetV2  # type: ignore
    except Exception:
        return None
    try:
        model = TransNetV2()
        # Some builds require .load() to fetch weights
        try:
            model.load()
        except Exception:
            pass
        # Newer API may offer predict_frames -> predictions
        preds = model.predict_frames(frames_bgr)
        # Convert predictions to scenes
        try:
            scenes = model.predictions_to_scenes(preds)
        except Exception:
            # Some versions return (preds, scenes)
            return None
        # scenes as list[[start,end], ...]
        out: List[Tuple[int, int]] = []
        for s, e in scenes:
            out.append((int(s), int(e)))
        return out
    except Exception:
        return None


def detect_shots_naive(frames_bgr: List[np.ndarray], thresh: float = 0.35) -> List[Tuple[int, int]]:
    """Simple content-based shot detection using HSV histogram deltas.
    Returns list of (start_idx, end_idx) over the provided frames.
    """
    if not frames_bgr:
        return []
    # Compute normalized HSV histograms per frame
    hists: List[np.ndarray] = []
    for fr in frames_bgr:
        hsv = cv2.cvtColor(fr, cv2.COLOR_BGR2HSV)
        h = cv2.calcHist([hsv], [0, 1], None, [16, 16], [0, 180, 0, 256])
        h = h.astype('float32')  # Convert to float32 before normalize
        h = cv2.normalize(h, h).flatten()
        hists.append(h)
    # Shot boundaries when histogram distance > thresh
    boundaries = [0]
    for i in range(1, len(hists)):
        d = cv2.compareHist(hists[i - 1], hists[i], cv2.HISTCMP_BHATTACHARYYA)
        if d > thresh:
            boundaries.append(i)
    boundaries.append(len(hists))
    shots: List[Tuple[int, int]] = []
    for i in range(len(boundaries) - 1):
        s = boundaries[i]
        e = max(s, boundaries[i + 1] - 1)
        shots.append((s, e))
    return shots


def run_shots(
    videos_dir: str,
    out_csv: str,
    shot_decode_fps: float,
    shot_long_sec: float,
    shot_per_long: int,
    exts: Tuple[str, ...],
    recursive: bool = False,
) -> None:
    vids = list_videos(videos_dir, exts, recursive=recursive)
    if not vids:
        print(f"[ERROR] No videos found in {videos_dir} with extensions {exts}")
        return
    # Truncate output
    open(out_csv, "w").close()
    for path in vids:
        vid = basename_no_ext(path)
        info = probe_video(path)
        decode_indices = sample_decode_indices(info, shot_decode_fps)
        if not decode_indices:
            print(f"[WARN] {vid}: no decodable frames; skipping")
            continue
        cap = cv2.VideoCapture(path)
        frames: List[np.ndarray] = []
        for idx in decode_indices:
            fr = read_frame_at(cap, idx)
            if fr is None:
                break
            # Downscale to speed up shot detection & memory
            h, w = fr.shape[:2]
            scale = 256.0 / max(h, w)
            if scale < 1.0:
                fr = cv2.resize(fr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
            frames.append(fr)
        cap.release()
        if not frames:
            print(f"[WARN] {vid}: failed to decode; skipping")
            continue

        shots = detect_shots_transnet(frames)
        used = "transnetv2" if shots is not None else "naive"
        if shots is None:
            shots = detect_shots_naive(frames)
        if not shots:
            print(f"[WARN] {vid}: no shots detected; keeping first frame only")
            write_selected(out_csv, vid, [0])
            continue

        kept: List[int] = []
        for s, e in shots:
            # Convert shot frame idx → original frame index via decode_indices mapping
            span = max(1, e - s + 1)
            span_sec = span / max(1e-6, shot_decode_fps)
            if span_sec >= shot_long_sec and shot_per_long > 1:
                n = int(shot_per_long)
                for k in range(n):
                    rel = int(round(s + (k + 0.5) * span / n))
                    rel = min(max(rel, s), e)
                    kept.append(decode_indices[rel])
            else:
                mid = (s + e) // 2
                kept.append(decode_indices[mid])

        # Deduplicate & sort
        kept = sorted(set(int(x) for x in kept))
        write_selected(out_csv, vid, kept)
        print(f"[shots-{used}] {vid}: shots={len(shots)} kept={len(kept)}")


def run_clip_delta(
    videos_dir: str,
    out_csv: str,
    model_name: str,
    pretrained: str,
    device: str,
    decode_fps: float,
    target_fps: float,
    min_gap_sec: float,
    exts: Tuple[str, ...],
    recursive: bool = False,
) -> None:
    vids = list_videos(videos_dir, exts, recursive=recursive)
    if not vids:
        print(f"[ERROR] No videos found in {videos_dir} with extensions {exts}")
        return

    # Truncate output file
    open(out_csv, "w").close()

    print(f"Loading CLIP model {model_name} ({pretrained}) on {device} ...")
    model, preprocess, use_open_clip = load_clip(model_name, device, pretrained)

    for path in vids:
        vid = basename_no_ext(path)
        info = probe_video(path)
        decode_indices = sample_decode_indices(info, decode_fps)
        if not decode_indices:
            print(f"[WARN] {vid}: no decodable frames; skipping")
            continue

        cap = cv2.VideoCapture(path)
        frames_bgr: List[np.ndarray] = []
        for idx in decode_indices:
            fr = read_frame_at(cap, idx)
            if fr is None:
                break
            frames_bgr.append(fr)
        cap.release()

        if not frames_bgr:
            print(f"[WARN] {vid}: failed to decode; skipping")
            continue

        embs = encode_images_clip(model, preprocess, frames_bgr, device, use_open_clip)
        kept_indices = select_by_delta(
            embs, decode_indices, info, target_fps=target_fps, min_gap_sec=min_gap_sec
        )
        write_selected(out_csv, vid, kept_indices)
        print(
            f"[clip-delta] {vid}: src_fps={info.fps:.2f} dur={info.duration_sec:.1f}s "
            f"decode_fps={decode_fps} decoded={len(decode_indices)} kept={len(kept_indices)}"
        )


def main() -> int:
    p = argparse.ArgumentParser(description="Smart video sampling with AI models")
    p.add_argument("--videos-dir", type=str, required=True, help="Directory containing input videos")
    p.add_argument("--strategy", type=str, choices=["clip-delta", "shots"], default="clip-delta")
    p.add_argument("--decode-fps", type=float, default=2.0, help="Decode frames at this FPS for analysis")
    p.add_argument("--target-fps", type=float, default=1.0, help="Target average kept frames per second")
    p.add_argument("--min-gap-sec", type=float, default=0.5, help="Minimum temporal gap between kept frames")
    p.add_argument("--model", type=str, default="ViT-B-32", help="CLIP model name for clip-delta")
    p.add_argument("--pretrained", type=str, default="laion2b_s34b_b79k", help="open_clip pretrained tag (e.g., webli or hf-hub:<repo>)")
    p.add_argument("--device", type=str, default=("cuda" if torch.cuda.is_available() else "cpu"))
    p.add_argument("--out-csv", type=str, default="selected_frames.csv")
    p.add_argument("--exts", type=str, default=".mp4,.avi,.mov,.mkv,.webm", help="Comma-separated video extensions")
    p.add_argument("--recursive", action="store_true", default=True, help="Recursively search for videos under --videos-dir (default: on)")
    # Shot-aware params
    p.add_argument("--shot-decode-fps", type=float, default=10.0, help="Decode FPS for shot boundary detection")
    p.add_argument("--shot-long-sec", type=float, default=4.0, help="Consider shots >= this length as long (more samples)")
    p.add_argument("--shot-per-long", type=int, default=3, help="Samples per long shot (e.g., 3 → start/mid/end)")
    args = p.parse_args()

    exts = tuple(e.strip().lower() for e in args.exts.split(",") if e.strip())
    if args.strategy == "clip-delta":
        run_clip_delta(
            videos_dir=args.videos_dir,
            out_csv=args.out_csv,
            model_name=args.model,
            pretrained=args.pretrained,
            device=args.device,
            decode_fps=args.decode_fps,
            target_fps=args.target_fps,
            min_gap_sec=args.min_gap_sec,
            exts=exts,
            recursive=args.recursive,
        )
        print(f"Wrote selected frames -> {args.out_csv}")
        print(
            "Next: extract images with crop_frame.py --frame-list to keep original indices."
        )
        return 0
    elif args.strategy == "shots":
        run_shots(
            videos_dir=args.videos_dir,
            out_csv=args.out_csv,
            shot_decode_fps=args.shot_decode_fps,
            shot_long_sec=args.shot_long_sec,
            shot_per_long=args.shot_per_long,
            exts=exts,
            recursive=args.recursive,
        )
        print(f"Wrote selected frames -> {args.out_csv}")
        print(
            "Next: extract images with crop_frame.py --frame-list to keep original indices."
        )
        return 0
    else:
        print(f"[ERROR] Strategy not implemented: {args.strategy}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
