import os
import cv2
import csv
import math
import argparse
import multiprocessing
from typing import Dict, Iterable, List, Tuple, Set
from tqdm.contrib.concurrent import thread_map


def ensure_dir(path: str) -> None:
    if not os.path.exists(path):
        os.makedirs(path)


def list_videos(input_dir: str, exts: Tuple[str, ...]) -> List[str]:
    out: List[str] = []
    for name in sorted(os.listdir(input_dir)):
        if any(name.lower().endswith(e) for e in exts):
            out.append(os.path.join(input_dir, name))
    return out


def basename_no_ext(path: str) -> str:
    base = os.path.basename(path)
    return os.path.splitext(base)[0]


def parse_frame_list(frame_list_path: str) -> Dict[str, Set[int]]:
    """Parse a CSV listing frames to extract.

    Supported formats per line (header optional):
      - video_id,frame_idx
      - video_id,frame1,frame2,frame3,...
    video_id should match the basename (no extension) of the video file.
    """
    mapping: Dict[str, Set[int]] = {}
    with open(frame_list_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            # Skip header heuristically
            if row[0].lower() in {"video", "video_id", "vid"}:
                continue
            vid = row[0].strip()
            frames: Set[int] = mapping.setdefault(vid, set())
            for x in row[1:]:
                x = x.strip()
                if not x:
                    continue
                try:
                    frames.add(int(float(x)))
                except Exception:
                    continue
    return mapping


def extract_uniform_frames(video_path: str, out_dir: str, sampling_fps: float) -> None:
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or math.isnan(fps):
        fps = 25.0
    # Avoid zero or negative interval
    interval = max(1, int(round(fps / max(1e-6, sampling_fps))))

    vid = basename_no_ext(video_path)
    video_frame_dir = os.path.join(out_dir, vid)
    ensure_dir(video_frame_dir)

    frame_idx = 0
    saved = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % interval == 0:
            out_path = os.path.join(video_frame_dir, f"{frame_idx}.jpg")
            cv2.imwrite(out_path, frame)
            saved += 1
        frame_idx += 1

    cap.release()
    # Save sampling info for traceability
    info_csv = os.path.join(video_frame_dir, "fps_info.csv")
    with open(info_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["fps", "frame_interval", "total_frames", "saved_frames"]) 
        w.writerow([fps, interval, total_frames, saved])
    print(f"[uniform] {vid}: src_fps={fps:.3f} interval={interval} total={total_frames} saved={saved}")


def extract_specific_frames(video_path: str, out_dir: str, frames_to_keep: Iterable[int]) -> None:
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or math.isnan(fps):
        fps = 25.0

    vid = basename_no_ext(video_path)
    video_frame_dir = os.path.join(out_dir, vid)
    ensure_dir(video_frame_dir)

    # Clamp and sort unique frames
    clamp = lambda x: int(max(0, min(total_frames - 1, int(x))))
    unique_sorted = sorted({clamp(i) for i in frames_to_keep})

    saved = 0
    last_pos = -1
    for idx in unique_sorted:
        if idx == last_pos:
            continue
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue
        out_path = os.path.join(video_frame_dir, f"{idx}.jpg")
        cv2.imwrite(out_path, frame)
        saved += 1
        last_pos = idx

    cap.release()
    info_csv = os.path.join(video_frame_dir, "fps_info.csv")
    with open(info_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["fps", "total_frames", "saved_frames"]) 
        w.writerow([fps, total_frames, saved])
    print(f"[list] {vid}: src_fps={fps:.3f} requested={len(unique_sorted)} saved={saved}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Extract frames (uniform or from list) while keeping original frame indices.")
    ap.add_argument("--input-dir", type=str, default="./raw/vids/video", help="Directory containing input videos")
    ap.add_argument("--output-dir", type=str, default="./raw/frames/", help="Directory to write extracted frames")
    ap.add_argument("--fps", type=float, default=1.0, help="Uniform sampling rate in frames per second (ignored if --frame-list provided)")
    ap.add_argument("--frame-list", type=str, help="CSV file of frames to keep: video_id,frame or video_id,frame1,frame2,...")
    ap.add_argument("--workers", type=int, default=max(1, multiprocessing.cpu_count()), help="Max parallel worker threads for decoding")
    ap.add_argument("--exts", type=str, default=".mp4,.avi,.mov,.mkv,.webm", help="Comma-separated list of video extensions to include")
    args = ap.parse_args()

    ensure_dir(args.output_dir)

    exts = tuple(e.strip().lower() for e in args.exts.split(",") if e.strip())
    videos = list_videos(args.input_dir, exts)
    if not videos:
        print(f"[ERROR] No videos found in {args.input_dir} with extensions {exts}")
        return

    if args.frame_list:
        mapping = parse_frame_list(args.frame_list)
        # Build basename -> path map
        name_to_path = {basename_no_ext(v): v for v in videos}

        tasks: List[Tuple[str, List[int]]] = []
        missing: List[str] = []
        for vid, frames in mapping.items():
            path = name_to_path.get(vid)
            if not path:
                missing.append(vid)
                continue
            tasks.append((path, sorted(frames)))
        if missing:
            print(f"[WARN] {len(missing)} videos from frame list not found under {args.input_dir}: {', '.join(missing[:10])}{' ...' if len(missing)>10 else ''}")

        def _run_task(t: Tuple[str, List[int]]):
            vp, frs = t
            extract_specific_frames(vp, args.output_dir, frs)

        thread_map(_run_task, tasks, max_workers=max(1, int(args.workers)))
    else:
        print(f"Extracting frames uniformly at {args.fps} fps ...")
        def _run_uniform(vp: str):
            extract_uniform_frames(vp, args.output_dir, args.fps)
        thread_map(_run_uniform, videos, max_workers=max(1, int(args.workers)))

    print("Frame extraction complete.")


if __name__ == "__main__":
    main()
