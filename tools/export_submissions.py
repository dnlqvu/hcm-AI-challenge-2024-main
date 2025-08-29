#!/usr/bin/env python3
"""
Export competition submissions (KIS + TRAKE) for AIC‑25 from query text files.

Usage examples:
  python tools/export_submissions.py --queries ./queries_round_1 --api http://localhost:8000 --outdir submission

Conventions:
  - Query files are named like: query-1-kis.txt, query-4-trake.txt
  - Output CSVs are written as: submission/query-1-kis.csv, submission/query-4-trake.csv
  - Backend API should be running from aic-24-BE at --api (default http://localhost:8000)

KIS:
  Calls POST /query { text, top } and parses each result's img_path into (video_id, frame_idx)

TRAKE:
  Calls POST /asr { text, top } and falls back to POST /heading if empty.
  Emits lines of: video_id,frame1,frame2,... using returned listFrameId
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

try:
    import requests  # type: ignore
except Exception:
    requests = None  # Will fallback to urllib

import urllib.request


def http_post_json(url: str, payload: dict, timeout: int = 60) -> dict:
    """POST JSON via requests if available; otherwise use urllib."""
    data = json.dumps(payload).encode("utf-8")
    if requests is not None:
        r = requests.post(url, json=payload, timeout=timeout)
        r.raise_for_status()
        return r.json()
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def slurp_query_file(path: Path) -> str:
    text = path.read_text(encoding="utf-8").strip()
    # Normalize whitespace/newlines into single spaces to form one query string
    text = re.sub(r"\s+", " ", text)
    return text


def infer_task_from_name(name: str) -> Optional[str]:
    lower = name.lower()
    if lower.endswith("-kis.txt"):
        return "kis"
    if lower.endswith("-trake.txt"):
        return "trake"
    return None


def parse_img_path_to_vid_frame(img_path: str) -> Optional[Tuple[str, int]]:
    # Accept forward or backward slashes
    parts = re.split(r"[/\\]", img_path)
    if len(parts) < 2:
        return None
    vid = parts[-2]
    fname = parts[-1]
    stem = os.path.splitext(fname)[0]
    try:
        frame_idx = int(stem)
    except ValueError:
        # Sometimes filename may be like "12345.webp" or "12345.jpg"; above handles.
        return None
    return vid, frame_idx


def export_kis(api_base: str, query_text: str, max_lines: int = 100) -> List[Tuple[str, int]]:
    url = api_base.rstrip("/") + "/query"
    data = http_post_json(url, {"text": query_text, "top": max_lines})
    items = data.get("data", [])
    results: List[Tuple[str, int]] = []
    for it in items[:max_lines]:
        img_path = it.get("img_path")
        if not img_path:
            continue
        parsed = parse_img_path_to_vid_frame(img_path)
        if not parsed:
            continue
        results.append(parsed)
    return results


def export_trake(api_base: str, query_text: str, max_lines: int = 100, prefer: str = "asr") -> List[Tuple[str, List[int]]]:
    def call(kind: str) -> List[dict]:
        url = api_base.rstrip("/") + ("/asr" if kind == "asr" else "/heading")
        return http_post_json(url, {"text": query_text, "top": max_lines})

    # Prefer ASR; fallback to heading if empty
    results = call(prefer)
    if not results:
        results = call("heading")

    out: List[Tuple[str, List[int]]] = []
    for r in results[:max_lines]:
        vid = r.get("video_id")
        frames = r.get("listFrameId") or []
        if not vid or not isinstance(frames, list) or not frames:
            continue
        # Ensure ints and ascending order
        cleaned = []
        for f in frames:
            try:
                cleaned.append(int(f))
            except Exception:
                continue
        if not cleaned:
            continue
        cleaned = sorted(set(cleaned))
        out.append((vid, cleaned))
    return out


def write_csv_kis(path: Path, pairs: Iterable[Tuple[str, int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        for vid, frame in pairs:
            w.writerow([vid, frame])


def write_csv_trake(path: Path, sequences: Iterable[Tuple[str, List[int]]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        for vid, frames in sequences:
            w.writerow([vid] + frames)


def main() -> int:
    p = argparse.ArgumentParser(description="Export AIC‑25 submissions for KIS/TRAKE.")
    p.add_argument("--queries", type=str, required=True, help="Directory containing query-*-kis.txt and query-*-trake.txt")
    p.add_argument("--api", type=str, default="http://localhost:8000", help="Base URL of aic-24-BE backend (default: %(default)s)")
    p.add_argument("--outdir", type=str, default="submission", help="Output directory for CSVs (default: %(default)s)")
    p.add_argument("--max-per-query", type=int, default=100, help="Max lines per CSV (default: %(default)s)")
    p.add_argument("--trake-prefer", type=str, choices=["asr", "heading"], default="asr", help="Prefer ASR or heading search for TRAKE (default: %(default)s)")
    args = p.parse_args()

    qdir = Path(args.queries)
    outdir = Path(args.outdir)
    if not qdir.is_dir():
        print(f"[ERROR] Queries directory not found: {qdir}", file=sys.stderr)
        return 2

    txt_files = sorted(qdir.glob("*.txt"))
    if not txt_files:
        print(f"[ERROR] No .txt query files found in {qdir}", file=sys.stderr)
        return 2

    produced = []
    for qf in txt_files:
        # Ensure task detection uses the full filename (with .txt)
        task = infer_task_from_name(qf.name)
        if task not in {"kis", "trake"}:
            # Skip non-target tasks (e.g., qa)
            continue
        query_text = slurp_query_file(qf)
        out_csv = outdir / f"{qf.stem}.csv"
        try:
            if task == "kis":
                pairs = export_kis(args.api, query_text, args.max_per_query)
                write_csv_kis(out_csv, pairs)
                print(f"[KIS] Wrote {len(pairs)} lines -> {out_csv}")
            elif task == "trake":
                seqs = export_trake(args.api, query_text, args.max_per_query, prefer=args.trake_prefer)
                write_csv_trake(out_csv, seqs)
                print(f"[TRAKE] Wrote {len(seqs)} lines -> {out_csv}")
            produced.append(out_csv)
        except Exception as e:
            print(f"[ERROR] Failed for {qf.name}: {e}", file=sys.stderr)

    if not produced:
        print("[WARN] No outputs produced. Ensure filenames end with -kis.txt or -trake.txt.")
        return 1
    print(f"Done. Zip the '{outdir.name}' folder for submission.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
