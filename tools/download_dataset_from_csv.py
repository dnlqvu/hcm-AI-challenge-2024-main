#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import os
import re
import sys
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    import requests  # type: ignore
except Exception as e:
    requests = None


def parse_csv(path: Path) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    with path.open('r', encoding='utf-8') as f:
        rdr = csv.DictReader(f)
        for row in rdr:
            # normalize keys
            norm = { (k or '').strip().lower(): (v or '').strip() for k, v in row.items() }
            # expected keys: url, name (filename), subdir (optional), md5/sha256 (optional)
            if not norm.get('url'):
                # try other aliases commonly seen
                for k in ('url', 'link', 'download', 'href', 'download link', 'download_link'):
                    if norm.get(k):
                        norm['url'] = norm[k]
                        break
            # filename
            if not norm.get('name'):
                # try common filename headers
                for k in ('name', 'filename', 'filenames', 'file'):
                    if norm.get(k):
                        norm['name'] = norm[k]
                        break
                # otherwise use filename from URL path
                if not norm.get('name'):
                    norm['name'] = derive_filename_from_url(norm.get('url', ''))
            # subdir (optional)
            if not norm.get('subdir'):
                for k in ('subdir', 'folder', 'dir', 'target', 'extract_dir'):
                    if norm.get(k):
                        norm['subdir'] = norm[k]
                        break
            rows.append(norm)
    return rows


def derive_filename_from_url(url: str) -> str:
    if not url:
        return 'download.bin'
    # Handle Google Drive shared links by extracting a sensible name fallback
    # Otherwise, take the last path segment and strip query string
    url_no_q = url.split('?')[0]
    segs = [s for s in url_no_q.split('/') if s]
    if segs:
        return segs[-1]
    return 'download.bin'


def is_google_drive_url(url: str) -> bool:
    return 'drive.google.com' in url


def extract_drive_file_id(url: str) -> Optional[str]:
    # Patterns: /file/d/<id>/view, open?id=<id>, uc?export=download&id=<id>
    m = re.search(r'/file/d/([^/]+)', url)
    if m:
        return m.group(1)
    m = re.search(r'[?&]id=([^&]+)', url)
    if m:
        return m.group(1)
    return None


def download_google_drive(url: str, out_path: Path, chunk: int = 1 << 20):
    if requests is None:
        raise RuntimeError('requests is required for Google Drive downloads')
    session = requests.Session()
    file_id = extract_drive_file_id(url)
    if not file_id:
        # Try direct
        resp = session.get(url, stream=True)
        resp.raise_for_status()
        _stream_to_file(resp, out_path, chunk)
        return
    base = 'https://drive.google.com/uc?export=download'
    params = {'id': file_id}
    resp = session.get(base, params=params, stream=True)
    token = _get_confirm_token(resp)
    if token:
        params['confirm'] = token
        resp = session.get(base, params=params, stream=True)
    resp.raise_for_status()
    _stream_to_file(resp, out_path, chunk)


def _get_confirm_token(resp) -> Optional[str]:
    for k, v in resp.cookies.items():
        if k.startswith('download_warning'):
            return v
    return None


def download_http(url: str, out_path: Path, chunk: int = 1 << 20):
    if requests is None:
        # Best-effort fallback using urllib
        import urllib.request
        with urllib.request.urlopen(url) as r, out_path.open('wb') as f:
            while True:
                b = r.read(chunk)
                if not b:
                    break
                f.write(b)
        return
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        _stream_to_file(r, out_path, chunk)


def _stream_to_file(resp, out_path: Path, chunk: int):
    total = int(resp.headers.get('Content-Length', '0') or '0')
    done = 0
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open('wb') as f:
        for b in resp.iter_content(chunk_size=chunk):
            if not b:
                continue
            f.write(b)
            done += len(b)
            if total:
                pct = (done / total) * 100
                print(f"\rDownloading {out_path.name}: {done}/{total} bytes ({pct:.1f}%)", end='')
    print()


def verify_checksum(path: Path, algo: str, expected: str) -> bool:
    h = hashlib.new(algo)
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    ok = h.hexdigest().lower() == expected.lower()
    print(f"[{algo}] {path.name}: {'OK' if ok else 'MISMATCH'}")
    return ok


def infer_extract_dir(filename: str) -> Optional[str]:
    """Infer a reasonable extract subfolder from the archive name.

    Order matters: check specific patterns (map-keyframes) before generic ones (keyframes)
    to avoid routing the maps archive into the keyframes folder.
    """
    name = filename.lower()
    # Map CSVs
    if (
        'map-keyframes' in name or 'map_keyframes' in name or 'mapkeyframes' in name
        or 'map-keyframe' in name or 'map_keyframe' in name or 'mapkeyframe' in name
    ):
        return 'map-keyframes'
    # Media-info JSONs
    if 'media-info' in name or 'mediainfo' in name:
        return 'media-info'
    # Precomputed CLIP features
    if 'clip' in name and 'feature' in name:
        return 'clip-features-32'
    # Object JSONs
    if 'object' in name:
        return 'objects'
    # Keyframes (generic) — check after map-keyframes
    if 'keyframe' in name:
        return 'keyframes'
    # Fallback: base name without .zip
    if name.endswith('.zip'):
        return os.path.splitext(os.path.basename(filename))[0]
    return None


def extract_zip(zip_path: Path, out_root: Path, target_subdir: Optional[str] = None):
    subdir = target_subdir or infer_extract_dir(zip_path.name)
    if not subdir:
        print(f"[SKIP] No extract target inferred for {zip_path.name}")
        return
    dst = out_root / subdir
    dst.mkdir(parents=True, exist_ok=True)
    print(f"Extracting {zip_path.name} -> {dst}")
    with zipfile.ZipFile(zip_path, 'r') as z:
        z.extractall(dst)
    # Flatten if archive contains a redundant top-level folder matching subdir
    nested = dst / subdir
    try:
        if nested.is_dir():
            for p in nested.iterdir():
                p.rename(dst / p.name)
            nested.rmdir()
    except Exception:
        # Non-fatal; continue
        pass


def main() -> int:
    ap = argparse.ArgumentParser(description='Download dataset assets from a CSV and stage them for setup')
    ap.add_argument('--csv', default='AIC_2025_dataset_download_link.csv', help='CSV file with columns: url,name,subdir,md5/sha256 (flexible headers)')
    ap.add_argument('--outdir', default='example_dataset', help='Output folder to stage assets')
    ap.add_argument('--extract', action='store_true', help='Extract zip archives into expected subfolders')
    args = ap.parse_args()

    csv_path = Path(args.csv)
    out_root = Path(args.outdir)
    out_root.mkdir(parents=True, exist_ok=True)

    rows = parse_csv(csv_path)
    if not rows:
        print(f"No rows parsed from {csv_path}")
        return 2

    for row in rows:
        url = row.get('url', '')
        name = row.get('name') or derive_filename_from_url(url)
        subdir = row.get('subdir') or None
        if not url:
            print(f"[SKIP] missing url in row: {row}")
            continue
        out_path = out_root / name
        if out_path.exists():
            print(f"[SKIP] exists: {out_path}")
        else:
            print(f"Downloading: {url} -> {out_path}")
            try:
                if is_google_drive_url(url):
                    download_google_drive(url, out_path)
                else:
                    download_http(url, out_path)
            except Exception as e:
                print(f"[ERROR] failed to download {url}: {e}")
                continue
        # checksum
        if row.get('md5'):
            verify_checksum(out_path, 'md5', row['md5'])
        if row.get('sha256'):
            verify_checksum(out_path, 'sha256', row['sha256'])
        # extract
        if args.extract and out_path.suffix.lower() == '.zip':
            try:
                extract_zip(out_path, out_root, target_subdir=subdir)
                # Remove archive after successful extraction to save space
                try:
                    out_path.unlink()
                    print(f"Removed archive: {out_path.name}")
                except Exception as e:
                    print(f"[WARN] could not remove {out_path.name}: {e}")
            except Exception as e:
                print(f"[ERROR] extract failed for {out_path}: {e}")

    print(f"Done. Staged assets in: {out_root}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
