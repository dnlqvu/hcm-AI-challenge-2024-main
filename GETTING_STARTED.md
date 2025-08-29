# Getting Started (AIC‑25 – KIS & TRAKE)

This guide gets you from the provided example dataset to a running backend and ready‑to‑submit CSVs for Textual KIS and TRAKE.

## Prerequisites
- Python 3.10+
- pip (and optionally a virtualenv)
- Docker (for Sonic – needed for TRAKE)

## Quick Start with the Example Dataset
The repo includes tools to download/extract assets into `example_dataset/`, then wire them into the backend: copy media‑info and keyframes, rename keyframes by original frame index, pack CLIP features, and build the retrieval model.

1) Get the dataset (recommended via CLI):
```
python hcm-AI-challenge-2024-main/tools/aic_cli.py download-dataset \
  --csv AIC_2025_dataset_download_link.csv \
  --outdir example_dataset \
  --extract
```
Notes:
- The downloader auto‑extracts to the correct subfolders (including `map-keyframes`) and deletes ZIPs after extraction. Nested folders are flattened.

2) Ensure the dataset looks like this:
```
example_dataset/
  keyframes/           # Lxx_Vxxx/{nnn.jpg}
  map-keyframes/       # Lxx_Vxxx.csv (n -> frame_idx)
  media-info/          # Lxx_Vxxx.json (watch_url, fps, ...)
  clip-features-32/    # Lxx_Vxxx.npy (per-video features)
```

3) Run the setup script:
```
python hcm-AI-challenge-2024-main/tools/setup_example_dataset.py \
  --example-dir example_dataset \
  --be-dir hcm-AI-challenge-2024-main/aic-24-BE
```

What it does:
- Copies media info → `aic-24-BE/data/media-info`
- Copies keyframes → `aic-24-BE/data/video_frames`
- Copies map CSVs → `aic-24-BE/data/map-keyframes`
- Renames keyframes from order `n` to original `frame_idx` using the map CSVs
- Packs `.npy` features into retrieval shards for the backend
- Builds a model pickle `aic-24-BE/models/clip_vit_b32_nitzche.pkl` and updates `.env`

4) Start the backend API:
Recommended (daemon mode so your shell/notebook remains responsive):
```
python hcm-AI-challenge-2024-main/tools/aic_cli.py serve --be-dir hcm-AI-challenge-2024-main/aic-24-BE --port 8000 --run --daemon --no-reload
# Check status and tail logs
python hcm-AI-challenge-2024-main/tools/aic_cli.py serve-status
tail -n 120 hcm-AI-challenge-2024-main/aic-24-BE/uvicorn.log
```
Or run directly in the BE folder (foreground):
```
cd hcm-AI-challenge-2024-main/aic-24-BE
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

5) Test KIS:
- Open http://localhost:8000/docs and invoke `POST /query` with `{ "text": "...", "top": 5 }`.

## Enabling TRAKE (Temporal Retrieval)
TRAKE uses Sonic and ingested transcripts/heading OCR. If/when you have these JSONs:

1) Place JSONs:
```
aic-24-BE/data_processing/raw/transcript/{video_id}.json
aic-24-BE/data_processing/raw/headings_ocr/{video_id}.json
```
Each JSON should include `segments` with: `text`, `start` (seconds), `fps`, `prefix`, and `frame_list` (list of original frame indices).

2) Start Sonic and ingest:
```
cd hcm-AI-challenge-2024-main/aic-24-BE
docker compose up -d
python sonic.py
python sonic_heading.py
```

3) Test TRAKE:
- In docs, try `POST /asr` (or `/heading`) with `{ "text": "...", "top": 5 }`. Responses include `video_id` and `listFrameId`.

## Exporting Submission CSVs (KIS & TRAKE)
Option A — Folder of query files (one per query):
```
queries_round_1/
  query-1-kis.txt
  query-2-trake.txt
```

Run the exporter:
```
python hcm-AI-challenge-2024-main/tools/export_submissions.py \
  --queries ./queries_round_1 \
  --api http://localhost:8000 \
  --outdir submission
```

Option B — Single inline query (no files needed):
```
python hcm-AI-challenge-2024-main/tools/export_submissions.py \
  --text "tin tức thời sự" \
  --task kis \
  --name query-1 \
  --api http://localhost:8000 \
  --outdir submission
```
Tip: add `--wait-api 30` to wait for the backend to come up.

Outputs:
- KIS: `submission/query-*.csv` lines are `video_id,frame_idx` (no header, ≤100 lines)
- TRAKE: `submission/query-*.csv` lines are `video_id,frame1,frame2,...` (no header, ≤100 lines)

Zip the `submission/` folder for Codabench.

## Tools Overview
- `tools/setup_example_dataset.py` – end‑to‑end wiring from `example_dataset` into the backend.
- `tools/rename_keyframes_from_map.py` – rename keyframes by original `frame_idx` using map CSVs.
- `tools/pack_features_from_npy.py` – pack per‑video `.npy` features into retrieval shards.
- `tools/export_submissions.py` – generate KIS/TRAKE CSVs from query files or a single inline `--text`.
- `tools/download_dataset_from_csv.py` – download + extract into correct subfolders; deletes ZIPs after extraction.
- `tools/aic_cli.py serve` – start backend; add `--daemon/--no-reload`, `serve-stop`, `serve-status`.

## Notes
- The backend is configured to use OpenCLIP ViT‑B/32 for text, aligned with the provided example features (B/32). If you switch feature backbones, update `aic-24-BE/nitzche_clip.py` accordingly and rebuild the model.
- Image paths now use `.jpg` directly. No `.webp` conversion is required.
- Keep filenames inside each video folder as the original frame indices to match competition scoring.
- If `media-info` lacks `fps`, the backend falls back to the fps column in `map-keyframes/{video_id}.csv`.
- TRAKE is optional: backend startup now skips ASR/heading preload if their folders are absent.

## Troubleshooting
- Empty KIS results: verify `aic-24-BE/models/clip_vit_b32_nitzche.pkl` exists; run `/query` in docs; ensure media‑info is copied.
- Empty TRAKE results: ensure Sonic is running and JSONs are ingested; test `/asr` in docs.
- Connection refused during export: start the server with `tools/aic_cli.py serve --run --daemon --no-reload`; check `serve-status` and tail `aic-24-BE/uvicorn.log`.
- Mismatched counts (features vs keyframes): the packer aligns to the minimum count per video; confirm your maps and keyframe folders are complete.

CLI alternative (downloads directly from a CSV of links and extracts for you):
```
# Download assets listed in AIC_2025_dataset_download_link.csv to example_dataset/
python hcm-AI-challenge-2024-main/tools/aic_cli.py download-dataset --csv AIC_2025_dataset_download_link.csv --outdir example_dataset --extract

# Wire example_dataset into the backend
python hcm-AI-challenge-2024-main/tools/aic_cli.py setup-example --example-dir example_dataset --be-dir hcm-AI-challenge-2024-main/aic-24-BE

# Start backend in background and export a single inline KIS query
python hcm-AI-challenge-2024-main/tools/aic_cli.py serve --port 8000 --run --daemon --no-reload
python hcm-AI-challenge-2024-main/tools/aic_cli.py export --text "tin tức thời sự" --task kis --outdir submission --wait-api 30
```
