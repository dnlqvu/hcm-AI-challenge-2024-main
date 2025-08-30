# CLI Workflow (No Frontend)

This guide shows a pure command‑line flow to set up the system, preprocess assets, run the backend, and export Codabench‑ready CSVs for Textual KIS and TRAKE — without any UI.

## Prerequisites
- Python 3.10+ with pip
- Docker (for Sonic; required for TRAKE)
- Recommended: a virtual environment

Install deps:
```
cd hcm-AI-challenge-2024-main/aic-24-BE
python -m venv .venv && source .venv/bin/activate  # (Windows: .venv\Scripts\activate)
pip install -r requirements.txt
```

## 1) Download Dataset From CSV (optional)
If you have a CSV of dataset links (e.g., `AIC_2025_dataset_download_link.csv`), use the CLI to fetch and extract files into `example_dataset/`:

```
cd hcm-AI-challenge-2024-main
python tools/aic_cli.py download-dataset \
  --csv AIC_2025_dataset_download_link.csv \
  --outdir example_dataset \
  --extract
```

Expected subfolders in `example_dataset/` after extraction:
- `keyframes/` (Lxx_Vxxx/nnn.jpg)
- `map-keyframes/` (Lxx_Vxxx.csv; columns include n → frame_idx)
- `media-info/` (Lxx_Vxxx.json; must include `watch_url`, `fps`)
- `clip-features-32/` (Lxx_Vxxx.npy)

## 2) Wire Dataset Into Backend
Automate copy, rename (keyframe order → original frame_idx), pack features, and build the retrieval model:

```
python tools/aic_cli.py setup-example \
  --example-dir example_dataset \
  --be-dir hcm-AI-challenge-2024-main/aic-24-BE
```

What it does:
- Copies `media-info` → `aic-24-BE/data/media-info`
- Copies `keyframes` → `aic-24-BE/data/video_frames`
- Renames files from `{n}.jpg` to `{frame_idx}.jpg` using `map-keyframes`
- Packs per‑video `.npy` features into shards the backend can load
- Builds model pickle `aic-24-BE/models/clip_vit_b32_nitzche.pkl` and updates `.env`

Tip — Smart sampling (optional):
```
python tools/aic_cli.py sample-smart --strategy clip-delta \
  --videos-dir <videos_root> \
  --frames-dir aic-24-BE/data/video_frames
```
Recursively finds videos; extracts only selected frames (original indices preserved).

## 3) Start the Backend API
Foreground:
```
python tools/aic_cli.py serve --be-dir hcm-AI-challenge-2024-main/aic-24-BE --port 8000 --run
```
Background (recommended in CI/Colab/terminals):
```
python tools/aic_cli.py serve --be-dir hcm-AI-challenge-2024-main/aic-24-BE --port 8000 --run --daemon --no-reload
python tools/aic_cli.py serve-status
tail -n 120 hcm-AI-challenge-2024-main/aic-24-BE/uvicorn.log
```
The API serves on `http://localhost:8000`.

## 3b) Recompute Features (Recommended: SigLIP2)
If you want higher recall/precision, recompute features and align the backend text encoder:

1) Extract features (no Docker), choose backend and model:
```
python tools/aic_cli.py clip-extract-colab \
  --videos-dir <videos_root> \
  --outdir hero_colab_out/clip-vit_features \
  --clip-len 1.5 \
  --backend lighthouse-clip \
  --model ViT-L-14-SigLIP-384 \
  --pretrained webli
```
2) Convert to shards, extract exact frames, build model:
```
python tools/convert_hero_clip_to_shards.py --hero-clip-dir hero_colab_out/clip-vit_features \
  --media-info aic-24-BE/data/media-info --clip-len 1.5 \
  --outdir aic-24-BE/data/clip_features --emit-frame-list selected_frames_from_clip.csv
python aic-24-BE/data_processing/crop_frame.py --input-dir <videos_root> --recursive \
  --output-dir aic-24-BE/data/video_frames --frame-list selected_frames_from_clip.csv
python tools/aic_cli.py build-model-from-shards --be-dir aic-24-BE --shards-dir aic-24-BE/data/clip_features --model-name clip_siglip.pkl
```
3) Set backend text encoder in `.env` to match:
```
CLIP_MODEL_NAME="ViT-L-14-SigLIP-384"
CLIP_PRETRAINED="webli"
```

## 4) Enable TRAKE (if you have transcripts/headings)
Place JSONs with `segments` that include `text`, `start` (seconds), `fps`, `prefix`, and `frame_list` (original frame indices) into:
- `aic-24-BE/data_processing/raw/transcript/{video_id}.json`
- `aic-24-BE/data_processing/raw/headings_ocr/{video_id}.json`

Start Sonic and ingest:
```
python tools/aic_cli.py ingest-sonic --be-dir hcm-AI-challenge-2024-main/aic-24-BE --up --heading
```

## 5) Prepare Queries
Option A — a folder of plain‑text files, one per query, named by task suffix:
```
mkdir -p queries_round_1
echo "người lính chào cờ" > queries_round_1/query-1-kis.txt
echo "chuỗi sự kiện: mở đầu -> phát biểu -> vỗ tay" > queries_round_1/query-2-trake.txt
```
Option B — a single inline query (no files):
```
python tools/aic_cli.py export --text "người lính chào cờ" --task kis --outdir submission --name query-1
```

## 6) Export Submission CSVs
```
python tools/aic_cli.py export \
  --queries ./queries_round_1 \
  --api http://localhost:8000 \
  --outdir submission
```

Outputs:
- KIS: `submission/query-*-kis.csv` lines are `video_id,frame_idx` (no header, ≤100 lines)
- TRAKE: `submission/query-*-trake.csv` lines are `video_id,frame1,frame2,...` (no header, ≤100 lines)

## 7) Zip for Codabench
```
python tools/aic_cli.py zip-submission --outdir submission --name team_ABC_round1.zip
```

The archive must contain a folder named `submission/` with your CSVs.

## Notes & Troubleshooting
- Downloader needs `requests` for Google Drive links; HTTP links fall back to stdlib.
- If KIS results are empty, ensure the model pickle exists (`aic-24-BE/models/clip_vit_b32_nitzche.pkl`) and that media‑info was copied.
- If TRAKE results are empty, ensure Sonic is running (`docker compose ps` in `aic-24-BE`) and that you ingested transcripts/headings.
- Ensure your runtime text encoder matches the recomputed features (see `.env`: `CLIP_MODEL_NAME`, `CLIP_PRETRAINED`).
- If the exporter says connection refused, start the server with `serve --run --daemon --no-reload`, check `serve-status`, and tail `aic-24-BE/uvicorn.log`. You can add `--wait-api 30` to the export command.
