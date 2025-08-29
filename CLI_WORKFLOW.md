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

## 3) Start the Backend API
```
python tools/aic_cli.py serve --be-dir hcm-AI-challenge-2024-main/aic-24-BE --port 8000 --run
```

The API serves on `http://localhost:8000`.

## 4) Enable TRAKE (if you have transcripts/headings)
Place JSONs with `segments` that include `text`, `start` (seconds), `fps`, `prefix`, and `frame_list` (original frame indices) into:
- `aic-24-BE/data_processing/raw/transcript/{video_id}.json`
- `aic-24-BE/data_processing/raw/headings_ocr/{video_id}.json`

Start Sonic and ingest:
```
python tools/aic_cli.py ingest-sonic --be-dir hcm-AI-challenge-2024-main/aic-24-BE --up --heading
```

## 5) Prepare Query Files
Create a folder of plain‑text queries, one file per query, named by task suffix:
```
mkdir -p queries_round_1
echo "người lính chào cờ" > queries_round_1/query-1-kis.txt
echo "chuỗi sự kiện: mở đầu -> phát biểu -> vỗ tay" > queries_round_1/query-2-trake.txt
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
- The backend uses OpenCLIP ViT‑B/32 text encoder to match the provided example features. If you switch feature backbones, update `aic-24-BE/nitzche_clip.py` and rebuild the model.

