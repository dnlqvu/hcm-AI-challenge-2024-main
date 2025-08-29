# New Dataset Setup Guide (AIC‑25 System)

## Goal
- Set up the AIC‑25 retrieval system on a new dataset (same `Lxx_Vxxx` naming style) and export Codabench-ready CSVs for Textual KIS and TRAKE.

## What You’ll Build
- Textual KIS: Text → top frames using CLIP features.
- TRAKE: Text → multi-moment frame sequences using ASR/heading search via Sonic.
- Submission CSVs: One per query in `submission/`, no header, ≤100 lines.

## Prerequisites
- Python 3.10+ and pip.
- Docker (for Sonic search server).
- Optional but helpful: FFmpeg (for audio extraction), GPU.
- Repo available locally: `hcm-AI-challenge-2024-main`.

## Dataset Assumptions
- Videos named like `L21_V003.mp4`.
- Frame rate ≈ 25 fps (your media-info JSON should include exact fps).
- You can either use provided keyframes/features/transcripts or generate them from raw videos.

## Folder Conventions (Backend: `aic-24-BE`)
- Frames: `aic-24-BE/data/video_frames/{video_id}/{frame_idx}.jpg|webp` (filename is original frame index).
- Packed CLIP features: `aic-24-BE/data/clip_features/` (each file contains a Python pickle tuple `(file_path_list, image_feature)`).
- Model pickle: `aic-24-BE/models/clip_vit_h14_nitzche.pkl` (created from the packed features).
- Media info JSONs: `aic-24-BE/data/media-info/{video_id}.json` (must include `watch_url`, `fps`).
- ASR JSONs: `aic-24-BE/data_processing/raw/transcript/{video_id}.json`.
- Heading OCR JSONs: `aic-24-BE/data_processing/raw/headings_ocr/{video_id}.json`.

## Step 1: Prepare Media Info
For each video, create `{video_id}.json` in `aic-24-BE/data/media-info/` with at least:

```json
{ "watch_url": "https://...", "fps": 25 }
```

## Step 2: Prepare Frames
Choose one:
- Using provided keyframes:
  - Place images into `aic-24-BE/data/video_frames/{video_id}/{frame_idx}.jpg`.
  - Ensure filenames are original frame indices (ints).
- From raw videos (DIY):
  - Extract frames preserving original frame index numbers (e.g., every 25th frame or your stride):
    - Edit and run `python aic-24-BE/data_processing/crop_frame.py` (sets input/output paths inside the script), or write a custom extractor that saves to `data/video_frames/Lxx_Vxxx/<original_frame_idx>.jpg`.
  - Optional: convert to `.webp`. Not required for submissions.

Notes:
- Maintain original frame indices in filenames to match scoring.
- The system parses `video_id` and `frame_idx` from frame paths; extension doesn’t matter.

## Step 3: Compute CLIP Features (Consistency Matters)
Use the same CLIP backbone for both image features (offline) and text encoding (runtime in `nitzche_clip.py`). By default, runtime uses OpenCLIP DFN5B ViT‑H/14.

Options:
- If you already have packed feature pickles (each file stores `(file_path_list, image_feature)`): place them under `aic-24-BE/data/clip_features/` and skip to Step 4.
- If you have per‑image `.npy` features or need to compute features:
  - Compute with your chosen backbone (recommended: OpenCLIP H/14 to match runtime), then pack into shards where each shard includes:
    - `file_path_list`: frame paths like `./data/video_frames/L21_V003/12345.jpg`
    - `image_feature`: `[N, D]` float32 array of features aligned to the paths

Important: If your features are bigG/B32, either switch runtime to the same text backbone or recompute features with H/14.

## Step 4: Build the Model Pickle
From repo root:

```bash
cd hcm-AI-challenge-2024-main/aic-24-BE
# Ensure data/clip_features/ contains packed pickles
python -c "from nitzche_clip import NitzcheCLIP; import os; os.makedirs('./models', exist_ok=True); m=NitzcheCLIP('./data/clip_features'); m.save('./models/clip_vit_h14_nitzche.pkl')"
```

Verify `.env` points to the model:

```
MODEL_PATH="./models"
MODEL_16="clip_vit_h14_nitzche.pkl"
```

## Step 5: Prepare ASR and Heading JSONs (for TRAKE)
For each `{video_id}.json` file (ASR or heading), minimal structure:

```json
{
  "segments": [
    { "text": "...", "start": 123.45, "fps": 25.0, "prefix": "", "frame_list": [1200, 1850, 2100, 2450] }
  ]
}
```

- Place ASR files under `aic-24-BE/data_processing/raw/transcript/`.
- Place heading OCR files under `aic-24-BE/data_processing/raw/headings_ocr/`.

If you don’t have these yet, you can still run KIS. To generate later:
- Extract audio (`extract_sound.py`) → run Whisper → produce transcripts JSONs.
- Run your heading OCR pipeline → headings JSONs.
- Map segments to `frame_list` (list of original frame indices in chronological order).

## Step 6: Start Sonic and Ingest
```bash
cd hcm-AI-challenge-2024-main/aic-24-BE
docker compose up -d   # starts Sonic on 127.0.0.1:1491
python sonic.py        # ingest ASR JSONs
python sonic_heading.py# ingest heading OCR JSONs
```

## Step 7: Start Backend API
```bash
cd hcm-AI-challenge-2024-main/aic-24-BE
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Sanity tests: open `http://localhost:8000/docs`
- `/query`: `{ "text": "your description", "top": 5 }` → Should return frames (KIS).
- `/asr`: `{ "text": "keyword", "top": 5 }` → Should return items with `video_id` and `listFrameId` (TRAKE).
- `/heading`: Similar to `/asr`.

## Step 8: Prepare Query Files
Create a folder `./queries_round_1/` with files:
- `query-1-kis.txt` → contains your KIS text
- `query-2-trake.txt` → contains your TRAKE text

Each file contains a single textual query (plain text).

## Step 9: Export Submission CSVs
From repo root:

```bash
python hcm-AI-challenge-2024-main/tools/export_submissions.py \
  --queries ./queries_round_1 \
  --api http://localhost:8000 \
  --outdir submission
```

Outputs:
- KIS: `submission/query-1-kis.csv` with lines like `L21_V003,12345` (no header, ≤100 lines).
- TRAKE: `submission/query-2-trake.csv` with lines like `L21_V003,1200,1850,2100,2450` (no header, ≤100 lines).

Zip for Codabench: zip the folder named exactly `submission/`.

## Troubleshooting
- KIS CSV empty:
  - Model pickle missing or features/backbone mismatch.
  - Media info missing (`data/media-info/*.json`).
  - Fix: Verify `./models/clip_vit_h14_nitzche.pkl` exists and `/query` returns results.
- TRAKE CSV empty:
  - Sonic not running or not ingested.
  - `transcript/` or `headings_ocr/` empty or wrong schema.
  - Fix: `docker compose up -d`, rerun `python sonic.py` and `python sonic_heading.py`, test `/asr` in docs.
- Frame indices off:
  - Ensure frame filenames are original indices.
  - Ensure fps in media-info matches reality; segment → frames mapping should align to original frames.
- CLIP backbone mismatch:
  - Use the same backbone for features and text. Either recompute features or switch runtime text model accordingly.

## Nice-To-Haves (Optional)
- Standardize `.env` and add `.env.example`.
- Add a packer tool to convert per-image features (`.npy`) into pickled shards for `nitzche_clip.py`.
- Add `--trake-events N` to the exporter to enforce a specific number of frames per line (trim or sample).
- Switch image path extension handling to `.jpg` only if you won’t use `.webp`.
