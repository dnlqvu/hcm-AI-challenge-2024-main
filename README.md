## AIC‑25 (2025) – Event Querying from Videos

### Third prize - Top 5/800

This codebase is prepared for AIC‑25 (2025). It originated from the 2024 system and has been updated for 2025 workflows and tooling (CLI-first). Paths remain the same for compatibility.

<img src="https://github.com/user-attachments/assets/d63cf2d1-3168-4381-8d09-ecb5568dcb25" width="500" height="370">

<img src="https://github.com/user-attachments/assets/c0aaad23-d1d6-479a-a730-72b9cba059f6" width="500" height="300">

<img src="https://github.com/user-attachments/assets/237442dc-030e-4b5e-b281-2fea223dc682" width="500" height="300">

<img src="https://github.com/user-attachments/assets/1f3321ef-0401-42e3-bf6a-c4ee0d7f8b11" width="500" height="300">



### Submission Export (KIS & TRAKE)

This repo includes a helper to generate Codabench‑ready CSVs for Textual KIS and TRAKE from query text files.

Prerequisites:
- Run the backend API in `aic-24-BE` (default: `http://localhost:8000`).
- Prepare a folder of queries named like `query-1-kis.txt`, `query-2-trake.txt`, etc. Each file contains the query text.

Export:
```
python tools/export_submissions.py \
  --queries ./path_to_queries \
  --api http://localhost:8000 \
  --outdir submission
```

Outputs:
- Writes CSVs to `submission/` with names matching the query files (e.g., `query-1-kis.csv`).
- KIS format: `video_id,frame_idx` (no header, up to 100 lines).
- TRAKE format: `video_id,frame1,frame2,...` (no header, up to 100 lines). Uses `/asr` and falls back to `/heading` if needed.

### Getting Started & Dataset Setup

- See `GETTING_STARTED.md` for a quick CLI-only path from the provided `example_dataset` to a running backend and CSV exports.
- For a deeper walkthrough on new datasets, see `NEW_DATASET_SETUP_GUIDE.md`.

### What’s New (2025 refresh)
- Smart sampling: `tools/smart_sampling.py` and `tools/aic_cli.py sample-smart` select frames by semantic change (CLIP‑delta) or shots; preserves original indices.
- Colab/local recompute: `tools/aic_cli.py clip-extract-colab` re‑encodes features without Docker. Supports OpenCLIP, SigLIP/SigLIP2 models via `--model/--pretrained`, and Lighthouse decoders.
- HERO automation: `tools/aic_cli.py hero-recompute-clip` runs HERO Docker extraction end‑to‑end, converts to shards, extracts frames, and builds the model.
- Backend alignment: text encoder is configurable via `.env` to match recomputed features (`CLIP_MODEL_NAME`, `CLIP_PRETRAINED`, optional `CLIP_DEVICE`).
- Notebook: `AIC-25_Colab_Textual_KIS.ipynb` reorganized with a simple path switch — Quickstart vs. Recompute (SigLIP2 by default).
