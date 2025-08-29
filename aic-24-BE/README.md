# AIC‑24 Backend API (KIS & TRAKE)

This service powers Textual KIS (CLIP retrieval) and optional TRAKE (temporal retrieval via Sonic).

## Setup
```bash
# From repo root
cd hcm-AI-challenge-2024-main/aic-24-BE
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
```

Model path is configured via `.env`:
```
MODEL_PATH="./models/"
MODEL_16="clip_vit_b32_nitzche.pkl"
```
The model pickle can be produced by the setup tools in `tools/` (see GETTING_STARTED.md) and placed into `aic-24-BE/models/`.

## Run
Foreground (dev reload):
```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Or via CLI with daemon mode (recommended in Colab/CI):
```bash
cd ..  # repo root
python tools/aic_cli.py serve --be-dir hcm-AI-challenge-2024-main/aic-24-BE --port 8000 --run --daemon --no-reload
python tools/aic_cli.py serve-status
tail -n 120 hcm-AI-challenge-2024-main/aic-24-BE/uvicorn.log
```
Stop:
```bash
python tools/aic_cli.py serve-stop
```

## Optional TRAKE (Sonic)
Start Sonic and ingest transcripts/heading OCR if you have them:
```bash
docker compose up -d
python sonic.py
python sonic_heading.py
```
The backend now tolerates missing ASR/heading folders and will still serve KIS.

## API
Open docs at http://localhost:8000/docs

- POST `/query` → KIS results by CLIP similarity
- POST `/asr`   → Temporal results from ASR (requires Sonic + transcripts)
- POST `/heading` → Temporal results from heading OCR (requires Sonic + OCR)

Quick test:
```bash
python - << 'PY'
import requests
r = requests.post('http://localhost:8000/query', json={'text':'tin tức thời sự', 'top':5}, timeout=30)
print(r.status_code, r.json())
PY
```

## Notes
- `media-info` fps is used if present; otherwise fps falls back to the `map-keyframes/{video_id}.csv` column.
- Image filenames in `data/video_frames/{video_id}/` must be original frame indices like `1234.jpg`.
- Logs are written to `uvicorn.log` if using the CLI daemon mode.
