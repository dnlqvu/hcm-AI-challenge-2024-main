#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def run(cmd: list[str], cwd: Path | None = None):
    print("$", " ".join(cmd))
    subprocess.check_call(cmd, cwd=str(cwd) if cwd else None)


def cmd_setup_example(args):
    script = ROOT / "tools" / "setup_example_dataset.py"
    run([
        sys.executable, str(script),
        "--example-dir", args.example_dir,
        "--be-dir", args.be_dir,
        "--model-name", args.model_name,
    ])


def cmd_rename_keyframes(args):
    script = ROOT / "tools" / "rename_keyframes_from_map.py"
    run([
        sys.executable, str(script),
        "--frames-dir", args.frames_dir,
        "--map-dir", args.map_dir,
        *( ["--pattern", args.pattern] if args.pattern else [] ),
        *( ["--allow-overwrite"] if args.allow_overwrite else [] ),
        *( ["--dry-run"] if args.dry_run else [] ),
        *([] if not args.videos else ["--videos", *args.videos]),
    ])


def cmd_pack_features(args):
    script = ROOT / "tools" / "pack_features_from_npy.py"
    run([
        sys.executable, str(script),
        "--features-dir", args.features_dir,
        "--map-dir", args.map_dir,
        "--frames-prefix", args.frames_prefix,
        "--outdir", args.outdir,
        "--ext", args.ext,
    ])


def cmd_serve(args):
    be_dir = Path(args.be_dir)
    if args.run:
        cmd = [sys.executable, "-m", "uvicorn", "main:app"]
        if not args.no_reload:
            cmd.append("--reload")
        cmd += ["--host", "0.0.0.0", "--port", str(args.port)]
        if args.daemon:
            logfile = be_dir / args.logfile
            pidfile = be_dir / args.pidfile
            logfile.parent.mkdir(parents=True, exist_ok=True)
            print("$ (daemon)", " ".join(cmd), f"cwd={be_dir}")
            with open(logfile, "ab") as f:
                p = subprocess.Popen(
                    cmd,
                    cwd=str(be_dir),
                    stdout=f,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
            pidfile.write_text(str(p.pid))
            print(f"Started uvicorn PID {p.pid} (logs: {logfile}, pidfile: {pidfile})")
        else:
            run(cmd, cwd=be_dir)
    else:
        print(f"Run: uvicorn main:app --reload --host 0.0.0.0 --port {args.port}  (in {be_dir})")


def cmd_serve_stop(args):
    be_dir = Path(args.be_dir)
    pidfile = be_dir / args.pidfile
    if not pidfile.exists():
        print(f"No pidfile found: {pidfile}")
        return
    try:
        pid = int(pidfile.read_text().strip())
    except Exception:
        print(f"Invalid pidfile: {pidfile}")
        return
    try:
        os.kill(pid, 15)
        print(f"Sent SIGTERM to PID {pid}")
    except ProcessLookupError:
        print(f"Process not found: {pid}")
    except Exception as e:
        print(f"Failed to stop PID {pid}: {e}")
    try:
        pidfile.unlink()
    except Exception:
        pass


def cmd_serve_status(args):
    be_dir = Path(args.be_dir)
    pidfile = be_dir / args.pidfile
    if not pidfile.exists():
        print("stopped")
        return
    try:
        pid = int(pidfile.read_text().strip())
        os.kill(pid, 0)
        print(f"running (PID {pid})")
    except Exception:
        print("stale pidfile")


def cmd_ingest_sonic(args):
    be_dir = Path(args.be_dir)
    if args.up:
        # Start Sonic via docker compose
        run(["docker", "compose", "up", "-d"], cwd=be_dir)
    # Ingest ASR and optionally heading
    run([sys.executable, "sonic.py"], cwd=be_dir)
    if args.heading:
        run([sys.executable, "sonic_heading.py"], cwd=be_dir)


def cmd_export(args):
    script = ROOT / "tools" / "export_submissions.py"
    cmd = [
        sys.executable, str(script),
        "--api", args.api,
        "--outdir", args.outdir,
        "--max-per-query", str(args.max_per_query),
        "--trake-prefer", args.trake_prefer,
        "--wait-api", str(args.wait_api),
    ]
    if args.queries:
        cmd += ["--queries", args.queries]
    if args.text:
        cmd += ["--text", args.text, "--task", args.task, "--name", args.name]
    run(cmd)


def cmd_zip(args):
    outdir = Path(args.outdir)
    zipname = Path(args.name)
    if zipname.suffix != ".zip":
        zipname = zipname.with_suffix(".zip")
    if not outdir.is_dir():
        raise SystemExit(f"Output dir not found: {outdir}")
    # Ensure the archive contains the folder 'submission/'
    if outdir.name != "submission":
        print("[WARN] Codabench expects a folder named 'submission/' inside the zip.")
    if zipname.exists():
        zipname.unlink()
    # Create zip
    shutil.make_archive(zipname.with_suffix("").as_posix(), "zip", root_dir=outdir.parent.as_posix(), base_dir=outdir.name)
    print(f"Wrote {zipname}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="AIC‑25 CLI (setup, serve, ingest, export, zip)")
    sub = p.add_subparsers(dest="cmd", required=True)

    # setup-example
    sp = sub.add_parser("setup-example", help="Wire example_dataset into backend")
    sp.add_argument("--example-dir", default="example_dataset")
    sp.add_argument("--be-dir", default=str(ROOT / "aic-24-BE"))
    sp.add_argument("--model-name", default="clip_vit_b32_nitzche.pkl")
    sp.set_defaults(func=cmd_setup_example)

    # download-dataset
    sp = sub.add_parser("download-dataset", help="Download dataset assets from CSV to example_dataset")
    sp.add_argument("--csv", default="AIC_2025_dataset_download_link.csv")
    sp.add_argument("--outdir", default="example_dataset")
    sp.add_argument("--extract", action="store_true")
    def _dl(args):
        script = ROOT / "tools" / "download_dataset_from_csv.py"
        run([
            sys.executable, str(script),
            "--csv", args.csv,
            "--outdir", args.outdir,
            *(["--extract"] if args.extract else []),
        ])
    sp.set_defaults(func=_dl)

    # rename-keyframes
    sp = sub.add_parser("rename-keyframes", help="Rename keyframes using map CSVs")
    sp.add_argument("--frames-dir", required=True)
    sp.add_argument("--map-dir", required=True)
    sp.add_argument("--pattern", default=r"(?P<n>\d+)$")
    sp.add_argument("--allow-overwrite", action="store_true")
    sp.add_argument("--dry-run", action="store_true")
    sp.add_argument("--videos", nargs="*")
    sp.set_defaults(func=cmd_rename_keyframes)

    # pack-features
    sp = sub.add_parser("pack-features", help="Pack per-video .npy into shards")
    sp.add_argument("--features-dir", required=True)
    sp.add_argument("--map-dir", required=True)
    sp.add_argument("--frames-prefix", default="./data/video_frames")
    sp.add_argument("--outdir", default=str(ROOT / "aic-24-BE" / "data" / "clip_features"))
    sp.add_argument("--ext", default=".jpg")
    sp.set_defaults(func=cmd_pack_features)

    # serve
    sp = sub.add_parser("serve", help="Start backend API (or print command)")
    sp.add_argument("--be-dir", default=str(ROOT / "aic-24-BE"))
    sp.add_argument("--port", type=int, default=8000)
    sp.add_argument("--run", action="store_true", help="Run uvicorn in foreground")
    sp.add_argument("--daemon", action="store_true", help="Run uvicorn in background and write pidfile")
    sp.add_argument("--no-reload", action="store_true", help="Disable --reload (recommended in daemon mode)")
    sp.add_argument("--logfile", default="uvicorn.log")
    sp.add_argument("--pidfile", default="uvicorn.pid")
    sp.set_defaults(func=cmd_serve)

    # serve-stop
    sp = sub.add_parser("serve-stop", help="Stop uvicorn using pidfile")
    sp.add_argument("--be-dir", default=str(ROOT / "aic-24-BE"))
    sp.add_argument("--pidfile", default="uvicorn.pid")
    sp.set_defaults(func=cmd_serve_stop)

    # serve-status
    sp = sub.add_parser("serve-status", help="Report uvicorn status via pidfile")
    sp.add_argument("--be-dir", default=str(ROOT / "aic-24-BE"))
    sp.add_argument("--pidfile", default="uvicorn.pid")
    sp.set_defaults(func=cmd_serve_status)

    # ingest-sonic
    sp = sub.add_parser("ingest-sonic", help="Start Sonic (optional) and ingest ASR/heading")
    sp.add_argument("--be-dir", default=str(ROOT / "aic-24-BE"))
    sp.add_argument("--up", action="store_true", help="Run 'docker compose up -d' before ingest")
    sp.add_argument("--heading", action="store_true", help="Also ingest heading OCR JSONs")
    sp.set_defaults(func=cmd_ingest_sonic)

    # export
    sp = sub.add_parser("export", help="Export KIS/TRAKE CSVs from query files")
    sp.add_argument("--queries")
    sp.add_argument("--text", help="Inline query text (bypass --queries)")
    sp.add_argument("--task", choices=["kis", "trake"], default="kis")
    sp.add_argument("--name", default="query-cli")
    sp.add_argument("--api", default="http://localhost:8000")
    sp.add_argument("--outdir", default="submission")
    sp.add_argument("--max-per-query", type=int, default=100)
    sp.add_argument("--trake-prefer", choices=["asr", "heading"], default="asr")
    sp.add_argument("--wait-api", type=int, default=15, help="Wait up to N seconds for the backend to be reachable before exporting")
    sp.set_defaults(func=cmd_export)

    # zip-submission
    sp = sub.add_parser("zip-submission", help="Zip the submission folder")
    sp.add_argument("--outdir", default="submission")
    sp.add_argument("--name", default="submission.zip")
    sp.set_defaults(func=cmd_zip)

    # sample-smart
    sp = sub.add_parser("sample-smart", help="AI-driven frame sampling (CLIP delta / shots) and extraction")
    sp.add_argument("--videos-dir", required=True, help="Directory containing input videos")
    sp.add_argument("--frames-dir", default=str(ROOT / "aic-24-BE" / "data" / "video_frames"), help="Directory to write extracted frames")
    sp.add_argument("--strategy", choices=["clip-delta", "shots"], default="clip-delta", help="Sampling strategy")
    # clip-delta params
    sp.add_argument("--decode-fps", type=float, default=2.0, help="Decode FPS for analysis (clip-delta)")
    sp.add_argument("--target-fps", type=float, default=1.0, help="Target kept frames/sec (clip-delta)")
    sp.add_argument("--min-gap-sec", type=float, default=0.5, help="Minimum time gap (clip-delta)")
    sp.add_argument("--model", default="ViT-L-16-SigLIP-256", help="CLIP model for analysis (clip-delta)")
    sp.add_argument("--pretrained", default="webli", help="open_clip pretrained tag (e.g., webli or hf-hub:<repo>)")
    sp.add_argument("--adaptive", action="store_true", help="Enable intelligent content-aware adaptive sampling")
    sp.add_argument("--batch-size", type=int, help="Batch size for CLIP encoding (default: auto-determine based on GPU memory)")
    # shots params
    sp.add_argument("--shot-decode-fps", type=float, default=10.0, help="Decode FPS for shot detection")
    sp.add_argument("--shot-long-sec", type=float, default=4.0, help="Long shot threshold (sec)")
    sp.add_argument("--shot-per-long", type=int, default=3, help="Samples per long shot")
    # shared
    sp.add_argument("--device", default="auto", help="cuda|cpu|auto")
    sp.add_argument("--exts", default=".mp4,.avi,.mov,.mkv,.webm", help="Comma-separated video extensions")
    sp.add_argument("--out-csv", default="selected_frames.csv", help="Path to write selected frame indices")
    sp.add_argument("--recursive", action="store_true", default=True, help="Recursively search for videos under --videos-dir")

    def _sample_smart(args):
        device = args.device
        if device == "auto":
            device = "cuda" if shutil.which("nvidia-smi") or os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu"
        smart = ROOT / "tools" / "smart_sampling.py"
        cmd = [
            sys.executable, str(smart),
            "--videos-dir", args.videos_dir,
            "--strategy", args.strategy,
            "--device", device,
            "--out-csv", args.out_csv,
            "--exts", args.exts,
        ]
        if args.strategy == "clip-delta":
            cmd += [
                "--decode-fps", str(args.decode_fps),
                "--target-fps", str(args.target_fps),
                "--min-gap-sec", str(args.min_gap_sec),
                "--model", args.model,
                "--pretrained", args.pretrained,
            ]
            if args.adaptive:
                cmd.append("--adaptive")
            if args.batch_size is not None:
                cmd.extend(["--batch-size", str(args.batch_size)])
        else:
            cmd += [
                "--shot-decode-fps", str(args.shot_decode_fps),
                "--shot-long-sec", str(args.shot_long_sec),
                "--shot-per-long", str(args.shot_per_long),
            ]
        run(cmd)
        crop = ROOT / "aic-24-BE" / "data_processing" / "crop_frame.py"
        run([
            sys.executable, str(crop),
            "--input-dir", args.videos_dir,
            "--output-dir", args.frames_dir,
            "--frame-list", args.out_csv,
        ])
    sp.set_defaults(func=_sample_smart)

    # build-model-from-shards
    def _build_model(args):
        be_dir = Path(args.be_dir)
        sys.path.insert(0, str(be_dir))
        from nitzche_clip import NitzcheCLIP  # type: ignore
        models_dir = be_dir / "models"
        models_dir.mkdir(parents=True, exist_ok=True)
        print(f"Building model from shards: {args.shards_dir}")
        m = NitzcheCLIP(args.shards_dir)
        out = models_dir / args.model_name
        m.save(str(out))
        print(f"Saved: {out}")
        # Patch .env
        envp = be_dir / ".env"
        lines = []
        if envp.exists():
            lines = envp.read_text(encoding="utf-8").splitlines()
        saw_path = saw_16 = False
        out_lines = []
        for line in lines:
            if line.strip().startswith("MODEL_PATH="):
                out_lines.append('MODEL_PATH="./models/"')
                saw_path = True
            elif line.strip().startswith("MODEL_16="):
                out_lines.append(f'MODEL_16="{args.model_name}"')
                saw_16 = True
            else:
                out_lines.append(line)
        if not saw_path:
            out_lines.append('MODEL_PATH="./models/"')
        if not saw_16:
            out_lines.append(f'MODEL_16="{args.model_name}"')
        envp.write_text("\n".join(out_lines) + "\n", encoding="utf-8")
        print(f"Patched {envp}")

    sp = sub.add_parser("build-model-from-shards", help="Build model pickle from shards in aic-24-BE/data/clip_features")
    sp.add_argument("--be-dir", default=str(ROOT / "aic-24-BE"))
    sp.add_argument("--shards-dir", default=str(ROOT / "aic-24-BE" / "data" / "clip_features"))
    sp.add_argument("--model-name", default="clip_vit_b32_nitzche.pkl")
    sp.set_defaults(func=_build_model)

    # hero-clone
    def _hero_clone(args):
        vendor = ROOT / "vendor" / "HERO_Video_Feature_Extractor"
        if vendor.exists():
            print(f"Already present: {vendor}")
            return
        vendor.parent.mkdir(parents=True, exist_ok=True)
        run(["git", "clone", "https://github.com/linjieli222/HERO_Video_Feature_Extractor.git", str(vendor)])
        print(f"Cloned HERO into {vendor}")

    sp = sub.add_parser("hero-clone", help="Clone HERO Video Feature Extractor into vendor/")
    sp.set_defaults(func=_hero_clone)

    # hero-extract-clip
    def _hero_extract_clip(args):
        vendor = ROOT / "vendor" / "HERO_Video_Feature_Extractor"
        if not vendor.exists():
            _hero_clone(args)
        outdir = Path(args.outdir).resolve()
        outdir.mkdir(parents=True, exist_ok=True)
        vids = Path(args.videos_dir).resolve()
        if not vids.exists():
            raise SystemExit(f"Videos dir not found: {vids}")
        image = "linjieli222/hero-video-feature-extractor:clip"
        inner = (
            "cd /src/clip && "
            "python gather_video_paths.py && "
            f"python extract.py --csv /output/csv/clip-vit_info.csv --num_decoding_thread {args.threads} "
            f"--model_version {args.model_version} --clip_len {args.clip_len}"
        )
        cmd = [
            "docker", "run", "--gpus", "all", "--ipc=host", "--network=host", "--rm", "-t",
            "--mount", f"src={vendor.as_posix()},dst=/src,type=bind",
            "--mount", f"src={vids.as_posix()},dst=/video,type=bind,readonly",
            "--mount", f"src={outdir.as_posix()},dst=/output,type=bind",
            "-w", "/src",
            image, "bash", "-lc", inner,
        ]
        run(cmd)
        print(f"HERO CLIP features written under {outdir}/clip-vit_features")

    sp = sub.add_parser("hero-extract-clip", help="Run HERO CLIP extractor via Docker")
    sp.add_argument("--videos-dir", required=True, help="Folder with input videos")
    sp.add_argument("--outdir", required=True, help="Output folder to mount as /output")
    sp.add_argument("--clip-len", default="1.5", help="Seconds per feature (e.g., 1.5 or 2)")
    sp.add_argument("--model-version", default="ViT-B/32", help="HERO CLIP model version")
    sp.add_argument("--threads", type=int, default=4, help="Decoding threads")
    sp.set_defaults(func=_hero_extract_clip)

    # hero-recompute-clip: extract -> convert -> extract frames -> build model
    def _hero_recompute_clip(args):
        _hero_extract_clip(args)
        hero_out = Path(args.outdir).resolve() / "clip-vit_features"
        conv = ROOT / "tools" / "convert_hero_clip_to_shards.py"
        frame_list = Path(args.frame_list or "selected_frames_from_hero.csv").resolve()
        run([
            sys.executable, str(conv),
            "--hero-clip-dir", hero_out.as_posix(),
            "--media-info", str(ROOT / "aic-24-BE" / "data" / "media-info"),
            "--clip-len", str(args.clip_len),
            "--frames-prefix", "./data/video_frames",
            "--outdir", str(ROOT / "aic-24-BE" / "data" / "clip_features"),
            "--emit-frame-list", str(frame_list),
        ])
        crop = ROOT / "aic-24-BE" / "data_processing" / "crop_frame.py"
        run([
            sys.executable, str(crop),
            "--input-dir", args.videos_dir,
            "--output-dir", str(ROOT / "aic-24-BE" / "data" / "video_frames"),
            "--frame-list", str(frame_list),
        ])
        be_dir = ROOT / "aic-24-BE"
        sys.path.insert(0, str(be_dir))
        from nitzche_clip import NitzcheCLIP  # type: ignore
        models_dir = be_dir / "models"
        models_dir.mkdir(parents=True, exist_ok=True)
        print(f"Building model from shards: {ROOT / 'aic-24-BE' / 'data' / 'clip_features'}")
        m = NitzcheCLIP(str(ROOT / "aic-24-BE" / "data" / "clip_features"))
        out = models_dir / args.model_name
        m.save(str(out))
        print(f"Saved: {out}")
        # Patch .env
        envp = be_dir / ".env"
        lines = []
        if envp.exists():
            lines = envp.read_text(encoding="utf-8").splitlines()
        saw_path = saw_16 = False
        out_lines = []
        for line in lines:
            if line.strip().startswith("MODEL_PATH="):
                out_lines.append('MODEL_PATH="./models/"')
                saw_path = True
            elif line.strip().startswith("MODEL_16="):
                out_lines.append(f'MODEL_16="{args.model_name}"')
                saw_16 = True
            else:
                out_lines.append(line)
        if not saw_path:
            out_lines.append('MODEL_PATH="./models/"')
        if not saw_16:
            out_lines.append(f'MODEL_16="{args.model_name}"')
        envp.write_text("\n".join(out_lines) + "\n", encoding="utf-8")
        print(f"Patched {envp}")
        print("All done. Start the backend and test /query.")

    sp = sub.add_parser("hero-recompute-clip", help="End-to-end: HERO extract → convert → extract frames → build model")
    sp.add_argument("--videos-dir", required=True, help="Folder with input videos")
    sp.add_argument("--outdir", required=True, help="HERO output folder to mount as /output")
    sp.add_argument("--clip-len", default="1.5", help="Seconds per feature (e.g., 1.5 or 2)")
    sp.add_argument("--model-version", default="ViT-B/32", help="HERO CLIP model version")
    sp.add_argument("--threads", type=int, default=4, help="Decoding threads")
    sp.add_argument("--frame-list", help="Path to write selected frames CSV (default: selected_frames_from_hero.csv)")
    sp.add_argument("--model-name", default="clip_vit_b32_nitzche.pkl")
    sp.set_defaults(func=_hero_recompute_clip)

    # clip-extract-colab: no Docker, simple per-clip_len features using open_clip
    def _clip_extract_colab(args):
        script = ROOT / "tools" / "extract_clip_features_colab.py"
        device = args.device
        if device == "auto":
            try:
                import torch  # type: ignore
                device = "cuda" if torch.cuda.is_available() else "cpu"
            except Exception:
                device = "cpu"
        cmd = [
            sys.executable, str(script),
            "--videos-dir", args.videos_dir,
            "--outdir", args.outdir,
            "--clip-len", str(args.clip_len),
            "--model", args.model,
            "--pretrained", args.pretrained,
            "--device", device,
            "--exts", args.exts,
        ]
        if args.backend and args.backend != "auto":
            cmd += ["--backend", args.backend]
        elif args.use_lighthouse:
            cmd.append("--use-lighthouse")
        if getattr(args, "frames_per_clip", None) is not None:
            cmd += ["--frames-per-clip", str(args.frames_per_clip)]
        if args.recursive:
            cmd.append("--recursive")
        run(cmd)

    sp = sub.add_parser("clip-extract-colab", help="Colab-friendly CLIP feature extractor (no Docker)")
    sp.add_argument("--videos-dir", required=True)
    sp.add_argument("--outdir", default=str(ROOT / "hero_colab_out" / "clip-vit_features"))
    sp.add_argument("--clip-len", default="1.5")
    sp.add_argument("--model", default="ViT-B-32")
    sp.add_argument("--pretrained", default="laion2b_s34b_b79k", help="open_clip pretrained tag or hf-hub:<repo>")
    sp.add_argument("--device", default="auto", help="cuda|cpu|auto")
    sp.add_argument("--exts", default=".mp4,.avi,.mov,.mkv,.webm")
    sp.add_argument("--use-lighthouse", action="store_true", help="Use vendor/lighthouse CLIPLoader for decoding (deprecated; use --backend)")
    sp.add_argument("--backend", choices=["auto", "clip", "lighthouse-clip", "slowfast"], default="auto")
    sp.add_argument("--frames-per-clip", type=int, default=8, help="When backend=slowfast: frames per clip to encode (averaged)")
    sp.set_defaults(func=_clip_extract_colab)

    return p


def main() -> int:
    p = build_parser()
    args = p.parse_args()
    args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
