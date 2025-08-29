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

    return p


def main() -> int:
    p = build_parser()
    args = p.parse_args()
    args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
