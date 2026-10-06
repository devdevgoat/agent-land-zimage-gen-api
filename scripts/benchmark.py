#!/usr/bin/env python3
"""Benchmark a running server and save the results with the hardware and config.

    python scripts/benchmark.py                                  # server from .env (127.0.0.1:$IMAGE_GEN_PORT)
    python scripts/benchmark.py --runs 5 --label "my-gpu, q4" --notes "TTS idle on the same GPU"

Measures (house pixel-art style, fixed seeds, after one warm-up that also loads the LoRA):
  * text-to-image at 512 / 768 / 1024 (median of --runs)
  * image-to-image at 768, strength 0.78 (the sprite-sheet cell setting)
  * one default sprite sheet (4 cells) from text
  * GPU memory: idle before and peak during (whole card, all processes)

Writes benchmarks/results/<date>_<gpu>_<model>.json and prints Markdown rows for BENCHMARKS.md.
Needs: pip install requests. Run it on the server machine to capture GPU info.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import platform
import re
import statistics
import subprocess
import threading
import time
from pathlib import Path

import requests

import envfile

LORA = "tarn59/pixel_art_style_lora_z_image_turbo"
PROMPT = ("Pixel art style. Close-up chest-up portrait of an elf archer with a green hood, calm expression, "
          "16-bit game character portrait, simple abstract background")
CHARACTER = "a young office worker with short black hair, brown eyes, round glasses, wearing a white shirt and a navy tie"


def nvidia(query: str) -> list[str]:
    try:
        return subprocess.run(["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"],
                              capture_output=True, text=True, check=True).stdout.strip().splitlines()
    except (OSError, subprocess.CalledProcessError):
        return []


def cpu_name() -> str:
    try:
        if platform.system() == "Windows":
            return subprocess.run(["powershell", "-NoProfile", "-Command",
                                   "(Get-CimInstance Win32_Processor | Select-Object -First 1).Name"],
                                  capture_output=True, text=True, timeout=20).stdout.strip()
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except (OSError, subprocess.SubprocessError):
        pass
    return platform.processor() or platform.machine()


def os_name() -> str:
    if platform.system() == "Windows":
        build = int(platform.version().split(".")[-1])
        return f"Windows {'11' if build >= 22000 else '10'} (build {build})"
    return platform.platform()


class VramSampler(threading.Thread):
    def __init__(self) -> None:
        super().__init__(daemon=True)
        self.peak = 0
        self.stop = threading.Event()

    def run(self) -> None:
        while not self.stop.is_set():
            used = nvidia("memory.used")
            if used:
                self.peak = max(self.peak, max(int(x) for x in used))
            time.sleep(0.25)


def post(url: str, path: str, body: dict, headers: dict) -> tuple[float, dict]:
    t0 = time.perf_counter()
    r = requests.post(f"{url}{path}", json={**body, "format": "json"}, headers=headers, timeout=900)
    r.raise_for_status()
    return time.perf_counter() - t0, r.json()


def main() -> None:
    cfg = envfile.settings()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default=f"http://127.0.0.1:{cfg.get('IMAGE_GEN_PORT', '8000')}")
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--label", default="")
    ap.add_argument("--notes", default="", help="conditions worth recording (e.g. other GPU load)")
    args = ap.parse_args()
    headers = {"Authorization": f"Bearer {cfg['API_TOKEN']}"} if cfg.get("API_TOKEN") else {}

    health = requests.get(f"{args.url}/health", timeout=10).json()
    if health.get("status") != "ok":
        raise SystemExit(f"server not ready: {health}")
    gpu = nvidia("name,compute_cap,memory.total,driver_version")
    system = {"gpu": gpu[0] if gpu else "unknown (nvidia-smi not available here)", "cpu": cpu_name(), "os": os_name()}
    config = {"model": health.get("model"), **{k: cfg.get(k) for k in ("OFFLOAD", "PYTORCH_IMAGE", "MAX_SIDE")}}
    vram_idle = max((int(x) for x in nvidia("memory.used")), default=None)

    sampler = VramSampler()
    sampler.start()
    post(args.url, "/generate", {"prompt": PROMPT, "width": 512, "height": 512, "seed": 1, "lora": LORA}, headers)
    t2i = {}
    for size in (512, 768, 1024):
        times = [post(args.url, "/generate", {"prompt": PROMPT, "width": size, "height": size, "seed": 3,
                                              "lora": LORA}, headers)[0] for _ in range(args.runs)]
        t2i[size] = {"median_s": round(statistics.median(times), 2), "runs_s": [round(t, 2) for t in times]}
    _, ref = post(args.url, "/generate", {"prompt": PROMPT, "width": 768, "height": 768, "seed": 3, "lora": LORA},
                  headers)
    i2i = [post(args.url, "/generate", {"prompt": PROMPT, "width": 768, "height": 768, "seed": 3, "lora": LORA,
                                        "image": ref["image"], "strength": 0.78}, headers)[0]
           for _ in range(args.runs)]
    sheet_s, sheet = post(args.url, "/sprite-sheet", {"character": CHARACTER, "seed": 21}, headers)
    sampler.stop.set()

    results = {
        "date": dt.datetime.now().isoformat(timespec="seconds"),
        "label": args.label, "notes": args.notes, "url": args.url, "system": system, "config": config,
        "vram_mb": {"idle": vram_idle, "peak": sampler.peak or None},
        "text_to_image": {str(k): v for k, v in t2i.items()},
        "image_to_image_768": {"median_s": round(statistics.median(i2i), 2), "runs_s": [round(t, 2) for t in i2i]},
        "sprite_sheet_default": {"seconds": round(sheet_s, 1), "cells": sheet.get("expressions"),
                                 "render_size": 768, "cell_size": sheet.get("cell_size")},
    }
    gpu_name = re.sub(r"[^A-Za-z0-9]+", "-", system["gpu"].split(",")[0]).strip("-").lower()
    out = envfile.REPO / "benchmarks" / "results" / f"{dt.date.today()}_{gpu_name}_{config['model']}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")

    print(f"saved {out.relative_to(envfile.REPO)}\n")
    print("| Model | 512 | 768 | 1024 | img2img 768 | 4-cell sheet | GPU memory idle / peak (whole card) |")
    print("|---|---|---|---|---|---|---|")
    print(f"| `{config['model']}` (OFFLOAD={config['OFFLOAD']}) | {t2i[512]['median_s']} s | {t2i[768]['median_s']} s"
          f" | {t2i[1024]['median_s']} s | {results['image_to_image_768']['median_s']} s | {sheet_s:.1f} s"
          f" | {vram_idle} / {sampler.peak} MB |")


if __name__ == "__main__":
    main()
