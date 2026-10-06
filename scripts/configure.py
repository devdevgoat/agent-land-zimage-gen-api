#!/usr/bin/env python3
"""Detect the GPU and write matching settings into .env.

    python scripts/configure.py                  # detect, print, write .env
    python scripts/configure.py --gpu 1          # size for GPU index 1 and use only that GPU
    python scripts/configure.py --dry-run        # print only

Sets DEFAULT_MODEL and OFFLOAD from the free GPU memory, PYTORCH_IMAGE from the driver's
CUDA version, and IMAGE_GEN_GPU. Everything else in .env is left as it is.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys

import envfile

# (min free GB, DEFAULT_MODEL, OFFLOAD): measured peaks on an RTX 4090, plus headroom.
MODEL_BY_VRAM = [
    (20.0, "z-image-turbo-bf16-te4", "none"),   # ~14 GB idle, ~18 GB peak
    (11.0, "z-image-turbo-q4", "none"),         # ~8.5 GB, ~10 GB peak
    (7.0, "z-image-turbo-q3", "model"),         # ~6 GB peak, ~0 idle
]
IMAGES = [  # (min driver CUDA version, base image)
    ((12, 8), "pytorch/pytorch:2.9.1-cuda12.8-cudnn9-runtime"),   # tested
    ((12, 6), "pytorch/pytorch:2.9.1-cuda12.6-cudnn9-runtime"),
]


def smi(*args: str) -> str:
    try:
        return subprocess.run(["nvidia-smi", *args], capture_output=True, text=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError) as error:
        sys.exit(f"nvidia-smi failed ({error}). An NVIDIA GPU and driver are required.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gpu", help="GPU index to use (default: all visible, sized on GPU 0)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    rows = []
    for line in smi("--query-gpu=index,name,compute_cap,memory.total,memory.used,driver_version",
                    "--format=csv,noheader,nounits").strip().splitlines():
        index, name, cap, total, used, driver = (x.strip() for x in line.split(","))
        rows.append({"index": index, "name": name, "cap": cap, "total": int(total) / 1024,
                     "free": (int(total) - int(used)) / 1024, "driver": driver})
        print(f"GPU {index}: {name}, compute {cap}, {int(total) / 1024:.1f} GB total, "
              f"{(int(total) - int(used)) / 1024:.1f} GB free, driver {driver}")
    gpu = next((g for g in rows if g["index"] == args.gpu), rows[0])
    match = re.search(r"CUDA Version:\s*(\d+)\.(\d+)", smi())
    cuda = (int(match.group(1)), int(match.group(2))) if match else (0, 0)
    print(f"driver supports CUDA {cuda[0]}.{cuda[1]}")

    model = next(((m, o) for need, m, o in MODEL_BY_VRAM if gpu["free"] >= need), None)
    if model is None:
        print(f"warning: only {gpu['free']:.1f} GB free; even z-image-turbo-q3 with OFFLOAD=model may not fit.")
        model = MODEL_BY_VRAM[-1][1:]
    image = next((img for need, img in IMAGES if cuda >= need), None)
    if image is None:
        print("warning: driver older than CUDA 12.6; update the NVIDIA driver.")
        image = IMAGES[-1][1]
    if gpu["cap"].startswith("12.") and cuda < (12, 8):
        print("warning: RTX 50xx (Blackwell) needs a driver with CUDA >= 12.8 and the cu128 image.")

    changes = {"DEFAULT_MODEL": model[0], "OFFLOAD": model[1], "PYTORCH_IMAGE": image,
               "IMAGE_GEN_GPU": args.gpu if args.gpu is not None else "all"}
    print("\nsettings:")
    for k, v in changes.items():
        print(f"  {k}={v}")
    if gpu["free"] < gpu["total"] - 1.5:
        print(f"\nnote: {gpu['total'] - gpu['free']:.1f} GB is already in use by other processes. If they also "
              "generate while images render, both slow down a lot (see BENCHMARKS.md, 'Sharing the GPU').")
    if args.dry_run:
        print("\n(dry run: .env not changed)")
        return
    envfile.update(changes)
    print(f"\nwrote {envfile.ENV}. Next: docker compose up -d --build")


if __name__ == "__main__":
    main()
