#!/usr/bin/env python3
"""Pre-download a model into the Hugging Face cache the container uses ($HF_CACHE_DIR).

    python scripts/download_models.py                       # DEFAULT_MODEL from .env
    python scripts/download_models.py --model z-image-turbo-q4 --lora

Optional: the server downloads on first start anyway. Doing it here shows progress and
keeps the first `docker compose up` short. Needs: pip install "huggingface_hub[hf_xet]".
The model list is read from app/models.py, so it always matches the server.
"""

from __future__ import annotations

import argparse
import ast
import os
from pathlib import Path

import envfile


def registry() -> dict[str, dict]:
    """Parse the ModelSpec(...) entries in app/models.py without importing torch."""
    tree = ast.parse((envfile.REPO / "app" / "models.py").read_text(encoding="utf-8"))
    specs = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "ModelSpec":
            fields = {kw.arg: ast.literal_eval(kw.value) for kw in node.keywords}
            specs[fields["id"]] = fields
    return specs


def main() -> None:
    specs = registry()
    cfg = envfile.settings()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=cfg.get("DEFAULT_MODEL", "z-image-turbo-bf16-te4"), choices=sorted(specs))
    ap.add_argument("--lora", action="store_true", help="also fetch the LoRAs listed in LORAS")
    args = ap.parse_args()

    cache = Path(cfg.get("HF_CACHE_DIR", "./hf-cache"))
    if not cache.is_absolute():
        cache = envfile.REPO / cache
    os.environ["HF_HOME"] = str(cache)  # same layout the container reads from /hf-cache
    from huggingface_hub import hf_hub_download, snapshot_download

    spec = specs[args.model]
    skip = ["assets/*", "*.pdf", "*.png", "*.webp", "*.jpg"]
    if spec.get("transformer_gguf"):
        skip.append("transformer/*.safetensors")  # the GGUF file replaces the 12 GB bf16 transformer
    print(f"{args.model}: {spec['repo']} -> {snapshot_download(spec['repo'], ignore_patterns=skip)}")
    if spec.get("transformer_gguf"):
        print(f"  transformer: {hf_hub_download(*spec['transformer_gguf'])}")
    if args.lora:
        for repo in filter(None, (x.strip() for x in cfg.get("LORAS", "").split(","))):
            print(f"  LoRA: {snapshot_download(repo, allow_patterns=['*.safetensors'])}")


if __name__ == "__main__":
    main()
