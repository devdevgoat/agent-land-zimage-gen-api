# agent-land-zimage-gen-api

A self-hosted image-generation API for [Z-Image Turbo](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo), in Docker on an NVIDIA GPU. It's built for pixel-art game portraits and character sprite sheets, generated from a text description, a reference sprite, or both.

- **`POST /generate`:**
  - **Modes:** text-to-image, and image-to-image from a reference picture (diffusers' native `ZImageImg2ImgPipeline`).
  - **LoRAs:** optional, from any Hugging Face repo (safetensors only).
  - **Output:** PNG, or base64 in JSON.
- **`POST /sprite-sheet`:**
  - **Default cells:** one character in 4 moods: `stressed` (working frantically), `accomplished` (finally done), `chill` (waiting for the next task), `confused` (asking a question).
  - **Look:** 1024 px cells on an abstract background that fits the character's mood.
  - **Inputs:** text, a reference `sprite` plus text, an optional real 16-bit `pixelate` pass, and up to 4 caller-chosen `activities`.
  - **Stable output:** names and order are the same on every request, and the same seed reproduces the sheet exactly.
- **Swappable models:** one registry entry per model; switch with `DEFAULT_MODEL` or per request.
- **Built for sharing a GPU safely:**
  - **Queueing:** one GPU job at a time, in a queue.
  - **Limits:** a 2-minute limit per job, and a hang watchdog.
  - **Recovery:** the service restarts itself after a CUDA fault.
- **One config file:** model, memory mode, GPU, base image, port and limits are all in `.env`. `scripts/configure.py` detects the GPU and sizes the model to the free memory.
- **Self-documenting:** `GET /usage` serves [API.md](API.md).

> **Deploying with an AI agent?** Point it at **[AGENTS.md](AGENTS.md)**: a step-by-step deploy, adapt and verify guide.

## Quick start

```bash
git clone https://github.com/devdevgoat/agent-land-zimage-gen-api.git && cd agent-land-zimage-gen-api
pip install -U "huggingface_hub[hf_xet]" requests
python scripts/configure.py               # detect the GPU, pick the model for its free memory, write .env
python scripts/download_models.py --lora  # optional: pre-download (13–21 GB) with progress
docker compose up -d --build
curl http://127.0.0.1:8000/health
curl -sS http://127.0.0.1:8000/sprite-sheet -H "Content-Type: application/json" \
  -d '{"character":"a young office worker with short black hair, round glasses, white shirt and navy tie","seed":21,"labels":true}' -o sheet.png
```

## Performance (RTX 4090 24 GB, i9-12900K, Windows 11 + Docker Desktop/WSL2)

| Model / mode | 512 | 768 | 1024 | 4-cell sprite sheet | GPU memory |
|---|---|---|---|---|---|
| `z-image-turbo-bf16-te4`, weights on GPU (default) | 1.4 s | 2.9 s | 6.1 s | ~14–17 s | ~14 GB idle, ~18 GB peak |
| `z-image-turbo-q4` (GGUF 4-bit), weights on GPU | ~3.3 s | 4.3 s | 6.9 s | ~25 s (estimated) | ~8.5 GB |
| `z-image-turbo-q3`, weights moved per job | 8.5 s | 16.5 s | 22.5 s | ~50 s | ~6 GB peak, ~0 idle |

**Sharing a GPU:** running this next to another heavy GPU service, such as a TTS server, can push total memory past the card's capacity. When that happens, both slow down several times. That's why `configure.py` sizes the model to the *free* memory. [BENCHMARKS.md](BENCHMARKS.md) has every measurement: the models tested, per-step timings, cold starts and GPU-sharing results. `scripts/benchmark.py` measures your own hardware.

## Layout

| Path | What |
|---|---|
| `.env.example` → `.env` | **Every setting** you might change, commented: model, memory mode, GPU, base image, port, auth, limits |
| `compose.yaml` | Build and run config; reads only `.env` |
| `app/main.py` | FastAPI app: endpoints, validation, optional bearer auth, GPU job queue, timeouts and watchdog |
| `app/models.py` | Model registry (`MODELS`) and backends (`BACKENDS`); add a model here |
| `app/sprites.py` | Sprite sheets: presets, prompt template, sprite prep, pixelation, grid layout |
| `Dockerfile`, `requirements.txt` | Container build (base image set by `PYTORCH_IMAGE`) |
| `scripts/` | `configure.py` (GPU detection), `download_models.py`, `benchmark.py`, `token_usage.py` |
| `benchmarks/results/` | Raw benchmark results (JSON) from `scripts/benchmark.py` |
| [API.md](API.md) | Full API reference, including the house pixel-art style recipe |
| [AGENTS.md](AGENTS.md), [BENCHMARKS.md](BENCHMARKS.md) | Deploy guide for AI agents; measurements |
| [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md), [CITATION.cff](CITATION.cff) | Licences and citations of everything this builds on |
| [TOKEN_LOG.md](TOKEN_LOG.md), `scripts/token_usage.py` | What it cost in Claude tokens to build this repo, and the script that computes it |

## Licence

- **This repo's code and docs:** [Apache-2.0](LICENSE).
- **Z-Image Turbo, the GGUF quantizations and the pixel-art LoRA it uses by default:** also Apache-2.0, so commercial use is allowed.
- **Details and citations:** see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

## How this was built

This repo was built by Claude (Opus 5.5) in Claude Code, working with the repo owner over one long session. That covered the API, the sprite-sheet design and prompt tuning, sprite input, and performance work. [TOKEN_LOG.md](TOKEN_LOG.md) has the token and cost breakdown.
