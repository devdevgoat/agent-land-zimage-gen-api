# AGENTS.md: deploying agent-land-zimage-gen-api

This is a step-by-step guide for an AI agent (or a person) to deploy this image-generation API on a fresh machine, adapt it to the hardware, and confirm it works. Follow the steps in order. Each step says how to check it before moving on.

## The one rule for changes: edit `.env`, not code

Everything that varies between machines lives in **`.env`** (created from the commented `.env.example`): model, memory mode, GPU, base image, port, auth and limits. `compose.yaml` reads only those variables. Edit code only to change behaviour, for example adding a model to `app/models.py`.

| To change... | Set in `.env` | Then |
|---|---|---|
| Model / memory use | `DEFAULT_MODEL`, `OFFLOAD`; `scripts/configure.py` picks them from free GPU memory | `docker compose up -d` |
| Which GPU | `IMAGE_GEN_GPU=all` or an index / UUID from `nvidia-smi -L` | `docker compose up -d` |
| CUDA / driver compatibility | `PYTORCH_IMAGE`; `configure.py` picks it from the driver | `docker compose up -d --build` |
| Port / LAN exposure | `IMAGE_GEN_PORT`, `IMAGE_GEN_BIND` (`127.0.0.1` = this machine only) | `docker compose up -d` |
| Authentication | `API_TOKEN` (empty = none) | `docker compose up -d` |
| Where models are cached | `HF_CACHE_DIR` | `docker compose up -d` |
| Limits | `MAX_SIDE`, `JOB_TIMEOUT_SECONDS`, `QUEUE_MAX`, `LORA_ALLOWLIST` and others | `docker compose up -d` |
| A second copy side by side | `COMPOSE_PROJECT_NAME`, `IMAGE_GEN_CONTAINER_NAME`, `IMAGE_GEN_PORT` | `docker compose up -d --build` |

## What you are deploying

- **What it is:** a FastAPI service in Docker running [Z-Image Turbo](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo) on an NVIDIA GPU, on port 8000 by default.
- **Endpoints:**
  - `POST /generate`: text-to-image and image-to-image, with LoRAs.
  - `POST /sprite-sheet`: one character in several moods, from text and/or a reference sprite.
- **Docs:** [API.md](API.md), also served at `GET /usage`.

Measured on an RTX 4090 ([BENCHMARKS.md](BENCHMARKS.md) has the full numbers and hardware):

| GPU memory free | `DEFAULT_MODEL` + `OFFLOAD` | Speed (768 px image / 4-cell sheet) | Download |
|---|---|---|---|
| ≥ 20 GB | `z-image-turbo-bf16-te4` + `none` | 2.9 s / ~15–17 s | ~21 GB |
| 11–20 GB, or **sharing the GPU** | `z-image-turbo-q4` + `none` | 4.3 s / ~25 s (est.) | ~13 GB |
| 7–11 GB | `z-image-turbo-q3` + `model` | 16.5 s / ~50 s | ~12 GB |

## Rules

- **Licences allow commercial use:** the model, its GGUF quantizations and the pixel-art LoRA are all Apache-2.0. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
- **Nothing large or secret in git:** never commit `.env`, `hf-cache/` or `outputs/`.
- **No internet exposure:** authentication is **off** by default. Keep it on a LAN or VPN, or set `API_TOKEN`. Never forward the port to the internet.
- **Ask before risky changes:** get the user's go-ahead before changing a firewall, stopping containers you didn't start, or **starting this next to another GPU service**. When total GPU memory passes the card's capacity, everything on it slows down several times; during this repo's build that also crashed a neighbouring TTS engine. See BENCHMARKS.md, "Sharing the GPU".

## 1. Check prerequisites

```bash
nvidia-smi                                          # NVIDIA GPU; note free memory and "CUDA Version"
docker version && docker compose version            # compose v2.30+ (uses `gpus: all`)
docker run --rm --gpus all nvidia/cuda:12.8.1-base-ubuntu22.04 nvidia-smi   # GPU visible in containers
git --version && python3 --version                  # 3.10+ for the scripts
python3 -m pip install -U "huggingface_hub[hf_xet]" requests
```

- **GPU not visible in containers:** install the NVIDIA Container Toolkit (Linux), or enable WSL2 GPU support in Docker Desktop. Stop and tell the user if you can't.
- **Disk:** about 25–35 GB.
- **System RAM:** 32 GB recommended.

## 2. Get the code and configure for this machine

```bash
git clone https://github.com/devdevgoat/agent-land-zimage-gen-api.git
cd agent-land-zimage-gen-api
python3 scripts/configure.py      # detects the GPU, sizes the model to FREE memory, writes .env
```

Check the printed settings. If other GPU workloads are running, it picks a smaller model on purpose; it prints a note when that happens. Set `API_TOKEN` in `.env` if the user wants authentication.

## 3. Download the model (optional, recommended)

```bash
python3 scripts/download_models.py --lora     # model from .env + the default LoRA, into $HF_CACHE_DIR
```

This shows progress and makes the first start quick. Without it, the server downloads on first start, which can take 10–30 minutes with no progress shown.

## 4. Build and start

```bash
docker compose up -d --build
until curl -fsS http://127.0.0.1:8000/health | grep -q '"status":"ok"'; do sleep 10; done
curl -s http://127.0.0.1:8000/health      # {"status":"ok","model_loaded":true,"model":"...","gpu":{...}}
docker compose logs --tail 20             # "loaded <model> in N s"
```

Use your `IMAGE_GEN_PORT` if you changed it. Loading from the cache takes about 1.5–3.5 minutes. If `/health` says `"status":"error"`, read the logs: usually it's out of GPU memory (pick a smaller model in `.env`).

## 5. Verify, then benchmark

```bash
curl -sS http://127.0.0.1:8000/generate -H "Content-Type: application/json" \
  -d '{"prompt":"Pixel art style. Portrait of an elf archer with green hood, 16-bit game character portrait, plain dark background","width":512,"height":512,"seed":5,"lora":"tarn59/pixel_art_style_lora_z_image_turbo"}' \
  -o portrait.png
curl -sS --max-time 900 http://127.0.0.1:8000/sprite-sheet -H "Content-Type: application/json" -D sheet.h \
  -d '{"character":"a young office worker with short black hair, brown eyes, round glasses, wearing a white shirt and a navy tie","seed":21,"labels":true,"gap":8}' \
  -o sheet.png
grep -i x-expressions sheet.h            # stressed,accomplished,chill,confused
python3 scripts/benchmark.py --label "<gpu>, <model>" --notes "<other GPU load, if any>"
```

- **Look at the images:** `portrait.png` should be a 512×512 pixel-art portrait, and `sheet.png` a labelled 2×2 sheet of the same character in four moods. A 200 response alone doesn't prove the output is right.
- **Benchmark:** `benchmark.py` saves `benchmarks/results/<date>_<gpu>_<model>.json` and prints a Markdown row. Compare it with BENCHMARKS.md and report it to the user.

## 6. Make it reachable on the LAN (ask the user first)

- **Linux:** usually reachable as is. Check from another machine with `curl http://<server-ip>:<port>/health`.
- **Windows:** inbound traffic is blocked by default. **Ask the user** before adding a rule. The command, for an Administrator PowerShell, allows the local subnet only:

```powershell
New-NetFirewallRule -DisplayName "Image Gen 8000" -Direction Inbound -Protocol TCP -LocalPort 8000 -Action Allow -Profile Private -RemoteAddress LocalSubnet
```

## Operate

```bash
docker compose stop            # stop (stays stopped across reboots; the model cache is kept)
docker compose start           # start again
docker compose logs -f         # per-request size/steps/time, plus a per-stage timing line per job
docker compose up -d           # apply .env changes
docker compose up -d --build   # after changing PYTORCH_IMAGE, requirements.txt or app code
```

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `503` | Model loading, or queue full. Retry with backoff |
| `504` | A job went over `JOB_TIMEOUT_SECONDS`, usually GPU memory spilling into system RAM because something else uses the GPU. Rerun `configure.py` (it sizes for free memory), or free the GPU |
| `507` | Out of GPU memory: smaller size or smaller model |
| `500` with a LoRA on `z-image-turbo-fp8` | Known: FP8 storage doesn't work with LoRAs. Use `bf16-te4` or `q4` |
| Everything slow, `nvidia-smi` near full | Memory spilled into system RAM. Stop other GPU jobs or pick a smaller model |
| Other GPU services slow down while images render | They share the card. Use `q4`, or stop image-gen when not needed |
| Service restarted by itself | Intended after a CUDA fault or a stuck job (watchdog). Check `docker compose logs` for `FATAL CUDA error` or `stuck` |

## When you're done

Report to the user:
- the URL (`http://<server-ip>:<port>`) and `/usage`,
- the model and memory mode,
- the GPU and the benchmark row,
- what you verified, including that you looked at the images,
- anything you skipped (for example the firewall) and why.
