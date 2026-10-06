# AGENTS.md: deploying agent-land-zimage-gen-api

This is a step-by-step guide for an AI agent (or a person) to deploy this image-generation API on a fresh machine and confirm it works. Follow the steps in order. Each step says how to check it before moving on.

## What you are deploying

- **What it is:** a FastAPI service in Docker that runs [Z-Image Turbo](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo) on an NVIDIA GPU, on port **8000**.
- **Endpoints:**
  - `POST /generate`: text-to-image and image-to-image, with optional LoRAs.
  - `POST /sprite-sheet`: one character in several moods, from text and/or a reference sprite.
- **Docs:** the full API is in [API.md](API.md); a running server also serves it at `GET /usage`.

Pick a model for the GPU (set with `DEFAULT_MODEL` and `OFFLOAD`):

| GPU memory free | `DEFAULT_MODEL` | `OFFLOAD` | Speed (RTX 4090, 768 px) | Download |
|---|---|---|---|---|
| ≥ 20 GB (default) | `z-image-turbo-bf16-te4` | `none` | ~2.9 s per image; 4-cell sheet ~15 s | ~21 GB |
| 12–20 GB, or sharing the GPU | `z-image-turbo-q4` | `none` | ~4.3 s per image | ~13 GB |
| 8–12 GB | `z-image-turbo-q3` | `model` | ~8–17 s per image | ~12 GB |

## Rules

- **Licences allow commercial use:** the model (Z-Image Turbo), its GGUF quantizations and the pixel-art LoRA are all Apache-2.0. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
- **Nothing large or secret in git:** never commit `hf-cache/` (the model weights), `outputs/` or `.env`.
- **No internet exposure:** authentication is **off** by default (`API_TOKEN` empty). Keep it on a LAN or VPN, or set `API_TOKEN`. Never forward port 8000 to the internet.
- **Ask first:** get the user's go-ahead before changing a firewall or stopping a container you didn't start.

## 1. Check prerequisites

```bash
nvidia-smi                                         # NVIDIA GPU; note memory used/total (pick a model above)
docker version                                     # Docker Engine (Linux) or Docker Desktop (Windows, WSL2)
docker run --rm --gpus all nvidia/cuda:12.8.1-base-ubuntu22.04 nvidia-smi   # GPU visible in containers
docker compose version                             # v2.30+ (the compose file uses `gpus: all`)
git --version
```

- **GPU not visible in containers:** install the NVIDIA Container Toolkit (Linux), or enable WSL2 GPU support in Docker Desktop. Stop and tell the user if you can't.
- **Disk:** about 25–35 GB free for the model cache and the image.
- **System RAM:** 32 GB recommended.

## 2. Get the code and configure

```bash
git clone https://github.com/devdevgoat/agent-land-zimage-gen-api.git
cd agent-land-zimage-gen-api
cp .env.example .env
```

In `.env`, set `DEFAULT_MODEL` / `OFFLOAD` if the default doesn't fit the GPU (see the table above). Leave `API_TOKEN` empty for no authentication, or set a long random string.

## 3. Build and start

```bash
docker compose up -d --build
```

- **Build:** a few minutes. The base image is `pytorch/pytorch:2.9.1-cuda12.8-cudnn9-runtime`, plus pip packages.
- **First start:** downloads the model into `./hf-cache/` (about 13–21 GB depending on the model), then loads it. That can take **10–30 minutes**. Later starts take about 1.5–3.5 minutes, mostly reading the weights.

Wait until it's ready:

```bash
until curl -fsS http://127.0.0.1:8000/health | grep -q '"status":"ok"'; do sleep 10; done
curl -s http://127.0.0.1:8000/health      # {"status":"ok","model_loaded":true,"model":"...","gpu":{...}}
docker compose logs --tail 20             # "loaded <model> in N s"
```

If `/health` reports `"status":"error"`, read `docker compose logs`. The usual causes are out of GPU memory (pick a smaller model) or a failed download (rerun `docker compose up -d`).

## 4. Verify it works

```bash
# Text to image (house pixel-art style; the LoRA downloads on first use)
curl -sS http://127.0.0.1:8000/generate -H "Content-Type: application/json" \
  -d '{"prompt":"Pixel art style. Portrait of an elf archer with green hood, 16-bit game character portrait, plain dark background","width":512,"height":512,"seed":5,"lora":"tarn59/pixel_art_style_lora_z_image_turbo"}' \
  -o portrait.png

# Sprite sheet: 4 moods (stressed, accomplished, chill, confused), 2x2 grid
curl -sS --max-time 900 http://127.0.0.1:8000/sprite-sheet -H "Content-Type: application/json" -D sheet.h \
  -d '{"character":"a young office worker with short black hair, brown eyes, round glasses, wearing a white shirt and a navy tie","seed":21,"labels":true,"gap":8}' \
  -o sheet.png
grep -i x-expressions sheet.h            # stressed,accomplished,chill,confused

curl -s http://127.0.0.1:8000/usage | head -5     # the API guide
```

- **`portrait.png`:** a 512×512 PNG.
- **`sheet.png`:** about 2072×2200 with labels and gaps.
- **Look at both:** open the images and check they show what was asked. A 200 response alone doesn't prove the output is right.

## 5. Make it reachable on the LAN (ask the user first)

- **Linux:** Docker publishes `0.0.0.0:8000`. Check from another machine with `curl http://<server-ip>:8000/health`.
- **Windows:** inbound traffic is blocked by default. **Ask the user** before adding a rule. The command, for an Administrator PowerShell, allows the local subnet only:

```powershell
New-NetFirewallRule -DisplayName "Image Gen 8000" -Direction Inbound -Protocol TCP -LocalPort 8000 -Action Allow -Profile Private -RemoteAddress LocalSubnet
```

## 6. Operate it

```bash
docker compose stop        # stop (stays stopped across reboots; the model cache is kept)
docker compose start       # start again
docker compose logs -f     # per-request size/steps/time, and a timing breakdown per job
```

**Settings** (`.env` or environment):

| Variable | Default | Meaning |
|---|---|---|
| `API_TOKEN` | empty | Empty: no auth. Set: `Authorization: Bearer <token>` required (except `/health`, `/usage`) |
| `DEFAULT_MODEL` | `z-image-turbo-bf16-te4` | See the model table above. Also `z-image-turbo-fp8` (~8 GB, fast, **no LoRA support**) and `z-image-turbo` (all bf16, ~21 GB) |
| `OFFLOAD` | `none` | `none`: weights stay on the GPU. `model`: moved per job (small, slow) |
| `MAX_SIDE` | `1024` | Maximum width and height |
| `JOB_TIMEOUT_SECONDS` | `120` | Per GPU job; a stuck job past this plus 60 s restarts the service |
| `LORAS` | the pixel-art LoRA | LoRAs listed by `/models` |
| `LORA_ALLOWLIST` | (any) | Restrict which LoRA repos callers may load |

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `503` | Model loading (first start can take 10–30 min), or queue full; retry |
| `504` | A job went over 120 s, usually GPU memory spilling into system RAM because another workload shares the GPU. Use `z-image-turbo-q4`, or free the GPU |
| `507` | Out of GPU memory; smaller size or smaller model |
| `500` with a LoRA on `z-image-turbo-fp8` | Known: FP8 storage doesn't work with LoRAs; use `z-image-turbo-bf16-te4` or `-q4` |
| Very slow everything, `nvidia-smi` near full | Spill into system RAM. Stop other GPU jobs or pick a smaller model |
| Other GPU apps slow down badly while images generate | Same cause: they share memory and compute. Pick a smaller model, or stop image-gen when not needed |

## When you're done

Report to the user:
- the URL (`http://<server-ip>:8000`) and that `GET /usage` serves the API guide,
- the model and memory mode,
- what you verified (health, one image, one sprite sheet, and that you looked at them),
- anything you skipped (for example the firewall rule) and why.
