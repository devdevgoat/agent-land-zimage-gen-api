# Benchmarks

Everything measured while building this repo (2026-10-05 to 2026-10-06), on one machine. **Time** is end-to-end request time (queue + generation + PNG encoding), except where a per-step figure is given. **GPU memory** is for the whole card from `nvidia-smi`. During these runs the card was also hosting a Breeze TTS server: about 17–18 GB (PyTorch) at first, about 5 GB (GGUF) later. The image service's own share is estimated by subtraction.

To add results for your hardware, run `python scripts/benchmark.py --label "<gpu>, <model>" --notes "<conditions>"` on the server. It writes `benchmarks/results/<date>_<gpu>_<model>.json`; add a row below. **That script was written after these measurements and hasn't been run on this machine yet.** The numbers below come from equivalent manual runs (same prompt, LoRA, seeds and sizes).

## Test system

| | |
|---|---|
| GPU | NVIDIA GeForce RTX 4090, 24 GB (24,564 MiB), compute capability 8.9, 450 W limit, PCIe 4.0 x16 |
| Driver | 591.86 (reports CUDA 13.1); image `pytorch/pytorch:2.9.1-cuda12.8-cudnn9-runtime` |
| CPU | Intel Core i9-12900K, 16 cores / 24 threads |
| RAM | 32 GB (Docker VM: 16 CPUs, ~31 GB) |
| OS | Windows 11 Pro (build 26200), Docker Desktop 28.4.0 on WSL2 (kernel 6.18) |
| Libraries | diffusers 0.40.0, transformers 5.18.0, torch 2.9.1+cu128, bitsandbytes 0.50.2, gguf 0.19.0, peft 0.21.2 |

## Models tested

All variants are Z-Image Turbo (6.15 B parameters, Apache-2.0). They differ in how the transformer and text encoder are stored.

| Registry id | Transformer | Text encoder (Qwen3, 4 B) | Download | Tested |
|---|---|---|---|---|
| `z-image-turbo-bf16-te4` (default) | bf16, 12.3 GB | NF4 4-bit (bitsandbytes) | ~21 GB | yes |
| `z-image-turbo-q4` | GGUF Q4_K_M (unsloth), 5.0 GB | NF4 | ~13 GB | yes |
| `z-image-turbo-q3` | GGUF Q3_K_M (unsloth), 4.2 GB | NF4 | ~12 GB | yes |
| `z-image-turbo-fp8` | bf16 weights **stored** as FP8 and converted back to bf16 layer by layer while running | NF4 | ~21 GB | partly: works without LoRA; **fails with any LoRA** |
| `z-image-turbo` | bf16 | bf16 (8 GB) | ~21 GB | no (needs ~21 GB+ free) |

The LoRA used throughout is `tarn59/pixel_art_style_lora_z_image_turbo` (Apache-2.0), at strength 1.0.

## Summary: weights kept on the GPU (`OFFLOAD=none`) beat everything else

Text-to-image with the LoRA, 9 steps, median of 3. Image-to-image is 768 px at strength 0.78, the sprite-sheet cell setting.

| Model / mode | 512 | 768 | 1024 | img2img 768 | Image service GPU memory |
|---|---|---|---|---|---|
| **`bf16-te4`, `OFFLOAD=none`** | **1.4 s** | **2.9 s** | **6.1 s** | **2.7 s** | ~14 GB idle, ~18 GB peak |
| `q4`, `OFFLOAD=none` | ~3.3 s (from step time; 6.5 s measured while TTS was busy) | 4.3 s | 6.9 s | 4.0 s | ~8.5 GB idle |
| `q4`, `OFFLOAD=model` | 6.6–7.0 s | not measured | 10.1–10.6 s | 6.7 s (512) | ~0 idle, +5.7 GB (512) / +6.1 GB (1024) peak |
| `q3`, `OFFLOAD=model` | 8.5 s | 16.5 s | 22.5 s | 17.1 s | ~1 GB idle, ~6 GB peak |
| `fp8`, `OFFLOAD=none` (no LoRA) | ~1.6 s (0.156 s per step) | not measured | not measured | not measured | ~8 GB |

**Per-step time** (from the server's timing log), weights on the GPU:

| Model | 512 | 768 | 1024 |
|---|---|---|---|
| `bf16-te4` | 0.131 s | 0.279 s | 0.550 s |
| `q4` (GGUF unpacks its weights every step) | 0.356 s | 0.450 s | 0.730 s |

Text encoding (4-bit) takes about 0.1–0.2 s and VAE decoding about 0.04–0.18 s, so the diffusion steps dominate. With `OFFLOAD=model`, moving the weights onto the GPU for each request adds about 3–5 s, which is why it's slow at every size.

## Sprite sheets (`POST /sprite-sheet`)

| Setup | Sheet | Time |
|---|---|---|
| `bf16-te4`, weights on GPU, render 768 → cells 1024 | 4 cells, from a sprite + `pixelate` | **13.5 s** |
| same | 4 cells, from text | **16.9 s** |
| same, `render_size` 1024 | 4 cells, from text | 27.5 s |
| `q3`, `OFFLOAD=model`, render 768 | 4 cells, from a sprite / from text | 53 s / 47 s |
| `q4`, `OFFLOAD=model`, 512 cells | 9 cells | ~67 s from text; 88 s from a reference image (JSON mode) |
| same | 12 cells / 16 cells | ~105 s / ~128–132 s |
| same, while the TTS server was generating | 9 cells | 180 s (3 cells took ~40 s each instead of ~7 s) |
| `q4`, `OFFLOAD=model`, render 1024, TTS busy | 4 cells | timed out (504): GPU memory spilled into system RAM |

**First request after a start:** about 2–12 s extra, while the LoRA is loaded from the cache and its adapter is applied.

## Cold start (loading the model from the Hugging Face cache)

| Model | First load | Later loads |
|---|---|---|
| `q4` | 204 s (reading about 9 GB through the Windows bind mount) | 90–125 s |
| `q3` | 173 s (including the Q3 GGUF download) | ~100 s |
| `bf16-te4` | not timed (12 GB transformer through the bind mount) | 100–125 s |

The first-ever start also downloads the model, which is about 13–21 GB.

## Sharing the GPU with a TTS server

| Situation | Effect |
|---|---|
| Image service `bf16-te4` (weights on GPU) next to Breeze TTS GGUF (~5 GB), image rendering while TTS speaks | GPU peaked at **23.8 / 24.6 GB**. TTS slowed to RTF 3.6–5.0. A 1024 sprite sheet took 74 s instead of 27.5 s |
| `q3` (`OFFLOAD=model`) next to TTS | Fits easily; TTS about 1.6× slower during image bursts |
| `q4` (`OFFLOAD=model`) next to PyTorch TTS (~18 GB) | Peak 23.9 / 24.6 GB: works, but speech during image jobs spilled and slowed, and image jobs timed out |
| A model job aborted by the timeout, before the fix | About 5 GB stayed on the GPU, and every later job spilled and timed out. **Fixed:** weights are now released after any job |

**The cause:** when total GPU memory goes past the card's capacity, Windows/WSL2 spills into system RAM instead of failing, and everything on the card slows down several times. `scripts/configure.py` sizes the model to the **free** memory for this reason. On this machine, with the TTS server running, it picks `z-image-turbo-q4`.

Two other GPU faults were seen during the build: a CUDA "unknown error" around an NVIDIA driver reset. The service now exits on a CUDA fault so Docker restarts it clean, and a hang watchdog restarts it if a step stalls.

## Image-to-image strength (quality, not speed)

Measured on sprite-sheet cells, same seed:

| Strength | Effect |
|---|---|
| 0.5 | Close to a copy; expressions barely change |
| 0.65 | Identity holds; expressions change but hands stay where the base portrait had them |
| 0.72 | Hands still frozen in the base pose |
| **0.78** (default) | Expressions plus hand gestures, with identity held, **given** a chest-up base and a concrete outfit description |
| 0.8+ | A different character |

## Build

| | Time |
|---|---|
| Docker image (pip installs on top of the PyTorch runtime image) | a few minutes; first base-image pull ~4 GB |
