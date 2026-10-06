# zimage-gen API spec

Pixel-art game portraits and character sprite sheets (expressions and props). Z-Image Turbo (full bf16 transformer, kept on the GPU) runs on a local NVIDIA GPU.

- **Base URL:** `http://<server-ip>:8000` (LAN), `http://127.0.0.1:8000` (on the host)
- **Auth:** none. The service is LAN-only. (If `API_TOKEN` is ever set in the server's `.env`, every endpoint except `/health` and `/usage` requires `Authorization: Bearer <token>`.)
- **Request bodies:** JSON (`Content-Type: application/json`).
- **Images in:** base64 PNG (a `data:image/png;base64,` prefix is fine).
- **Images out:** raw PNG by default. Add `"format":"json"` (or the header `Accept: application/json`) to get base64 in JSON instead.

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/generate` | One picture: text-to-image, or image-to-image from a reference |
| `POST` | `/sprite-sheet` | One character in many expressions, as a grid |
| `GET` | `/sprite-sheet/expressions` | Expression presets and the default set |
| `GET` | `/models` | Models and LoRAs |
| `GET` | `/health` | Readiness and GPU state (no token) |
| `GET` | `/usage` | This guide as Markdown (no token) |
| `GET` | `/openapi.json`, `/docs` | Machine-readable schema and an interactive page (no token) |

---

## The house style ("LoRA 512" look)

The anime-leaning pixel-art portrait style comes from **four settings together**:

| Setting | Value | Why |
|---|---|---|
| `lora` | `tarn59/pixel_art_style_lora_z_image_turbo` | The pixel-art LoRA (Apache 2.0). Without it you get chunkier, chibi-style sprites |
| `lora_strength` | `1.0` | Full style. 0.7 is softer, and above 1.2 tends to break up |
| Prompt starts with | `Pixel art style.` | The LoRA's trigger phrase. Keep it first |
| Prompt ends with | `16-bit game character portrait, plain dark background` | Gives a framed bust and a clean background that's easy to cut out |

Size is **512 × 512**: the style holds best there, and it takes about 1.5 s. 1024 works too (about 6 s), but it's not more "pixel". Steps stay at the default of 9.

Prompt template:

```
Pixel art style. Portrait of <who: species, role, hair, eyes, outfit>, <expression>, 16-bit game character portrait, plain dark background
```

The exact request that produced the reference portrait for this style:

```json
POST /generate
{
  "prompt": "Pixel art style. Portrait of an elf archer with green hood, 16-bit game character portrait, plain dark background",
  "width": 512, "height": 512, "seed": 5,
  "lora": "tarn59/pixel_art_style_lora_z_image_turbo", "lora_strength": 1.0
}
```

**Same seed + same prompt + same settings gives a byte-identical PNG**, so store the seed to get a picture back. Changing only the seed gives a new character in the same style.

**Prompt tips:**
- **Visual details:** name the colours of hair, eyes and clothing. The model holds onto concrete visual details better than abstract traits like "brave" or "mysterious".
- **Ears and accessories:** pointed ears, horns and similar features can disappear under the LoRA. Mention them explicitly ("pointed elf ears").
- **`negative_prompt`:** it does nothing at the default `guidance_scale` of 0, so leave it out.

---

## POST /generate

| Field | Type | Default | Notes |
|---|---|---|---|
| `prompt` | string | required | 1–4000 characters |
| `width`, `height` | int | 512 | 256–1024, multiples of 16 |
| `seed` | int | random | 0 to 2³²−1. Returned in `X-Seed` |
| `steps` | int | 9 | 1–50. Turbo is tuned for 9 |
| `image` | base64 PNG | — | Reference picture, resized to `width` × `height` |
| `strength` | float | 0.6 | 0–1 (above 0): how much to change the reference. `steps × strength` must be ≥ 1 |
| `lora` | string | — | Hugging Face repo id (safetensors only) |
| `lora_strength` | float | 1.0 | 0–2 |
| `negative_prompt` | string | — | Only has an effect when `guidance_scale` > 0 |
| `guidance_scale` | float | 0 | Leave at 0 for Turbo |
| `model` | string | `z-image-turbo-bf16-te4` | See `/models` |
| `format` | `png` \| `json` | `png` | |

**Response `200`:**
- **PNG:** `image/png`, with headers `X-Seed`, `X-Model`, `X-Elapsed-Ms`.
- **JSON:** `{"image", "mime_type", "seed", "width", "height", "steps", "model", "elapsed_ms"}`.

**Image-to-image strength guide** (same character, changed pose or details):

| Strength | Effect |
|---|---|
| 0.3–0.5 | Recolour or clean up; the composition is kept |
| 0.6–0.65 | Change details or the expression, keeping identity (bust framing) |
| 0.7 | Bigger changes; identity starts to drift (eye colour first) |
| 0.78 | Expression plus hand gestures. Needs chest-up framing and a concrete outfit description (see `/sprite-sheet`) |
| 0.8 and up | Effectively a new character |

---

## POST /sprite-sheet

Generates one character in N expressions and returns a grid (and, in JSON mode, each cell separately).

**How it works:**
- **Base portrait:** the server makes a close-up, chest-up portrait of `character` in a neutral pose (hands clasped in front of the chest), in one of three ways:
  - **From text only:** just `character`.
  - **From a `sprite` + `character`** (recommended when you have art): the sprite sets the look, and the text pins down the identity and adds what a tiny sprite can't show (face detail, hands). See below.
  - **From `image`:** a ready-made base portrait, used as-is.

  The base is not a cell.
- **Cells:** each expression is image-to-image from that base at strength 0.78, using the same seed. The base only anchors identity; every cell goes through the same process, so they share one rendering.
- **Queueing:** each cell is its own GPU job, so other callers' `/generate` requests can run between cells.

**Default sheet: four work moods, 1024 × 1024 per cell** (rendered at 768 and scaled up). Every sheet has the same cells in the same order (a 2×2 grid):

| Cell | Name | Mood | What it shows |
|---|---|---|---|
| 1 | `stressed` | working frantically | panicked wide eyes, sweat, typing furiously on a laptop |
| 2 | `accomplished` | finally done | triumphant relieved grin, eyes shut, fist pump |
| 3 | `chill` | waiting for the next task | calm half-smile, sipping a mug of coffee |
| 4 | `confused` | asking a question | raised eyebrow, head tilt, "?" above the head |

- **Background:** a simple abstract backdrop in colours that fit the character's mood (no room or scenery). Every cell shares the base portrait's backdrop.
- **Colours:** bold and saturated, from the prompt plus the `vibrance` boost.
- **Optional extras:** office presets and props, plus up to 4 caller-chosen `activities`, can be added to the sheet (see below).

The gestures and props depend on three things working together (the defaults handle all three):

1. **Framing:** a close-up, chest-up shot with the hands near the face. From a tight bust with no hands visible (like the single portrait above), gestures can't appear, and pushing the strength high enough to add them gives you a different character.
2. **Strength around 0.78:** at 0.72 and below the hands stay frozen in the base pose. Above about 0.8 the identity starts to drift.
3. **A concrete `character`, outfit included:** for example `"an elf archer with long green hair, purple eyes, pointed elf ears, wearing a plain dark green hooded cloak with no trim"`. Vague outfits drift between cells (stray trim, changing colours).

| Field | Type | Default | Notes |
|---|---|---|---|
| `character` | string | required | Who to draw. Name the hair, eyes, distinctive features and **outfit** (see above) |
| `expressions` | list | the 12 defaults (4 expressions + 8 props) | 0–16 items, unique names. Each item is a **preset name** (`"coffee"`), **free text** (`"pouting, puffed cheeks"`), or an **object** `{"name", "prompt"?, "strength"?}`. Send a shorter list to make a smaller sheet |
| `activities` | list of strings | `[]` | Up to 4 caller-chosen props or activities, e.g. `"eating instant noodles from a cup with chopsticks"`. Appended after `expressions` as cells `activity_1`…`activity_4`. The server adds "with the prop clearly visible and held in the hands near the face"; without that, props were often missing. Total cells ≤ 16 |
| `sprite` | base64 PNG | — | Reference sprite: any size, transparency fine, full body fine. Used **together with** `character` |
| `sprite_strength` | float | 0.8 | How far the base may move away from the sprite. 0.55 is close to a copy (too little to work with), 0.8 keeps the sprite's look while adding a face and hands |
| `image` | base64 PNG | — | Ready-made base portrait, used as-is (not with `sprite`). It must already be a chest-up framing with the hands visible near the chest; otherwise gestures won't appear |
| `pixelate` | bool | false | Snap every cell to a real pixel grid with one palette shared across the sheet, for a genuine 16-bit look |
| `pixel_grid` | int | 128 | Pixels per cell side when pixelating (32–256). 64–96 is chunkier, 128 a detailed 16-bit look, 192+ finer |
| `vibrance` | float | 1.35 | Colour boost applied to every cell (saturation × vibrance, plus a little contrast) before pixelating. 1.0 turns it off, 1.6 gives very punchy colours |
| `palette_colors` | int | 32 | Colours in the shared palette (2–256) |
| `strength` | float | 0.78 | Default image-to-image strength for every cell |
| `seed` | int | random | Used for the base and every cell |
| `cell_size` | int | 1024 | Output size of each cell (256–1024, multiple of 16) |
| `render_size` | int | 768 | Size each cell is actually generated at, then scaled to `cell_size` (through the pixel grid when `pixelate` is on, smoothly otherwise). 768 is faster and lighter; 1024 adds detail (about 2× the time per cell) |
| `columns` | int | ceil(√N) | 1–8 |
| `gap` | int | 0 | Pixels between cells (0–64) |
| `background` | `#rrggbb` | `#000000` | Colour of the gaps and label strip |
| `labels` | bool | false | Write each expression's name under its cell |
| `prompt_template` | string | see below | Must contain `{character}` and `{expression}` |
| `lora` | string \| null | `"auto"` | `auto` uses the pixel-art LoRA for text-only sheets and **no LoRA when a `sprite` is given** (the LoRA's anime look overrides the sprite's). Send a repo id to force one, or `null` for none |
| `lora_strength` | float | 1.0 | |
| `steps`, `model`, `format` | | | As in `/generate` |

Default `prompt_template`:

```
Pixel art style. Close-up chest-up portrait of {character}, face large in frame, {expression}, hands visible near the face, exaggerated expressive anime reaction, 16-bit game character portrait, vibrant saturated bold colors, high contrast, simple abstract background whose colors and shapes fit the character's mood in this moment, no room, no scenery
```

### From your own sprite (image + text)

Send both `sprite` and `character`. The server:
1. flattens transparency onto a dark background;
2. crops the head and chest (the top 60% of the visible figure);
3. scales it up with hard pixel edges;
4. redraws it at `sprite_strength` into the chest-up base, guided by `character`.

Every cell is then made from that base, as usual.

- **Describe what the sprite shows** (colours, hood, ears, outfit), plus anything you want that it doesn't show clearly (hair colour, eye colour). Text that contradicts the sprite usually loses to the sprite.
- **Add `"pixelate": true`** to bring the cells back to a real 16-bit pixel grid. Without it, cells are smooth large pixel art.

```bash
printf '{"character":"an elf archer with a green hood, brown hair, pointed elf ears, wearing a green tunic","sprite":"%s","seed":7,"pixelate":true,"labels":true,"gap":6,"background":"#1a1a24"}' \
  "$(base64 -w0 my_sprite.png)" > req.json
curl -sS --max-time 900 http://<server-ip>:8000/sprite-sheet \
  -H "Content-Type: application/json" -d @req.json -o sheet.png
```

### Cells: stable names, fixed order

- **Default set** (the same four, in this order, on every request that omits `expressions`): `stressed, accomplished, chill, confused`. Then `activity_1`…`activity_N`, one for each entry in `activities`.
- **Order:** cells are always in that order, left to right and then top to bottom.
- **Names:** a preset is always named by its lowercase key (`"Chill "` comes back as `chill`). Activity cells are always `activity_1`…`activity_4`, whatever text was sent; the text is in that cell's `prompt`. A free-text item in `expressions` keeps its trimmed text as its name.
- **Unique names:** repeating a name is rejected with `422`. So are more than 4 `activities` and more than 16 cells in total.
- **Reproducible:** the same request with the same `seed` returns byte-identical cells and sheet. Without a seed, you get the same cells, in the same order and with the same names, but a new random character.

**Optional presets.** These aren't in the default sheet; add them by name to `expressions`, e.g. `["stressed","accomplished","chill","confused","coffee","phone"]`:

| Group | Presets |
|---|---|
| office (talking to the boss) | `listening`, `confident`, `nervous`, `apologetic` |
| props | `reading`, `coffee`, `coding`, `phone`, `notes`, `files`, `lunch`, `presenting` |

Activities (`activities`) work best when they're **held props near the face or chest**: noodles with chopsticks, a headset video call and a water bottle all came out clearly. Arms overhead fall outside the chest-up frame.

`GET /sprite-sheet/expressions` returns `{"expressions": {...}, "office": {...}, "props": {...}, "default": [...], "max_activities": 4}`, with the exact prompt wording of each preset.

**Response `200`:**
- **PNG:** the sheet. Headers:
  - `X-Seed`, `X-Model`, `X-Elapsed-Ms`
  - `X-Grid` (for example `3x3`)
  - `X-Expressions`: the cell names in order, comma-separated and percent-encoded, for example `stressed,accomplished,chill,confused` or `stressed,accomplished,chill,confused,activity_1`
- **JSON:**

```json
{
  "image": "<sheet base64>", "mime_type": "image/png",
  "width": 2048, "height": 2048, "columns": 2, "rows": 2, "cell_size": 1024,
  "expressions": ["stressed", "accomplished", "chill", "confused"],
  "seed": 5, "model": "z-image-turbo-q4", "lora": "tarn59/pixel_art_style_lora_z_image_turbo",
  "elapsed_ms": 70120,
  "cells": [
    {"name": "stressed", "x": 0, "y": 0, "width": 1024, "height": 1024,
     "prompt": "Pixel art style. Close-up chest-up portrait of …, stressed and frazzled, working frantically, …",
     "strength": 0.78, "image": "<cell base64>"}
  ],
  "base_image": "<base64, only when the server generated the base>"
}
```

`cells[i].name == expressions[i]`. Cell positions (`x`, `y`) account for `gap` and `labels`, so you can cut cells out of the sheet without decoding each one.

**Timing (RTX 4090, default model):** about 3 s per cell at the default 768 render size (about 6 s at `render_size: 1024`), plus one step for the base portrait. The default 4-cell sheet takes **about 14–17 s**. Use an HTTP timeout of at least 5 minutes anyway: if another job is using the same GPU, cells can slow down a lot (see the limits section).

**Tuning:**
- **A cell drifted** (outfit changed, eye colour off): describe that detail more concretely in `character`, or lower that one expression to `"strength": 0.74`.
- **A gesture is missing:** raise that expression to `0.8`, or describe the hand position concretely in a custom `prompt`.
- **Small cells:** 256 px cells are about 2× faster per cell but lose detail in the hands.

**Example: the full flow**

```bash
# default 4-cell sheet plus one caller-chosen activity
curl -sS --max-time 900 http://<server-ip>:8000/sprite-sheet \
  -H "Content-Type: application/json" -o sheet.png -D headers.txt \
  -d '{"character":"a young office worker with short black hair, brown eyes, round glasses, wearing a white shirt and a navy tie","seed":21,"gap":6,"labels":true,"background":"#1a1a24","activities":["eating instant noodles from a cup with chopsticks"]}'
grep -i x-expressions headers.txt
```

Python, saving each cell under its expression name:

```python
import base64, requests
r = requests.post("http://<server-ip>:8000/sprite-sheet", timeout=600,
                  json={"character": "a young office worker with short black hair, brown eyes, "
                                     "round glasses, wearing a white shirt and a navy tie",
                        "activities": ["eating instant noodles from a cup with chopsticks"],
                        "seed": 21, "format": "json"})
r.raise_for_status()
d = r.json()
open("sheet.png", "wb").write(base64.b64decode(d["image"]))
for c in d["cells"]:                       # stressed.png, accomplished.png, chill.png, confused.png, activity_1.png
    open(f"{c['name']}.png", "wb").write(base64.b64decode(c["image"]))
```

---

## GET /health (no token)

```json
{"status":"ok","model_loaded":true,"model":"z-image-turbo-q4","gpu_free":true,"queued":0,
 "gpu":{"name":"NVIDIA GeForce RTX 4090","used_mb":17942,"free_mb":6622,"total_mb":24564}}
```

`status` is `loading` (HTTP 503) for 1.5–3.5 minutes after a restart. `gpu_free` is false while a job is running. `gpu` covers every process on the card, not just this service.

## GET /models

Returns the model registry (id, licence, default steps, approximate VRAM, whether it's loaded) and the LoRAs (loaded, suggested, allowlist).

## Errors

The body is `{"detail": ...}`.

| Code | Meaning | What to do |
|---|---|---|
| 401 | Missing or wrong token (only when the server has `API_TOKEN` set) | |
| 413 | Reference image or sprite over 10 MB or 4096² pixels | |
| 422 | Invalid input (the message names the field) | Fix the request |
| 503 | Model loading, or queue full (8 waiting) | Retry after `Retry-After` |
| 504 | A job ran over 120 s | Retry; it usually means GPU contention |
| 507 | GPU out of memory | Smaller size, or retry later |
| 500 | Anything else | |

## Limits and behaviour

- **One GPU job at a time**, with a queue of up to 8. A sprite sheet is N separate jobs.
- **GPU memory:** with the default model (`z-image-turbo-bf16-te4`, `OFFLOAD=none`) this service holds about 14 GB and peaks around 18 GB while generating. If another heavy GPU workload shares the card (a TTS server, an LLM) and the total goes past the card's memory, Windows/WSL2 spills GPU memory into system RAM. Jobs then slow down a lot (6× or worse) instead of failing, and can hit the 120 s per-job limit (`504`); the other workload slows down too. On a shared or smaller GPU, use `DEFAULT_MODEL=z-image-turbo-q4` (about 8.5 GB, about 1.6× slower per step) or `z-image-turbo-q3` with `OFFLOAD=model` (smallest, slowest). If a job is stuck more than 60 s past its limit, the service restarts itself (about 2 minutes of `503`), so callers should retry `503` and `504` with backoff.
- **Reproducible:** the same request with the same seed gives the same PNG.
- **Licence:** the model (Apache 2.0), the GGUF quantization (Apache 2.0) and the pixel-art LoRA (Apache 2.0) all allow commercial use.
