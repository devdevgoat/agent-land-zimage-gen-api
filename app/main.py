"""Self-hosted image generation API.

POST /generate   text-to-image or image-to-image (bearer token)
GET  /models     available models and LoRAs (bearer token)
GET  /health     model loaded? GPU busy? (no token)
"""

from __future__ import annotations

import asyncio
import base64
import binascii
import hmac
import io
import math
import logging
import os
import pathlib
import random
import re
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from urllib.parse import quote

import torch
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, PlainTextResponse, Response
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel, Field, field_validator

from app.models import MODELS, Job, JobTimeout, ModelManager
from app.sprites import (
    DEFAULT_EXPRESSIONS,
    EXPRESSION_PRESETS,
    MAX_ACTIVITIES,
    OFFICE_PRESETS,
    PROP_PRESETS,
    SpriteSheetRequest,
    boost_colors,
    compose_sheet,
    pixelate,
    prep_sprite,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("image-gen")
logging.getLogger("httpx").setLevel(logging.WARNING)  # per-file Hugging Face HEAD requests


def _env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, default))


API_TOKEN = os.environ.get("API_TOKEN", "")
DEFAULT_MODEL = os.environ.get("DEFAULT_MODEL", "z-image-turbo-bf16-te4")
OFFLOAD = os.environ.get("OFFLOAD", "none")  # "none" (fastest) or "model" (share a crowded GPU)
MAX_SIDE = _env_int("MAX_SIDE", 1024)
MIN_SIDE = 256
MAX_STEPS = _env_int("MAX_STEPS", 50)
JOB_TIMEOUT = _env_int("JOB_TIMEOUT_SECONDS", 120)
QUEUE_MAX = _env_int("QUEUE_MAX", 8)
HANG_GRACE = _env_int("HANG_GRACE_SECONDS", 60)
MAX_IMAGE_BYTES = _env_int("MAX_IMAGE_MB", 10) * 1024 * 1024
SUGGESTED_LORAS = [x.strip() for x in os.environ.get("LORAS", "").split(",") if x.strip()]
LORA_ALLOWLIST = {x.strip() for x in os.environ.get("LORA_ALLOWLIST", "").split(",") if x.strip()}
REPO_ID = re.compile(r"^[A-Za-z0-9][\w.-]{0,95}/[\w.-]{1,96}$")

if not API_TOKEN:
    log.warning("API_TOKEN is empty: authentication is off (fine on a LAN-only host)")
if DEFAULT_MODEL not in MODELS:
    raise SystemExit(f"DEFAULT_MODEL '{DEFAULT_MODEL}' is not one of {sorted(MODELS)}")
if OFFLOAD not in ("model", "none"):
    raise SystemExit("OFFLOAD must be 'model' or 'none'")

manager = ModelManager(DEFAULT_MODEL, OFFLOAD)
executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="gpu")
state = {"status": "loading", "error": None, "busy": False}
queue: asyncio.Queue | None = None


# ---------------------------------------------------------------- request model


class GenerateRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=4000)
    negative_prompt: str | None = Field(None, max_length=4000)
    width: int = Field(512, ge=MIN_SIDE, le=MAX_SIDE)
    height: int = Field(512, ge=MIN_SIDE, le=MAX_SIDE)
    steps: int | None = Field(None, ge=1, le=MAX_STEPS)
    guidance_scale: float | None = Field(None, ge=0, le=20)
    seed: int | None = Field(None, ge=0, le=2**32 - 1)
    image: str | None = Field(None, description="Reference picture, base64 PNG (data: URL prefix allowed)")
    strength: float = Field(0.6, gt=0, le=1)
    lora: str | None = Field(None, description="Hugging Face repo id of a LoRA")
    lora_strength: float = Field(1.0, ge=0, le=2)
    model: str | None = None
    format: str | None = Field(None, pattern="^(png|json)$")

    @field_validator("width", "height")
    @classmethod
    def multiple_of_16(cls, v: int) -> int:
        if v % 16:
            raise ValueError("must be a multiple of 16")
        return v

    @field_validator("model")
    @classmethod
    def known_model(cls, v: str | None) -> str | None:
        return _check_model(v)

    @field_validator("lora")
    @classmethod
    def lora_repo(cls, v: str | None) -> str | None:
        return _check_lora(v)


def _check_model(v: str | None) -> str | None:
    if v is not None and v not in MODELS:
        raise ValueError(f"unknown model; available: {sorted(MODELS)}")
    return v


def _check_lora(v: str | None) -> str | None:
    if v is None:
        return v
    if not REPO_ID.match(v):
        raise ValueError("must be a Hugging Face repo id like 'owner/name'")
    if LORA_ALLOWLIST and v not in LORA_ALLOWLIST:
        raise ValueError(f"LoRA not allowed; allowed: {sorted(LORA_ALLOWLIST)}")
    return v


def _decode_image(data: str) -> Image.Image:
    if data.startswith("data:"):
        data = data.split(",", 1)[-1]
    try:
        raw = base64.b64decode(data, validate=True)
    except (binascii.Error, ValueError):
        raise HTTPException(422, "image: not valid base64")
    if len(raw) > MAX_IMAGE_BYTES:
        raise HTTPException(413, f"image: larger than {MAX_IMAGE_BYTES // 2**20} MB")
    try:
        image = Image.open(io.BytesIO(raw))
        image.load()
    except (UnidentifiedImageError, OSError):
        raise HTTPException(422, "image: not a readable picture (PNG expected)")
    if image.width * image.height > 4096 * 4096:
        raise HTTPException(413, "image: more than 4096 x 4096 pixels")
    return image


# ---------------------------------------------------------------- auth


bearer = HTTPBearer(auto_error=False)


def require_token(creds: HTTPAuthorizationCredentials | None = Depends(bearer)) -> None:
    if not API_TOKEN:  # auth disabled
        return
    if creds is None or not hmac.compare_digest(creds.credentials.encode(), API_TOKEN.encode()):
        raise HTTPException(401, "missing or invalid bearer token", headers={"WWW-Authenticate": "Bearer"})


# ---------------------------------------------------------------- GPU worker


async def gpu_worker() -> None:
    """Runs queued jobs one at a time on the single GPU thread."""
    loop = asyncio.get_running_loop()
    while True:
        model_id, job, future, queued_at = await queue.get()
        if future.cancelled():
            continue
        waited = time.monotonic() - queued_at
        if waited > JOB_TIMEOUT:
            future.set_exception(JobTimeout(f"waited {waited:.0f}s in the queue"))
            continue
        job.deadline = time.monotonic() + JOB_TIMEOUT
        job.queue_wait = waited
        state["busy"] = True
        try:
            running = loop.run_in_executor(executor, _run, model_id, job)
            try:
                result = await asyncio.wait_for(asyncio.shield(running), JOB_TIMEOUT + HANG_GRACE)
            except asyncio.TimeoutError:
                # The step-level deadline can't fire while a single step is stuck
                # (e.g. GPU memory spilled to system RAM under contention). The GPU
                # thread can't be cancelled, so restart the process instead of
                # leaving the service frozen.
                log.critical("job stuck for %ss past its limit; exiting for a clean restart", HANG_GRACE)
                os._exit(1)
            if not future.done():
                future.set_result(result)
        except BaseException as error:  # noqa: BLE001 - forwarded to the caller
            if not future.done():
                future.set_exception(error)
        finally:
            state["busy"] = False


def _run(model_id: str, job: Job) -> tuple[bytes, float]:
    start = time.perf_counter()
    try:
        image = manager.get(model_id).generate(job)
        torch.cuda.empty_cache()  # give memory back to other GPU users (e.g. the TTS server)
    except torch.AcceleratorError as error:
        # A CUDA fault (illegal access, driver reset) poisons this process's CUDA
        # context; every later job would fail. Exit so Docker restarts us clean.
        log.critical("FATAL CUDA error, exiting for a clean restart: %s", error)
        os._exit(1)
    except Exception:
        torch.cuda.empty_cache()
        raise
    elapsed = time.perf_counter() - start
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue(), elapsed


def _load_default() -> None:
    try:
        manager.get(DEFAULT_MODEL)
        torch.cuda.empty_cache()
        state["status"] = "ok"
    except Exception as error:  # noqa: BLE001
        log.exception("loading %s failed", DEFAULT_MODEL)
        state.update(status="error", error=f"{type(error).__name__}: {error}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    global queue
    queue = asyncio.Queue(maxsize=QUEUE_MAX)
    worker = asyncio.create_task(gpu_worker())
    # Load in the GPU thread; /health answers "loading" meanwhile.
    asyncio.get_running_loop().run_in_executor(executor, _load_default)
    yield
    worker.cancel()


app = FastAPI(title="Image generation API", lifespan=lifespan)


# ---------------------------------------------------------------- endpoints


def _gpu_memory() -> dict | None:
    # nvidia-smi counts every process on the GPU (e.g. other containers);
    # torch.cuda.mem_get_info() under WSL2 only sees this process.
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.used,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=3, check=True,
        ).stdout.splitlines()[0]
        name, used, total = (x.strip() for x in out.split(","))
        return {"name": name, "used_mb": int(used), "free_mb": int(total) - int(used), "total_mb": int(total)}
    except Exception:  # noqa: BLE001
        return None


USAGE_PATH = pathlib.Path(__file__).resolve().parent.parent / "API.md"


@app.get("/usage", response_class=PlainTextResponse)
def usage() -> PlainTextResponse:
    """The full how-to-use guide (Markdown). No token, so a client can read it first."""
    return PlainTextResponse(USAGE_PATH.read_text(encoding="utf-8"), media_type="text/markdown; charset=utf-8")


@app.get("/health")
def health() -> JSONResponse:
    gpu = _gpu_memory()
    body = {
        "status": state["status"],
        "model_loaded": manager.backend is not None,
        "model": manager.loaded_id,
        "gpu_free": not state["busy"],
        "queued": queue.qsize() if queue else 0,
        "gpu": gpu,
    }
    if state["error"]:
        body["error"] = state["error"]
    return JSONResponse(body, status_code=200 if state["status"] == "ok" else 503)


@app.get("/models", dependencies=[Depends(require_token)])
def models() -> dict:
    loaded = manager.backend
    return {
        "default": DEFAULT_MODEL,
        "loaded": manager.loaded_id,
        "offload": OFFLOAD,
        "models": [
            {
                "id": s.id,
                "description": s.description,
                "license": s.license,
                "default_steps": s.default_steps,
                "default_guidance_scale": s.default_guidance,
                "approx_vram_gb": s.approx_vram_gb,
                "loaded": loaded is not None and loaded.spec.id == s.id,
            }
            for s in MODELS.values()
        ],
        "max_side": MAX_SIDE,
        "loras": {
            "loaded": sorted(loaded.loras) if loaded else [],
            "suggested": SUGGESTED_LORAS,
            "allowlist": sorted(LORA_ALLOWLIST) or None,
        },
    }


async def _submit(model_id: str, job: Job, label: str = "generate") -> tuple[bytes, float]:
    """Queue one GPU job, wait for it, and map failures to HTTP errors."""
    future = asyncio.get_running_loop().create_future()
    try:
        queue.put_nowait((model_id, job, future, time.monotonic()))
    except asyncio.QueueFull:
        raise HTTPException(503, "queue is full, try again shortly", headers={"Retry-After": "10"})

    try:
        png, elapsed = await future
    except JobTimeout as error:
        raise HTTPException(504, f"timed out after {JOB_TIMEOUT}s: {error}")
    except torch.cuda.OutOfMemoryError:
        raise HTTPException(507, "GPU out of memory; try a smaller size or free GPU memory")
    except ValueError as error:  # e.g. a LoRA repo without safetensors
        raise HTTPException(422, str(error))
    except Exception as error:
        log.exception("generation failed")
        raise HTTPException(500, f"{type(error).__name__}: {error}")

    kind = f"img2img strength={job.strength}" if job.image is not None else "txt2img"
    lora = f" lora={job.lora}@{job.lora_strength}" if job.lora else ""
    log.info(
        "%s %dx%d steps=%d %s%s model=%s seed=%d queued=%.1fs took=%.2fs",
        label, job.width, job.height, job.steps, kind, lora, model_id, job.seed, job.queue_wait, elapsed,
    )
    return png, elapsed


def _wants_json(fmt: str | None, request: Request) -> bool:
    return fmt == "json" or (fmt is None and "application/json" in request.headers.get("accept", ""))


def _check_ready() -> None:
    if state["status"] != "ok":
        raise HTTPException(503, f"model not ready ({state['status']})")


@app.post("/generate", dependencies=[Depends(require_token)])
async def generate(req: GenerateRequest, request: Request) -> Response:
    _check_ready()
    model_id = req.model or DEFAULT_MODEL
    spec = MODELS[model_id]
    steps = req.steps or spec.default_steps
    if req.image is not None and int(steps * req.strength) < 1:
        raise HTTPException(422, f"steps x strength must be >= 1 (got {steps} x {req.strength})")
    job = Job(
        prompt=req.prompt,
        negative_prompt=req.negative_prompt,
        width=req.width,
        height=req.height,
        steps=steps,
        guidance=spec.default_guidance if req.guidance_scale is None else req.guidance_scale,
        seed=random.randrange(2**32) if req.seed is None else req.seed,
        image=_decode_image(req.image) if req.image is not None else None,
        strength=req.strength,
        lora=req.lora,
        lora_strength=req.lora_strength,
    )
    png, elapsed = await _submit(model_id, job)

    headers = {
        "X-Seed": str(job.seed),
        "X-Model": model_id,
        "X-Elapsed-Ms": str(round(elapsed * 1000)),
    }
    if _wants_json(req.format, request):
        return JSONResponse(
            {
                "image": base64.b64encode(png).decode(),
                "mime_type": "image/png",
                "seed": job.seed,
                "width": job.width,
                "height": job.height,
                "steps": job.steps,
                "model": model_id,
                "elapsed_ms": round(elapsed * 1000),
            },
            headers=headers,
        )
    return Response(png, media_type="image/png", headers=headers)


@app.get("/sprite-sheet/expressions", dependencies=[Depends(require_token)])
def sprite_expressions() -> dict:
    return {
        "expressions": EXPRESSION_PRESETS,
        "office": OFFICE_PRESETS,
        "props": PROP_PRESETS,
        "default": DEFAULT_EXPRESSIONS,
        "max_activities": MAX_ACTIVITIES,
    }


@app.post("/sprite-sheet", dependencies=[Depends(require_token)])
async def sprite_sheet(req: SpriteSheetRequest, request: Request) -> Response:
    """One character, many expressions. Each cell is its own GPU job, so other
    callers can interleave between cells instead of waiting for the whole sheet."""
    _check_ready()
    try:
        _check_model(req.model)
        lora = req.effective_lora()
        _check_lora(lora)
        expressions = req.resolved_expressions()
    except ValueError as error:
        raise HTTPException(422, str(error))
    if not expressions:
        raise HTTPException(422, "nothing to draw: expressions and activities are both empty")
    model_id = req.model or DEFAULT_MODEL
    spec = MODELS[model_id]
    steps = req.steps or spec.default_steps
    for e in expressions:
        if int(steps * e.strength) < 1:
            raise HTTPException(422, f"steps x strength must be >= 1 for '{e.name}'")
    seed = random.randrange(2**32) if req.seed is None else req.seed
    common = dict(
        negative_prompt=None,
        width=req.render_size,
        height=req.render_size,
        steps=steps,
        guidance=spec.default_guidance,
        seed=seed,
        lora=lora,
        lora_strength=req.lora_strength,
    )
    started = time.perf_counter()
    total = len(expressions)

    # The base portrait only anchors identity. Every cell, including the first,
    # goes through image-to-image so they all share one rendering; the raw
    # text-to-image base looks noticeably different from its derivatives.
    if req.image is not None:
        base = _decode_image(req.image).convert("RGB")
    elif req.sprite is not None:
        # The sprite sets the look; `character` pins identity and adds what a
        # tiny sprite can't show (face detail, hands).
        reference = prep_sprite(_decode_image(req.sprite), req.render_size)
        png, _ = await _submit(
            model_id,
            Job(prompt=req.base_prompt(), image=reference, strength=req.sprite_strength, **common),
            "sprite base (from sprite)",
        )
        base = Image.open(io.BytesIO(png)).convert("RGB")
    else:
        png, _ = await _submit(model_id, Job(prompt=req.base_prompt(), **common), "sprite base")
        base = Image.open(io.BytesIO(png)).convert("RGB")

    cells: list[Image.Image] = []
    for i, e in enumerate(expressions):
        png, _ = await _submit(
            model_id,
            Job(prompt=req.prompt_for(e), image=base, strength=e.strength, **common),
            f"sprite {i + 1}/{total} {e.name}",
        )
        cells.append(Image.open(io.BytesIO(png)).convert("RGB"))

    # Cells render at `render_size` (less GPU memory) and are scaled to
    # `cell_size` here: through the pixel grid when pixelating, smoothly otherwise.
    cells = boost_colors(cells, req.vibrance)
    if req.pixelate:
        cells = pixelate(cells, req.pixel_grid, req.palette_colors, req.cell_size)
    else:
        cells = [c.resize((req.cell_size, req.cell_size), Image.LANCZOS) for c in cells]
    columns = req.columns or math.ceil(math.sqrt(total))
    sheet, boxes = compose_sheet(
        cells, [e.name for e in expressions], columns=columns, cell=req.cell_size,
        gap=req.gap, background=req.background, labels=req.labels,
    )
    elapsed = time.perf_counter() - started
    log.info("sprite-sheet %d cells of %d px, seed=%d, took %.1fs", total, req.cell_size, seed, elapsed)

    def png_bytes(image: Image.Image) -> bytes:
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        return buffer.getvalue()

    sheet_png = png_bytes(sheet)
    headers = {
        "X-Seed": str(seed),
        "X-Model": model_id,
        "X-Elapsed-Ms": str(round(elapsed * 1000)),
        "X-Grid": f"{columns}x{math.ceil(total / columns)}",
        # Cell order, left to right then top to bottom (percent-encoded names).
        "X-Expressions": ",".join(quote(e.name, safe="") for e in expressions),
    }
    if not _wants_json(req.format, request):
        return Response(sheet_png, media_type="image/png", headers=headers)
    return JSONResponse(
        {
            "image": base64.b64encode(sheet_png).decode(),
            "mime_type": "image/png",
            "width": sheet.width,
            "height": sheet.height,
            "columns": columns,
            "rows": math.ceil(total / columns),
            "cell_size": req.cell_size,
            "expressions": [e.name for e in expressions],
            "seed": seed,
            "model": model_id,
            "lora": lora,
            "elapsed_ms": round(elapsed * 1000),
            "cells": [
                {
                    **box,
                    "prompt": req.prompt_for(e),
                    "strength": e.strength,
                    "image": base64.b64encode(png_bytes(cell)).decode(),
                }
                for box, e, cell in zip(boxes, expressions, cells)
            ],
            "base_image": None if req.image is not None else base64.b64encode(png_bytes(base)).decode(),
        },
        headers=headers,
    )
