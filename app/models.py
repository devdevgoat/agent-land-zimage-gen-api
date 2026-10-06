"""Model registry and backends.

To add a model, add a ModelSpec to MODELS. A new architecture family also needs
a Backend subclass registered in BACKENDS; a new variant of an existing family
(another quantization, a fine-tune in diffusers format) only needs the spec.
"""

from __future__ import annotations

import gc
import logging
import re
import threading
import time
from dataclasses import dataclass

import torch
from PIL import Image

log = logging.getLogger("image-gen")


@dataclass(frozen=True)
class ModelSpec:
    id: str
    family: str  # selects the Backend class
    description: str
    license: str
    repo: str  # diffusers-format repo: configs, tokenizer, text encoder, VAE, scheduler
    transformer_gguf: tuple[str, str] | None = None  # (repo, file): quantized transformer
    text_encoder_4bit: bool = False  # load the text encoder with bitsandbytes NF4
    # Store transformer weights as FP8 and upcast each layer to bf16 as it runs.
    # Half the memory of bf16, and far cheaper per step than GGUF dequantization.
    transformer_fp8_storage: bool = False
    default_steps: int = 9
    default_guidance: float = 0.0
    approx_vram_gb: float = 0.0  # weights only, all components resident


MODELS: dict[str, ModelSpec] = {
    spec.id: spec
    for spec in [
        ModelSpec(
            id="z-image-turbo-q4",
            family="z-image",
            description="Z-Image Turbo, 4-bit GGUF transformer (Q4_K_M) + NF4 text encoder",
            license="apache-2.0",
            repo="Tongyi-MAI/Z-Image-Turbo",
            transformer_gguf=("unsloth/Z-Image-Turbo-GGUF", "z-image-turbo-Q4_K_M.gguf"),
            text_encoder_4bit=True,
            approx_vram_gb=8.0,
        ),
        ModelSpec(
            id="z-image-turbo-q3",
            family="z-image",
            description="Z-Image Turbo, 3-bit GGUF transformer (Q3_K_M) + NF4 text encoder; smallest usable",
            license="apache-2.0",
            repo="Tongyi-MAI/Z-Image-Turbo",
            transformer_gguf=("unsloth/Z-Image-Turbo-GGUF", "z-image-turbo-Q3_K_M.gguf"),
            text_encoder_4bit=True,
            approx_vram_gb=7.2,
        ),
        ModelSpec(
            id="z-image-turbo-fp8",
            family="z-image",
            # Nearly bf16 speed at half the memory, but fails with LoRAs ("addmm_cuda"
            # not implemented for Float8): the LoRA layer multiplies the FP8 weight directly.
            description="Z-Image Turbo, bf16 transformer stored as FP8 + NF4 text encoder (no LoRA support)",
            license="apache-2.0",
            repo="Tongyi-MAI/Z-Image-Turbo",
            text_encoder_4bit=True,
            transformer_fp8_storage=True,
            approx_vram_gb=9.5,
        ),
        ModelSpec(
            id="z-image-turbo-bf16-te4",
            family="z-image",
            description="Z-Image Turbo, full bf16 transformer + NF4 text encoder",
            license="apache-2.0",
            repo="Tongyi-MAI/Z-Image-Turbo",
            text_encoder_4bit=True,
            approx_vram_gb=15.5,
        ),
        ModelSpec(
            id="z-image-turbo",
            family="z-image",
            description="Z-Image Turbo, full bf16 (best quality; downloads the 12 GB transformer)",
            license="apache-2.0",
            repo="Tongyi-MAI/Z-Image-Turbo",
            approx_vram_gb=21.0,
        ),
    ]
}


@dataclass
class Job:
    prompt: str
    negative_prompt: str | None
    width: int
    height: int
    steps: int
    guidance: float
    seed: int
    image: Image.Image | None = None
    strength: float = 0.6
    lora: str | None = None
    lora_strength: float = 1.0
    deadline: float = 0.0  # time.monotonic() after which the job is aborted
    queue_wait: float = 0.0  # seconds spent waiting for the GPU


class JobTimeout(Exception):
    pass


class Backend:
    def __init__(self, spec: ModelSpec, offload: str):
        self.spec = spec
        self.offload = offload  # "none" | "model"
        self.loras: set[str] = set()  # adapters loaded into the model
        self.active_lora: tuple[str, float] | None = None  # (adapter, strength) in use

    def load(self) -> None:
        raise NotImplementedError

    def generate(self, job: Job) -> Image.Image:
        raise NotImplementedError

    def unload(self) -> None:
        pass


def _adapter_name(repo: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]", "_", repo)


def _lora_weight_file(repo: str) -> str:
    """Pick the LoRA's safetensors file. Pickle formats are refused."""
    from huggingface_hub import list_repo_files

    files = [f for f in list_repo_files(repo) if f.endswith(".safetensors")]
    if not files:
        raise ValueError(f"LoRA repo '{repo}' has no .safetensors file")
    files.sort(key=lambda f: ("lora" not in f.lower(), f.count("/"), len(f)))
    return files[0]


class ZImageBackend(Backend):
    def load(self) -> None:
        from diffusers import (
            GGUFQuantizationConfig,
            ZImageImg2ImgPipeline,
            ZImagePipeline,
            ZImageTransformer2DModel,
        )

        dtype = torch.bfloat16
        kwargs: dict = {"torch_dtype": dtype}
        if self.spec.transformer_gguf:
            from huggingface_hub import hf_hub_download

            path = hf_hub_download(*self.spec.transformer_gguf)
            kwargs["transformer"] = ZImageTransformer2DModel.from_single_file(
                path,
                quantization_config=GGUFQuantizationConfig(compute_dtype=dtype),
                config=self.spec.repo,
                subfolder="transformer",
                torch_dtype=dtype,
            )
        if self.spec.text_encoder_4bit:
            from transformers import AutoModelForCausalLM, BitsAndBytesConfig

            kwargs["text_encoder"] = AutoModelForCausalLM.from_pretrained(
                self.spec.repo,
                subfolder="text_encoder",
                torch_dtype=dtype,
                quantization_config=BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=dtype,
                ),
            )
        self.txt2img = ZImagePipeline.from_pretrained(self.spec.repo, **kwargs)
        if self.spec.transformer_fp8_storage:
            self.txt2img.transformer.enable_layerwise_casting(
                storage_dtype=torch.float8_e4m3fn, compute_dtype=dtype
            )
        # Shares every component (no extra memory) with the text-to-image pipeline.
        self.img2img = ZImageImg2ImgPipeline(**self.txt2img.components)
        for pipe in (self.txt2img, self.img2img):
            pipe.set_progress_bar_config(disable=True)
        if self.offload == "model":
            # Components move to the GPU only while they run. Hooks live on the
            # shared modules, so enabling it once covers both pipelines.
            self.img2img.enable_model_cpu_offload()
            self.txt2img.enable_model_cpu_offload()
        else:
            self.txt2img.to("cuda")

    def _apply_lora(self, repo: str | None, strength: float) -> None:
        wanted = None if repo is None else (_adapter_name(repo), strength)
        if wanted == self.active_lora:
            return
        if wanted is None:
            self.txt2img.disable_lora()
        else:
            name = wanted[0]
            if name not in self.loras:
                weight = _lora_weight_file(repo)
                log.info("loading LoRA %s (%s)", repo, weight)
                self.txt2img.load_lora_weights(repo, weight_name=weight, adapter_name=name)
                self.loras.add(name)
            self.txt2img.enable_lora()
            self.txt2img.set_adapters([name], [strength])
        self.active_lora = wanted

    def generate(self, job: Job) -> Image.Image:
        t_lora = time.perf_counter()
        self._apply_lora(job.lora, job.lora_strength)
        t_start = time.perf_counter()
        step_ends: list[float] = []

        def check_deadline(pipe, step, timestep, callback_kwargs):
            torch.cuda.synchronize()  # so the timestamps reflect GPU work, not queued kernels
            step_ends.append(time.perf_counter())
            if time.monotonic() > job.deadline:
                raise JobTimeout(f"job exceeded its time limit at step {step}")
            return callback_kwargs

        args = dict(
            prompt=job.prompt,
            negative_prompt=job.negative_prompt,
            width=job.width,
            height=job.height,
            num_inference_steps=job.steps,
            guidance_scale=job.guidance,
            generator=torch.Generator("cpu").manual_seed(job.seed),
            callback_on_step_end=check_deadline,
        )
        # Not inference_mode: offload moves LoRA weights between devices during the
        # call, which would turn them into inference tensors that a later
        # enable_lora()/set_adapters() can no longer modify.
        try:
            with torch.no_grad():
                if job.image is not None:
                    image = job.image.convert("RGB").resize((job.width, job.height), Image.LANCZOS)
                    result = self.img2img(image=image, strength=job.strength, **args).images[0]
                else:
                    result = self.txt2img(**args).images[0]
            end = time.perf_counter()
            if step_ends:
                steps = len(step_ends)
                per_step = (step_ends[-1] - step_ends[0]) / (steps - 1) if steps > 1 else 0.0
                # Before the first step ends: text encoding (+ VAE encode for img2img) + one step.
                log.info(
                    "timing %dx%d: lora %.2fs, prep+encode %.2fs, %d steps x %.3fs, decode %.2fs",
                    job.width, job.height, t_start - t_lora,
                    step_ends[0] - t_start - per_step, steps, per_step, end - step_ends[-1],
                )
            return result
        finally:
            # A finished call offloads its weights back to system RAM itself; an
            # aborted one (timeout, error) skips that and leaves ~5 GB on the GPU,
            # which made every later job spill into shared memory and time out too.
            if self.offload == "model":
                self.txt2img.maybe_free_model_hooks()

    def unload(self) -> None:
        for name in ("txt2img", "img2img"):
            pipe = getattr(self, name, None)
            if pipe is not None:
                pipe.remove_all_hooks()
            setattr(self, name, None)
        gc.collect()
        torch.cuda.empty_cache()


BACKENDS: dict[str, type[Backend]] = {"z-image": ZImageBackend}


class ModelManager:
    """Holds one loaded model at a time. Only the GPU worker thread calls into it."""

    def __init__(self, default_id: str, offload: str):
        self.default_id = default_id
        self.offload = offload
        self.backend: Backend | None = None
        self.lock = threading.Lock()

    @property
    def loaded_id(self) -> str | None:
        return self.backend.spec.id if self.backend else None

    def get(self, model_id: str) -> Backend:
        with self.lock:
            if self.backend and self.backend.spec.id == model_id:
                return self.backend
            if self.backend:
                log.info("unloading %s", self.backend.spec.id)
                self.backend.unload()
                self.backend = None
            spec = MODELS[model_id]
            log.info("loading %s (offload=%s)", model_id, self.offload)
            start = time.perf_counter()
            backend = BACKENDS[spec.family](spec, self.offload)
            backend.load()
            log.info("loaded %s in %.1fs", model_id, time.perf_counter() - start)
            self.backend = backend
            return backend
