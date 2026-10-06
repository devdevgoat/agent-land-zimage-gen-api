# Third-party notices and citations

This repository's own code and docs are licensed under Apache-2.0 (see [LICENSE](LICENSE)). It builds on the work below. **No model weights are included in this repository.** They are downloaded from Hugging Face into `./hf-cache/` on your machine, under their own licences.

## Models (downloaded, not redistributed)

| Component | Used as | Licence | Source |
|---|---|---|---|
| Z-Image Turbo (Tongyi-MAI): transformer, text encoder (Qwen3-based), VAE, tokenizer | Image model (all registry entries) | Apache-2.0 | https://huggingface.co/Tongyi-MAI/Z-Image-Turbo |
| Z-Image Turbo GGUF quantizations (unsloth), Q4_K_M and Q3_K_M | Transformer for `z-image-turbo-q4` / `-q3` | Apache-2.0 | https://huggingface.co/unsloth/Z-Image-Turbo-GGUF |
| Pixel Art Style LoRA for Z-Image Turbo (tarn59) | Default LoRA for text-only sprite sheets; trigger phrase "Pixel art style." | Apache-2.0 | https://huggingface.co/tarn59/pixel_art_style_lora_z_image_turbo |

Callers can load other LoRAs by Hugging Face repo id. Each LoRA has its own licence, which the caller is responsible for. Set `LORA_ALLOWLIST` to restrict them.

## Software

| Component | Licence | Source |
|---|---|---|
| diffusers (`ZImagePipeline`, `ZImageImg2ImgPipeline`, GGUF loading) | Apache-2.0 | https://github.com/huggingface/diffusers |
| transformers | Apache-2.0 | https://github.com/huggingface/transformers |
| accelerate | Apache-2.0 | https://github.com/huggingface/accelerate |
| PEFT (LoRA loading) | Apache-2.0 | https://github.com/huggingface/peft |
| huggingface_hub | Apache-2.0 | https://github.com/huggingface/huggingface_hub |
| safetensors | Apache-2.0 | https://github.com/huggingface/safetensors |
| gguf (Python) | MIT | https://github.com/ggml-org/llama.cpp/tree/master/gguf-py |
| bitsandbytes (NF4 text encoder) | MIT | https://github.com/bitsandbytes-foundation/bitsandbytes |
| PyTorch (`pytorch/pytorch` base image) | BSD-3-Clause | https://github.com/pytorch/pytorch |
| NVIDIA CUDA libraries (in the base image) | NVIDIA Deep Learning Container License / CUDA EULA | https://catalog.ngc.nvidia.com/ |
| FastAPI | MIT | https://github.com/fastapi/fastapi |
| Uvicorn | BSD-3-Clause | https://github.com/encode/uvicorn |
| Pydantic | MIT | https://github.com/pydantic/pydantic |
| Pillow | MIT-CMU (HPND) | https://github.com/python-pillow/Pillow |

## Citations

Z-Image, as requested by its authors (https://github.com/Tongyi-MAI/Z-Image):

```bibtex
@article{team2025zimage,
  title={Z-Image: An Efficient Image Generation Foundation Model with Single-Stream Diffusion Transformer},
  author={Z-Image Team},
  journal={arXiv preprint arXiv:2511.22699},
  year={2025}
}

@article{liu2025decoupled,
  title={Decoupled DMD: CFG Augmentation as the Spear, Distribution Matching as the Shield},
  author={Dongyang Liu and Peng Gao and David Liu and Ruoyi Du and Zhen Li and Qilong Wu and Xin Jin and Sihan Cao and Shifeng Zhang and Hongsheng Li and Steven Hoi},
  journal={arXiv preprint arXiv:2511.22677},
  year={2025}
}

@article{jiang2025distribution,
  title={Distribution Matching Distillation Meets Reinforcement Learning},
  author={Jiang, Dengyang and Liu, Dongyang and Wang, Zanyi and Wu, Qilong and Jin, Xin and Liu, David and Li, Zhen and Wang, Mengmeng and Gao, Peng and Yang, Harry},
  journal={arXiv preprint arXiv:2511.13649},
  year={2025}
}
```

Also credit:
- **unsloth:** *Z-Image-Turbo-GGUF*. https://huggingface.co/unsloth/Z-Image-Turbo-GGUF.
- **tarn59:** *Pixel Art Style LoRA for Z-Image Turbo*. https://huggingface.co/tarn59/pixel_art_style_lora_z_image_turbo.
- **Hugging Face:** *Diffusers: State-of-the-art diffusion models*. https://github.com/huggingface/diffusers.

[CITATION.cff](CITATION.cff) has the same references in machine-readable form.
