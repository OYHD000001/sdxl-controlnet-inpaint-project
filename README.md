# SDXL ControlNet Inpaint Training Project

This project scaffolds a Hugging Face `diffusers` training workflow for:

- SDXL
- single ControlNet condition
- inpainting-style latent concatenation

Current scope is the first-stage engineering skeleton only:

- project layout
- config and shell scripts
- dataset/transforms/loss utilities
- SDXL + ControlNet + inpaint training framework
- validation and inference placeholders

The intended training target is:

`(masked_source_image, mask_image, conditioning_image, text) -> target_image`

## Goals

- edit real human model images into mannequin images
- preserve composition, background, garment identity, and garment appearance as much as possible
- train a single-condition SDXL + ControlNet + inpaint pipeline on local data

## Current local environment check

Checked from the existing `/data/ouyanghaodong/oyhd_20260318_extracted/oyhd/env`:

- `torch` is already available
- `transformers` is already available
- `diffusers` is not installed yet
- `accelerate` is not installed yet
- `datasets` is not installed yet

This project therefore ships a `requirements.txt` for a fresh project-local install.

## Planned training input schema

Each training sample should provide:

- `target_image`
- `source_image`
- `mask_image`
- `conditioning_image`
- `text`

Optional field:

- `masked_source_image`

If `masked_source_image` is absent, it can be generated dynamically from `source_image` and `mask_image`.

## Suggested metadata format

Use a JSONL file where each line looks like:

```json
{
  "target_image": "/abs/path/to/target.png",
  "source_image": "/abs/path/to/source.png",
  "mask_image": "/abs/path/to/mask.png",
  "conditioning_image": "/abs/path/to/conditioning.png",
  "text": "a mannequin wearing the same clothes"
}
```

See [examples/metadata_example.jsonl](examples/metadata_example.jsonl).

## Directory layout

```text
project_root/
  README.md
  requirements.txt
  .gitignore
  configs/
    train_sdxl_inpaint_controlnet.yaml
  scripts/
    train.sh
    infer.sh
  src/
    __init__.py
    data/
      __init__.py
      dataset.py
      transforms.py
    models/
      __init__.py
      pipeline.py
    training/
      __init__.py
      train_controlnet_sdxl_inpaint.py
      losses.py
      validate.py
    utils/
      __init__.py
      io.py
      masks.py
      prompts.py
  examples/
    metadata_example.jsonl
```

## Design notes

- Based conceptually on the official Hugging Face `diffusers` `train_controlnet_sdxl.py` example.
- First version supports only one ControlNet condition.
- First version assumes segmentation-like conditioning images.
- First version uses only standard diffusion MSE noise-prediction loss.
- No frontend.
- No distributed training.
- No W&B.
- No LoRA.

## Core training flow

1. encode `target_image` to latents
2. sample noise and timestep
3. add noise to target latents
4. encode `masked_source_image` to latents
5. resize `mask_image` to latent size
6. concatenate `[noisy_latents, mask, masked_image_latents]`
7. feed `conditioning_image` to ControlNet
8. predict noise with UNet
9. optimize standard MSE loss against the sampled noise

## What is intentionally left as TODO

- final argument surface parity with official `diffusers` example
- fp16/bf16 and memory optimization tuning
- exact SDXL inpaint latent-channel verification against the final chosen base model
- validation image generation with a complete custom inpaint + ControlNet SDXL pipeline
- local pretrained weight path wiring for your final dataset run
- dataset-specific prompt cleaning and augmentation policy

## Install

```bash
cd /data/ouyanghaodong/oyhd_20260318_extracted/sdxl_controlnet_inpaint_project
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Train

```bash
bash scripts/train.sh
```

## Infer

```bash
bash scripts/infer.sh
```
