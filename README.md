# SDXL Inpaint + Pose ControlNet Real-to-Mannequin Pipeline

This repository now contains a **canonical** training and inference path for building aligned `(human image, mannequin image)` pairs with:

- `SDXL Inpainting`
- **one** `Pose ControlNet`
- fixed mannequin prompt
- pure-noise inpaint inference with `strength = 1.0`

Older experimental branches are still kept in `configs/`, `scripts/`, and `outputs/`, but the default entrypoints below are the ones that follow the final design contract.

## Core Design

These rules are enforced by the canonical path:

1. `x0 / target_image` is the **full mannequin image**.
2. `mask_image` is the **inverse clothes mask**.
   - clothes keep region = `0`
   - mannequin body + background redraw region = `1`
3. `masked_source_image` is the **clothes-only image**.
4. Only **one** ControlNet is used, and it is **pose**.
5. No clothes ControlNet branch is allowed in the canonical path.
6. Training and inference share the same resize, mask conversion, and masked-image construction code through [preprocess/extract.py](preprocess/extract.py).
7. Timestep sampling is full-range uniform: `t ~ U[0, T)`.
8. Canonical inference asserts `strength == 1.0`.

## Forbidden

- Do **not** use clothes image as diffusion target `x0`.
- Do **not** keep a second clothes ControlNet in the canonical path.
- Do **not** truncate timestep sampling.
- Do **not** run canonical inference with `strength < 1.0`.
- Do **not** let the clothes keep-region mask leak human/mannequin body pixels.
- Do **not** implement separate train/infer preprocessing paths.

## Repository Layout

```text
sdxl_controlnet_inpaint_project/
├── README.md
├── requirements.txt
├── train.py
├── infer.py
├── configs/
│   └── default.yaml
├── data/
│   └── dataset.py
├── preprocess/
│   ├── extract.py
│   └── pose.py
├── models/
│   ├── pipeline.py
│   ├── controlnet-canny-sdxl-1.0/
│   └── controlnet-openpose-sdxl-1.0-xinsir/
├── scripts/
│   └── build_pairs.py
└── src/
    ├── data/
    ├── models/
    ├── training/
    └── utils/
```

The files under `src/` contain the actual implementation used by the top-level entrypoints.

## Canonical Data Contract

Each record in the canonical JSONL metadata must provide:

- `target_image`: full mannequin image
- `source_image`: original human image
- `mask_image`: clothes mask (`white = clothes`)
- `conditioning_image`: pose image
- `text`: optional per-record prompt; canonical path overrides it with a fixed prompt when configured

The canonical pipeline constructs:

- `inpaint_mask = invert(mask_image)`
- `masked_source_image = clothes-only image built from source_image + inpaint_mask`

Example metadata files are provided in:

- [examples/metadata_canonical_train.jsonl](examples/metadata_canonical_train.jsonl)
- [examples/metadata_canonical_val.jsonl](examples/metadata_canonical_val.jsonl)

## Shared Preprocessing

Shared preprocessing lives in [preprocess/extract.py](preprocess/extract.py):

- `build_inpaint_mask_from_clothes_mask`
- `build_masked_clothes_image`
- `prepare_training_tensors`
- `prepare_inference_inputs`
- `assert_pipeline_consistency`

The repository uses a single canonical preprocessing path for:

- mask polarity
- interpolation modes
- clothes-only masked-image construction
- tensor conversion

`assert_pipeline_consistency()` is called before canonical inference starts and raises if train/infer preprocessing diverges on the same sample.

## Training

Default config:

- [configs/default.yaml](configs/default.yaml)

Canonical training command:

```bash
cd /data/ouyanghaodong/oyhd_20260318_extracted/sdxl_controlnet_inpaint_project
python train.py --config configs/default.yaml
```

This entrypoint validates that:

- `model.base_mode == inpaint`
- `project.canonical_pose_inpaint == true`
- exactly one pose ControlNet is configured

Implementation entry:

- [train.py](train.py)
- [src/training/train_controlnet_sdxl_inpaint.py](src/training/train_controlnet_sdxl_inpaint.py)

### Training Notes

- `target_image` is encoded to latents and noised.
- `masked_source_image` is encoded without adding noise.
- The 9-channel UNet input is:
  - `cat([noisy_latents(4), resized_mask(1), masked_source_latents(4)])`
  - this follows the native `diffusers` SDXL inpainting channel contract
- The single ControlNet condition is `pose`.
- Loss is standard diffusion MSE against sampled noise.
- Timestep sampling uses the full scheduler range with no truncation.

## Inference

Canonical inference command:

```bash
cd /data/ouyanghaodong/oyhd_20260318_extracted/sdxl_controlnet_inpaint_project
python infer.py --config configs/default.yaml
```

Implementation entry:

- [infer.py](infer.py)
- [src/training/validate.py](src/training/validate.py)

The canonical inference wrapper enforces:

- `strength == 1.0`
- one pose ControlNet only
- shared preprocessing consistency check

The generated images are written to:

- `inference.output_dir`

If `generate_pairs: true`, original-vs-generated comparisons are also written to:

- `inference.pair_output_dir`

## Batch Pair Generation

To batch-generate aligned pair folders:

```bash
cd /data/ouyanghaodong/oyhd_20260318_extracted/sdxl_controlnet_inpaint_project
python scripts/build_pairs.py --config configs/default.yaml --pairs-dir ./outputs/pairs_example
```

This script:

1. runs inference
2. copies originals into `pairs_dir/original`
3. copies generated mannequin images into `pairs_dir/generated`
4. writes side-by-side comparisons into `pairs_dir/original_generated_pairs`

## Condition-Side Domain-Gap Augmentation

The canonical path supports lightweight augmentation only on condition-side inputs:

- clothes-only image affine jitter
- clothes-only image color jitter
- mask morphology jitter
- pose affine jitter

These are controlled from:

- `data.condition_augmentation` in [configs/default.yaml](configs/default.yaml)

`target_image / x0` always remains clean.

## Installation

```bash
cd /data/ouyanghaodong/oyhd_20260318_extracted/sdxl_controlnet_inpaint_project
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Assumptions

- The canonical path uses a zero-valued RGB background when constructing clothes-only masked images.
- Pose extraction itself is project-specific; load/export helpers live in [preprocess/pose.py](preprocess/pose.py), but the actual detector backend is left pluggable.
- Historical dual-control, LoRA, T2I, and other experimental files remain in the repository for reference, but they are not the default path described here.
