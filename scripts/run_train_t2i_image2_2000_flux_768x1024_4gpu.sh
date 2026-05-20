#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEFAULT_PYTHON="$ROOT/../oyhd/env/bin/python"
if [[ -x "$DEFAULT_PYTHON" ]]; then
  PYTHON="${PYTHON:-$DEFAULT_PYTHON}"
else
  PYTHON="${PYTHON:-python}"
fi

CONFIG_PATH="${CONFIG_PATH:-$ROOT/configs/train_sdxl_t2i_controlnet_dualcontrol_clothesrgb_headlesspose_image2_2000_flux_768x1024_50ep.yaml}"
MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-29534}"
NUM_PROCESSES="${NUM_PROCESSES:-4}"
BASE_MODEL_REPO="${BASE_MODEL_REPO:-stabilityai/stable-diffusion-xl-base-1.0}"
FORCE_RECACHE="${FORCE_RECACHE:-false}"

cd "$ROOT"

export NO_PROXY="${NO_PROXY:-127.0.0.1,localhost}"
export no_proxy="${no_proxy:-127.0.0.1,localhost}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"

DATA_SOURCE_ROOT="$ROOT/datasets/data"
TARGET_DIR="$DATA_SOURCE_ROOT/image2_2000_flux模特图"
MASK_DIR="$DATA_SOURCE_ROOT/image2_2000_flux模特图_服装mask/masks"
CUTOUT_DIR="$DATA_SOURCE_ROOT/image2_2000_flux模特图_服装mask/cutouts"
POSE_DIR="$DATA_SOURCE_ROOT/image2_2000_flux模特图_mmpose/headless_pose"
DATASET_ROOT="$ROOT/datasets/image2_2000_flux_clothesrgb_headlesspose_t2i_dualcontrol_768x1024"

TRAIN_METADATA="$DATASET_ROOT/metadata_train.jsonl"
LATENT_METADATA="$DATASET_ROOT/metadata_train_latent_cached_768x1024.jsonl"
LATENT_CACHE_DIR="$DATASET_ROOT/latent_cache_train"

PROMPT="${PROMPT:-replace the human model with a smooth glossy white plastic retail mannequin, clearly non-human mannequin body, rigid mannequin limbs, no human skin, no flesh, no realistic face, no realistic hands, keep the exact clothes, garment texture, garment silhouette, pose, composition, and studio background}"

echo "== Prepare T2I dataset =="
"$PYTHON" "$ROOT/scripts/prepare_t2i_dualcontrol_flux_dataset.py" \
  --target-dir "$TARGET_DIR" \
  --mask-dir "$MASK_DIR" \
  --cutout-dir "$CUTOUT_DIR" \
  --pose-dir "$POSE_DIR" \
  --output-root "$DATASET_ROOT" \
  --output-subdir "image2_2000_flux模特图" \
  --prompt "$PROMPT"

echo "== Prefetch SDXL T2I base =="
"$PYTHON" - <<PY
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="${BASE_MODEL_REPO}",
    allow_patterns=[
        "model_index.json",
        "scheduler/scheduler_config.json",
        "text_encoder/config.json",
        "text_encoder/model.fp16.safetensors",
        "text_encoder_2/config.json",
        "text_encoder_2/model.fp16.safetensors",
        "tokenizer/*",
        "tokenizer_2/*",
        "unet/config.json",
        "unet/diffusion_pytorch_model.fp16.safetensors",
        "vae/config.json",
        "vae/diffusion_pytorch_model.fp16.safetensors",
    ],
)
print("prefetched ${BASE_MODEL_REPO}")
PY

if [[ "$FORCE_RECACHE" =~ ^(1|true|yes|on)$ ]]; then
  echo "== Latent cache =="
  echo "force recache enabled, removing stale cache"
  rm -f "$LATENT_METADATA"
  rm -rf "$LATENT_CACHE_DIR"
fi

if [[ -f "$LATENT_METADATA" ]] && [[ -d "$LATENT_CACHE_DIR" ]] && [[ "$(wc -l < "$LATENT_METADATA")" -gt 0 ]]; then
  echo "== Latent cache =="
  echo "latent cache already exists, skip recaching"
else
  echo "== Latent cache =="
  "$PYTHON" "$ROOT/scripts/precompute_training_latent_cache.py" \
    --config "$CONFIG_PATH" \
    --metadata "$TRAIN_METADATA" \
    --output-metadata "$LATENT_METADATA" \
    --cache-dir "$LATENT_CACHE_DIR" \
    --batch-size 1
fi

echo "== 4GPU Train =="
"$PYTHON" -m accelerate.commands.launch \
  --num_processes "$NUM_PROCESSES" \
  --num_machines 1 \
  --mixed_precision bf16 \
  --main_process_port "$MAIN_PROCESS_PORT" \
  -m src.training.train_controlnet_sdxl_inpaint \
  --config "$CONFIG_PATH"
