#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEFAULT_PYTHON="$ROOT/../oyhd/env/bin/python"
if [[ -x "$DEFAULT_PYTHON" ]]; then
  PYTHON="${PYTHON:-$DEFAULT_PYTHON}"
else
  PYTHON="${PYTHON:-python}"
fi

CONFIG_PATH="${CONFIG_PATH:-$ROOT/configs/lora_train_sdxl_inpaint_controlnet_dualcontrol_clothesrgb_pose_sourcetarget2000_from_run22_4gpu_50ep.yaml}"
MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-29523}"
NUM_PROCESSES="${NUM_PROCESSES:-4}"

cd "$ROOT"

export NO_PROXY="${NO_PROXY:-127.0.0.1,localhost}"
export no_proxy="${no_proxy:-127.0.0.1,localhost}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"

DATASET_ROOT="$ROOT/datasets/source_target_2000_clothes_pose_dualcontrol"
LATENT_METADATA="$DATASET_ROOT/metadata_train_latent_cached.jsonl"
LATENT_CACHE_DIR="$DATASET_ROOT/latent_cache_train"
TRAIN_METADATA="$DATASET_ROOT/metadata_train.jsonl"

if [[ -f "$LATENT_METADATA" ]] && [[ -d "$LATENT_CACHE_DIR" ]] && [[ "$(wc -l < "$LATENT_METADATA")" -gt 0 ]]; then
  echo "== Latent cache =="
  echo "latent cache already exists, skip recaching"
else
  echo "== Latent cache =="
  "$PYTHON" scripts/precompute_training_latent_cache.py \
    --config "$CONFIG_PATH" \
    --metadata "$TRAIN_METADATA" \
    --output-metadata "$LATENT_METADATA" \
    --cache-dir "$LATENT_CACHE_DIR" \
    --batch-size 1
fi

echo "== 4GPU LoRA Train =="
"$PYTHON" -m accelerate.commands.launch \
  --num_processes "$NUM_PROCESSES" \
  --num_machines 1 \
  --mixed_precision bf16 \
  --main_process_port "$MAIN_PROCESS_PORT" \
  -m src.training.lora_train_controlnet_sdxl_inpaint \
  --config "$CONFIG_PATH"
