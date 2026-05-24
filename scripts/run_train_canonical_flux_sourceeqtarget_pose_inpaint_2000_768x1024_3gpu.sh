#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEFAULT_PYTHON="$ROOT/../oyhd/env/bin/python"
if [[ -x "$DEFAULT_PYTHON" ]]; then
  PYTHON="${PYTHON:-$DEFAULT_PYTHON}"
else
  PYTHON="${PYTHON:-python}"
fi

CONFIG_PATH="${CONFIG_PATH:-$ROOT/configs/train_canonical_flux_sourceeqtarget_pose_inpaint_2000_768x1024_50ep.yaml}"
MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-29542}"
NUM_PROCESSES="${NUM_PROCESSES:-3}"

cd "$ROOT"

export NO_PROXY="${NO_PROXY:-127.0.0.1,localhost}"
export no_proxy="${no_proxy:-127.0.0.1,localhost}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export DIFFUSERS_SKIP_DEP_VERSION_CHECK="${DIFFUSERS_SKIP_DEP_VERSION_CHECK:-1}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1,2,3}"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"

DATASET_ROOT="$ROOT/datasets/canonical_flux_sourceeqtarget_pose_inpaint_2000_768x1024"
LATENT_METADATA="$DATASET_ROOT/metadata_train_latent_cached_768x1024.jsonl"
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

echo "== 3GPU Train =="
"$PYTHON" -m accelerate.commands.launch \
  --num_processes "$NUM_PROCESSES" \
  --num_machines 1 \
  --mixed_precision bf16 \
  --main_process_port "$MAIN_PROCESS_PORT" \
  -m src.training.train_controlnet_sdxl_inpaint \
  --config "$CONFIG_PATH"
