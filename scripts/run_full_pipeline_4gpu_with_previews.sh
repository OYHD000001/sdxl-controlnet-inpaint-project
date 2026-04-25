#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEFAULT_PYTHON="$ROOT/../oyhd/env/bin/python"
if [[ -x "$DEFAULT_PYTHON" ]]; then
  PYTHON="${PYTHON:-$DEFAULT_PYTHON}"
else
  PYTHON="${PYTHON:-python}"
fi

CONFIG_PATH="${CONFIG_PATH:-$ROOT/configs/train_sdxl_inpaint_controlnet_dualcontrol_pretrained_canny_pose_4gpu_50ep.yaml}"
NUM_PROCESSES="${NUM_PROCESSES:-4}"
MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-29516}"
PREVIEW_CUDA_VISIBLE_DEVICES="${PREVIEW_CUDA_VISIBLE_DEVICES:-3}"

cd "$ROOT"

export NO_PROXY="${NO_PROXY:-127.0.0.1,localhost}"
export no_proxy="${no_proxy:-127.0.0.1,localhost}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

echo "== Preview watcher =="
(
  export CUDA_VISIBLE_DEVICES="$PREVIEW_CUDA_VISIBLE_DEVICES"
  "$PYTHON" scripts/watch_training_previews.py --config "$CONFIG_PATH" --python "$PYTHON"
) &
PREVIEW_PID=$!
trap 'kill "$PREVIEW_PID" 2>/dev/null || true' EXIT

echo "== 4GPU Train =="
"$PYTHON" -m accelerate.commands.launch \
  --num_processes "$NUM_PROCESSES" \
  --num_machines 1 \
  --mixed_precision bf16 \
  --main_process_port "$MAIN_PROCESS_PORT" \
  -m src.training.train_controlnet_sdxl_inpaint \
  --config "$CONFIG_PATH"

echo "== Infer =="
CUDA_VISIBLE_DEVICES="${INFER_CUDA_VISIBLE_DEVICES:-0}" "$PYTHON" -m src.training.validate --config "$CONFIG_PATH" --mode infer

wait "$PREVIEW_PID" || true
