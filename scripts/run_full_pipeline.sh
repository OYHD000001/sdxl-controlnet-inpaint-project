#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEFAULT_PYTHON="$ROOT/../oyhd/env/bin/python"
if [[ -x "$DEFAULT_PYTHON" ]]; then
  PYTHON="${PYTHON:-$DEFAULT_PYTHON}"
else
  PYTHON="${PYTHON:-python}"
fi

CONFIG_PATH="${CONFIG_PATH:-$ROOT/configs/train_sdxl_inpaint_controlnet.yaml}"

cd "$ROOT"

export NO_PROXY="${NO_PROXY:-127.0.0.1,localhost}"
export no_proxy="${no_proxy:-127.0.0.1,localhost}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

echo "== Train =="
"$PYTHON" -m src.training.train_controlnet_sdxl_inpaint --config "$CONFIG_PATH"

echo "== Infer =="
"$PYTHON" -m src.training.validate --config "$CONFIG_PATH" --mode infer
