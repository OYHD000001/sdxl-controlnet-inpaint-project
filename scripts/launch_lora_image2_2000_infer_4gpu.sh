#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEFAULT_PYTHON="$ROOT/../oyhd/env/bin/python"
if [[ -x "$DEFAULT_PYTHON" ]]; then
  PYTHON="${PYTHON:-$DEFAULT_PYTHON}"
else
  PYTHON="${PYTHON:-python}"
fi

BASE_CONFIG="${BASE_CONFIG:-$ROOT/configs/infer_run23_lora_image2_2000_clothesrgb_pose.yaml}"
METADATA_PATH="${METADATA_PATH:-$ROOT/datasets/external_image2_infer/run23_image2_2000_clothesrgb_pose/metadata_all.jsonl}"
RUN_NAME="${RUN_NAME:-run24_image2_2000_run23_lora}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$ROOT/outputs/$RUN_NAME/generated}"
NUM_GPUS="${NUM_GPUS:-4}"
PROGRESS_INTERVAL="${PROGRESS_INTERVAL:-30}"
SESSION_PREFIX="${SESSION_PREFIX:-run24_image2_2000_lora}"

LAUNCH_ROOT="$ROOT/outputs/$RUN_NAME"
SHARD_ROOT="$LAUNCH_ROOT/shards"
CONFIG_ROOT="$LAUNCH_ROOT/configs"
LOG_ROOT="$LAUNCH_ROOT/logs"

mkdir -p "$SHARD_ROOT" "$CONFIG_ROOT" "$LOG_ROOT" "$OUTPUT_ROOT"

export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
export NO_PROXY="${NO_PROXY:-127.0.0.1,localhost}"
export no_proxy="${no_proxy:-127.0.0.1,localhost}"

"$PYTHON" "$ROOT/scripts/shard_metadata_jsonl.py" \
  --input "$METADATA_PATH" \
  --output-dir "$SHARD_ROOT" \
  --num-shards "$NUM_GPUS"

for ((gpu=0; gpu<NUM_GPUS; gpu++)); do
  export SHARD_PATH="$SHARD_ROOT/shard_$(printf '%02d' "$gpu").jsonl"
  export CONFIG_PATH="$CONFIG_ROOT/infer_gpu_${gpu}.yaml"
  export SHARED_OUTPUT_ROOT="$OUTPUT_ROOT"
  export BASE_CONFIG_PATH="$BASE_CONFIG"
  "$PYTHON" - <<'PY'
import os
from pathlib import Path
import yaml

base = Path(os.environ["BASE_CONFIG_PATH"])
config_path = Path(os.environ["CONFIG_PATH"])
shard_path = Path(os.environ["SHARD_PATH"])
output_root = Path(os.environ["SHARED_OUTPUT_ROOT"])

config = yaml.safe_load(base.read_text(encoding="utf-8"))
config["inference"]["metadata_path"] = str(shard_path.resolve())
config["inference"]["output_dir"] = str(output_root.resolve())
config["inference"]["enable_model_cpu_offload"] = False
config_path.write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8")
print(config_path)
PY
done

launch_session() {
  local session_name="$1"
  local command="$2"
  if tmux has-session -t "$session_name" 2>/dev/null; then
    echo "tmux session exists, keep running: $session_name"
  else
    tmux new-session -d -s "$session_name" "$command"
    echo "started tmux session: $session_name"
  fi
}

for ((gpu=0; gpu<NUM_GPUS; gpu++)); do
  session="${SESSION_PREFIX}_gpu_${gpu}"
  log_path="$LOG_ROOT/gpu_${gpu}.log"
  cmd="cd '$ROOT' && CUDA_VISIBLE_DEVICES=$gpu '$PYTHON' -m src.training.lora_infer_controlnet_sdxl_inpaint --config '$CONFIG_ROOT/infer_gpu_${gpu}.yaml' 2>&1 | tee '$log_path'"
  launch_session "$session" "$cmd"
done

watch_session="${SESSION_PREFIX}_watch"
watch_cmd="cd '$ROOT' && '$PYTHON' '$ROOT/scripts/watch_external_infer_progress.py' --metadata '$METADATA_PATH' --output-root '$OUTPUT_ROOT' --interval '$PROGRESS_INTERVAL'"
launch_session "$watch_session" "$watch_cmd"

echo "metadata: $METADATA_PATH"
echo "output root: $OUTPUT_ROOT"
echo "config root: $CONFIG_ROOT"
echo "watch session: $watch_session"
