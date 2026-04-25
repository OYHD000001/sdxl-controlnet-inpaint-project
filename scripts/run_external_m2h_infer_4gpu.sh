#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEFAULT_PYTHON="$ROOT/../oyhd/env/bin/python"
if [[ -x "$DEFAULT_PYTHON" ]]; then
  PYTHON="${PYTHON:-$DEFAULT_PYTHON}"
else
  PYTHON="${PYTHON:-python}"
fi

BASE_CONFIG="${BASE_CONFIG:-$ROOT/configs/train_sdxl_inpaint_controlnet_dualcontrol_pretrained_canny_headlesspose_whiteplastic800_4gpu_cached_fast_50ep.yaml}"
RUN_NAME="${RUN_NAME:-run20_external_m2h_infer}"
WORK_ROOT="${WORK_ROOT:-$ROOT/datasets/external_m2h_infer/$RUN_NAME}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$ROOT/outputs/$RUN_NAME/generated}"
CONFIG_ROOT="${CONFIG_ROOT:-$ROOT/outputs/$RUN_NAME/configs}"
LOG_ROOT="${LOG_ROOT:-$ROOT/outputs/$RUN_NAME/logs}"
SHARD_ROOT="${SHARD_ROOT:-$ROOT/outputs/$RUN_NAME/shards}"
COMBINED_METADATA="${COMBINED_METADATA:-$ROOT/outputs/$RUN_NAME/metadata_all.jsonl}"
NUM_GPUS="${NUM_GPUS:-4}"
MASK_WORKERS="${MASK_WORKERS:-16}"
COND_WORKERS="${COND_WORKERS:-16}"
PROGRESS_INTERVAL="${PROGRESS_INTERVAL:-30}"

mkdir -p "$WORK_ROOT" "$OUTPUT_ROOT" "$CONFIG_ROOT" "$LOG_ROOT" "$SHARD_ROOT"

export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
export NO_PROXY="${NO_PROXY:-127.0.0.1,localhost}"
export no_proxy="${no_proxy:-127.0.0.1,localhost}"

prepare_subset() {
  local dataset_tag="$1"
  local image_dir="$2"
  local pose_dir="$3"
  local subset_key
  subset_key="$(echo "$dataset_tag" | tr '/' '_')"
  local subset_root="$WORK_ROOT/$subset_key"
  local mask_dir="$subset_root/person_masks"
  local meta_root="$subset_root/metadata"

  echo "== prepare masks: $dataset_tag =="
  "$PYTHON" "$ROOT/scripts/prepare_pose_guided_person_masks.py" \
    --image-dir "$image_dir" \
    --pose-dir "$pose_dir" \
    --output-dir "$mask_dir" \
    --workers "$MASK_WORKERS"

  echo "== build metadata: $dataset_tag =="
  "$PYTHON" "$ROOT/scripts/build_external_infer_metadata.py" \
    --image-dir "$image_dir" \
    --mask-dir "$mask_dir" \
    --pose-dir "$pose_dir" \
    --output-root "$meta_root" \
    --dataset-tag "$dataset_tag" \
    --workers "$COND_WORKERS"
}

prepare_subset "zalando_M2H/singleimage" "/data/ouyanghaodong/zalando_M2H/singleimage" "/data/ouyanghaodong/zalando_M2H/zalando_M2H_mmpose/singleimage"
prepare_subset "zalando_M2H/multiimages" "/data/ouyanghaodong/zalando_M2H/multiimages" "/data/ouyanghaodong/zalando_M2H/zalando_M2H_mmpose/multiimages"
prepare_subset "VITON-HD_M2H/train" "/data/ouyanghaodong/VITON-HD_M2H/train/image" "/data/ouyanghaodong/VITON-HD_M2H/train/mmpose"
prepare_subset "VITON-HD_M2H/test" "/data/ouyanghaodong/VITON-HD_M2H/test/image" "/data/ouyanghaodong/VITON-HD_M2H/test/mmpose"

cat \
  "$WORK_ROOT/zalando_M2H_singleimage/metadata/metadata_all.jsonl" \
  "$WORK_ROOT/zalando_M2H_multiimages/metadata/metadata_all.jsonl" \
  "$WORK_ROOT/VITON-HD_M2H_train/metadata/metadata_all.jsonl" \
  "$WORK_ROOT/VITON-HD_M2H_test/metadata/metadata_all.jsonl" \
  > "$COMBINED_METADATA"

echo "== shard metadata =="
"$PYTHON" "$ROOT/scripts/shard_metadata_jsonl.py" \
  --input "$COMBINED_METADATA" \
  --output-dir "$SHARD_ROOT" \
  --num-shards "$NUM_GPUS"

for ((gpu=0; gpu<NUM_GPUS; gpu++)); do
  export GPU_INDEX="$gpu"
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
  session="m2h_infer_gpu_${gpu}"
  log_path="$LOG_ROOT/gpu_${gpu}.log"
  cmd="cd '$ROOT' && CUDA_VISIBLE_DEVICES=$gpu '$PYTHON' -m src.training.validate --config '$CONFIG_ROOT/infer_gpu_${gpu}.yaml' --mode infer 2>&1 | tee '$log_path'"
  launch_session "$session" "$cmd"
done

watch_session="m2h_infer_watch"
watch_cmd="cd '$ROOT' && '$PYTHON' '$ROOT/scripts/watch_external_infer_progress.py' --metadata '$COMBINED_METADATA' --output-root '$OUTPUT_ROOT' --interval '$PROGRESS_INTERVAL'"
launch_session "$watch_session" "$watch_cmd"

echo "combined metadata: $COMBINED_METADATA"
echo "output root: $OUTPUT_ROOT"
echo "watch session: $watch_session"
