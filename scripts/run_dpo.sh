#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
MASTER_PORT="${MASTER_PORT:-29600}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" /home/songyl/anaconda3/envs/salra/bin/torchrun \
  --standalone \
  --nproc_per_node="$NPROC_PER_NODE" \
  --master_port="$MASTER_PORT" \
  -m src.training.run_dpo \
  --config configs/default.yaml \
  "$@"
