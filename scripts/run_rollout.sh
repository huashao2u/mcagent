#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
conda run -n salra python -m src.rollout.generate_rollouts --config configs/default.yaml "$@"
