#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
conda run -n salra python -m src.eval.aggregate_results --config configs/default.yaml "$@"
