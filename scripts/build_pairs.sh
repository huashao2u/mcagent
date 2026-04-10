#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
conda run -n salra python -m src.pairs.build_pairs --config configs/default.yaml "$@"
