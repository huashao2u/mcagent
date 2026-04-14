#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
/home/songyl/anaconda3/envs/salra/bin/python -m src.baselines.run_baselines --config configs/default.yaml "$@"
