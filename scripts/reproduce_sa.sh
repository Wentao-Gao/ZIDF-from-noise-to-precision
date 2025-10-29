#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export MPLBACKEND=Agg

# Reproduce South Australia experiment (requires local NetCDF paths in configs/sa_precip.yaml)

python train.py \
  --config configs/sa_precip.yaml

python evaluate.py \
  --config configs/sa_precip.yaml \
  --zidf_checkpoint ./checkpoints_sa/best_model.pt \
  --save_dir ./results_sa

echo "SA reproduction finished. Results in ./results_sa"

