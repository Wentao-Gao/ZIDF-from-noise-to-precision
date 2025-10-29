#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export MPLBACKEND=Agg

# Ablation examples on synthetic data
SAVE=./results_ablation
mkdir -p "$SAVE"

# 1) Different zero ratios
for zr in 0.0 0.3 0.5 0.7 0.9; do
  python train.py --config configs/synthetic_70zeros.yaml --zero_ratio $zr --save_dir ./checkpoints_zr_${zr//./}
  python evaluate.py --n_samples 10000 --n_features 5 --zero_ratio $zr \
    --input_len 48 --output_len 24 --batch_size 64 --num_workers 4 \
    --diffusion_steps 1000 --beta_schedule cosine \
    --zidf_checkpoint ./checkpoints_zr_${zr//./}/best_model.pt \
    --save_dir $SAVE/zr_${zr//./}
done

# 2) Noise level ablation
for an in 0.001 0.01 0.1 0.2 0.5; do
  python train.py --config configs/synthetic_70zeros.yaml --alpha_noise $an --save_dir ./checkpoints_alpha_${an//./}
  python evaluate.py --config configs/synthetic_70zeros.yaml \
    --zidf_checkpoint ./checkpoints_alpha_${an//./}/best_model.pt \
    --save_dir $SAVE/alpha_${an//./}
done

echo "Ablations completed. Results in $SAVE"

