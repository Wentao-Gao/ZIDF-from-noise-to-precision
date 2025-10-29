#!/bin/bash

# Example script to run ZIDF training and evaluation

echo "==================================="
echo "ZIDF Training and Evaluation Script"
echo "==================================="

# Create necessary directories
mkdir -p checkpoints
mkdir -p results

# Step 1: Train ZIDF model
echo ""
echo "Step 1: Training ZIDF model..."
python train.py \
    --use_synthetic \
    --n_samples 10000 \
    --n_features 5 \
    --zero_ratio 0.7 \
    --input_len 48 \
    --output_len 24 \
    --batch_size 64 \
    --d_model 512 \
    --n_heads 8 \
    --pred_epochs 50 \
    --diff_epochs 30 \
    --pred_lr 1e-4 \
    --diff_lr 1e-4 \
    --alpha_noise 0.1 \
    --save_dir ./checkpoints \
    --seed 42

# Step 2: Evaluate and compare with baselines
echo ""
echo "Step 2: Evaluating model and comparing with baselines..."
python evaluate.py \
    --n_samples 10000 \
    --n_features 5 \
    --zero_ratio 0.7 \
    --input_len 48 \
    --output_len 24 \
    --batch_size 64 \
    --zidf_checkpoint ./checkpoints/best_model.pt \
    --save_dir ./results \
    --seed 42

echo ""
echo "==================================="
echo "Training and evaluation completed!"
echo "Results saved to ./results/"
echo "==================================="
