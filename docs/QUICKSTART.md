# Quick Start Guide

This guide will help you get started with ZIDF quickly.

## Installation

### Step 1: Clone the Repository

```bash
git clone https://github.com/Wentao-Gao/ZIDF-from-noise-to-precision.git
cd ZIDF-from-noise-to-precision
```

### Step 2: Set Up Environment

```bash
# Create conda environment
conda create -n zidf python=3.9
conda activate zidf

# Install dependencies
pip install -r requirements.txt
```

### Step 3: Verify Installation

```bash
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

## Running Your First Experiment

### Option 1: Using the Example Script

The easiest way to get started:

```bash
./run_example.sh
```

This script will:
1. Train ZIDF on synthetic zero-inflated data
2. Evaluate the model and compare with baselines
3. Save results to `./results/`

### Option 2: Manual Training

Train the predictive component:

```bash
python train.py \
    --use_synthetic \
    --n_samples 10000 \
    --zero_ratio 0.7 \
    --pred_epochs 50 \
    --diff_epochs 30 \
    --save_dir ./checkpoints
```

### Option 3: Using Jupyter Notebook

For interactive exploration:

```bash
jupyter notebook notebooks/demo.ipynb
```

## Understanding the Output

After training, you'll find:

- `checkpoints/`: Saved model weights
  - `best_predictive_model.pt`: Predictive component
  - `best_diffusion_model.pt`: Diffusion model

- `results/`: Evaluation results
  - `test_metrics.json`: Performance metrics
  - `predictions.png`: Visualization of predictions
  - `training_history.png`: Training curves

## Key Parameters

### Data Parameters

- `--n_samples`: Number of samples (default: 10000)
- `--zero_ratio`: Proportion of zeros in data (default: 0.7)
- `--input_len`: Input sequence length (default: 48)
- `--output_len`: Output sequence length (default: 24)

### Model Parameters

- `--d_model`: Model dimension (default: 512)
- `--n_heads`: Number of attention heads (default: 8)
- `--alpha_noise`: Noise injection level (default: 0.1)
- `--diffusion_steps`: Number of diffusion steps (default: 1000)

### Training Parameters

- `--pred_epochs`: Epochs for predictive component (default: 50)
- `--diff_epochs`: Epochs for diffusion model (default: 30)
- `--pred_lr`: Learning rate for predictor (default: 1e-4)
- `--diff_lr`: Learning rate for diffusion (default: 1e-4)

## Customizing for Your Data

To use your own precipitation data:

1. Prepare data in the format: `(features, targets)`
   - Features: `(n_samples, n_features)` numpy array
   - Targets: `(n_samples,)` numpy array with precipitation values

2. Load data in `train.py`:

```python
from data.preprocessing import preprocess_precipitation_data

# Load your data
features, targets = load_your_data()

# Preprocess
train_features, train_targets, val_features, val_targets, \
test_features, test_targets, metadata = preprocess_precipitation_data(
    features, targets,
    train_ratio=0.7,
    val_ratio=0.15
)
```

3. Run training:

```bash
python train.py --save_dir ./my_checkpoints
```

## Troubleshooting

### CUDA Out of Memory

Reduce batch size or model dimensions:

```bash
python train.py --batch_size 32 --d_model 256
```

### Slow Training

- Use GPU if available
- Reduce diffusion steps during training
- Use smaller model dimensions

### Poor Performance

- Increase training epochs
- Adjust noise level (`--alpha_noise`)
- Try different learning rates
- Check data quality and zero ratio

## Next Steps

1. **Explore the Demo Notebook**: `notebooks/demo.ipynb` provides interactive examples
2. **Read the Paper**: Understand the theory behind ZIDF
3. **Experiment with Hyperparameters**: Try different configurations
4. **Use Your Own Data**: Apply ZIDF to your precipitation dataset

## Getting Help

- Check [Issues](https://github.com/Wentao-Gao/ZIDF-from-noise-to-precision/issues) for common problems
- Read the [Contributing Guide](../CONTRIBUTING.md) to contribute
- Contact the authors for research collaborations

## Example Results

Expected performance on synthetic data (70% zeros):

| Model | MSE | MAE |
|-------|-----|-----|
| ZIDF | 0.38 | 0.25 |
| Non-stationary Transformer | 0.87 | 0.30 |
| ZIP | 1.56 | 0.56 |

Your results may vary based on:
- Data characteristics
- Random seed
- Hyperparameters
- Hardware

Happy forecasting! 🌧️
