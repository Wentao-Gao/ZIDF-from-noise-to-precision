"""
Evaluation script for comparing ZIDF with baseline models
"""

import torch
import numpy as np
import argparse
import os
import json
import pandas as pd
from tqdm import tqdm

from models.zidf import ZIDF
from models.diffusion import DiffusionModel
from models.non_stationary_transformer import NonStationaryTransformer
from models.official_nst_adapter import OfficialNSTWrapper
from models.baselines import ZeroInflatedPoisson, HurdleModel, ZeroInflatedGaussian
from data.data_loader import create_dataloaders
from data.preprocessing import generate_synthetic_data
from utils.metrics import compute_metrics, compute_zero_inflation_metrics
from utils.visualization import plot_predictions, plot_zero_inflation_analysis
from utils.config import load_config, apply_config_to_args


def evaluate_all_models(args):
    """Evaluate ZIDF and baseline models."""

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using device: {device}')

    # Generate data
    print('Generating test data...')
    features, targets = generate_synthetic_data(
        n_samples=args.n_samples,
        n_features=args.n_features,
        zero_ratio=args.zero_ratio
    )

    # Split data
    n = len(features)
    train_end = int(n * 0.7)
    val_end = int(n * 0.85)

    test_features = features[val_end:]
    test_targets = targets[val_end:]

    print(f'Test samples: {len(test_features)}')
    print(f'Zero ratio in test set: {np.sum(test_targets == 0) / len(test_targets):.3f}')

    # Prepare data for baselines (flatten for simple models)
    test_features_flat = test_features[args.input_len:]
    test_targets_flat = test_targets[args.input_len:]

    # Ensure output directory exists early (before plotting)
    os.makedirs(args.save_dir, exist_ok=True)

    results = {}

    # Evaluate baseline models
    print('\n' + '='*50)
    print('Evaluating Baseline Models')
    print('='*50)

    # Zero-Inflated Poisson
    print('\n[1/3] Zero-Inflated Poisson')
    zip_model = ZeroInflatedPoisson(input_dim=args.n_features, hidden_dim=64).to(device)

    # Simple training
    optimizer = torch.optim.Adam(zip_model.parameters(), lr=1e-3)
    X_train = torch.FloatTensor(features[:train_end]).to(device)
    y_train = torch.FloatTensor(targets[:train_end]).to(device)

    zip_model.train()
    for _ in tqdm(range(100), desc='Training ZIP'):
        optimizer.zero_grad()
        loss = zip_model.compute_loss(X_train, y_train)
        loss.backward()
        optimizer.step()

    zip_model.eval()
    with torch.no_grad():
        X_test = torch.FloatTensor(test_features_flat).to(device)
        zip_predictions = zip_model.predict(X_test).cpu().numpy().flatten()

    zip_metrics = compute_metrics(zip_predictions, test_targets_flat, prefix='zip_')
    results['ZIP'] = zip_metrics
    print('ZIP Metrics:', {k: f'{v:.6f}' for k, v in zip_metrics.items()})

    # Hurdle Model
    print('\n[2/3] Hurdle Model')
    hurdle_model = HurdleModel(input_dim=args.n_features, hidden_dim=64).to(device)

    optimizer = torch.optim.Adam(hurdle_model.parameters(), lr=1e-3)
    hurdle_model.train()
    for _ in tqdm(range(100), desc='Training Hurdle'):
        optimizer.zero_grad()
        loss = hurdle_model.compute_loss(X_train, y_train)
        loss.backward()
        optimizer.step()

    hurdle_model.eval()
    with torch.no_grad():
        hurdle_predictions = hurdle_model.predict(X_test).cpu().numpy().flatten()

    hurdle_metrics = compute_metrics(hurdle_predictions, test_targets_flat, prefix='hurdle_')
    results['Hurdle'] = hurdle_metrics
    print('Hurdle Metrics:', {k: f'{v:.6f}' for k, v in hurdle_metrics.items()})

    # Zero-Inflated Gaussian
    print('\n[3/3] Zero-Inflated Gaussian')
    zig_model = ZeroInflatedGaussian(input_dim=args.n_features, hidden_dim=64).to(device)

    optimizer = torch.optim.Adam(zig_model.parameters(), lr=1e-3)
    zig_model.train()
    for _ in tqdm(range(100), desc='Training ZIG'):
        optimizer.zero_grad()
        loss = zig_model.compute_loss(X_train, y_train)
        loss.backward()
        optimizer.step()

    zig_model.eval()
    with torch.no_grad():
        zig_predictions = zig_model.predict(X_test).cpu().numpy().flatten()

    zig_metrics = compute_metrics(zig_predictions, test_targets_flat, prefix='zig_')
    results['ZIG'] = zig_metrics
    print('ZIG Metrics:', {k: f'{v:.6f}' for k, v in zig_metrics.items()})

    # Load and evaluate ZIDF model if checkpoint exists
    if os.path.exists(args.zidf_checkpoint):
        print('\n' + '='*50)
        print('Evaluating ZIDF Model')
        print('='*50)

        # Create dataloaders
        _, _, test_loader = create_dataloaders(
            features[:train_end], targets[:train_end],
            features[train_end:val_end], targets[train_end:val_end],
            test_features, test_targets,
            input_len=args.input_len,
            output_len=args.output_len,
            batch_size=args.batch_size,
            num_workers=args.num_workers
        )

        # Initialize ZIDF
        # Build predictive component (official NST optional)
        if getattr(args, 'use_official_nst', False):
            import importlib
            if not args.nst_module or not args.nst_class:
                print("[WARN] use_official_nst=True but nst_module/nst_class not provided; falling back to internal NST.")
                predictive_model = NonStationaryTransformer(
                    input_dim=args.n_features,
                    output_len=args.output_len,
                    device=device
                ).to(device)
            else:
                print(f"Loading official NST: {args.nst_module}.{args.nst_class} (defaults)")
                mod = importlib.import_module(args.nst_module)
                cls = getattr(mod, args.nst_class)
                base_model = cls()  # default official config
                base_model = base_model.to(device)
                predictive_model = OfficialNSTWrapper(base_model, output_len=args.output_len, device=device).to(device)
        else:
            predictive_model = NonStationaryTransformer(
                input_dim=args.n_features,
                output_len=args.output_len,
                device=device
            ).to(device)

        diffusion_model = DiffusionModel(
            input_dim=args.output_len,
            hidden_dim=128,
            num_steps=args.diffusion_steps,
            device=device,
            beta_schedule=args.beta_schedule
        ).to(device)

        zidf_model = ZIDF(
            predictive_model=predictive_model,
            diffusion_model=diffusion_model,
            alpha_noise=0.1,
            device=device
        ).to(device)

        # Load checkpoint
        checkpoint = torch.load(args.zidf_checkpoint)
        predictive_model.load_state_dict(checkpoint['predictive_model'])
        diffusion_model.load_state_dict(checkpoint['diffusion_model'])

        # Evaluate
        zidf_model.eval()
        all_predictions = []
        all_targets = []

        with torch.no_grad():
            for x, y_history, y_target in tqdm(test_loader, desc='Evaluating ZIDF'):
                x = x.to(device)
                y_history = y_history.to(device)

                predictions = zidf_model.forward_inference(
                    x, y_history, num_diffusion_steps=args.diffusion_steps
                )
                all_predictions.append(predictions.cpu().numpy())
                all_targets.append(y_target.numpy())

        zidf_predictions = np.concatenate(all_predictions, axis=0).flatten()
        zidf_targets = np.concatenate(all_targets, axis=0).flatten()

        zidf_metrics = compute_metrics(zidf_predictions, zidf_targets, prefix='zidf_')
        results['ZIDF'] = zidf_metrics
        print('ZIDF Metrics:', {k: f'{v:.6f}' for k, v in zidf_metrics.items()})

        # Plot ZIDF predictions
        plot_predictions(
            zidf_predictions,
            zidf_targets,
            save_path=os.path.join(args.save_dir, 'zidf_predictions.png')
        )

        plot_zero_inflation_analysis(
            zidf_predictions,
            zidf_targets,
            save_path=os.path.join(args.save_dir, 'zidf_zero_inflation.png')
        )

    # Create comparison table
    print('\n' + '='*50)
    print('Comparison Table')
    print('='*50)

    comparison_df = pd.DataFrame({
        model: {
            'MSE': metrics.get(f'{model.lower()}_mse', metrics.get('mse', np.nan)),
            'MAE': metrics.get(f'{model.lower()}_mae', metrics.get('mae', np.nan)),
            'RMSE': metrics.get(f'{model.lower()}_rmse', metrics.get('rmse', np.nan)),
        }
        for model, metrics in results.items()
    }).T

    print(comparison_df.to_string())

    # Save results

    # Ensure JSON-serializable
    results_serializable = {}
    for model, mets in results.items():
        results_serializable[model] = {k: float(v) for k, v in mets.items()}

    with open(os.path.join(args.save_dir, 'comparison_results.json'), 'w') as f:
        json.dump(results_serializable, f, indent=4)

    comparison_df.to_csv(os.path.join(args.save_dir, 'comparison_table.csv'))

    print(f'\nResults saved to {args.save_dir}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate ZIDF and baseline models')

    parser.add_argument('--n_samples', type=int, default=10000, help='Number of samples')
    parser.add_argument('--n_features', type=int, default=5, help='Number of features')
    parser.add_argument('--zero_ratio', type=float, default=0.7, help='Zero inflation ratio')
    parser.add_argument('--input_len', type=int, default=48, help='Input sequence length')
    parser.add_argument('--output_len', type=int, default=24, help='Output sequence length')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--zidf_checkpoint', type=str, default='./checkpoints/best_model.pt',
                        help='Path to ZIDF checkpoint')
    parser.add_argument('--save_dir', type=str, default='./results', help='Directory to save results')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--diffusion_steps', type=int, default=1000, help='Diffusion steps for ZIDF')
    parser.add_argument('--beta_schedule', type=str, default='linear', choices=['linear','cosine'], help='Diffusion beta schedule')
    parser.add_argument('--config', type=str, default='', help='Path to YAML/JSON config')
    parser.add_argument('--use_official_nst', action='store_true', help='Use official Non-stationary Transformer')
    parser.add_argument('--nst_module', type=str, default='', help='Python module path for official NST (e.g., thuml.Nonstationary_Transformers.xxx)')
    parser.add_argument('--nst_class', type=str, default='', help='Class name for official NST')
    parser.add_argument('--num_workers', type=int, default=0, help='Dataloader workers (0 for compatibility)')

    args = parser.parse_args()

    # Apply config overrides
    if args.config:
        cfg = load_config(args.config)
        args = apply_config_to_args(args, cfg)

    # Set random seed and deterministic flags
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    try:
        import random
        random.seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass

    evaluate_all_models(args)
