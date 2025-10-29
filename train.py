"""
Training script for ZIDF model
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import argparse
import os
import json
from tqdm import tqdm

from models.zidf import ZIDF
from models.diffusion import DiffusionModel
from models.non_stationary_transformer import NonStationaryTransformer
from models.official_nst_adapter import OfficialNSTWrapper
from data.data_loader import create_dataloaders
from data.preprocessing import (
    generate_synthetic_data,
    preprocess_precipitation_data,
    create_sliding_windows,
    build_sa_dataset_from_netcdf,
    assemble_features_targets,
)
from utils.config import load_config, apply_config_to_args
from utils.metrics import evaluate_model, compute_metrics
from utils.visualization import plot_training_history, plot_predictions


def train_predictive_component(
    model: ZIDF,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_epochs: int,
    learning_rate: float,
    device: str,
    save_dir: str
):
    """
    Train the predictive component of ZIDF.

    Args:
        model: ZIDF model
        train_loader: Training dataloader
        val_loader: Validation dataloader
        num_epochs: Number of epochs
        learning_rate: Learning rate
        device: Device to use
        save_dir: Directory to save checkpoints
    """
    optimizer = optim.AdamW(model.predictive_model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    history = {
        'train_loss': [],
        'val_loss': []
    }

    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        # Training
        model.train()
        train_losses = []

        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]')
        for x, y_history, y_target in pbar:
            x = x.to(device)
            y_history = y_history.to(device)
            y_target = y_target.to(device)

            optimizer.zero_grad()

            # Forward pass with noisy targets
            predictions, y_target_noisy = model.forward_train(x, y_history, y_target)

            # Compute loss
            loss = model.compute_loss(predictions, y_target_noisy, loss_type='mse')

            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.predictive_model.parameters(), max_norm=1.0)
            optimizer.step()

            train_losses.append(loss.item())
            pbar.set_postfix({'loss': loss.item()})

        # Validation
        model.eval()
        val_losses = []

        with torch.no_grad():
            for x, y_history, y_target in val_loader:
                x = x.to(device)
                y_history = y_history.to(device)
                y_target = y_target.to(device)

                predictions, y_target_noisy = model.forward_train(x, y_history, y_target)
                loss = model.compute_loss(predictions, y_target_noisy, loss_type='mse')

                val_losses.append(loss.item())

        # Update history
        avg_train_loss = np.mean(train_losses)
        avg_val_loss = np.mean(val_losses)

        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)

        print(f'Epoch {epoch+1}/{num_epochs} - Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}')

        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(
                model.predictive_model.state_dict(),
                os.path.join(save_dir, 'best_predictive_model.pt')
            )
            print(f'Saved best model with val loss: {best_val_loss:.6f}')

        scheduler.step()

    return history


def train_diffusion_model(
    diffusion_model: DiffusionModel,
    train_targets: np.ndarray,
    input_len: int,
    output_len: int,
    num_epochs: int,
    learning_rate: float,
    batch_size: int,
    device: str,
    save_dir: str
):
    """
    Train the diffusion model for denoising.

    Args:
        diffusion_model: Diffusion model
        train_targets: Training target values
        num_epochs: Number of epochs
        learning_rate: Learning rate
        batch_size: Batch size
        device: Device to use
        save_dir: Directory to save checkpoints
    """
    optimizer = optim.Adam(diffusion_model.parameters(), lr=learning_rate)

    # Create sliding windows to train diffusion on output sequences
    # Shape: Y_windows -> (N, output_len, 1) -> squeeze to (N, output_len)
    ts = train_targets.reshape(-1, 1)
    _, Y_windows = create_sliding_windows(ts, input_len=input_len, output_len=output_len, stride=1)
    Y_windows = Y_windows.squeeze(-1)
    train_targets_tensor = torch.FloatTensor(Y_windows).to(device)
    dataset = torch.utils.data.TensorDataset(train_targets_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    best_loss = float('inf')

    for epoch in range(num_epochs):
        diffusion_model.train()
        epoch_losses = []

        pbar = tqdm(dataloader, desc=f'Epoch {epoch+1}/{num_epochs} [Diffusion]')
        for batch in pbar:
            x0 = batch[0]

            optimizer.zero_grad()

            # Compute diffusion loss
            loss = diffusion_model.compute_loss(x0)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(diffusion_model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_losses.append(loss.item())
            pbar.set_postfix({'loss': loss.item()})

        avg_loss = np.mean(epoch_losses)
        print(f'Epoch {epoch+1}/{num_epochs} - Diffusion Loss: {avg_loss:.6f}')

        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(
                diffusion_model.state_dict(),
                os.path.join(save_dir, 'best_diffusion_model.pt')
            )
            print(f'Saved best diffusion model with loss: {best_loss:.6f}')


def main(args):
    """Main training function."""
    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using device: {device}')

    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)

    # Generate or load data
    if args.use_synthetic:
        print('Generating synthetic data...')
        features, targets = generate_synthetic_data(
            n_samples=args.n_samples,
            n_features=args.n_features,
            zero_ratio=args.zero_ratio
        )

        # Split data
        n = len(features)
        train_end = int(n * 0.7)
        val_end = int(n * 0.85)

        train_features = features[:train_end]
        train_targets = targets[:train_end]
        val_features = features[train_end:val_end]
        val_targets = targets[train_end:val_end]
        test_features = features[val_end:]
        test_targets = targets[val_end:]

        num_features = args.n_features

    else:
        # Real data path: expects config specifying netcdf_paths, variables, region
        if not hasattr(args, 'netcdf_paths'):
            raise NotImplementedError(
                "Real data requires config with 'netcdf_paths', 'variables', and 'region'"
            )
        print('Loading real precipitation data from NetCDF...')
        df = build_sa_dataset_from_netcdf(
            netcdf_paths=args.netcdf_paths,
            region=args.region,
            variables=args.variables,
        )
        # Target assumed to be precipitation variable (e.g., 'prate')
        target_var = args.target_var if hasattr(args, 'target_var') else args.variables[0]
        features, targets = assemble_features_targets(df, target_var)

        # Split data
        n = len(features)
        train_end = int(n * 0.7)
        val_end = int(n * 0.85)

        train_features = features[:train_end]
        train_targets = targets[:train_end]
        val_features = features[train_end:val_end]
        val_targets = targets[train_end:val_end]
        test_features = features[val_end:]
        test_targets = targets[val_end:]

        num_features = train_features.shape[1]

    # Create dataloaders
    train_loader, val_loader, test_loader = create_dataloaders(
        train_features, train_targets,
        val_features, val_targets,
        test_features, test_targets,
        input_len=args.input_len,
        output_len=args.output_len,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )

    print(f'Train samples: {len(train_loader.dataset)}')
    print(f'Val samples: {len(val_loader.dataset)}')
    print(f'Test samples: {len(test_loader.dataset)}')

    # Initialize models
    print('Initializing models...')

    # Predictive component
    # Predictive component: official NST (optional) or internal implementation
    if getattr(args, 'use_official_nst', False):
        import importlib
        if not args.nst_module or not args.nst_class:
            print("[WARN] use_official_nst=True but nst_module/nst_class not provided; falling back to internal NST.")
            predictive_model = NonStationaryTransformer(
                input_dim=num_features,
                output_len=args.output_len,
                d_model=args.d_model,
                n_heads=args.n_heads,
                e_layers=args.e_layers,
                d_layers=args.d_layers,
                dropout=args.dropout,
                device=device
            ).to(device)
        else:
            print(f"Loading official NST: {args.nst_module}.{args.nst_class} (defaults)")
            mod = importlib.import_module(args.nst_module)
            cls = getattr(mod, args.nst_class)
            base_model = cls()  # use official defaults
            base_model = base_model.to(device)
            predictive_model = OfficialNSTWrapper(base_model, output_len=args.output_len, device=device).to(device)
    else:
        predictive_model = NonStationaryTransformer(
            input_dim=num_features,
            output_len=args.output_len,
            d_model=args.d_model,
            n_heads=args.n_heads,
            e_layers=args.e_layers,
            d_layers=args.d_layers,
            dropout=args.dropout,
            device=device
        ).to(device)

    # Diffusion model
    diffusion_model = DiffusionModel(
        input_dim=args.output_len,
        hidden_dim=args.diffusion_hidden_dim,
        num_steps=args.diffusion_steps,
        device=device,
        beta_schedule=args.beta_schedule
    ).to(device)

    # ZIDF framework
    zidf_model = ZIDF(
        predictive_model=predictive_model,
        diffusion_model=diffusion_model,
        alpha_noise=args.alpha_noise,
        device=device
    ).to(device)

    # Train predictive component
    print('\n' + '='*50)
    print('Training Predictive Component')
    print('='*50)

    history = train_predictive_component(
        zidf_model,
        train_loader,
        val_loader,
        num_epochs=args.pred_epochs,
        learning_rate=args.pred_lr,
        device=device,
        save_dir=args.save_dir
    )

    # Plot training history
    plot_training_history(history, save_path=os.path.join(args.save_dir, 'training_history.png'))

    # Train diffusion model
    print('\n' + '='*50)
    print('Training Diffusion Model')
    print('='*50)

    train_diffusion_model(
        diffusion_model,
        train_targets,
        input_len=args.input_len,
        output_len=args.output_len,
        num_epochs=args.diff_epochs,
        learning_rate=args.diff_lr,
        batch_size=args.batch_size,
        device=device,
        save_dir=args.save_dir
    )

    # Load best models
    print('\nLoading best models for evaluation...')
    predictive_model.load_state_dict(
        torch.load(os.path.join(args.save_dir, 'best_predictive_model.pt'))
    )
    diffusion_model.load_state_dict(
        torch.load(os.path.join(args.save_dir, 'best_diffusion_model.pt'))
    )

    # Save a combined checkpoint for evaluation convenience
    combined_ckpt_path = os.path.join(args.save_dir, 'best_model.pt')
    torch.save({
        'predictive_model': predictive_model.state_dict(),
        'diffusion_model': diffusion_model.state_dict()
    }, combined_ckpt_path)
    print(f"Saved combined checkpoint to {combined_ckpt_path}")

    # Evaluate on test set
    print('\n' + '='*50)
    print('Evaluating on Test Set')
    print('='*50)

    predictions, targets, metrics = evaluate_model(
        zidf_model,
        test_loader,
        device=device,
        diffusion_steps=args.diffusion_steps
    )

    # Print metrics
    print('\nTest Metrics:')
    for key, value in metrics.items():
        print(f'{key}: {value:.6f}')

    # Save metrics (ensure JSON-serializable)
    metrics_serializable = {k: float(v) for k, v in metrics.items()}
    with open(os.path.join(args.save_dir, 'test_metrics.json'), 'w') as f:
        json.dump(metrics_serializable, f, indent=4)

    # Plot predictions
    plot_predictions(
        predictions,
        targets,
        save_path=os.path.join(args.save_dir, 'predictions.png')
    )

    print(f'\nTraining completed! Results saved to {args.save_dir}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train ZIDF model')
    parser.add_argument('--config', type=str, default='', help='Path to YAML/JSON config')

    # Data parameters
    parser.add_argument('--use_synthetic', action='store_true', help='Use synthetic data')
    parser.add_argument('--n_samples', type=int, default=10000, help='Number of samples (synthetic)')
    parser.add_argument('--n_features', type=int, default=5, help='Number of features (synthetic)')
    parser.add_argument('--zero_ratio', type=float, default=0.7, help='Zero inflation ratio (synthetic)')
    parser.add_argument('--input_len', type=int, default=48, help='Input sequence length')
    parser.add_argument('--output_len', type=int, default=24, help='Output sequence length')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--num_workers', type=int, default=0, help='Dataloader workers (0 for compatibility)')

    # Model parameters
    parser.add_argument('--d_model', type=int, default=512, help='Model dimension')
    parser.add_argument('--n_heads', type=int, default=8, help='Number of attention heads')
    parser.add_argument('--e_layers', type=int, default=2, help='Number of encoder layers')
    parser.add_argument('--d_layers', type=int, default=1, help='Number of decoder layers')
    parser.add_argument('--dropout', type=float, default=0.05, help='Dropout rate')
    parser.add_argument('--alpha_noise', type=float, default=0.1, help='Noise injection parameter')
    parser.add_argument('--diffusion_hidden_dim', type=int, default=128, help='Diffusion hidden dimension')
    parser.add_argument('--diffusion_steps', type=int, default=1000, help='Number of diffusion steps')
    parser.add_argument('--beta_schedule', type=str, default='linear', choices=['linear','cosine'], help='Diffusion beta schedule')
    # Official NST integration (optional)
    parser.add_argument('--use_official_nst', action='store_true', help='Use official Non-stationary Transformer')
    parser.add_argument('--nst_module', type=str, default='', help='Python module path for official NST (e.g., repo.module)')
    parser.add_argument('--nst_class', type=str, default='', help='Class name for official NST')

    # Training parameters
    parser.add_argument('--pred_epochs', type=int, default=50, help='Epochs for predictive component')
    parser.add_argument('--diff_epochs', type=int, default=30, help='Epochs for diffusion model')
    parser.add_argument('--pred_lr', type=float, default=1e-4, help='Learning rate for predictive component')
    parser.add_argument('--diff_lr', type=float, default=1e-4, help='Learning rate for diffusion model')

    # Other parameters
    parser.add_argument('--save_dir', type=str, default='./checkpoints', help='Directory to save models')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')

    args = parser.parse_args()

    # Load config if provided
    if args.config:
        cfg = load_config(args.config)
        args = apply_config_to_args(args, cfg)

    # Set random seed
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

    main(args)
