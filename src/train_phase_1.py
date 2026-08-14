from argparse import ArgumentParser
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import optim
from torch.utils.data import TensorDataset, DataLoader
from scipy.io import loadmat

from data_loading import load_all_takes_except, load_subject
from model import Encoder
from loss import CrossCorrLoss
from train import train_model

def main(
    output_dir: Path,
    data_dir: Path,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    optim_name: str,
    print_freq: int,
):
    # Setup logging
    output_dir.mkdir(parents=True, exist_ok=True)
    log_file = Path(output_dir / 'logs.txt')
    if log_file.exists():
        log_file.unlink()
    log_file.touch()

    # Setup device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Using', device)
    with open(log_file, 'a') as writer:
        writer.write(f'Using {device}\n')
    
    # Load data
    test_subject = 10
    print('Training stage 1 without subject', test_subject)
    with open(log_file, 'a') as writer:
        writer.write(f'Training stage 1 without subject {test_subject}\n')
    train_val_a, train_val_b, _ = load_all_takes_except(data_dir, test_subject, wrapper_fn=torch.tensor)
    test_a, test_b, _ = load_subject(data_dir, test_subject, wrapper_fn=torch.tensor)

    # Split training, validation, and testing rows
    all_indices = np.random.permutation(train_val_a.shape[0])
    train_samples_a = train_val_a[all_indices[: int(all_indices.shape[0] * 0.9)]]
    train_samples_b = train_val_b[all_indices[: int(all_indices.shape[0] * 0.9)]]
    val_samples_a = train_val_a[all_indices[int(all_indices.shape[0] * 0.9) :]]
    val_samples_b = train_val_b[all_indices[int(all_indices.shape[0] * 0.9) :]]

    # Create dataloaders
    train_ds = TensorDataset(train_samples_a, train_samples_b)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_ds = TensorDataset(val_samples_a, val_samples_b)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_ds = TensorDataset(test_a, test_b)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    # Create model
    model = Encoder()
    model.to(device)

    # Create loss criterion
    criterion = CrossCorrLoss()

    # Create optimizer
    if optim_name == 'SGD':
        optimizer = optim.SGD(
            model.parameters(),
            lr=lr / batch_size,
            momentum=0.9,
            weight_decay=weight_decay,
            nesterov=False
        )
    elif optim_name == 'AdamW':
        optimizer = optim.AdamW(
            model.parameters(),
            lr=lr / batch_size,
            weight_decay=weight_decay
        )

    # Train model
    model, train_loss_history, val_loss_history, test_loss = train_model(
        model,
        optimizer,
        criterion,
        train_loader,
        val_loader,
        test_loader,
        epochs,
        device,
        output_dir,
        log_file,
        print_freq
    )

    # Create loss plot
    fig, ax = plt.subplots(1, 2, figsize=(14, 6))
    x_vals, y_vals = zip(*train_loss_history)
    ax[0].plot(x_vals, y_vals, label='Train loss')
    ax[1].loglog(x_vals, y_vals, label='Train loss')
    x_vals, y_vals = zip(*val_loss_history)
    ax[0].plot(x_vals, y_vals, label='Val loss')
    ax[1].loglog(x_vals, y_vals, label='Val loss')
    ax[0].plot(
        epochs * len(train_loader),
        test_loss,
        marker='o',
        label='Test loss'
    )
    ax[1].loglog(
        epochs * len(train_loader),
        test_loss,
        marker='o',
        label='Test loss'
    )
    ax[0].set_title('Loss vs step')
    ax[1].set_title('Loss vs step')
    ax[0].set_xlabel('Steps')
    ax[1].set_xlabel('Steps')
    ax[0].set_ylabel('Loss')
    ax[1].set_ylabel('Loss')
    ax[0].legend()
    ax[1].legend()
    fig.tight_layout()
    fig.savefig(output_dir / 'loss_plot.png')

    # Save model
    torch.save(model, output_dir / 'model.pt')
    torch.save(model.state_dict(), output_dir / 'model_state_dict.pt')

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument(
        '--output-dir',
        type=Path,
        default='outputs_phase_1',
        help='Output path.'
    )
    parser.add_argument(
        '--data-dir',
        type=Path,
        default='AlignmentTransferCode/TMMEvaluation/Patches',
        help='Path to data.'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=15,
        help='Number of training epochs.'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=128,
        help='Batch size.'
    )
    parser.add_argument(
        '--lr',
        type=float,
        default=0.01,
        help='Learning rate. This is scaled by batch size.'
    )
    parser.add_argument(
        '--weight-decay',
        type=float,
        default=0.0005,
        help='Weight decay.'
    )
    parser.add_argument(
        '--optimizer',
        choices=['SGD', 'Adam'],
        default='SGD',
        help='Optimizer.'
    )
    parser.add_argument(
        '--print-freq',
        type=int,
        default=10,
        help='How often (in batches) to print training loss.'
    )
    args = parser.parse_args()

    main(
        args.output_dir,
        args.data_dir,
        args.epochs,
        args.batch_size,
        args.lr,
        args.weight_decay,
        args.optimizer,
        args.print_freq
    )
