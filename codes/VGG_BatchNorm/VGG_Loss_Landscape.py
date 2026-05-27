import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn

from data.loaders import get_cifar_loader
from models.vgg import VGG_A, VGG_A_BatchNorm
from training_framework import DEVICE, build_min_max_curves, set_random_seeds, train


# ## Constants (parameters) initialization
NUM_WORKERS = 4
BATCH_SIZE = 128
EPOCHS = 20
SEED = 2020
# Select a list of learning rates to represent different step sizes.
LEARNING_RATES = [1e-3, 2e-3, 1e-4, 5e-4]
MODEL_CONFIGS = [("vgg_a", VGG_A), ("vgg_a_bn", VGG_A_BatchNorm)]
RESULTS_PATH = Path(__file__).resolve().parent / "results" / "bn_loss_landscape"


def parse_args():
    parser = argparse.ArgumentParser(description="Plot VGG-A loss landscape with and without BatchNorm.")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--num-workers", type=int, default=NUM_WORKERS)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--learning-rates", type=float, nargs="+", default=LEARNING_RATES)
    parser.add_argument("--results-dir", type=Path, default=RESULTS_PATH)
    return parser.parse_args()


def print_loader_sample(train_loader):
    # Initialize your data loader and make sure it works as expected by observing one sample batch.
    for x, y in train_loader:
        print(f"Sample batch images: {x.shape}, labels: {y.shape}")
        print(f"Image value range after normalization: [{x.min():.3f}, {x.max():.3f}]")
        print(f"First labels: {y[:8].tolist()}")
        break


def plot_loss_landscape(min_curve, max_curve, output_path):
    # Plot the final loss landscape. The filled area shows loss variation across step sizes.
    fig, ax = plt.subplots(figsize=(10, 5))
    labels = {"vgg_a": "VGG-A without BN", "vgg_a_bn": "VGG-A with BN"}
    colors = {"vgg_a": "tab:orange", "vgg_a_bn": "tab:blue"}
    for model_name in labels:
        steps = np.arange(len(min_curve[model_name]))
        ax.plot(steps, min_curve[model_name], color=colors[model_name], linewidth=1.2)
        ax.plot(steps, max_curve[model_name], color=colors[model_name], linewidth=1.2)
        # Fill the area between max_curve and min_curve for this model.
        ax.fill_between(steps, min_curve[model_name], max_curve[model_name], color=colors[model_name], alpha=0.25, label=labels[model_name])
    ax.set_title("Loss landscape comparison")
    ax.set_xlabel("Training step")
    ax.set_ylabel("Training loss")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def main():
    args = parse_args()
    models_path = args.results_dir / "models"
    os.makedirs(args.results_dir, exist_ok=True)
    os.makedirs(models_path, exist_ok=True)
    print(f"Using device: {DEVICE}")
    train_loader, val_loader = get_cifar_loader(batch_size=args.batch_size, num_workers=args.num_workers, train=True)
    test_loader = get_cifar_loader(batch_size=args.batch_size, num_workers=args.num_workers, train=False)
    print_loader_sample(train_loader)
    criterion = nn.CrossEntropyLoss()
    experiment_results = {}

    # Train VGG-A and VGG-A with BN under each learning rate, saving every batch loss.
    for model_name, model_cls in MODEL_CONFIGS:
        experiment_results[model_name] = []
        for lr in args.learning_rates:
            set_random_seeds(args.seed, DEVICE)
            model = model_cls()
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            run_name = f"{model_name}_lr_{lr:g}".replace(".", "p")
            losses, grads, history, test_metrics = train(
                model, optimizer, criterion, train_loader, val_loader,
                test_loader=test_loader, epochs_n=args.epochs,
                best_model_path=str(models_path / f"{run_name}_best.pth")
            )
            experiment_results[model_name].append({"lr": lr, "losses": losses, "grads": grads, "history": history, "test_metrics": test_metrics})
            np.savetxt(args.results_dir / f"{run_name}_loss.txt", np.array(losses), fmt="%.8f")
            np.savetxt(args.results_dir / f"{run_name}_grads.txt", np.array(grads), fmt="%.8f")

    # Maintain two lists: max_curve and min_curve. At the same training step,
    # select the maximum and minimum loss values among all learning-rate runs.
    min_curve = {}
    max_curve = {}
    
    for model_name, runs in experiment_results.items():
        min_curve[model_name], max_curve[model_name] = build_min_max_curves([run["losses"] for run in runs])
        np.savetxt(args.results_dir / f"{model_name}_min_curve.txt", min_curve[model_name], fmt="%.8f")
        np.savetxt(args.results_dir / f"{model_name}_max_curve.txt", max_curve[model_name], fmt="%.8f")

    plot_loss_landscape(min_curve, max_curve, args.results_dir / "loss_landscape_comparison.png")


if __name__ == "__main__":
    main()
