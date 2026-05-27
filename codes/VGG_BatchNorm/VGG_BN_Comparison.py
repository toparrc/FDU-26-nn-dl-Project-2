import argparse
import csv
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch import nn

from data.loaders import get_cifar_loader
from models.vgg import VGG_A, VGG_A_BatchNorm
from training_framework import DEVICE, set_random_seeds, train


NUM_WORKERS = 4
BATCH_SIZE = 128
EPOCHS = 20
LR = 1e-3
SEED = 2020
MODEL_CONFIGS = [("vgg_a", VGG_A, "VGG-A without BN"), ("vgg_a_bn", VGG_A_BatchNorm, "VGG-A with BN")]
RESULTS_PATH = Path(__file__).resolve().parent / "results" / "bn_comparison"


def parse_args():
    parser = argparse.ArgumentParser(description="Train VGG-A with and without BatchNorm.")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--num-workers", type=int, default=NUM_WORKERS)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--results-dir", type=Path, default=RESULTS_PATH)
    return parser.parse_args()


def save_history(path, history):
    with path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=["epoch", "train_loss", "train_acc", "val_loss", "val_acc"])
        writer.writeheader()
        writer.writerows(history)


def plot_histories(histories, output_path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for _, label, history in histories:
        epochs = [row["epoch"] for row in history]
        axes[0].plot(epochs, [row["val_loss"] for row in history], label=label)
        axes[1].plot(epochs, [row["val_acc"] for row in history], label=label)
    axes[0].set_title("Validation loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[1].set_title("Validation accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    for ax in axes:
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
    criterion = nn.CrossEntropyLoss()
    histories = []
    summary_rows = []

    for model_name, model_cls, label in MODEL_CONFIGS:
        set_random_seeds(args.seed, DEVICE)
        model = model_cls()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        _, _, history, metrics = train(
            model, optimizer, criterion, train_loader, val_loader,
            test_loader=test_loader, epochs_n=args.epochs,
            best_model_path=str(models_path / f"{model_name}_best.pth")
        )
        save_history(args.results_dir / f"{model_name}_history.csv", history)
        histories.append((model_name, label, history))
        summary_rows.append({"model": model_name, **metrics})

    with (args.results_dir / "summary.csv").open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=["model", "best_epoch", "best_val_acc", "test_loss", "test_acc"])
        writer.writeheader()
        writer.writerows(summary_rows)
    plot_histories(histories, args.results_dir / "validation_comparison.png")


if __name__ == "__main__":
    main()
