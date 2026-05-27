import argparse
import csv
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.optim import AdamW, SGD
from torch.optim.lr_scheduler import CosineAnnealingLR

from model import CIFAR10CNN, CIFAR10CNN3, CIFAR10CNN4, CIFAR10ResidualCNN, count_parameters
from data import get_cifar10_loaders
from plots import plot_confusion_matrix, plot_validation_curves


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 50
BATCH_SIZE = 128
LR = 1e-3
WEIGHT_DECAY = 5e-4
MOMENTUM = 0.9
PATIENCE = 10
SEED = 42
CHECKPOINT_DIR = Path(__file__).resolve().parent / "checkpoints"
RESULTS_DIR = Path(__file__).resolve().parent / "results"
CIFAR10_CLASSES = (
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
)

MODEL_REGISTRY = {
    "cnn": CIFAR10CNN,
    "cnn4": CIFAR10CNN4,
    "cnn3": CIFAR10CNN3,
    "residual": CIFAR10ResidualCNN,
}


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)
        running_loss += loss.item() * batch_size
        correct += (outputs.argmax(dim=1) == labels).sum().item()
        total += batch_size

    return running_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        outputs = model(images)
        loss = criterion(outputs, labels)

        batch_size = labels.size(0)
        running_loss += loss.item() * batch_size
        correct += (outputs.argmax(dim=1) == labels).sum().item()
        total += batch_size

    return running_loss / total, correct / total


@torch.no_grad()
def compute_confusion_matrix(model, loader, num_classes, device):
    model.eval()
    matrix = torch.zeros(num_classes, num_classes, dtype=torch.int64)

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        outputs = model(images)
        predictions = outputs.argmax(dim=1).cpu()

        for target, prediction in zip(labels.view(-1), predictions.view(-1)):
            matrix[target.long(), prediction.long()] += 1

    return matrix.numpy()


def save_checkpoint(path, model, optimizer, scheduler, epoch, best_val_acc, args):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "best_val_acc": best_val_acc,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "args": vars(args),
        },
        path,
    )


def save_history_csv(path, history):
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "epoch",
        "train_loss",
        "train_acc",
        "val_loss",
        "val_acc",
        "best_val_acc",
        "epoch_time_sec",
        "elapsed_time_sec",
    ]
    with path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(history)


def save_summary_json(path, results):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as json_file:
        json.dump(results, json_file, indent=2)


def parse_args():
    parser = argparse.ArgumentParser(description="Train CIFAR-10 CNN.")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY)
    parser.add_argument("--optimizer", choices=["adamw", "sgd", "sgd_momentum"], default="adamw")
    parser.add_argument("--momentum", type=float, default=MOMENTUM)
    parser.add_argument("--patience", type=int, default=PATIENCE)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--val-size", type=int, default=5000)
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--checkpoint-dir", type=Path, default=CHECKPOINT_DIR)
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--model", choices=["cnn", "cnn4", "cnn3", "residual", "both"], default="both")
    parser.add_argument(
        "--activation",
        choices=["relu", "leaky_relu", "elu", "gelu", "silu"],
        default="relu",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Optional output subdirectory name for comparison experiments.",
    )
    return parser.parse_args()


def create_model(model_name, activation):
    model_cls = MODEL_REGISTRY[model_name]
    if model_name == "cnn":
        return model_cls()
    return model_cls(activation=activation)


def create_optimizer(model, args):
    if args.optimizer == "adamw":
        return AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    if args.optimizer == "sgd":
        return SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    if args.optimizer == "sgd_momentum":
        return SGD(model.parameters(),lr=args.lr,momentum=args.momentum,weight_decay=args.weight_decay,)
    raise ValueError(f"Unknown optimizer: {args.optimizer}")


def get_output_name(model_name, args):
    if args.run_name is None:
        return model_name
    if args.model == "both":
        return f"{args.run_name}_{model_name}"
    return args.run_name


def train_model(model_name, args, train_loader, val_loader, test_loader):
    set_seed(args.seed)
    model = create_model(model_name, args.activation).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = create_optimizer(model, args)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val_acc = -1.0
    best_epoch = 0
    epochs_without_improvement = 0
    output_name = get_output_name(model_name, args)
    model_checkpoint_dir = args.checkpoint_dir / output_name
    model_results_dir = args.results_dir / output_name
    best_path = model_checkpoint_dir / "best.pth"
    last_path = model_checkpoint_dir / "last.pth"
    history = []
    train_start_time = time.perf_counter()

    print(f"Using device: {DEVICE}")
    print(f"Training model: {model_name}")
    print(f"Activation: {args.activation}")
    print(f"Optimizer: {args.optimizer}")
    print(f"Learning rate: {args.lr:g}")
    print(f"Weight decay: {args.weight_decay:g}")
    if args.optimizer == "sgd_momentum":
        print(f"Momentum: {args.momentum:g}")
    print(f"Trainable parameters: {count_parameters(model):,}")
    for epoch in range(1, args.epochs + 1):
        epoch_start_time = time.perf_counter()
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE)
        val_loss, val_acc = evaluate(model, val_loader, criterion, DEVICE)
        scheduler.step()
        epoch_time = time.perf_counter() - epoch_start_time
        elapsed_time = time.perf_counter() - train_start_time

        improved = val_acc > best_val_acc
        if improved:
            best_val_acc = val_acc
            best_epoch = epoch
            epochs_without_improvement = 0
            save_checkpoint(best_path, model, optimizer, scheduler, epoch, best_val_acc, args)
        else:
            epochs_without_improvement += 1

        save_checkpoint(last_path, model, optimizer, scheduler, epoch, best_val_acc, args)
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "best_val_acc": best_val_acc,
                "epoch_time_sec": epoch_time,
                "elapsed_time_sec": elapsed_time,
            }
        )
        save_history_csv(model_results_dir / "history.csv", history)

        print(
            f"Epoch [{epoch:03d}/{args.epochs:03d}] "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} "
            f"best_val_acc={best_val_acc:.4f} "
            f"epoch_time={epoch_time:.1f}s elapsed={elapsed_time:.1f}s"
        )

        if args.patience > 0 and epochs_without_improvement >= args.patience:
            print(
                f"Early stopping: validation accuracy did not improve for "
                f"{args.patience} epochs."
            )
            break

    total_train_time = time.perf_counter() - train_start_time
    checkpoint = torch.load(best_path, map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_start_time = time.perf_counter()
    test_loss, test_acc = evaluate(model, test_loader, criterion, DEVICE)
    test_time = time.perf_counter() - test_start_time
    confusion_matrix = compute_confusion_matrix(model, test_loader, len(CIFAR10_CLASSES), DEVICE)

    plot_validation_curves(model_results_dir / "validation_curves.png", history, model_name)
    plot_confusion_matrix(
        model_results_dir / "test_confusion_matrix.png",
        confusion_matrix,
        CIFAR10_CLASSES,
        model_name,
    )

    summary = {
        "model": model_name,
        "output_name": output_name,
        "activation": args.activation,
        "optimizer": args.optimizer,
        "learning_rate": args.lr,
        "weight_decay": args.weight_decay,
        "momentum": args.momentum if args.optimizer == "sgd_momentum" else 0.0,
        "trainable_parameters": count_parameters(model),
        "epochs_completed": history[-1]["epoch"] if history else 0,
        "best_epoch": best_epoch,
        "best_val_acc": best_val_acc,
        "test_loss": test_loss,
        "test_acc": test_acc,
        "total_train_time_sec": total_train_time,
        "avg_epoch_time_sec": total_train_time / len(history) if history else 0.0,
        "test_time_sec": test_time,
        "best_checkpoint": str(best_path),
        "history_csv": str(model_results_dir / "history.csv"),
        "validation_curves": str(model_results_dir / "validation_curves.png"),
        "test_confusion_matrix": str(model_results_dir / "test_confusion_matrix.png"),
    }
    save_summary_json(model_results_dir / "summary.json", summary)

    print(f"Best checkpoint: {best_path}")
    print(f"Test loss={test_loss:.4f} test_acc={test_acc:.4f} test_time={test_time:.1f}s")
    print(f"Training time={total_train_time:.1f}s avg_epoch_time={summary['avg_epoch_time_sec']:.1f}s")
    print(f"Results saved to: {model_results_dir}")
    return summary


def main():
    args = parse_args()
    set_seed(args.seed)

    train_loader, val_loader, test_loader = get_cifar10_loaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        val_size=args.val_size,
        num_workers=args.num_workers,
        seed=args.seed,
    )

    model_names = list(MODEL_REGISTRY) if args.model == "both" else [args.model]
    results = {}
    for model_name in model_names:
        summary = train_model(
            model_name, args, train_loader, val_loader, test_loader
        )
        results[model_name] = summary

    save_summary_json(args.results_dir / "summary.json", results)
    if len(results) > 1:
        print("Summary:")
        for model_name, summary in results.items():
            print(
                f"{model_name}: best_val_acc={summary['best_val_acc']:.4f} "
                f"test_acc={summary['test_acc']:.4f} "
                f"train_time={summary['total_train_time_sec']:.1f}s"
            )


if __name__ == "__main__":
    main()
