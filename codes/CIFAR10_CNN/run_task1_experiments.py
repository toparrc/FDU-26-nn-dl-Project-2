import argparse
from copy import copy
from pathlib import Path
from types import SimpleNamespace

from data import get_cifar10_loaders
from plots import plot_history_comparison
from train import (
    BATCH_SIZE,
    CHECKPOINT_DIR,
    EPOCHS,
    LR,
    MOMENTUM,
    PATIENCE,
    RESULTS_DIR,
    SEED,
    WEIGHT_DECAY,
    save_summary_json,
    train_model,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Run Task 1 comparison experiments.")
    parser.add_argument(
        "--group",
        choices=["regularization", "activation", "depth", "optimizer", "all"],
        default="all",
    )
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--patience", type=int, default=PATIENCE)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--val-size", type=int, default=5000)
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--checkpoint-dir", type=Path, default=CHECKPOINT_DIR / "task1_comparisons")
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR / "task1_comparisons")
    return parser.parse_args()


def build_train_args(base_args, config):
    args = SimpleNamespace(**vars(base_args))
    args.model = config["model"]
    args.activation = config.get("activation", "relu")
    args.optimizer = config.get("optimizer", "adamw")
    args.momentum = config.get("momentum", MOMENTUM)
    args.weight_decay = config.get("weight_decay", WEIGHT_DECAY)
    args.run_name = config["run_name"]
    return args


def get_configs(group):
    configs = {
        "regularization": [
            {
                "label": "wd=0",
                "model": "cnn4",
                "activation": "relu",
                "weight_decay": 0.0,
                "run_name": "regularization/wd_0",
            },
            {
                "label": "wd=1e-4",
                "model": "cnn4",
                "activation": "relu",
                "weight_decay": 1e-4,
                "run_name": "regularization/wd_1e-4",
            },
            {
                "label": "wd=5e-4",
                "model": "cnn4",
                "activation": "relu",
                "weight_decay": 5e-4,
                "run_name": "regularization/wd_5e-4",
            },
            {
                "label": "wd=1e-3",
                "model": "cnn4",
                "activation": "relu",
                "weight_decay": 1e-3,
                "run_name": "regularization/wd_1e-3",
            },
        ],
        "activation": [
            {
                "label": "ReLU",
                "model": "cnn4",
                "activation": "relu",
                "weight_decay": WEIGHT_DECAY,
                "run_name": "activation/relu",
            },
            {
                "label": "LeakyReLU",
                "model": "cnn4",
                "activation": "leaky_relu",
                "weight_decay": WEIGHT_DECAY,
                "run_name": "activation/leaky_relu",
            },
            {
                "label": "ELU",
                "model": "cnn4",
                "activation": "elu",
                "weight_decay": WEIGHT_DECAY,
                "run_name": "activation/elu",
            },
        ],
        "depth": [
            {
                "label": "4 conv blocks",
                "model": "cnn4",
                "activation": "relu",
                "weight_decay": WEIGHT_DECAY,
                "run_name": "depth/cnn4",
            },
            {
                "label": "3 conv blocks",
                "model": "cnn3",
                "activation": "relu",
                "weight_decay": WEIGHT_DECAY,
                "run_name": "depth/cnn3",
            },
        ],
        "optimizer": [
            {
                "label": "AdamW",
                "model": "cnn4",
                "activation": "relu",
                "optimizer": "adamw",
                "weight_decay": WEIGHT_DECAY,
                "run_name": "optimizer/adamw",
            },
            {
                "label": "SGD",
                "model": "cnn4",
                "activation": "relu",
                "optimizer": "sgd",
                "weight_decay": WEIGHT_DECAY,
                "run_name": "optimizer/sgd",
            },
            {
                "label": "SGD+momentum",
                "model": "cnn4",
                "activation": "relu",
                "optimizer": "sgd_momentum",
                "momentum": MOMENTUM,
                "weight_decay": WEIGHT_DECAY,
                "run_name": "optimizer/sgd_momentum",
            },
        ],
    }
    if group == "all":
        return copy(configs)
    return {group: configs[group]}


def main():
    args = parse_args()
    train_loader, val_loader, test_loader = get_cifar10_loaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        val_size=args.val_size,
        num_workers=args.num_workers,
        seed=args.seed,
    )

    all_results = {}
    for group_name, configs in get_configs(args.group).items():
        history_items = []
        group_results = {}
        for config in configs:
            train_args = build_train_args(args, config)
            summary = train_model(
                config["model"], train_args, train_loader, val_loader, test_loader
            )
            summary["comparison_label"] = config["label"]
            group_results[config["label"]] = summary

            history_csv = Path(summary["history_csv"])
            history_items.append((config["label"], _load_history_for_plot(history_csv)))

        plot_path = args.results_dir / group_name / "validation_comparison.png"
        plot_history_comparison(plot_path, history_items, group_name)
        group_results["comparison_plot"] = str(plot_path)
        all_results[group_name] = group_results

    save_summary_json(args.results_dir / "summary.json", all_results)


def _load_history_for_plot(path):
    from plots import load_history_csv

    return load_history_csv(path)


if __name__ == "__main__":
    main()
