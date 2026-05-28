import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# ## Constants (parameters) initialization
NUM_WORKERS = 4
BATCH_SIZE = 128
EPOCHS = 50
SEED = 2020
SMOOTH_WINDOW = 50
PLOT_STRIDE = 10
# Select a list of learning rates to represent different step sizes.
LEARNING_RATES = [1e-3, 2e-3, 1e-4, 5e-4]
MODEL_NAMES = ["vgg_a", "vgg_a_bn"]
RESULTS_PATH = Path(__file__).resolve().parent / "results" / "bn_loss_landscape"


def parse_args():
    parser = argparse.ArgumentParser(description="Plot VGG-A loss landscape with and without BatchNorm.")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--num-workers", type=int, default=NUM_WORKERS)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--learning-rates", type=float, nargs="+", default=LEARNING_RATES)
    parser.add_argument("--results-dir", type=Path, default=RESULTS_PATH)
    parser.add_argument("--plot-only", action="store_true", help="Replot figures from saved curve files without training.")
    parser.add_argument("--smooth-window", type=int, default=SMOOTH_WINDOW, help="Moving-average window for plotted curves. Use 1 to disable smoothing.")
    parser.add_argument("--plot-stride", type=int, default=PLOT_STRIDE, help="Plot every Nth point after smoothing. Use 1 to disable downsampling.")
    parser.add_argument("--output-suffix", type=str, default="", help="Suffix appended to output figure names before the extension.")
    return parser.parse_args()


def print_loader_sample(train_loader):
    # Initialize your data loader and make sure it works as expected by observing one sample batch.
    for x, y in train_loader:
        print(f"Sample batch images: {x.shape}, labels: {y.shape}")
        print(f"Image value range after normalization: [{x.min():.3f}, {x.max():.3f}]")
        print(f"First labels: {y[:8].tolist()}")
        break


def moving_average(values, window):
    values = np.asarray(values, dtype=float)
    if window <= 1:
        return values
    left = window // 2
    right = window - 1 - left
    padded = np.pad(values, (left, right), mode="edge")
    kernel = np.ones(window) / window
    return np.convolve(padded, kernel, mode="valid")


def plot_min_max_curves(min_curve, max_curve, output_path, title, ylabel, smooth_window, plot_stride):
    # Plot min/max envelopes. The filled area shows variation across step sizes.
    fig, ax = plt.subplots(figsize=(10, 5))
    labels = {"vgg_a": "Standard VGG", "vgg_a_bn": "Standard VGG + BatchNorm"}
    colors = {"vgg_a": "#55a868", "vgg_a_bn": "#c44e52"}
    for model_name in labels:
        smooth_min = moving_average(min_curve[model_name], smooth_window)
        smooth_max = moving_average(max_curve[model_name], smooth_window)
        stride = max(1, plot_stride)
        steps = np.arange(len(smooth_min))[::stride]
        smooth_min = smooth_min[::stride]
        smooth_max = smooth_max[::stride]
        # Fill the area between max_curve and min_curve for this model.
        ax.fill_between(
            steps,
            smooth_min,
            smooth_max,
            facecolor=colors[model_name],
            alpha=0.35,
            edgecolor=colors[model_name],
            linewidth=0.4,
            label=labels[model_name],
            zorder=1,
        )
    ax.set_title(title)
    ax.set_xlabel("Training step")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def load_curve_pair(results_dir, min_suffix, max_suffix):
    min_curve = {}
    max_curve = {}
    for model_name in MODEL_NAMES:
        min_curve[model_name] = np.loadtxt(results_dir / f"{model_name}_{min_suffix}.txt")
        max_curve[model_name] = np.loadtxt(results_dir / f"{model_name}_{max_suffix}.txt")
    return min_curve, max_curve


def figure_path(results_dir, stem, suffix):
    return results_dir / f"{stem}{suffix}.png"


def plot_saved_curves(results_dir, smooth_window, plot_stride, output_suffix):
    min_curve, max_curve = load_curve_pair(results_dir, "min_curve", "max_curve")
    grad_min_curve, grad_max_curve = load_curve_pair(results_dir, "grad_norm_min_curve", "grad_norm_max_curve")
    grad_change_paths = [
        results_dir / f"{model_name}_grad_change_{kind}_curve.txt"
        for model_name in MODEL_NAMES
        for kind in ("min", "max")
    ]
    grad_smoothness_paths = [
        results_dir / f"{model_name}_grad_smoothness_{kind}_curve.txt"
        for model_name in MODEL_NAMES
        for kind in ("min", "max")
    ]
    plot_min_max_curves(
        min_curve,
        max_curve,
        figure_path(results_dir, "loss_landscape_comparison", output_suffix),
        "Loss landscape comparison",
        "Training loss",
        smooth_window,
        plot_stride,
    )
    if all(path.exists() for path in grad_change_paths):
        grad_change_min_curve, grad_change_max_curve = load_curve_pair(results_dir, "grad_change_min_curve", "grad_change_max_curve")
        plot_min_max_curves(
            grad_change_min_curve,
            grad_change_max_curve,
            figure_path(results_dir, "grad_change_landscape_comparison", output_suffix),
            "Gradient change comparison",
            "Full-gradient change norm",
            smooth_window,
            plot_stride,
        )
    else:
        print("Skipping gradient-change plot: saved full-gradient change curves were not found.")
    if all(path.exists() for path in grad_smoothness_paths):
        grad_smoothness_min_curve, grad_smoothness_max_curve = load_curve_pair(results_dir, "grad_smoothness_min_curve", "grad_smoothness_max_curve")
        plot_min_max_curves(
            grad_smoothness_min_curve,
            grad_smoothness_max_curve,
            figure_path(results_dir, "grad_smoothness_landscape_comparison", output_suffix),
            "Gradient smoothness comparison",
            "||delta gradient|| / ||delta parameters||",
            smooth_window,
            plot_stride,
        )
    else:
        print("Skipping gradient-smoothness plot: saved full-gradient smoothness curves were not found.")
    plot_min_max_curves(
        grad_min_curve,
        grad_max_curve,
        figure_path(results_dir, "grad_norm_landscape_comparison", output_suffix),
        "Full-gradient norm comparison",
        "Full-gradient norm",
        smooth_window,
        plot_stride,
    )


def main():
    args = parse_args()
    if args.plot_only:
        plot_saved_curves(args.results_dir, args.smooth_window, args.plot_stride, args.output_suffix)
        return

    import torch
    from torch import nn

    from data.loaders import get_cifar_loader
    from models.vgg import VGG_A, VGG_A_BatchNorm
    from training_framework import DEVICE, build_min_max_curves, set_random_seeds, train

    model_configs = [("vgg_a", VGG_A), ("vgg_a_bn", VGG_A_BatchNorm)]
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
    for model_name, model_cls in model_configs:
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
            experiment_results[model_name].append({
                "lr": lr,
                "losses": losses,
                "grads": grads,
                "grad_changes": test_metrics["grad_changes"],
                "grad_smoothness": test_metrics["grad_smoothness"],
                "history": history,
                "test_metrics": test_metrics,
            })
            np.savetxt(args.results_dir / f"{run_name}_loss.txt", np.array(losses), fmt="%.8f")
            np.savetxt(args.results_dir / f"{run_name}_grads.txt", np.array(grads), fmt="%.8f")
            np.savetxt(args.results_dir / f"{run_name}_grad_changes.txt", np.array(test_metrics["grad_changes"]), fmt="%.8f")
            np.savetxt(args.results_dir / f"{run_name}_grad_smoothness.txt", np.array(test_metrics["grad_smoothness"]), fmt="%.8f")

    # Maintain two lists: max_curve and min_curve. At the same training step,
    # select the maximum and minimum loss values among all learning-rate runs.
    min_curve = {}
    max_curve = {}
    
    for model_name, runs in experiment_results.items():
        min_curve[model_name], max_curve[model_name] = build_min_max_curves([run["losses"] for run in runs])
        np.savetxt(args.results_dir / f"{model_name}_min_curve.txt", min_curve[model_name], fmt="%.8f")
        np.savetxt(args.results_dir / f"{model_name}_max_curve.txt", max_curve[model_name], fmt="%.8f")

    grad_min_curve = {}
    grad_max_curve = {}
    for model_name, runs in experiment_results.items():
        grad_min_curve[model_name], grad_max_curve[model_name] = build_min_max_curves([run["grads"] for run in runs])
        np.savetxt(args.results_dir / f"{model_name}_grad_norm_min_curve.txt", grad_min_curve[model_name], fmt="%.8f")
        np.savetxt(args.results_dir / f"{model_name}_grad_norm_max_curve.txt", grad_max_curve[model_name], fmt="%.8f")

    grad_change_min_curve = {}
    grad_change_max_curve = {}
    for model_name, runs in experiment_results.items():
        grad_change_min_curve[model_name], grad_change_max_curve[model_name] = build_min_max_curves([run["grad_changes"] for run in runs])
        np.savetxt(args.results_dir / f"{model_name}_grad_change_min_curve.txt", grad_change_min_curve[model_name], fmt="%.8f")
        np.savetxt(args.results_dir / f"{model_name}_grad_change_max_curve.txt", grad_change_max_curve[model_name], fmt="%.8f")

    grad_smoothness_min_curve = {}
    grad_smoothness_max_curve = {}
    for model_name, runs in experiment_results.items():
        grad_smoothness_min_curve[model_name], grad_smoothness_max_curve[model_name] = build_min_max_curves([run["grad_smoothness"] for run in runs])
        np.savetxt(args.results_dir / f"{model_name}_grad_smoothness_min_curve.txt", grad_smoothness_min_curve[model_name], fmt="%.8f")
        np.savetxt(args.results_dir / f"{model_name}_grad_smoothness_max_curve.txt", grad_smoothness_max_curve[model_name], fmt="%.8f")

    plot_saved_curves(args.results_dir, args.smooth_window, args.plot_stride, args.output_suffix)


if __name__ == "__main__":
    main()
