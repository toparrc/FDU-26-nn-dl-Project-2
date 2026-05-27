import argparse
from pathlib import Path

from plots import load_history_csv, plot_history_comparison


def parse_args():
    parser = argparse.ArgumentParser(description="Plot CIFAR-10 history comparisons.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--title", type=str, required=True)
    parser.add_argument(
        "--history",
        action="append",
        nargs=2,
        metavar=("LABEL", "CSV_PATH"),
        required=True,
        help="Add one labeled history.csv file.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    history_items = [
        (label, load_history_csv(Path(csv_path))) for label, csv_path in args.history
    ]
    plot_history_comparison(args.output, history_items, args.title)
    print(f"Saved comparison plot to {args.output}")


if __name__ == "__main__":
    main()
