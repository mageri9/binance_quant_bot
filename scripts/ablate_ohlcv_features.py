"""Run the P2 OHLCV feature ablation on an existing dataset."""

import argparse
import json

import pandas as pd

from src.models.ablation import run_ohlcv_ablation


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare OHLCV feature families against technical indicators.")
    parser.add_argument("dataset", help="Path to dataset parquet file")
    parser.add_argument("--target", default="target_triple", choices=["target_binary", "target_triple"])
    parser.add_argument("--train-size", type=int, default=1000)
    parser.add_argument("--test-size", type=int, default=200)
    parser.add_argument("--output", default="ohlcv_ablation.json")
    args = parser.parse_args()

    report = run_ohlcv_ablation(
        pd.read_parquet(args.dataset), args.target, args.train_size, args.test_size,
    )
    with open(args.output, "w", encoding="utf-8") as output_file:
        json.dump(report, output_file, indent=2)

    for result in report["results"]:
        delta = result["f1_delta_vs_baseline"]
        delta_text = "baseline" if delta is None else f"{delta:+.4f}"
        print(f"{result['name']}: F1={result['f1']:.4f}, delta={delta_text}")


if __name__ == "__main__":
    main()
