"""CLI entrypoint for the P1 side-specific model comparison."""

import argparse
import json

import pandas as pd

from src.core.config import get_settings
from src.execution.kernel import ExecutionKernel, costs_from_settings
from src.models.side_comparison import SideComparisonConfig, run_side_model_comparison


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare side classifiers with net-return regressors")
    parser.add_argument("dataset")
    parser.add_argument("metadata")
    parser.add_argument("--train-size", type=int, default=3500)
    parser.add_argument("--test-size", type=int, default=300)
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--classifier-threshold", type=float, default=0.55)
    parser.add_argument("--quantile", type=float, default=0.25)
    parser.add_argument("--output")
    args = parser.parse_args()

    with open(args.metadata, encoding="utf-8") as metadata_file:
        features = json.load(metadata_file)["features"]
    result = run_side_model_comparison(
        pd.read_parquet(args.dataset), features,
        SideComparisonConfig(args.train_size, args.test_size, args.horizon, args.classifier_threshold, args.quantile),
        ExecutionKernel(costs_from_settings(get_settings())),
    )
    rendered = json.dumps(result, ensure_ascii=False, indent=2)
    print(rendered)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as output_file:
            output_file.write(rendered + "\n")


if __name__ == "__main__":
    main()
