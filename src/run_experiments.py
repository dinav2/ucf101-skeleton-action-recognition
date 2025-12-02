#!/usr/bin/python3
"""
Grid runner for PoseTransformer.
"""

import argparse
import json
from pathlib import Path
from typing import List

from config import (
    DIM_FEEDFORWARD,
    D_MODEL,
    DROPOUT,
    EXPERIMENTS_DIR,
    N_HEADS,
    NUM_EPOCHS_TRANSFORMER,
    NUM_LAYERS,
    PKL_PATH,
)
from train import parse_allowed_labels, train_pipeline, build_parser


def default_transformer_grid(prefix: str) -> List[dict]:
    """Set of configs to try"""
    return [
        {
            "name": f"{prefix}_d128_h2_l2_do01",
            "d_model": 128,
            "n_heads": 2,
            "num_layers": 2,
            "dim_feedforward": 256,
            "dropout": 0.1,
        },
        {
            "name": f"{prefix}_d192_h4_l3_do02",
            "d_model": 192,
            "n_heads": 4,
            "num_layers": 3,
            "dim_feedforward": 512,
            "dropout": 0.2,
        },
        {
            "name": f"{prefix}_d256_h4_l4_do03",
            "d_model": 256,
            "n_heads": 4,
            "num_layers": 4,
            "dim_feedforward": 768,
            "dropout": 0.3,
        },
    ]


def build_grid_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a grid of PoseTransformer experiments"
    )
    parser.add_argument(
        "--prefix", type=str, default="transformer", help="Prefix for experiment names"
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default=None,
        help="Optional global name suffix/prefix",
    )
    parser.add_argument("--pkl-path", type=str, default=str(PKL_PATH))
    parser.add_argument("--train-split", type=str, default="train1")
    parser.add_argument("--val-split", type=str, default="test1")
    parser.add_argument("--test-split", type=str, default=None)
    parser.add_argument("--allowed-labels", type=str, default=None)
    parser.add_argument("--limit-first-n-labels", type=int, default=None)
    return parser


def run_single_experiment(base_args, overrides: dict):
    args = argparse.Namespace(**vars(base_args))  # shallow copy
    args.model = "transformer"
    name = overrides["name"]
    if base_args.experiment_name:
        name = f"{base_args.experiment_name}_{name}"
    args.experiment_name = name
    args.d_model = overrides.get("d_model", D_MODEL)
    args.n_heads = overrides.get("n_heads", N_HEADS)
    args.num_layers = overrides.get("num_layers", NUM_LAYERS)
    args.dim_feedforward = overrides.get("dim_feedforward", DIM_FEEDFORWARD)
    args.dropout = overrides.get("dropout", DROPOUT)
    args.num_epochs = overrides.get("num_epochs", NUM_EPOCHS_TRANSFORMER)
    args.output_dir = str(Path(EXPERIMENTS_DIR) / args.experiment_name)
    args.allowed_labels = base_args.allowed_labels
    args.limit_first_n_labels = base_args.limit_first_n_labels
    args.pkl_path = base_args.pkl_path
    args.train_split = base_args.train_split
    args.val_split = base_args.val_split
    args.test_split = base_args.test_split
    print(f"\n>>> Running experiment: {args.experiment_name}")
    summary = train_pipeline(args)
    return {
        "name": args.experiment_name,
        "best_val_acc": summary.get("best_val_acc"),
        "best_epoch": summary.get("best_epoch"),
        "checkpoint": summary.get("checkpoint"),
        "config": summary.get("config"),
    }


def main():
    parser = build_grid_parser()
    args = parser.parse_args()

    allowed = parse_allowed_labels(args.allowed_labels, args.limit_first_n_labels)
    args.allowed_labels = ",".join(map(str, allowed)) if allowed is not None else None

    grid = default_transformer_grid(args.prefix)

    results = []
    for cfg_overrides in grid:
        results.append(run_single_experiment(args, cfg_overrides))

    results_path = Path(EXPERIMENTS_DIR) / "transformer_results.json"
    results_path.write_text(json.dumps(results, indent=2))
    print(f"\nSaved aggregated results to {results_path}")


if __name__ == "__main__":
    main()
