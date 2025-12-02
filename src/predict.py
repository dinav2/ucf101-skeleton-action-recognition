#!/usr/bin/python3
"""
Simple inference script for baseline or PoseTransformer checkpoints.
It loads a saved checkpoint, runs a few samples from a split, and prints predicted classes and probabilities.
"""

import argparse
import json
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import DataLoader

from config import D_MODEL, DROPOUT, PKL_PATH, T_TARGET
from dataset_loader import UCFSkeletonDataset
from models import MLPBaseline, PoseTransformer
from train import parse_allowed_labels


def load_config(config_path: Optional[str]) -> dict:
    if config_path is None:
        return {}
    data = json.loads(Path(config_path).read_text())
    return data.get("config", data)


def build_model(model_name: str, input_dim: int, num_classes: int, cfg: dict):
    if model_name == "baseline":
        return MLPBaseline(
            input_dim=input_dim,
            num_classes=num_classes,
            hidden_dim=cfg.get("hidden_dim", 256),
            dropout=cfg.get("dropout", 0.5),
        )
    if model_name == "transformer":
        return PoseTransformer(
            input_dim=input_dim,
            num_classes=num_classes,
            d_model=cfg.get("d_model", D_MODEL),
            nhead=cfg.get("n_heads", 4),
            num_layers=cfg.get("num_layers", 3),
            dim_feedforward=cfg.get("dim_feedforward", 512),
            dropout=cfg.get("dropout", DROPOUT),
        )
    raise ValueError(f"Unknown model: {model_name}")


def predict_samples(
    model_name: str,
    checkpoint: str,
    pkl_path: str,
    split: str,
    T: int,
    num_samples: int,
    allowed_labels,
    device: str = "auto",
    config_path: Optional[str] = None,
):
    cfg = load_config(config_path)
    dataset = UCFSkeletonDataset(
        pkl_path=pkl_path,
        split_name=split,
        T=T,
        person_strategy="max_score",
        allowed_labels=allowed_labels,
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    input_dim = dataset[0][0].shape[-1]
    num_classes = len(dataset.label_map)
    device = torch.device(
        "cuda" if (device == "auto" and torch.cuda.is_available()) else device
    )

    model = build_model(model_name, input_dim, num_classes, cfg).to(device)
    state = torch.load(checkpoint, map_location=device)
    model.load_state_dict(state)
    model.eval()

    print(f"Loaded checkpoint: {checkpoint}")
    printed = 0
    label_names = dataset.class_name_map
    inv_label_map = dataset.inv_label_map

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            probs = torch.softmax(logits, dim=1)
            prob, pred = probs.max(dim=1)

            pred_idx = int(pred.item())
            pred_orig = inv_label_map.get(pred_idx, pred_idx)
            pred_name = label_names.get(pred_orig, f"class_{pred_orig}")

            true_idx = int(y.item())
            true_orig = inv_label_map.get(true_idx, true_idx)
            true_name = label_names.get(true_orig, f"class_{true_orig}")

            print(
                f"Sample {printed:03d} | pred={pred_name} "
                f"(p={prob.item():.3f}) | true={true_name} | raw_label={pred_orig}"
            )

            printed += 1
            if printed >= num_samples:
                break


def build_parser():
    parser = argparse.ArgumentParser(description="Run inference on a few samples.")
    parser.add_argument(
        "--model", choices=["baseline", "transformer"], default="transformer"
    )
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to summary.json or config.json saved during training",
    )
    parser.add_argument("--pkl-path", type=str, default=str(PKL_PATH))
    parser.add_argument("--split", type=str, default="test1")
    parser.add_argument("--T", type=int, default=T_TARGET)
    parser.add_argument("--num-samples", type=int, default=5)
    parser.add_argument("--allowed-labels", type=str, default=None)
    parser.add_argument("--limit-first-n-labels", type=int, default=None)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    allowed = parse_allowed_labels(args.allowed_labels, args.limit_first_n_labels)

    predict_samples(
        model_name=args.model,
        checkpoint=args.checkpoint,
        pkl_path=args.pkl_path,
        split=args.split,
        T=args.T,
        num_samples=args.num_samples,
        allowed_labels=allowed,
        device=args.device,
        config_path=args.config,
    )


if __name__ == "__main__":
    main()
