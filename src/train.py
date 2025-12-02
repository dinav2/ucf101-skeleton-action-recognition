#!/usr/bin/python3

import argparse
import json
import random
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from config import (
    BATCH_SIZE,
    DIM_FEEDFORWARD,
    D_MODEL,
    DROPOUT,
    EXPERIMENTS_DIR,
    HIDDEN_DIM,
    LEARNING_RATE,
    N_HEADS,
    NUM_EPOCHS_BASELINE,
    NUM_EPOCHS_TRANSFORMER,
    NUM_LAYERS,
    NUM_WORKERS,
    PKL_PATH,
    SEED,
    T_TARGET,
    WEIGHT_DECAY,
)
from dataset_loader import UCFSkeletonDataset
from models import MLPBaseline, PoseTransformer


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == y).sum().item()
        total += x.size(0)

    return total_loss / max(total, 1), correct / max(total, 1)


@torch.no_grad()
def eval_model(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    model.eval()
    total_loss, correct, total = 0.0, 0, 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits, y)

        total_loss += loss.item() * x.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == y).sum().item()
        total += x.size(0)

    return total_loss / max(total, 1), correct / max(total, 1)


def parse_allowed_labels(text: Optional[str], limit_first_n: Optional[int]) -> Optional[List[int]]:
    if limit_first_n is not None:
        return list(range(limit_first_n))
    if text is None:
        return None
    cleaned = text.strip()
    if not cleaned:
        return None
    return [int(x) for x in cleaned.split(",")]


def build_model(args, input_dim: int, num_classes: int) -> nn.Module:
    if args.model == "baseline":
        hidden_dim = getattr(args, "hidden_dim", HIDDEN_DIM)
        return MLPBaseline(
            input_dim=input_dim,
            num_classes=num_classes,
            hidden_dim=hidden_dim,
            dropout=args.dropout,
        )
    if args.model == "transformer":
        return PoseTransformer(
            input_dim=input_dim,
            num_classes=num_classes,
            d_model=args.d_model,
            nhead=args.n_heads,
            num_layers=args.num_layers,
            dim_feedforward=args.dim_feedforward,
            dropout=args.dropout,
        )
    raise ValueError(f"Unknown model: {args.model}")


def create_dataloaders(args, allowed_labels):
    train_ds = UCFSkeletonDataset(
        pkl_path=args.pkl_path,
        split_name=args.train_split,
        T=T_TARGET,
        person_strategy="max_score",
        allowed_labels=allowed_labels,
    )
    val_ds = UCFSkeletonDataset(
        pkl_path=args.pkl_path,
        split_name=args.val_split,
        T=T_TARGET,
        person_strategy="max_score",
        allowed_labels=allowed_labels,
    )
    test_ds = (
        UCFSkeletonDataset(
            pkl_path=args.pkl_path,
            split_name=args.test_split,
            T=T_TARGET,
            person_strategy="max_score",
            allowed_labels=allowed_labels,
        )
        if args.test_split
        else None
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )
    test_loader = (
        DataLoader(
            test_ds,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=NUM_WORKERS,
            pin_memory=True,
        )
        if test_ds
        else None
    )

    return train_loader, val_loader, test_loader, train_ds


def train_pipeline(args) -> dict:
    set_seed(SEED)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[{args.experiment_name}] device={device}")

    allowed_labels = parse_allowed_labels(args.allowed_labels, args.limit_first_n_labels)
    train_loader, val_loader, test_loader, train_ds = create_dataloaders(args, allowed_labels)
    input_dim = train_ds[0][0].shape[-1]
    num_classes = len(train_ds.label_map)

    model = build_model(args, input_dim, num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )

    history = []
    best_val_acc = -1.0
    best_epoch = 0
    ckpt_path = output_dir / f"{args.model}_best.pth"

    for epoch in range(1, args.num_epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device
        )
        val_loss, val_acc = eval_model(model, val_loader, criterion, device)

        improved = val_acc > best_val_acc
        if improved:
            best_val_acc = val_acc
            best_epoch = epoch
            torch.save(model.state_dict(), ckpt_path)

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
            }
        )

        print(
            f"[{args.model}] Ep {epoch:02d} | "
            f"train_loss={train_loss:.3f} val_loss={val_loss:.3f} | "
            f"train_acc={train_acc:.3f} val_acc={val_acc:.3f} "
            f"{'(best)' if improved else ''}"
        )

    summary = {
        "config": vars(args),
        "best_val_acc": best_val_acc,
        "best_epoch": best_epoch,
        "checkpoint": str(ckpt_path),
    }

    if test_loader is not None:
        test_loss, test_acc = eval_model(model, test_loader, criterion, device)
        summary["test_loss"] = test_loss
        summary["test_acc"] = test_acc
        print(f"Test acc: {test_acc:.3f}")

    history_path = output_dir / "metrics.json"
    history_path.write_text(json.dumps(history, indent=2))
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    print(f"Best val acc: {best_val_acc:.3f} at epoch {best_epoch}")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train baseline or PoseTransformer on UCF101 skeletons")
    parser.add_argument("--model", choices=["baseline", "transformer"], default="transformer")
    parser.add_argument("--experiment-name", type=str, default="debug")
    parser.add_argument("--pkl-path", type=str, default=str(PKL_PATH))
    parser.add_argument("--train-split", type=str, default="train1")
    parser.add_argument("--val-split", type=str, default="test1")
    parser.add_argument("--test-split", type=str, default=None)
    parser.add_argument("--num-epochs", type=int, default=None)
    parser.add_argument("--d-model", type=int, default=D_MODEL)
    parser.add_argument("--n-heads", type=int, default=N_HEADS)
    parser.add_argument("--num-layers", type=int, default=NUM_LAYERS)
    parser.add_argument("--dim-feedforward", type=int, default=DIM_FEEDFORWARD)
    parser.add_argument("--dropout", type=float, default=DROPOUT)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--allowed-labels", type=str, default=None, help="Comma separated list, e.g., 0,1,2")
    parser.add_argument("--limit-first-n-labels", type=int, default=None, help="Keep first N labels (sorted)")
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    args.num_epochs = args.num_epochs or (
        NUM_EPOCHS_BASELINE if args.model == "baseline" else NUM_EPOCHS_TRANSFORMER
    )
    args.output_dir = args.output_dir or str(Path(EXPERIMENTS_DIR) / args.experiment_name)
    train_pipeline(args)


if __name__ == "__main__":
    main()
