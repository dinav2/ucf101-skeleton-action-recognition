#!/usr/bin/python3

import random
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch import nn, optim

from config import (
    CHECKPOINT_DIR,
    PKL_PATH,
    SEED,
    BATCH_SIZE,
    NUM_WORKERS,
    T_TARGET,
    NUM_EPOCHS_BASELINE,
    NUM_EPOCHS_TRANSFORMER,
    LEARNING_RATE,
    WEIGHT_DECAY,
    D_MODEL,
    N_HEADS,
    NUM_LAYERS,
    DIM_FEEDFORWARD,
    DROPOUT,
)
from dataset_loader import UCFSkeletonDataset
from models import MLPBaseline
from models import PoseTransformer


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train_one_epoch(model, loader, optimizer, criterion, device):
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

    return total_loss / total, correct / total


@torch.no_grad()
def eval_model(model, loader, criterion, device):
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

    return total_loss / total, correct / total


def main():
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # ----- subset to train with -----
    selected_labels = range(5)

    # ----- datasets y loaders -----
    train_ds = UCFSkeletonDataset(
        pkl_path=str(PKL_PATH),
        split_name="train1",
        T=T_TARGET,
        augment=True,
        person_strategy="max_score",
        allowed_labels=selected_labels,
    )

    val_ds = UCFSkeletonDataset(
        pkl_path=str(PKL_PATH),
        split_name="test1",
        T=T_TARGET,
        augment=False,
        person_strategy="max_score",
        allowed_labels=selected_labels,
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

    input_dim = train_ds[0][0].shape[-1]  # V*2
    num_classes = len(train_ds.label_map)

    print(f"Input dim: {input_dim}, num_classes: {num_classes}")

    criterion = nn.CrossEntropyLoss()

    # Baseline MLP
    baseline = MLPBaseline(input_dim, num_classes).to(device)
    optimizer = optim.Adam(
        baseline.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    best_val_acc = 0.0
    print("\n=== Training MLP baseline ===")
    for epoch in range(1, NUM_EPOCHS_BASELINE + 1):
        tr_loss, tr_acc = train_one_epoch(
            baseline, train_loader, optimizer, criterion, device
        )
        val_loss, val_acc = eval_model(baseline, val_loader, criterion, device)
        print(
            f"[Baseline] Epoch {epoch:02d} | "
            f"train_acc={tr_acc:.3f}, val_acc={val_acc:.3f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(baseline.state_dict(), CHECKPOINT_DIR / "baseline_best.pth")

    # PoseTransformer
    model = PoseTransformer(
        input_dim=input_dim,
        num_classes=num_classes,
        d_model=D_MODEL,
        nhead=N_HEADS,
        num_layers=NUM_LAYERS,
        dim_feedforward=DIM_FEEDFORWARD,
        dropout=DROPOUT,
    ).to(device)

    optimizer = optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    best_val_acc = 0.0
    print("\n=== Training PoseTransformer ===")
    for epoch in range(1, NUM_EPOCHS_TRANSFORMER + 1):
        tr_loss, tr_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device
        )
        val_loss, val_acc = eval_model(model, val_loader, criterion, device)
        print(
            f"[Transformer] Epoch {epoch:02d} | "
            f"train_acc={tr_acc:.3f}, val_acc={val_acc:.3f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), CHECKPOINT_DIR / "transformer_best.pth")

    print("Done.")


if __name__ == "__main__":
    main()
