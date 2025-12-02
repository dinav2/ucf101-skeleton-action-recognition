#!/usr/bin/env python3
"""
Visualizations for trained models:
- Confusion matrix, per-class accuracy/F1
- t-SNE of CLS embeddings
- Temporal attention and head maps
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.manifold import TSNE

from config import (
    PKL_PATH,
    T_TARGET,
    BATCH_SIZE,
    NUM_WORKERS,
    D_MODEL,
    N_HEADS,
    NUM_LAYERS,
    DIM_FEEDFORWARD,
    DROPOUT,
    FIGURES_DIR,
)
from dataset_loader import UCFSkeletonDataset
from models.pose_transformer import PoseTransformer

sns.set_theme(style="whitegrid", context="notebook")

# Helpers


def build_val_loader(selected_labels, batch_size, num_workers):
    val_ds = UCFSkeletonDataset(
        pkl_path=str(PKL_PATH),
        split_name="test1",
        T=T_TARGET,
        person_strategy="max_score",
        allowed_labels=selected_labels,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return val_ds, val_loader


def load_transformer(val_ds, device, transformer_ckpt_path):
    input_dim = val_ds[0][0].shape[-1]
    num_classes = len(val_ds.label_map)

    transformer = PoseTransformer(
        input_dim=input_dim,
        num_classes=num_classes,
        d_model=D_MODEL,
        nhead=N_HEADS,
        num_layers=NUM_LAYERS,
        dim_feedforward=DIM_FEEDFORWARD,
        dropout=DROPOUT,
    ).to(device)
    transformer_ckpt = Path(transformer_ckpt_path)
    transformer.load_state_dict(torch.load(transformer_ckpt, map_location=device))
    transformer.eval()

    return transformer


def evaluate_model(model, loader, device):
    y_true = []
    y_pred = []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)

            logits = model(x)
            preds = logits.argmax(dim=-1)

            y_true.extend(y.cpu().numpy().tolist())
            y_pred.extend(preds.cpu().numpy().tolist())

    return np.array(y_true), np.array(y_pred)


def get_class_names(val_ds):
    names = []
    for new_id in range(len(val_ds.label_map)):
        orig_label = val_ds.inv_label_map[new_id]
        class_name = val_ds.class_name_map[orig_label]
        names.append(class_name)
    return names


# 1) Confusion matrix + per-class metrics


def _short_class_names(class_names):
    def shorten(name: str) -> str:
        if name.startswith("v_"):
            name2 = name[2:]
        else:
            name2 = name
        name2 = name2.split("_g")[0]
        return name2

    return [shorten(c) for c in class_names]


def plot_confusion_and_metrics(y_true, y_pred, class_names, prefix="transformer"):
    nice_names = _short_class_names(class_names)

    cm = confusion_matrix(y_true, y_pred, labels=np.arange(len(class_names)))
    df_cm = pd.DataFrame(cm, index=nice_names, columns=nice_names)

    plt.figure(figsize=(5, 4))
    ax = sns.heatmap(
        df_cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar_kws={"label": "Count"},
        square=True,
    )
    ax.set_xlabel("Prediction", fontsize=10)
    ax.set_ylabel("Ground truth", fontsize=10)
    ax.set_title(f"Confusion matrix - {prefix}", fontsize=11)
    plt.xticks(rotation=45, ha="right", fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    plt.tight_layout()

    out_path = FIGURES_DIR / f"confusion_{prefix}.png"
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Saved {out_path}")

    metrics = []
    for cid, cname in enumerate(class_names):
        mask = y_true == cid
        if mask.sum() == 0:
            acc_c = np.nan
        else:
            acc_c = (y_pred[mask] == y_true[mask]).mean()

        y_true_bin = (y_true == cid).astype(int)
        y_pred_bin = (y_pred == cid).astype(int)
        f1_c = f1_score(y_true_bin, y_pred_bin, average="binary", zero_division=0.0)

        metrics.append(
            {
                "class_full": cname,
                "class_short": _short_class_names([cname])[0],
                "accuracy": acc_c,
                "f1": f1_c,
            }
        )

    df = pd.DataFrame(metrics)

    plt.figure(figsize=(7, 4))
    ax = sns.barplot(data=df, x="class_short", y="accuracy")
    ax.set_ylim(0, 1.0)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Accuracy", fontsize=10)
    ax.set_xlabel("Class", fontsize=10)
    ax.set_title(f"Per-class accuracy - {prefix}", fontsize=11)
    plt.tight_layout()
    out_path = FIGURES_DIR / f"class_accuracy_{prefix}.png"
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Saved {out_path}")

    plt.figure(figsize=(7, 4))
    ax = sns.barplot(data=df, x="class_short", y="f1")
    ax.set_ylim(0, 1.0)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("F1-score", fontsize=10)
    ax.set_xlabel("Class", fontsize=10)
    ax.set_title(f"Per-class F1 - {prefix}", fontsize=11)
    plt.tight_layout()
    out_path = FIGURES_DIR / f"class_f1_{prefix}.png"
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Saved {out_path}")


# 2) t-SNE of CLS embeddings


def plot_tsne_embeddings(transformer, loader, val_ds, device):
    print("Collecting CLS embeddings for t-SNE...")
    emb_list = []
    label_list = []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            emb = transformer.forward_features(x)  # (B, d_model)
            emb_list.append(emb.cpu().numpy())
            label_list.append(y.numpy())

    embeddings = np.concatenate(emb_list, axis=0)
    labels = np.concatenate(label_list, axis=0)

    class_names = get_class_names(val_ds)
    label_names = [class_names[i] for i in labels]
    label_short = _short_class_names(label_names)

    print("Running t-SNE...")
    tsne = TSNE(n_components=2, init="pca", learning_rate="auto")
    z = tsne.fit_transform(embeddings)

    df = pd.DataFrame({"x": z[:, 0], "y": z[:, 1], "class": label_short})

    plt.figure(figsize=(6, 6))
    ax = sns.scatterplot(data=df, x="x", y="y", hue="class", s=25, palette="tab10")
    ax.set_xlabel("Dim 1", fontsize=10)
    ax.set_ylabel("Dim 2", fontsize=10)
    ax.set_title("t-SNE of CLS embeddings (Transformer)", fontsize=11)
    plt.legend(
        title="Class",
        bbox_to_anchor=(1.05, 1),
        loc="upper left",
        fontsize=8,
    )
    plt.tight_layout()
    out_path = FIGURES_DIR / "tsne_cls_transformer.png"
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Saved {out_path}")


# 3) Temporal attention and head maps


def get_example_batch(val_ds, device, index=0):
    x, y = val_ds[index]  # x: (T, D)
    x = x.unsqueeze(0).to(device)  # (1, T, D)
    y = torch.tensor([y], device=device)
    new_id = int(y.item())
    orig_label = val_ds.inv_label_map[new_id]
    class_name = val_ds.class_name_map[orig_label]
    return x, y, class_name


def capture_attention(transformer, x_sample, device):
    attn_maps = []

    def attn_hook(module, input, output):
        if isinstance(output, tuple) and len(output) == 2 and output[1] is not None:
            attn_weights = output[1]
        else:
            q = input[0]
            k = q
            v = q
            _, attn_weights = module(
                q,
                k,
                v,
                need_weights=True,
                average_attn_weights=False,
            )
        attn_maps.append(attn_weights.detach().cpu())

    last_layer = transformer.encoder.layers[-1]
    handle = last_layer.self_attn.register_forward_hook(attn_hook)

    with torch.no_grad():
        _ = transformer(x_sample.to(device))

    handle.remove()
    attn = attn_maps[0]  # (B, H, L, L)
    return attn


def plot_temporal_attention_and_headmaps(transformer, val_ds, device):
    x_sample, y_sample, class_name_full = get_example_batch(val_ds, device, index=0)
    class_name = _short_class_names([class_name_full])[0]
    attn = capture_attention(transformer, x_sample, device)

    B, H, L, _ = attn.shape
    print("Attention shape:", attn.shape)
    attn_mean = attn.mean(dim=1)  # (B, L, L)

    cls_to_all = attn_mean[0, 0, :]  # (L,)
    cls_to_frames = cls_to_all[1:]  # drop CLS token
    importance = cls_to_frames.numpy()
    T = importance.shape[0]

    df = pd.DataFrame({"frame": np.arange(T), "attention": importance})
    plt.figure(figsize=(7, 3))
    ax = sns.lineplot(data=df, x="frame", y="attention", marker="o")
    ax.set_xlabel("Frame index", fontsize=10)
    ax.set_ylabel("Attention weight (CLS -> frame)", fontsize=10)
    ax.set_title(
        f"Transformer temporal attention\nExample class: {class_name}",
        fontsize=11,
    )
    plt.tight_layout()
    out_path = FIGURES_DIR / "attention_temporal_example.png"
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Saved {out_path}")

    head = 0
    A = attn[0, head].numpy()  # (L, L)

    plt.figure(figsize=(6, 5))
    ax = sns.heatmap(
        A,
        cmap="magma",
        cbar_kws={"label": "Attention weight"},
        xticklabels=False,
        yticklabels=False,
    )
    ax.set_xlabel("Key token", fontsize=10)
    ax.set_ylabel("Query token", fontsize=10)
    ax.set_title(f"Attention map - head {head}", fontsize=11)
    plt.tight_layout()
    out_path = FIGURES_DIR / "attention_matrix_head0.png"
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Saved {out_path}")

    fig, axes = plt.subplots(1, H, figsize=(3 * H, 3), sharex=True, sharey=True)
    for h in range(H):
        A_h = attn[0, h].numpy()
        ax = axes[h]
        sns.heatmap(
            A_h,
            cmap="magma",
            cbar=False,
            xticklabels=False,
            yticklabels=False,
            ax=ax,
        )
        ax.set_title(f"Head {h}", fontsize=9)
    fig.suptitle("Attention patterns by head", fontsize=11)
    plt.tight_layout()
    out_path = FIGURES_DIR / "attention_heads_grid.png"
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Saved {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate visualization assets")
    parser.add_argument(
        "--transformer-checkpoint",
        type=str,
        required=True,
        help="Path to transformer checkpoint",
    )
    parser.add_argument(
        "--limit-first-n-labels",
        type=int,
        default=None,
        help="Keep first N labels (sorted); omit to use all labels",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    selected_labels = (
        list(range(args.limit_first_n_labels)) if args.limit_first_n_labels else None
    )
    val_ds, val_loader = build_val_loader(selected_labels, BATCH_SIZE, NUM_WORKERS)
    class_names = get_class_names(val_ds)

    transformer = load_transformer(val_ds, device, args.transformer_checkpoint)

    print("Evaluating Transformer on validation split...")
    y_true, y_pred_tr = evaluate_model(transformer, val_loader, device)
    plot_confusion_and_metrics(y_true, y_pred_tr, class_names, prefix="transformer")

    plot_tsne_embeddings(transformer, val_loader, val_ds, device)
    plot_temporal_attention_and_headmaps(transformer, val_ds, device)


if __name__ == "__main__":
    main()
