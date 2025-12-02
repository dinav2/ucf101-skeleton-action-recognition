# UCF101 Skeleton-Based Action Recognition

This project tackles human action recognition on the UCF101 dataset using 2D skeletons (keypoints). Working with pose sequences removes background and appearance noise, letting models focus on body dynamics. This repo provides a simple MLP baseline and a PoseTransformer that uses self-attention over time to capture richer temporal dependencies and joint interactions. All reported Transformer results here are trained both on a 5-class subset (`allowed_labels=0..4`) and the 101 classes.

## Dataset Description and Work Strategy
- **UCF101 overview**: 13,320 videos, 101 action classes spanning sports, daily activities, and human–object interactions (see official docs: https://www.crcv.ucf.edu/data/UCF101.php).
- **Skeleton/keypoint representation**: Each video is preprocessed into a sequence of 2D keypoints stored in a `.pkl` under `data/` (configurable via `PKL_PATH` in `config.py`). For each clip, we keep up to `T_TARGET=90` frames; each frame contains 17 joints with (x, y) coordinates. After person selection (highest average keypoint score), tensors are shaped `(T, 17, 2)` and flattened to `(T, 34)` for the models.
- **Preprocessing**:
  - Normalize per-sample (zero mean, unit variance).
  - Pad with zeros or randomly crop to `T_TARGET` frames.
  - Person strategy: choose the person with the highest keypoint confidence per clip.
- **Splits**: Uses the provided UCF101 skeleton splits (`train1`, `test1`); Validation is performed on `test1` for model selection in these experiments.

## Repository Structure
- `src/train.py`: Training/validation loop, checkpoint saving, JSON summaries.
- `src/predict.py`: Inference utility for trained checkpoints.
- `src/dataset_loader.py`: `UCFSkeletonDataset` with normalization, padding/cropping, and person selection.
- `src/models/`: `mlp_baseline.py` (MLPBaseline) and `pose_transformer.py` (PoseTransformer).
- `src/run_experiments.py`: Grid runner for PoseTransformer variants.
- `src/visualize_all.py`: Visualization utilities (confusion matrix, t-SNE, attention) for a given Transformer checkpoint.
- `src/config.py`: Hyperparameters and paths (`PKL_PATH`, `EXPERIMENTS_DIR`, `FIGURES_DIR`, batch size, epochs, Transformer dims/dropout, etc.).
- `experiments/`: One folder per run with `metrics.json` and `summary.json` (best checkpoint path, best val accuracy, best epoch). `transformer_results.json` aggregates grid runs.
- `figures/`: Plots already checked into the repo (loss/accuracy curves, confusion matrices, attention maps) so they can be viewed directly.

## Models
### Baseline: `MLPBaseline`
- Flattens each skeleton frame and applies a small stack of fully connected layers with dropout.
- Purpose: establish a lightweight reference to quantify Transformer gains.

### Main: `PoseTransformer`
- Applies a linear projection to each frame, prepends a CLS token, and feeds the sequence to a Transformer encoder. Self-attention models temporal relationships and inter-joint interactions.
- Key hyperparameters:
  - `D_MODEL`: embedding dimension.
  - `N_HEADS`: number of attention heads.
  - `NUM_LAYERS`: encoder depth.
  - `DIM_FEEDFORWARD`: FFN hidden size.
  - `DROPOUT`: dropout inside attention/FFN blocks.

## Training and Evaluation
- **Data pipeline**: `UCFSkeletonDataset` + PyTorch `DataLoader` with shuffling for training.
- **Objective/optimizer**: Cross-entropy loss, Adam optimizer (`LEARNING_RATE`, `WEIGHT_DECAY` from `config.py`).
- **Training length**: `NUM_EPOCHS_BASELINE` and `NUM_EPOCHS_TRANSFORMER` in `config.py`; batch size `BATCH_SIZE=64`.
- **Validation**: Top-1 accuracy on `val_split` (here `test1`). Best checkpoint selected by max validation accuracy.
- **Commands** (examples):
  ```bash
  # Baseline
  python src/train.py --model baseline --experiment-name mlp_baseline

  # Transformer
  python src/train.py --model transformer --experiment-name transformer_default

  # Custom Transformer
  python src/train.py --model transformer \
    --experiment-name xfmr_d192_h4_l3_do02 \
    --d-model 192 --n-heads 4 --num-layers 3 \
    --dim-feedforward 512 --dropout 0.2

  # Grid runner (PoseTransformer sweep)
  python src/run_experiments.py --prefix transformer
  ```

## Experimental Results and Baseline Comparison
Metrics below are derived from the JSON summaries created after the training process. Validation accuracy is reported on `test1` (used as validation here). Best epoch refers to the epoch with the highest validation accuracy. The first table corresponds to the 5-class subset (`limit_first_n_labels=5`). A second table summarizes full-class runs (all 101 classes).

| Experiment | Model | D_MODEL | N_HEADS | NUM_LAYERS | DIM_FF | DROPOUT | Weight Decay | Best Val Acc | Best Epoch |
|------------|-------|---------|---------|------------|--------|---------|--------------|--------------|------------|
| `baseline` | MLPBaseline | — | — | — | — | 0.3 | 1e-4 | 0.792 | 18 |
| `transformer_d128_h2_l2_do01` | PoseTransformer | 128 | 2 | 2 | 256 | 0.1 | 1e-4 | **0.814** | 10 |
| `transformer_d192_h4_l3_do02` | PoseTransformer | 192 | 4 | 3 | 512 | 0.2 | 1e-4 | 0.792 | 20 |
| `transformer_d256_h4_l4_do03` | PoseTransformer | 256 | 4 | 4 | 768 | 0.3 | 1e-4 | 0.792 | 8 |

Full-class runs (all 101 classes; no class limit):

| Experiment | Model | D_MODEL | N_HEADS | NUM_LAYERS | DIM_FF | DROPOUT | Best Val Acc | Best Epoch |
|------------|-------|---------|---------|------------|--------|---------|--------------|------------|
| `baseline` | MLPBaseline | — | — | — | — | 0.3 | 0.351 | 19 |
| `transformer_d128_h2_l2_do01` | PoseTransformer | 128 | 2 | 2 | 256 | 0.1 | 0.474 | 14 |
| `transformer_d192_h4_l3_do02` | PoseTransformer | 192 | 4 | 3 | 512 | 0.2 | **0.486** | 12 |
| `transformer_d256_h4_l4_do03` | PoseTransformer | 256 | 4 | 4 | 768 | 0.3 | 0.471 | 20 |

Key observations:
- The best Transformer (`transformer_d128_h2_l2_do01`) improves validation accuracy by **~+2.2 points** over the baseline (0.814 vs. 0.792).
- Mid-size and larger Transformers (192/256 dims) matched the baseline at ~0.792; higher capacity without extra regularization or data did not improve further on this 5-class subset.

## Model Improvements: What Changed and Why It Worked
### Change 1 – Light vs. mid-size capacity (D_MODEL 128 vs. 192/256)
- **What changed**: Compared a light Transformer (d128, 2 heads, 2 layers) to mid/large setups (d192/d256, 4 heads, 3–4 layers).
- **Effect (5 classes)**: The smallest model performed best (0.814), while mid/large models plateaued at ~0.792.
- **Effect (101 classes)**: Mid-size (d192) performed slightly better than the others (0.486 vs. 0.474/0.471), indicating larger capacity helps when more classes are present.
- **Why**: On small-class subsets, extra capacity overfits quickly.

### Change 2 – Depth and heads (2×2 vs. 4×3/4×4)
- **What changed**: Raised heads/layers from 2×2 to 4×3 and 4×4.
- **Effect (5 classes)**: No improvement over the light model; 4×3/4×4 matched the baseline (~0.792).
- **Effect (101 classes)**: 4×3 achieved the highest among the three (0.486) but only marginally.
- **Why**: Additional heads/layers increase expressiveness but also optimization difficulty; gains appear only when the task diversity grows (full 101 classes), and even then remain small without stronger regularization or more data.

### Change 3 – Regularization (dropout 0.1–0.3, weight decay 1e-4)
- **What changed**: Dropout 0.1 on the light model; 0.2–0.3 on larger ones; weight decay fixed at 1e-4.
- **Effect**: Higher dropout stabilized deeper models but did not lift accuracy beyond 0.792; low dropout on the small model preserved its lead.

### Change 4 – Learning rate and convergence
- **What changed**: Used Adam with lr=1e-3.
- **Effect**: Best epochs: 10 (d128), 20 (d192), 8 (d256), 18 (baseline).

## Inference and Predictions
Run predictions with an existing Transformer checkpoint:
```bash
python src/predict.py --model transformer \
  --checkpoint experiments/transformer_d192_h4_l3_do02/transformer_best.pth \
  --num-samples 5
```
Outputs include predicted class, probability, and ground-truth label for sampled clips. Use checkpoints from the 5-class runs to match label mappings.
