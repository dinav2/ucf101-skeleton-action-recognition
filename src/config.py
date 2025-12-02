#!/usr/bin/python3

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
PKL_PATH = DATA_DIR / "ucf101_2d.pkl"

CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

# Training
SEED = 42
BATCH_SIZE = 64
NUM_WORKERS = 4

T_TARGET = 90

NUM_EPOCHS_BASELINE = 5
NUM_EPOCHS_TRANSFORMER = 20

LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4

# Transformer
D_MODEL = 192
N_HEADS = 6
NUM_LAYERS = 3
DIM_FEEDFORWARD = 512
DROPOUT = 0.2
