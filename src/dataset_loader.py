#!/usr/bin/python3

import pickle
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path


class UCFSkeletonDataset(Dataset):
    """
    Dataset for UCF101 2D skeletons from .pickle
    """

    def __init__(
        self,
        pkl_path,
        split_name="train1",
        T=60,
        augment=False,
        person_strategy="max_score",
        allowed_labels=None,
    ):
        """
        Constructor
        """
        self.pkl_path = Path(pkl_path)
        self.T = T
        self.augment = augment
        self.person_strategy = person_strategy

        with open(self.pkl_path, "rb") as f:
            data = pickle.load(f)

        split_dict = data["split"]
        annotations = data["annotations"]

        if split_name not in split_dict:
            raise ValueError(f"split_name={split_name} not found")

        valid_frame_dirs = set(split_dict[split_name])

        # Filter by split
        samples = [a for a in annotations if a["frame_dir"] in valid_frame_dirs]

        # Filter by subset of classes
        if allowed_labels is not None:
            allowed_labels = set(allowed_labels)
            samples = [a for a in samples if a["label"] in allowed_labels]

        original_labels = sorted({int(a["label"]) for a in samples})
        self.label_map = {orig: i for i, orig in enumerate(original_labels)}
        self.inv_label_map = {v: k for k, v in self.label_map.items()}

        self.samples = samples

        self.class_name_map = {}

        for s in self.samples:
            orig = int(s["label"])
            if orig not in self.class_name_map:
                self.class_name_map[orig] = s["frame_dir"].split("/")[0]

    def __len__(self):
        return len(self.samples)

    def normalize(self, skel):
        mean = skel.mean(axis=(0, 1), keepdims=True)
        std = skel.std(axis=(0, 1), keepdims=True) + 1e-6
        return (skel - mean) / std

    def pad_or_crop(self, skel):
        if skel is None:
            return np.zeros((self.T, 17, 2), dtype=np.float32)

        T, V, C = skel.shape

        if T > self.T:
            start = np.random.randint(0, T - self.T + 1)
            skel = skel[start : start + self.T]
        elif T < self.T:
            pad = np.zeros((self.T - T, V, C), dtype=np.float32)
            skel = np.concatenate([skel, pad], axis=0)

        return skel

    def select_person(self, keypoint, keypoint_score):
        if keypoint.size == 0:
            return None

        M, T, V, C = keypoint.shape

        if M == 1 or self.person_strategy == "first":
            return keypoint[0]

        if keypoint_score is None or keypoint_score.size == 0:
            return keypoint[0]

        avg_scores = keypoint_score.mean(axis=(1, 2))
        best_idx = int(avg_scores.argmax())
        return keypoint[best_idx]

    def __getitem__(self, idx):
        a = self.samples[idx]

        real_label = int(a["label"])
        label = self.label_map[real_label]

        keypoint = np.array(a["keypoint"], dtype=np.float32)
        keypoint_score = np.array(a["keypoint_score"], dtype=np.float32)

        skel = self.select_person(keypoint, keypoint_score)
        skel = self.pad_or_crop(skel)
        skel = self.normalize(skel)

        T, V, C = skel.shape

        skel = skel.reshape(T, V * C)
        x = torch.from_numpy(skel)
        y = torch.tensor(label, dtype=torch.long)

        return x, y
