"""Dataset helpers for the SYN -> cropped-SVHN transfer experiment."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image
from torch.utils.data import Dataset


SPLIT_FILES = {
    "syn_train_50k": ("syn_train_50k_images.npy", "syn_train_50k_labels.npy"),
    "syn_test_full": ("syn_test_full_images.npy", "syn_test_full_labels.npy"),
    "svhn_test_10k": ("svhn_test_10k_images.npy", "svhn_test_10k_labels.npy"),
}


class SynSVHNNpyDataset(Dataset):
    """Read a prepared uint8 NCHW split and apply PIL-compatible transforms."""

    def __init__(self, root, split, transform=None):
        if split not in SPLIT_FILES:
            raise ValueError(f"Unknown SYN/SVHN split: {split}")
        processed = Path(root) / "processed"
        image_name, label_name = SPLIT_FILES[split]
        self.images = np.load(processed / image_name, mmap_mode="r")
        self.labels = np.load(processed / label_name, mmap_mode="r")
        if self.images.ndim != 4 or self.images.shape[1:] != (3, 32, 32):
            raise ValueError(f"Invalid image shape for {split}: {self.images.shape}")
        if self.images.dtype != np.uint8:
            raise ValueError(f"Invalid image dtype for {split}: {self.images.dtype}")
        if self.labels.shape != (len(self.images),):
            raise ValueError(f"Image/label count mismatch for {split}")
        if self.labels.dtype != np.int64:
            raise ValueError(f"Invalid label dtype for {split}: {self.labels.dtype}")
        if self.labels.min() < 0 or self.labels.max() > 9:
            raise ValueError(f"Invalid label range for {split}")
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        image = np.asarray(self.images[int(index)]).transpose(1, 2, 0)
        image = Image.fromarray(image, mode="RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, int(self.labels[int(index)])
