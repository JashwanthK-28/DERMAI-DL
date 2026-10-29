"""
DermAI: Skin Disease Detector from Mobile Images Using Deep Learning
PyTorch Dataset and Augmentation Pipeline for PAD-UFES-20
"""

import os
import glob
from typing import Dict, List, Optional, Tuple, Callable
import numpy as np
import pandas as pd
from PIL import Image, ImageOps
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T

DIAGNOSTIC_CLASSES = ["ACK", "BCC", "MEL", "NEV", "SCC", "SEK"]
CLASS_TO_IDX = {cls: idx for idx, cls in enumerate(DIAGNOSTIC_CLASSES)}
IDX_TO_CLASS = {idx: cls for idx, cls in enumerate(DIAGNOSTIC_CLASSES)}

# ImageNet statistics for CNN backbones
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def build_image_index(search_dirs: Optional[List[str]] = None) -> Dict[str, str]:
    """Build an O(1) index of image filename -> absolute path."""
    if search_dirs is None:
        search_dirs = [
            "data/images/imgs_part_1",
            "data/images/imgs_part_2",
            "data/images/imgs_part_3",
            "/home/user/Documents/DermAI Project/images/imgs_part_1",
            "/home/user/Documents/DermAI Project/images/imgs_part_2",
            "/home/user/Documents/DermAI Project/images/imgs_part_3",
        ]

    index: Dict[str, str] = {}
    for d in search_dirs:
        if os.path.exists(d):
            for fname in os.listdir(d):
                if fname.lower().endswith((".png", ".jpg", ".jpeg")):
                    if fname not in index:
                        index[fname] = os.path.abspath(os.path.join(d, fname))
    return index


def get_train_transforms(image_size: int = 224) -> T.Compose:
    """Training augmentations: Flip, rotation, color jitter, ImageNet normalize."""
    return T.Compose([
        T.Resize((image_size, image_size), interpolation=T.InterpolationMode.BICUBIC),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomVerticalFlip(p=0.5),
        T.RandomRotation(degrees=20),
        T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        T.ToTensor(),  # Maps [0, 255] PIL Image to float32 [0.0, 1.0]
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_eval_transforms(image_size: int = 224) -> T.Compose:
    """Evaluation / inference preprocessing: clean 224x224 bicubic resize + normalize."""
    return T.Compose([
        T.Resize((image_size, image_size), interpolation=T.InterpolationMode.BICUBIC),
        T.ToTensor(),  # Maps [0, 255] PIL Image to float32 [0.0, 1.0]
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


class PADUFESDataset(Dataset):
    """PyTorch Dataset for PAD-UFES-20 mobile skin images."""

    def __init__(
        self,
        df: pd.DataFrame,
        img_index: Dict[str, str],
        transforms: Optional[Callable] = None,
        is_training: bool = False,
    ):
        self.df = df.reset_index(drop=True)
        self.img_index = img_index
        self.is_training = is_training
        self.transforms = transforms or (
            get_train_transforms() if is_training else get_eval_transforms()
        )

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, str]:
        row = self.df.iloc[idx]
        img_id = str(row["img_id"])
        diagnostic = str(row["diagnostic"])
        label_idx = CLASS_TO_IDX[diagnostic]

        if img_id not in self.img_index:
            raise FileNotFoundError(f"Image {img_id} not found in indexed paths.")

        img_path = self.img_index[img_id]

        # Step 1: Open image and apply EXIF orientation correction
        img = Image.open(img_path)
        img = ImageOps.exif_transpose(img)

        # Step 2: Convert to 3-channel RGB
        if img.mode != "RGB":
            img = img.convert("RGB")

        # Step 3: Apply transforms (spatial resize 224x224, augmentations if training, normalization)
        tensor = self.transforms(img)

        return tensor, label_idx, img_id


def compute_class_weights(df: pd.DataFrame, device: torch.device) -> torch.Tensor:
    """
    Compute inverse-frequency class weights to combat severe class imbalance
    (MEL = 2.3% vs BCC = 36.8%).
    """
    counts = df["diagnostic"].value_counts()
    total = len(df)
    n_classes = len(DIAGNOSTIC_CLASSES)
    weights = []
    for cls in DIAGNOSTIC_CLASSES:
        count = counts.get(cls, 1)
        # Smooth balanced weight: N / (C * N_c)
        w = total / (n_classes * count)
        weights.append(w)

    weights_tensor = torch.tensor(weights, dtype=torch.float32, device=device)
    print(f"[*] Computed class weights: {dict(zip(DIAGNOSTIC_CLASSES, [round(w, 2) for w in weights]))}")
    return weights_tensor
