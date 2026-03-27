# dataloader/mbrset.py

import os
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from config.constants import BATCH_SIZE, NUM_WORKERS, USE_MIXUP, MIXUP_ALPHA, USE_MOSAIC, MOSAIC_PROB, USE_COPY_PASTE, COPY_PASTE_PROB
import numpy as np
import random
from sklearn.utils.class_weight import compute_class_weight
from utils.transforms import preprocess_image, mixup_data, mosaic_data, copy_paste_data


class MBRSETDataset(Dataset):
    def __init__(self, root, split="train", transform=None, val_split=0.1,
                 use_mixup=USE_MIXUP, mixup_alpha=MIXUP_ALPHA,
                 use_mosaic=USE_MOSAIC, mosaic_prob=MOSAIC_PROB,
                 use_copy_paste=USE_COPY_PASTE, copy_paste_prob=COPY_PASTE_PROB):
        self.root = root
        self.transform = transform
        self.use_mixup = use_mixup
        self.mixup_alpha = mixup_alpha
        self.use_mosaic = use_mosaic
        self.mosaic_prob = mosaic_prob
        self.use_copy_paste = use_copy_paste
        self.copy_paste_prob = copy_paste_prob

        img_dir = os.path.join(root, "images")
        csv_path = os.path.join(root, "labels_mbrset.csv")

        # Try reading with header
        df = pd.read_csv(csv_path)

        # If first column isn't called "file", fallback to no-header mode
        if "file" not in df.columns or "final_icdr" not in df.columns:
            df = pd.read_csv(csv_path, header=None, names=["file", "final_icdr"])

        # Drop bad rows
        df = df.dropna(subset=["file", "final_icdr"])

        # Force string filenames
        df["file"] = df["file"].astype(str).str.strip()

        # Build full paths
        df["path"] = df["file"].apply(lambda x: os.path.join(img_dir, x))

        # Keep only existing files
        df = df[df["path"].apply(os.path.exists)]

        if split == "full":
            pass
        elif split in ["train", "val"]:
            n = len(df)
            val_size = int(n * val_split)
            train_df = df.iloc[:-val_size]
            val_df   = df.iloc[-val_size:]
            df = train_df if split == "train" else val_df

        self.df = df.reset_index(drop=True)
        print(f"[MBRSET] Split: {split:<5} | Images: {len(self.df)}")

    def __len__(self):
        return len(self.df)

    def _load_single(self, idx):
        row = self.df.iloc[idx]
        path = row["path"]
        label = int(row["final_icdr"])

        img = preprocess_image(path)
        if self.transform:
            img = self.transform(img)

        return img, label, path

    def __getitem__(self, idx):
        if self.use_mosaic and random.random() < self.mosaic_prob and len(self.df) >= 4:
            idxs = random.sample(range(len(self.df)), 4)
            imgs, labels, paths = [], [], []
            for j in idxs:
                img_j, lbl_j, path_j = self._load_single(j)
                imgs.append(img_j); labels.append(lbl_j); paths.append(path_j)
            img_m, lbl_m = mosaic_data(imgs, labels)
            return img_m, lbl_m, paths

        if self.use_copy_paste and random.random() < self.copy_paste_prob:
            idx2 = random.randint(0, len(self.df) - 1)
            img1, lbl1, path1 = self._load_single(idx)
            img2, lbl2, _ = self._load_single(idx2)
            img_cp, lbl_cp = copy_paste_data(img1, img2, lbl1, lbl2)
            return img_cp, lbl_cp, path1

        if self.use_mixup:
            idx2 = random.randint(0, len(self.df) - 1)
            img1, lbl1, path1 = self._load_single(idx)
            img2, lbl2, _ = self._load_single(idx2)
            img_mx, lbl_mx, _ = mixup_data(img1, img2, lbl1, lbl2, self.mixup_alpha)
            return img_mx, lbl_mx, path1

        return self._load_single(idx)


class MBRSETModule(pl.LightningDataModule):
    def __init__(
        self,
        root,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        transform=None,
        use_mixup=USE_MIXUP,
        mixup_alpha=MIXUP_ALPHA,
        use_mosaic=USE_MOSAIC,
        mosaic_prob=MOSAIC_PROB,
        use_copy_paste=USE_COPY_PASTE,
        copy_paste_prob=COPY_PASTE_PROB,
    ):
        super().__init__()
        self.root = root
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.transform = transform
        self.use_mixup = use_mixup
        self.mixup_alpha = mixup_alpha
        self.use_mosaic = use_mosaic
        self.mosaic_prob = mosaic_prob
        self.use_copy_paste = use_copy_paste
        self.copy_paste_prob = copy_paste_prob

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            self.train_ds = MBRSETDataset(
                self.root, split="train", transform=self.transform,
                use_mixup=self.use_mixup, mixup_alpha=self.mixup_alpha,
                use_mosaic=self.use_mosaic, mosaic_prob=self.mosaic_prob,
                use_copy_paste=self.use_copy_paste, copy_paste_prob=self.copy_paste_prob,
            )
            self.val_ds = MBRSETDataset(
                self.root, split="val", transform=self.transform,
                use_mixup=self.use_mixup, mixup_alpha=self.mixup_alpha,
                use_mosaic=self.use_mosaic, mosaic_prob=self.mosaic_prob,
                use_copy_paste=self.use_copy_paste, copy_paste_prob=self.copy_paste_prob,
            )

        if stage == "test":
            self.test_ds = MBRSETDataset(
                self.root, split="val", transform=self.transform,
                use_mixup=self.use_mixup, mixup_alpha=self.mixup_alpha,
                use_mosaic=self.use_mosaic, mosaic_prob=self.mosaic_prob,
                use_copy_paste=self.use_copy_paste, copy_paste_prob=self.copy_paste_prob,
            )

        if stage == "full":
            # use whole dataset as validation
            self.val_ds = MBRSETDataset(
                self.root, split="full", transform=self.transform,
                use_mixup=self.use_mixup, mixup_alpha=self.mixup_alpha,
                use_mosaic=self.use_mosaic, mosaic_prob=self.mosaic_prob,
                use_copy_paste=self.use_copy_paste, copy_paste_prob=self.copy_paste_prob,
            )

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size,
                          shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size,
                          shuffle=False, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_ds, batch_size=self.batch_size,
                          shuffle=False, num_workers=self.num_workers)


def compute_mbrset_class_weights(root):
    """
    Computes class weights for the full MBRSET dataset.
    """
    csv_path = os.path.join(root, "labels_mbrset.csv")
    if not os.path.exists(csv_path):
        print(f"Warning: CSV not found at {csv_path}. Returning default weights.")
        return torch.tensor([1., 1., 1., 1., 1.])

    df = pd.read_csv(csv_path, header=None, names=["file", "final_icdr"])
    labels = df["final_icdr"].values
    classes = np.unique(labels)

    weights = compute_class_weight(class_weight="balanced",
                                   classes=classes,
                                   y=labels)

    return torch.tensor(weights, dtype=torch.float)