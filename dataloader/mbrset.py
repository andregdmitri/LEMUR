# dataloader/mbrset.py

import os
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from config.constants import BATCH_SIZE, NUM_WORKERS
import numpy as np
from sklearn.utils.class_weight import compute_class_weight


class MBRSETDataset(Dataset):
    def __init__(self, root, split="train", transform=None, val_split=0.1):
        self.root = root
        self.transform = transform

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

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        path = row["path"]
        label = int(row["final_icdr"])

        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)

        return img, label, path


class MBRSETModule(pl.LightningDataModule):
    def __init__(self, root, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, transform=None):
        super().__init__()
        self.root = root
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.transform = transform

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            self.train_ds = MBRSETDataset(self.root, split="train", transform=self.transform)
            self.val_ds   = MBRSETDataset(self.root, split="val",   transform=self.transform)

        if stage == "test":
            self.test_ds  = MBRSETDataset(self.root, split="val",   transform=self.transform)

        if stage == "full":
            # use whole dataset as validation
            self.val_ds   = MBRSETDataset(self.root, split="full",  transform=self.transform)

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