# dataloader/messidor.py

import os
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from config.constants import BATCH_SIZE, NUM_WORKERS


def _find_root_with_paths(root, required_paths, max_depth=3):
    root = os.path.abspath(root)
    if all(os.path.exists(os.path.join(root, path)) for path in required_paths):
        return root

    for current_root, dirs, files in os.walk(root):
        depth = current_root.count(os.sep) - root.count(os.sep)
        if depth > max_depth:
            continue
        if all(os.path.exists(os.path.join(current_root, path)) for path in required_paths):
            return current_root

    return root


class MessidorDataset(Dataset):
    def __init__(self, root, split="train", transform=None, val_split=0.1):
        self.root = _find_root_with_paths(root, ["messidor_data.csv", "messidor-2"])
        self.transform = transform
        self.split = split
        self.val_split = val_split

        csv_path = os.path.join(self.root, "messidor_data.csv")
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Messidor CSV not found at {csv_path}")

        self.img_dir = os.path.join(self.root, "messidor-2")
        if os.path.isdir(self.img_dir):
            image_files = [f for f in os.listdir(self.img_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
            if not image_files:
                nested_dir = os.path.join(self.img_dir, "messidor-2", "preprocess")
                if os.path.isdir(nested_dir):
                    self.img_dir = nested_dir
        if not os.path.isdir(self.img_dir):
            raise FileNotFoundError(f"Messidor image directory not found at {self.img_dir}")

        df = pd.read_csv(csv_path)
        if "id_code" not in df.columns or "diagnosis" not in df.columns:
            raise ValueError("Messidor CSV is missing required columns 'id_code' or 'diagnosis'.")

        df["raw_id_code"] = df["id_code"].astype(str)
        df["id_code"] = df["raw_id_code"].str.replace(".png", "", regex=False).str.replace(".jpg", "", regex=False)
        df["path"] = df["raw_id_code"].apply(lambda x: os.path.join(self.img_dir, x))
        if "adjudicated_gradable" in df.columns:
            df = df[df["adjudicated_gradable"] == 1]

        # Keep only existing image files.
        df = df[df["path"].apply(os.path.exists)].reset_index(drop=True)
        if df.empty:
            raise FileNotFoundError(f"No Messidor images were found in {self.img_dir}")

        if split != "full":
            n = len(df)
            val_count = max(1, int(n * self.val_split))
            if split == "train":
                df = df.iloc[:-val_count].reset_index(drop=True)
            elif split in ["val", "test"]:
                df = df.iloc[-val_count:].reset_index(drop=True)
            else:
                raise ValueError(f"Unsupported split: {split}")

        self.df = df
        self.label_map = dict(zip(df["id_code"], df["diagnosis"].astype(int)))
        print(f"[MESSIDOR] Split: {split:<5} | Images: {len(self.df)}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        path = row["path"]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)

        img_id = os.path.basename(path).split(".")[0]
        label = int(self.label_map.get(img_id, 0))
        return img, label, path


class MessidorModule(pl.LightningDataModule):
    def __init__(self, root, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, transform=None):
        super().__init__()
        self.root = root
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.transform = transform

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            self.train_ds = MessidorDataset(self.root, split="train", transform=self.transform)
            self.val_ds = MessidorDataset(self.root, split="val", transform=self.transform)
        if stage == "test":
            self.test_ds = MessidorDataset(self.root, split="test", transform=self.transform)
        if stage == "full":
            self.val_ds = MessidorDataset(self.root, split="full", transform=self.transform)

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_ds, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    def full_dataloader(self):
        ds = MessidorDataset(self.root, split="full", transform=self.transform)
        return DataLoader(ds, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)


def compute_messidor_class_weights(root):
    csv_path = os.path.join(_find_root_with_paths(root, ["messidor_data.csv", "messidor-2"]), "messidor_data.csv")
    if not os.path.exists(csv_path):
        print(f"Warning: Messidor CSV not found at {csv_path}. Returning default weights.")
        return torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0])

    df = pd.read_csv(csv_path)
    if "diagnosis" not in df.columns:
        print(f"Warning: Messidor CSV lacks diagnosis column. Returning default weights.")
        return torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0])

    if "adjudicated_gradable" in df.columns:
        df = df[df["adjudicated_gradable"] == 1]

    labels = df["diagnosis"].astype(int).values
    classes = sorted(set(labels))
    from sklearn.utils.class_weight import compute_class_weight

    weights = compute_class_weight(class_weight="balanced", classes=classes, y=labels)
    return torch.tensor(weights, dtype=torch.float)
