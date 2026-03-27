# dataloader/idrid.py

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

class IDRiDDataset(Dataset):
    def __init__(self, root, split="train", transform=None,
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
        
        # IDRiD Structure Mapping
        # Note: We treat 'test' as 'val' for internal validation during training
        paths = {
            "train": ("1. Original Images/a. Training Set", "2. Groundtruths/a. IDRiD_Disease Grading_Training Labels.csv"),
            "test":  ("1. Original Images/b. Testing Set",  "2. Groundtruths/b. IDRiD_Disease Grading_Testing Labels.csv"),
        }
        paths["val"] = paths["test"] # Map val to test set

        if split == "full":
            self.img_paths = []
            all_dfs = []
            for s in ["train", "test"]:
                img_sub, csv_sub = paths[s]
                d, c = os.path.join(root, img_sub), os.path.join(root, csv_sub)
                if os.path.exists(d):
                    self.img_paths.extend([os.path.join(d, f) for f in os.listdir(d) if f.lower().endswith(".jpg")])
                if os.path.exists(c):
                    all_dfs.append(pd.read_csv(c))
            df = pd.concat(all_dfs, ignore_index=True)
        else:
            img_sub, csv_sub = paths[split]
            img_dir = os.path.join(root, img_sub)
            self.img_paths = sorted([os.path.join(img_dir, f) for f in os.listdir(img_dir) if f.lower().endswith(".jpg")])
            df = pd.read_csv(os.path.join(root, csv_sub))

        # Cleanup ID codes (Image name vs Retinopathy grade)
        df["Image name"] = df["Image name"].str.replace(".jpg", "", regex=False)
        self.label_map = dict(zip(df["Image name"], df["Retinopathy grade"]))
        print(f"[IDRID] Split: {split:<5} | Images: {len(self.img_paths)}")

    def __len__(self):
        return len(self.img_paths)

    def _load_single(self, idx):
        path = self.img_paths[idx]
        img = preprocess_image(path)
        if self.transform:
            img = self.transform(img)
        img_id = os.path.basename(path).replace(".jpg", "")
        label = int(self.label_map.get(img_id, 0))
        return img, label, path

    def __getitem__(self, idx):
        if self.use_mosaic and random.random() < self.mosaic_prob and len(self.img_paths) >= 4:
            idxs = random.sample(range(len(self.img_paths)), 4)
            imgs = []
            labels = []
            paths = []
            for j in idxs:
                img_j, lbl_j, path_j = self._load_single(j)
                imgs.append(img_j)
                labels.append(lbl_j)
                paths.append(path_j)
            img_m, lbl_m = mosaic_data(imgs, labels)
            return img_m, lbl_m, paths

        if self.use_copy_paste and random.random() < self.copy_paste_prob:
            idx2 = random.randint(0, len(self.img_paths) - 1)
            img1, lbl1, path1 = self._load_single(idx)
            img2, lbl2, path2 = self._load_single(idx2)
            img_cp, lbl_cp = copy_paste_data(img1, img2, lbl1, lbl2)
            return img_cp, lbl_cp, path1

        if self.use_mixup and random.random() < 1.0:
            idx2 = random.randint(0, len(self.img_paths) - 1)
            img1, lbl1, path1 = self._load_single(idx)
            img2, lbl2, path2 = self._load_single(idx2)
            img_mx, lbl_mx, _ = mixup_data(img1, img2, lbl1, lbl2, self.mixup_alpha)
            return img_mx, lbl_mx, path1

        return self._load_single(idx)


class IDRiDModule(pl.LightningDataModule):
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
            self.train_ds = IDRiDDataset(
                self.root, split="train", transform=self.transform,
                use_mixup=self.use_mixup, mixup_alpha=self.mixup_alpha,
                use_mosaic=self.use_mosaic, mosaic_prob=self.mosaic_prob,
                use_copy_paste=self.use_copy_paste, copy_paste_prob=self.copy_paste_prob,
            )
            self.val_ds = IDRiDDataset(
                self.root, split="test", transform=self.transform,
                use_mixup=self.use_mixup, mixup_alpha=self.mixup_alpha,
                use_mosaic=self.use_mosaic, mosaic_prob=self.mosaic_prob,
                use_copy_paste=self.use_copy_paste, copy_paste_prob=self.copy_paste_prob,
            )
        if stage == "test":
            self.test_ds = IDRiDDataset(
                self.root, split="test", transform=self.transform,
                use_mixup=self.use_mixup, mixup_alpha=self.mixup_alpha,
                use_mosaic=self.use_mosaic, mosaic_prob=self.mosaic_prob,
                use_copy_paste=self.use_copy_paste, copy_paste_prob=self.copy_paste_prob,
            )
        if stage == "full":
            self.val_ds = IDRiDDataset(
                self.root, split="full", transform=self.transform,
                use_mixup=self.use_mixup, mixup_alpha=self.mixup_alpha,
                use_mosaic=self.use_mosaic, mosaic_prob=self.mosaic_prob,
                use_copy_paste=self.use_copy_paste, copy_paste_prob=self.copy_paste_prob,
            )

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_ds, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
    
def compute_idrid_class_weights(root):
    """
    Computes class weights for the training set to handle class imbalance.
    """
    csv_path = os.path.join(root, "2. Groundtruths/a. IDRiD_Disease Grading_Training Labels.csv")
    if not os.path.exists(csv_path):
        print(f"Warning: CSV not found at {csv_path}. Returning default weights.")
        return torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0])

    df = pd.read_csv(csv_path)
    labels = df["Retinopathy grade"].values
    classes = np.unique(labels)
    
    # Calculate weights: weight = total_samples / (n_classes * samples_per_class)
    weights = compute_class_weight(class_weight='balanced', classes=classes, y=labels)
    
    return torch.tensor(weights, dtype=torch.float)