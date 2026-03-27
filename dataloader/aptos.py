# dataloader/aptos.py

import os
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from config.constants import BATCH_SIZE, NUM_WORKERS, USE_MIXUP, MIXUP_ALPHA, USE_MOSAIC, MOSAIC_PROB, USE_COPY_PASTE, COPY_PASTE_PROB
import random
from utils.transforms import preprocess_image, mixup_data, mosaic_data, copy_paste_data

class APTOSDataset(Dataset):
    def __init__(self, root, split="train", transform=None,
                 use_mixup=USE_MIXUP, mixup_alpha=MIXUP_ALPHA,
                 use_mosaic=USE_MOSAIC, mosaic_prob=MOSAIC_PROB,
                 use_copy_paste=USE_COPY_PASTE, copy_paste_prob=COPY_PASTE_PROB):
        self.root = root
        self.split = split
        self.transform = transform
        self.use_mixup = use_mixup
        self.mixup_alpha = mixup_alpha
        self.use_mosaic = use_mosaic
        self.mosaic_prob = mosaic_prob
        self.use_copy_paste = use_copy_paste
        self.copy_paste_prob = copy_paste_prob
        
        # Define internal mapping for standard splits
        split_map = {
            "train": ("train_images/train_images", "train_1.csv"),
            "val":   ("val_images/val_images", "valid.csv"),
            "test":  ("test_images/test_images", "test.csv")
        }

        if split == "full":
            # Combine all known sub-directories and CSVs
            self.img_paths = []
            all_dfs = []
            for sub_dir, csv_name in split_map.values():
                d = os.path.join(root, sub_dir)
                c = os.path.join(root, csv_name)
                if os.path.exists(d):
                    self.img_paths.extend([os.path.join(d, f) for f in os.listdir(d) if f.lower().endswith(('.png', '.jpg'))])
                if os.path.exists(c):
                    all_dfs.append(pd.read_csv(c))
            df = pd.concat(all_dfs, ignore_index=True)
        else:
            sub_dir, csv_name = split_map[split]
            img_dir = os.path.join(root, sub_dir)
            self.img_paths = sorted([os.path.join(img_dir, f) for f in os.listdir(img_dir) if f.lower().endswith(('.png', '.jpg'))])
            df = pd.read_csv(os.path.join(root, csv_name))

        # Cleanup ID codes and create label map
        df["id_code"] = df["id_code"].str.replace(".png", "", regex=False).str.replace(".jpg", "", regex=False)
        self.label_map = dict(zip(df["id_code"], df["diagnosis"]))
        print(f"[APTOS] Split: {split:<5} | Images: {len(self.img_paths)}")

    def __len__(self):
        return len(self.img_paths)

    def _load_single(self, idx):
        path = self.img_paths[idx]
        img = preprocess_image(path)
        if self.transform:
            img = self.transform(img)
        img_id = os.path.basename(path).split(".")[0]
        label = int(self.label_map.get(img_id, 0))
        return img, label, path

    def __getitem__(self, idx):
        if self.use_mosaic and random.random() < self.mosaic_prob and len(self.img_paths) >= 4:
            idxs = random.sample(range(len(self.img_paths)), 4)
            imgs, labels, paths = [], [], []
            for j in idxs:
                img_j, lbl_j, path_j = self._load_single(j)
                imgs.append(img_j); labels.append(lbl_j); paths.append(path_j)
            img_m, lbl_m = mosaic_data(imgs, labels)
            return img_m, lbl_m, paths

        if self.use_copy_paste and random.random() < self.copy_paste_prob:
            idx2 = random.randint(0, len(self.img_paths)-1)
            img1, lbl1, path1 = self._load_single(idx)
            img2, lbl2, _ = self._load_single(idx2)
            img_cp, lbl_cp = copy_paste_data(img1, img2, lbl1, lbl2)
            return img_cp, lbl_cp, path1

        if self.use_mixup:
            idx2 = random.randint(0, len(self.img_paths)-1)
            img1, lbl1, path1 = self._load_single(idx)
            img2, lbl2, _ = self._load_single(idx2)
            img_mx, lbl_mx, _ = mixup_data(img1, img2, lbl1, lbl2, self.mixup_alpha)
            return img_mx, lbl_mx, path1

        return self._load_single(idx)

class APTOSModule(pl.LightningDataModule):
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
        # stage can be 'fit', 'validate', 'test', or 'predict'
        if stage == "fit" or stage is None:
            self.train_ds = APTOSDataset(
                self.root, split="train", transform=self.transform,
                use_mixup=self.use_mixup, mixup_alpha=self.mixup_alpha,
                use_mosaic=self.use_mosaic, mosaic_prob=self.mosaic_prob,
                use_copy_paste=self.use_copy_paste, copy_paste_prob=self.copy_paste_prob,
            )
            self.val_ds = APTOSDataset(
                self.root, split="val", transform=self.transform,
                use_mixup=self.use_mixup, mixup_alpha=self.mixup_alpha,
                use_mosaic=self.use_mosaic, mosaic_prob=self.mosaic_prob,
                use_copy_paste=self.use_copy_paste, copy_paste_prob=self.copy_paste_prob,
            )
        if stage == "test":
            self.test_ds = APTOSDataset(
                self.root, split="test", transform=self.transform,
                use_mixup=self.use_mixup, mixup_alpha=self.mixup_alpha,
                use_mosaic=self.use_mosaic, mosaic_prob=self.mosaic_prob,
                use_copy_paste=self.use_copy_paste, copy_paste_prob=self.copy_paste_prob,
            )
        if stage == "full":
            self.val_ds = APTOSDataset(
                self.root, split="full", transform=self.transform,
                use_mixup=self.use_mixup, mixup_alpha=self.mixup_alpha,
                use_mosaic=self.use_mosaic, mosaic_prob=self.mosaic_prob,
                use_copy_paste=self.use_copy_paste, copy_paste_prob=self.copy_paste_prob,
            )

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_ds, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
    
    def full_dataloader(self):
        # Useful for extracting features from the entire dataset
        ds = APTOSDataset(self.root, split="full", transform=self.transform)
        return DataLoader(ds, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)