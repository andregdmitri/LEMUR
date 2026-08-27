# dataloader/mured.py

import os
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from config.constants import BATCH_SIZE, NUM_WORKERS

class MUREDDataset(Dataset):
    def __init__(self, root, split="train", transform=None):
        self.root = root
        self.split = split
        self.transform = transform
        
        # MURED's 20 multi-label classes
        self.classes = ['DR', 'NORMAL', 'MH', 'ODC', 'TSLN', 'ARMD', 'DN', 'MYA', 'BRVO', 
                        'ODP', 'CRVO', 'CNV', 'RS', 'ODE', 'LS', 'CSR', 'HTR', 'ASR', 'CRS', 'OTHER']
        
        # MURED uses a single directory for all images, split entirely via CSVs
        split_map = {
            "train": ("images/images", "train_data.csv"),
            "val":   ("images/images", "val_data.csv"), # Assumes standard val_data.csv exists
            "test":  ("images/images", "test_data.csv")
        }

        img_dir = os.path.join(root, "images/images")
        
        # Build an extension-agnostic mapping of available files
        # Resolves MURED's mix of .tif, .png, .jpg without hardcoding extensions in the CSV parser
        if os.path.exists(img_dir):
            available_images = [f for f in os.listdir(img_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff'))]
            self.base_to_path = {os.path.splitext(f)[0]: os.path.join(img_dir, f) for f in available_images}
        else:
            self.base_to_path = {}

        # Load appropriate CSV(s)
        if split == "full":
            all_dfs = []
            for _, csv_name in split_map.values():
                c = os.path.join(root, csv_name)
                if os.path.exists(c):
                    all_dfs.append(pd.read_csv(c))
            df = pd.concat(all_dfs, ignore_index=True) if all_dfs else pd.DataFrame()
        else:
            _, csv_name = split_map[split]
            csv_path = os.path.join(root, csv_name)
            df = pd.read_csv(csv_path) if os.path.exists(csv_path) else pd.DataFrame()

        if not df.empty:
            # Strip extensions from the CSV 'ID' column to match the dictionary keys
            df["ID_clean"] = df["ID"].astype(str).apply(lambda x: os.path.splitext(x)[0])
            
            # Intersection: Keep only records where the physical image file actually exists
            df = df[df["ID_clean"].isin(self.base_to_path.keys())].reset_index(drop=True)
            
            self.ids = df["ID_clean"].values
            self.labels = df[self.classes].values.astype(float)
        else:
            self.ids = []
            self.labels = []
            
        print(f"[MURED] Split: {split:<5} | Images: {len(self.ids)}")

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        img_id = self.ids[idx]
        path = self.base_to_path[img_id]
        
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        
        # Multi-label float tensor (shape: [20])
        label = torch.tensor(self.labels[idx], dtype=torch.float32) 
        return img, label, path


class MUREDModule(pl.LightningDataModule):
    def __init__(self, root, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, transform=None):
        super().__init__()
        self.root = root
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.transform = transform

        if isinstance(transform, (list, tuple)) and len(transform) == 2:
            self.train_transform, self.val_transform = transform
        else:
            self.train_transform = self.val_transform = transform

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            self.train_ds = MUREDDataset(self.root, split="train", transform=self.train_transform)
            self.val_ds   = MUREDDataset(self.root, split="val",   transform=self.val_transform)
        if stage == "test":
            self.test_ds  = MUREDDataset(self.root, split="test",  transform=self.val_transform)
        if stage == "full":
            self.val_ds   = MUREDDataset(self.root, split="full",  transform=self.val_transform)

    def train_dataloader(self):
        loader_kwargs = dict(batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, pin_memory=torch.cuda.is_available())
        if self.num_workers > 0:
            loader_kwargs.update({"persistent_workers": True, "prefetch_factor": 2})
        return DataLoader(self.train_ds, **loader_kwargs)

    def val_dataloader(self):
        loader_kwargs = dict(batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=torch.cuda.is_available())
        if self.num_workers > 0:
            loader_kwargs.update({"persistent_workers": True, "prefetch_factor": 2})
        return DataLoader(self.val_ds, **loader_kwargs)

    def test_dataloader(self):
        loader_kwargs = dict(batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=torch.cuda.is_available())
        if self.num_workers > 0:
            loader_kwargs.update({"persistent_workers": True, "prefetch_factor": 2})
        return DataLoader(self.test_ds, **loader_kwargs)
    
    def full_dataloader(self):
        ds = MUREDDataset(self.root, split="full", transform=self.transform)
        loader_kwargs = dict(batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=torch.cuda.is_available())
        if self.num_workers > 0:
            loader_kwargs.update({"persistent_workers": True, "prefetch_factor": 2})
        return DataLoader(ds, **loader_kwargs)