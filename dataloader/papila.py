# dataloader/papila.py

import os
import re
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from config.constants import BATCH_SIZE, NUM_WORKERS, NUM_CLASSES

try:
    import pandas as pd
except ImportError:
    pd = None


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


def _normalize_id(raw_id):
    if raw_id is None:
        return None
    text = str(raw_id).strip()
    if text == "ID":
        return None
    text = text.lstrip("#")
    match = re.match(r"^(\d+)", text)
    return match.group(1).zfill(3) if match else None


def _load_clinical_data(path):
    if pd is not None:
        try:
            # The PAPILA clinical tables use multi-row headers.
            df = pd.read_excel(path, header=[0, 1], engine="openpyxl")
            columns = []
            for first, second in df.columns:
                if second is not None and str(second).strip():
                    columns.append(str(second).strip())
                elif first is not None:
                    columns.append(str(first).strip())
                else:
                    columns.append("")
            df.columns = columns
            df = df.loc[3:].reset_index(drop=True)
            df = df[df.iloc[:, 0].notna()]
            records = []
            for _, row in df.iterrows():
                values = {col: row[col] for col in df.columns if col and not pd.isna(row[col])}
                if values:
                    records.append(values)
            return records
        except Exception:
            pass

    # Fallback: parse the XLSX package manually using XML.
    import zipfile
    import xml.etree.ElementTree as ET

    with zipfile.ZipFile(path) as zf:
        shared_strings = [None]
        if "xl/sharedStrings.xml" in zf.namelist():
            shared_root = ET.fromstring(zf.read("xl/sharedStrings.xml"))
            ns = {"ns": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
            for si in shared_root.findall("ns:si", ns):
                texts = "".join(t.text or "" for t in si.findall(".//ns:t", ns))
                shared_strings.append(texts)

        sheet = ET.fromstring(zf.read("xl/worksheets/sheet1.xml"))
        ns = {"ns": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
        rows = sheet.findall(".//ns:row", ns)

        header_rows = []
        for r in rows[:2]:
            row_values = []
            current_col = 1
            for c in r.findall("ns:c", ns):
                ref = c.attrib.get("r")
                col = re.sub(r"\d+", "", ref)
                while current_col < (ord(col[0]) - ord("A") + 1):
                    row_values.append(None)
                    current_col += 1
                cell_type = c.attrib.get("t")
                v = c.find("ns:v", ns)
                value = v.text if v is not None else None
                if cell_type == "s" and value is not None:
                    value = shared_strings[int(value)]
                row_values.append(value)
                current_col += 1
            header_rows.append(row_values)

        if len(header_rows) < 2:
            return []

        max_len = max(len(header_rows[0]), len(header_rows[1]))
        header1 = header_rows[0] + [None] * (max_len - len(header_rows[0]))
        header2 = header_rows[1] + [None] * (max_len - len(header_rows[1]))
        columns = []
        for first, second in zip(header1, header2):
            if second is not None and str(second).strip():
                columns.append(str(second).strip())
            elif first is not None:
                columns.append(str(first).strip())
            else:
                columns.append("")

        records = []
        for r in rows[3:]:
            row_values = []
            current_col = 1
            for c in r.findall("ns:c", ns):
                ref = c.attrib.get("r")
                col = re.sub(r"\d+", "", ref)
                while current_col < (ord(col[0]) - ord("A") + 1):
                    row_values.append(None)
                    current_col += 1
                cell_type = c.attrib.get("t")
                v = c.find("ns:v", ns)
                value = v.text if v is not None else None
                if cell_type == "s" and value is not None:
                    value = shared_strings[int(value)]
                row_values.append(value)
                current_col += 1
            row_values += [None] * (len(columns) - len(row_values))
            values = {col: row_values[i] for i, col in enumerate(columns) if col and row_values[i] is not None}
            if values:
                records.append(values)
        return records


class PAPILADataset(Dataset):
    def __init__(self, root, split="train", transform=None, val_split=0.1, label_col=None):
        self.root = _find_root_with_paths(root, ["ClinicalData", "FundusImages"])
        self.transform = transform
        self.split = split
        self.val_split = val_split
        self.label_col = label_col

        self.img_dir = os.path.join(self.root, "FundusImages")
        if not os.path.isdir(self.img_dir):
            raise FileNotFoundError(f"PAPILA image directory not found at {self.img_dir}")

        clinical_dir = os.path.join(self.root, "ClinicalData")
        if not os.path.isdir(clinical_dir):
            raise FileNotFoundError(f"PAPILA clinical data directory not found at {clinical_dir}")

        records = []
        for side in ["OD", "OS"]:
            path = os.path.join(clinical_dir, f"patient_data_{side.lower()}.xlsx")
            if os.path.exists(path):
                for record in _load_clinical_data(path):
                    record["_side"] = side
                    records.append(record)

        if not records:
            raise FileNotFoundError("No PAPILA clinical records could be loaded.")

        samples = []
        label_map = {}
        for record in records:
            raw_id = record.get("ID")
            image_id = _normalize_id(raw_id)
            if image_id is None:
                continue
            side = record.get("_side", "OD")
            filename = f"RET{image_id}{side}.jpg"
            path = os.path.join(self.img_dir, filename)
            if not os.path.exists(path):
                continue

            label = 0
            if self.label_col and self.label_col in record:
                raw_label = record[self.label_col]
                if raw_label is not None and str(raw_label).strip() != "":
                    try:
                        label = int(raw_label)
                    except (ValueError, TypeError):
                        try:
                            label = int(float(raw_label))
                        except Exception:
                            label = 0

            samples.append({"path": path, "label": label, "id": filename})
            label_map[filename] = label

        if not samples:
            # fallback: include all images with default label 0
            images = sorted([os.path.join(self.img_dir, x) for x in os.listdir(self.img_dir) if x.lower().endswith((".jpg", ".jpeg", ".png"))])
            samples = [{"path": p, "label": 0, "id": os.path.basename(p)} for p in images]
            print("[PAPILA] Warning: no clinical labels were found. Falling back to unlabeled image loading.")

        samples = sorted(samples, key=lambda x: x["id"])
        n = len(samples)
        val_count = max(1, int(n * self.val_split)) if n > 1 else 0
        if split == "train":
            samples = samples[:-val_count] if val_count else samples
        elif split in ["val", "test"]:
            samples = samples[-val_count:] if val_count else samples
        elif split != "full":
            raise ValueError(f"Unsupported split: {split}")

        self.samples = samples
        self.label_map = {s["id"]: s["label"] for s in self.samples}
        print(f"[PAPILA] Split: {split:<5} | Images: {len(self.samples)} | Labeled: {len(label_map)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        path = sample["path"]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)

        label = int(self.label_map.get(sample["id"], 0))
        return img, label, path


class PAPILAModule(pl.LightningDataModule):
    def __init__(self, root, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, transform=None, label_col=None):
        super().__init__()
        self.root = root
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.transform = transform
        self.label_col = label_col

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            self.train_ds = PAPILADataset(self.root, split="train", transform=self.transform, label_col=self.label_col)
            self.val_ds = PAPILADataset(self.root, split="val", transform=self.transform, label_col=self.label_col)
        if stage == "test":
            self.test_ds = PAPILADataset(self.root, split="test", transform=self.transform, label_col=self.label_col)
        if stage == "full":
            self.val_ds = PAPILADataset(self.root, split="full", transform=self.transform, label_col=self.label_col)

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_ds, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    def full_dataloader(self):
        ds = PAPILADataset(self.root, split="full", transform=self.transform, label_col=self.label_col)
        return DataLoader(ds, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)


def compute_papila_class_weights(root):
    # PAPILA does not expose an explicit classification label in the shipped clinical tables.
    print("[PAPILA] No explicit class labels available; returning uniform weights.")
    return torch.ones(NUM_CLASSES, dtype=torch.float)
