from dataloader.mbrset import MBRSETModule
import torch
import pytorch_lightning as pl
from torchvision import transforms
from pytorch_lightning.loggers import WandbLogger

from config.constants import *
from train.train_retfound import RETFoundTask
from dataloader.idrid import IDRiDModule
from dataloader.aptos import APTOSModule
from utils.flops import compute_flops
from eval.shared_eval import EvalWrapper
from utils.transforms import eval_transform

def run_eval_retfound(args):
    print(f"\n=== RETFOUND EVALUATION: {args.load_model} ===")

    # 1. Load the Task (handles Backbone + Head automatically)
    model = RETFoundTask.load_from_checkpoint(
        args.load_model,
        strict=False,           
        mode="finetune",
        class_weights=None
    )
    
    # 2. Setup Data
    tfm = eval_transform(IMG_SIZE)
    
    if args.dataset == "idrid":
        dm = IDRiDModule(root=IDRID_PATH, transform=tfm, batch_size=BATCH_SIZE)
    elif args.dataset == "aptos":
        dm = APTOSModule(root=APTOS_PATH, transform=tfm, batch_size=BATCH_SIZE)
    elif args.dataset == "mbrset":
        dm = MBRSETModule(root=MBRSET_PATH, transform=tfm, batch_size=BATCH_SIZE)
    dm.setup(stage="full")

    # 3. Evaluate
    wrapper = EvalWrapper(model)
    trainer = pl.Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        precision="16-mixed",
        logger=WandbLogger(project="retfound_eval")
    )
    
    val_results = trainer.validate(wrapper, dm)
    val_metrics = val_results[0] if len(val_results) > 0 else {}

    # -------------------------------------------------
    # 7. FLOPs
    # -------------------------------------------------
    flops, _ = compute_flops(model, IMG_SIZE)
    print(f"Total Complexity: {flops/1e9:.3f} GFLOPs")
    
    import csv
    import os
    import time

    row = {
        "model": "retfound_idrid",
        "mode": "eval",
        "mask_ratio": "unmasked",
        "dataset": args.dataset,
        "model_path": args.load_model,
        "seed": args.seed,
        **val_metrics
    }

    csv_path = "eval_results.csv"
    file_exists = os.path.isfile(csv_path)

    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())

        if not file_exists:
            writer.writeheader()

        writer.writerow(row)