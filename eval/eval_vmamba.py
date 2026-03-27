from dataloader.mbrset import MBRSETModule
import torch
import torch.nn as nn
import pytorch_lightning as pl
from torchvision import transforms
from pytorch_lightning.loggers import WandbLogger

from utils.transforms import eval_transform
from config.constants import *
from models.vmamba_backbone import VisualMamba
from dataloader.idrid import IDRiDModule
from timm import create_model
from dataloader.aptos import APTOSModule
from utils.flops import compute_flops
from eval.shared_eval import EvalWrapper

def run_evaluation(args):
    print(f"\n=== VMAMBA EVALUATION: {args.load_model} ===")

    # -------------------------------------------------
    # 1. Reconstruct Model Structure
    # -------------------------------------------------
    student_model = getattr(args, 'student_model', STUDENT_MODEL)
    if student_model == 'vmamba':
        backbone = VisualMamba(
            img_size=IMG_SIZE,
            patch_size=PATCH_SIZE,
            in_chans=IN_CHANS,
            embed_dim=VMAMBA_EMBED_DIM,
            depth=VMAMBA_DEPTH,
            learning_rate=0.0,
            mask_ratio=0.0,
            use_cls_token=False
        )
        backbone_embed_dim = backbone.embed_dim
    elif student_model == 'tinyvit':
        backbone = create_model('tinyvit_tiny_patch16_224', pretrained=False, num_classes=0)
        backbone_embed_dim = backbone.num_features
    elif student_model == 'mobilenet_v3_small':
        backbone = create_model('mobilenetv3_small_100', pretrained=False, num_classes=0)
        backbone_embed_dim = backbone.num_features
    elif student_model == 'efficientnet_b0':
        backbone = create_model('efficientnet_b0', pretrained=False, num_classes=0)
        backbone_embed_dim = backbone.num_features
    else:
        raise ValueError(f"Unknown student_model: {student_model}")

    head = nn.Sequential(
        nn.Linear(backbone_embed_dim, 512),
        nn.BatchNorm1d(512),
        nn.GELU(),
        nn.Dropout(0.3),
        nn.Linear(512, 128),
        nn.BatchNorm1d(128),
        nn.GELU(),
        nn.Dropout(0.2),
        nn.Linear(128, NUM_CLASSES)
    )

    # -------------------------------------------------
    # 2. Load Lightning Checkpoint
    # -------------------------------------------------
    ckpt = torch.load(args.load_model, map_location="cpu")
    state_dict = ckpt.get("state_dict", ckpt)

    new_state = {}

    for k, v in state_dict.items():
        if not k.startswith("backbone."):
            continue

        # case 1: backbone.backbone.X -> backbone.X
        if k.startswith("backbone.backbone."):
            new_key = k.replace("backbone.backbone.", "backbone.", 1)

        # case 2: backbone.pos_embed / backbone.patch_embed / backbone.norm
        elif k.startswith("backbone.pos_embed") \
        or k.startswith("backbone.mask_token") \
        or k.startswith("backbone.patch_embed") \
        or k.startswith("backbone.norm"):
            new_key = k.replace("backbone.", "", 1)

        # case 3: normal backbone layers (backbone.0., backbone.1., etc.)
        else:
            new_key = k

        new_state[new_key] = v

    backbone.load_state_dict(new_state, strict=True)

    # head (unchanged)
    head_state = {
        k.replace("head.", ""): v
        for k, v in state_dict.items()
        if k.startswith("head.")
    }
    if head_state:
        head.load_state_dict(head_state, strict=True)
    else:
        print("[!] No head weights found in checkpoint (using random head)")
    # -------------------------------------------------
    # 3. Combine into single inference module
    # -------------------------------------------------
    class VMambaInference(nn.Module):
        def __init__(self, b, h):
            super().__init__()
            self.backbone = b
            self.head = h

        def forward(self, x):
            feats = self.backbone.forward_features(x)
            return self.head(feats)

    model = VMambaInference(backbone, head)

    # -------------------------------------------------
    # 4. Data
    # -------------------------------------------------
    tfm = eval_transform(IMG_SIZE)

    if args.dataset == "idrid":
        dm = IDRiDModule(
            root=IDRID_PATH,
            transform=tfm,
            batch_size=BATCH_SIZE,
            use_mixup=USE_MIXUP,
            mixup_alpha=MIXUP_ALPHA,
            use_mosaic=USE_MOSAIC,
            mosaic_prob=MOSAIC_PROB,
            use_copy_paste=USE_COPY_PASTE,
            copy_paste_prob=COPY_PASTE_PROB,
        )
    elif args.dataset == "aptos":
        dm = APTOSModule(
            root=APTOS_PATH,
            transform=tfm,
            batch_size=BATCH_SIZE,
            use_mixup=USE_MIXUP,
            mixup_alpha=MIXUP_ALPHA,
            use_mosaic=USE_MOSAIC,
            mosaic_prob=MOSAIC_PROB,
            use_copy_paste=USE_COPY_PASTE,
            copy_paste_prob=COPY_PASTE_PROB,
        )
    elif args.dataset == "mbrset":
        dm = MBRSETModule(
            root=MBRSET_PATH,
            transform=tfm,
            batch_size=BATCH_SIZE,
            use_mixup=USE_MIXUP,
            mixup_alpha=MIXUP_ALPHA,
            use_mosaic=USE_MOSAIC,
            mosaic_prob=MOSAIC_PROB,
            use_copy_paste=USE_COPY_PASTE,
            copy_paste_prob=COPY_PASTE_PROB,
        )
    dm.setup(stage="full")

    # -------------------------------------------------
    # 5. Trainer
    # -------------------------------------------------
    trainer = pl.Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        logger=WandbLogger(project="vmamba_eval"),
        precision="16-mixed"
    )
    
    # -------------------------------------------------
    # 6. Run Evaluation
    # -------------------------------------------------
    wrapper = EvalWrapper(model)
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
        "model": f"{student_model}_{args.dataset}",
        "mode": "eval",
        "mask_ratio": "unmasked",
        "dataset": args.dataset,
        "student_model": student_model,
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
