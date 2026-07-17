import os
import time
import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import WandbLogger

from utils.transforms import train_transform_default
from config.constants import *
from dataloader import get_dataloader, get_class_weights
from optimizers.optimizer import warmup_cosine_optimizer
from models.mobilenet import MobileNetClassifier
from models.efficientnet import EfficientNetClassifier
from models.unet import UNetClassifier
from models.vmamba import VMambaClassifier


MODEL_REGISTRY = {
    "mobilenet": MobileNetClassifier,
    "efficientnet": EfficientNetClassifier,
    "unet": UNetClassifier,
    "vmamba": VMambaClassifier,
}


def run_light_model(args):
    seed = pl.seed_everything(args.seed or SEED)

    transform = train_transform_default(IMG_SIZE)
    datamodule = get_dataloader(args.dataset, transform, batch_size=BATCH_SIZE)
    class_weights = get_class_weights(args.dataset, compute_weights=True)
    datamodule.setup(stage="fit")

    if args.model not in MODEL_REGISTRY:
        raise ValueError(f"Unsupported model: {args.model}. Choose from {list(MODEL_REGISTRY)}")

    model_cls = MODEL_REGISTRY[args.model]
    model = model_cls(
        num_classes=NUM_CLASSES,
        pretrained=args.pretrained,
        lr=args.lr or LR,
        class_weights=class_weights,
    )

    ckpt_cb = ModelCheckpoint(monitor="val/f1", mode="max", save_top_k=1, filename=f"{args.model}_best")
    early_cb = EarlyStopping(monitor="val/f1", patience=PATIENCE, mode="max")

    run_name = f"{args.model}_{args.dataset}_{seed}_{int(time.time())}"
    logger = WandbLogger(project="light_models_retina", name=run_name)

    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        precision="16-mixed" if torch.cuda.is_available() else 32,
        callbacks=[ckpt_cb, early_cb],
        logger=logger,
        log_every_n_steps=10,
    )

    trainer.fit(model, datamodule)
    print(f"[✓] Training complete: {ckpt_cb.best_model_path}")
    return ckpt_cb.best_model_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train lightweight retina fundus classifiers")
    parser.add_argument("--model", type=str, required=True, choices=list(MODEL_REGISTRY.keys()))
    parser.add_argument("--dataset", type=str, required=True, choices=["idrid", "aptos", "messidor", "papila"])
    parser.add_argument("--epochs", type=int, default=HEAD_EPOCHS)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--pretrained", action="store_true", help="Use ImageNet pretrained backbone if available")
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    run_light_model(args)
