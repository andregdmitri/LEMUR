import os
import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import WandbLogger
from torchmetrics import Accuracy, AUROC, F1Score, AveragePrecision
from torchvision import transforms
from utils.transforms import train_transform_retfound_linear, train_transform_default

from config.constants import *
from models.retfound import RETFoundBackbone
from dataloader.idrid import IDRiDModule, compute_idrid_class_weights
from dataloader.aptos import APTOSModule
from optimizers.optimizer import warmup_cosine_optimizer

# -----------------------------------------------------------
#  Module: RETFoundTask
# -----------------------------------------------------------

class RETFoundTask(pl.LightningModule):
    def __init__(self, mode, lr, checkpoint_path, class_weights):
        super().__init__()
        self.save_hyperparameters(ignore=["class_weights"])
        self.mode = mode
        self._final_eval = False
        
        # 1. Load the RETFound Model
        # We use the existing RETFoundBackbone to get the ViT backbone
        full_model = RETFoundBackbone(
            num_classes=NUM_CLASSES,
            checkpoint_path=checkpoint_path or os.path.join(CHECKPOINT_DIR, "RETFound_cfp_weights.pth")
        )
        self.backbone = full_model.model
        self.embed_dim = self.backbone.head.weight.shape[1]
        
        # 2. Re-initialize Head
        self.head = nn.Linear(self.embed_dim, NUM_CLASSES)
        
        # 3. Handle Freezing logic
        if mode == "linear":
            for param in self.backbone.parameters():
                param.requires_grad = False
        else: # finetune
            for param in self.backbone.parameters():
                param.requires_grad = True

        # 4. Loss & Metrics
        self.loss_fn = nn.CrossEntropyLoss(weight=class_weights) if class_weights is not None else nn.CrossEntropyLoss()
        
        metric_args = {"task": "multiclass", "num_classes": NUM_CLASSES}
        self.train_metrics = nn.ModuleDict({
            "acc": Accuracy(**metric_args),
            "f1": F1Score(**metric_args, average="macro"),
            "auroc": AUROC(**metric_args),
            "aupr": AveragePrecision(**metric_args),
        })
        self.val_metrics = nn.ModuleDict({
            "acc": Accuracy(**metric_args),
            "f1": F1Score(**metric_args, average="macro"),
            "auroc": AUROC(**metric_args),
            "aupr": AveragePrecision(**metric_args),
        })

    def forward(self, x):
        if self.mode == "linear" and not self.training:
            with torch.no_grad():
                feats = self.backbone.forward_features(x)
        else:
            feats = self.backbone.forward_features(x)
        return self.head(feats)

    def shared_step(self, batch, stage):
        x, y, _ = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)

        probs = torch.softmax(logits, dim=1)
        preds = torch.argmax(probs, dim=1)

        metrics = self.train_metrics if stage == "train" else self.val_metrics

        metrics["acc"].update(preds, y)
        metrics["f1"].update(preds, y)
        try:
            metrics["auroc"].update(probs, y)
            metrics["aupr"].update(probs, y)
        except ValueError:
            pass

        self.log(f"{stage}/loss", loss, prog_bar=True, on_epoch=True, batch_size=x.size(0))
        return loss

    def training_step(self, batch, batch_idx):
        return self.shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self.shared_step(batch, "val")

    def test_step(self, batch, batch_idx):
        return self.shared_step(batch, "train")

    def on_train_epoch_end(self):
        self._log_metrics("train")

    def on_validation_epoch_end(self):
        self._log_metrics("val")

    def _log_metrics(self, stage):
        metrics = self.train_metrics if stage == "train" else self.val_metrics
        for name, m in metrics.items():
            try:
                self.log(f"{stage}/{name}", m.compute(), prog_bar=(name == "f1"))
            except:
                pass
            if not self._final_eval:
                m.reset()

    def configure_optimizers(self):
        params = list(self.head.parameters())
        if self.mode != "linear":
            params += list(self.backbone.parameters())

        optimizer, scheduler = warmup_cosine_optimizer(
            parameters=params,
            max_epochs=self.trainer.max_epochs,
            lr=self.hparams.lr,
            warmup_epochs=WARMUP_EPOCHS,
            final_lr=FINAL_LR,
            weight_decay=WEIGHT_DECAY
        )
        return {"optimizer": optimizer, "lr_scheduler": scheduler}
    
# -----------------------------------------------------------
#  Unified Train Entry
# -----------------------------------------------------------

def run_train_retfound(args):
    seed = pl.seed_everything(args.seed or 42)
    
    if args.retfound_mode == "linear":
        tfm = train_transform_retfound_linear(IMG_SIZE)
    else:
        tfm = train_transform_default(IMG_SIZE)

    if args.dataset == "idrid":
        dm = IDRiDModule(root=IDRID_PATH, transform=tfm, batch_size=BATCH_SIZE)
        csv_path = os.path.join(IDRID_PATH, "2. Groundtruths", "a. IDRiD_Disease Grading_Training Labels.csv")
        class_weights = compute_idrid_class_weights(csv_path)
    else:
        dm = APTOSModule(root=APTOS_PATH, transform=tfm, batch_size=BATCH_SIZE)
        class_weights = None

    dm.setup(stage="fit")

    # 2. Setup Model
    model = RETFoundTask(
        mode=args.retfound_mode, 
        lr=args.lr, 
        checkpoint_path=args.checkpoint,
        class_weights=class_weights
    )

    # 3. Trainer
    ckpt_cb = ModelCheckpoint(monitor="val/f1", mode="max", save_top_k=1, filename=f"retfound_{args.retfound_mode}_best")
    early_cb = EarlyStopping(monitor="val/f1", patience=50, mode="max")
    
    import time
    run_name = f"retfound_{args.dataset}_seed{seed}_{int(time.time())}"
    logger = WandbLogger(project="retfound_unified", name=run_name)

    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        precision="16-mixed",
        callbacks=[ckpt_cb, early_cb],
        logger=logger,
        log_every_n_steps=5
    )

    trainer.fit(model, dm)
    print(f"\n[✓] Training Complete. Best model: {ckpt_cb.best_model_path}")

    # load best weights into model and compute metrics
    try:
        best_ckpt = torch.load(ckpt_cb.best_model_path, map_location="cpu")["state_dict"]
        model.load_state_dict(best_ckpt, strict=False)

        dm.setup(stage="fit")
        train_dl = dm.train_dataloader()
        val_dl = dm.val_dataloader()

        model._final_eval = True

        for m in model.train_metrics.values(): m.reset()
        for m in model.val_metrics.values(): m.reset()

        # ---- VAL METRICS ----
        trainer.validate(model, dataloaders=val_dl)
        val_metrics = {
            "val/acc": model.val_metrics["acc"].compute().item(),
            "val/f1": model.val_metrics["f1"].compute().item(),
            "val/auroc": model.val_metrics["auroc"].compute().item(),
            "val/aupr": model.val_metrics["aupr"].compute().item(),
        }

        for m in model.train_metrics.values(): m.reset()
        for m in model.val_metrics.values(): m.reset()

        # ---- TRAIN METRICS ----
        trainer.test(model, dataloaders=train_dl)
        train_metrics = {
            "train/acc": model.train_metrics["acc"].compute().item(),
            "train/f1": model.train_metrics["f1"].compute().item(),
            "train/auroc": model.train_metrics["auroc"].compute().item(),
            "train/aupr": model.train_metrics["aupr"].compute().item(),
        }

        model._final_eval = False

    except Exception as e:
        print(f"[!] Metric computation failed: {e}")
        val_metrics = {}
        train_metrics = {}

    try:
        from utils.results import append_result
        row = {
            "timestamp": int(time.time()),
            "mode": f"retfound_{args.retfound_mode}",
            "dataset": args.dataset,
            "model_path": ckpt_cb.best_model_path,
            "run_name": run_name,
            "seed": args.seed,
            "monitor": ckpt_cb.monitor if hasattr(ckpt_cb, "monitor") else None,
            "monitor_value": float(ckpt_cb.best_model_score) if getattr(ckpt_cb, "best_model_score", None) is not None else None,
            "train_metrics": train_metrics[0] if train_metrics else {},
            "val_metrics": val_metrics[0] if val_metrics else {},
        }
        append_result(row)
    except Exception:
        pass