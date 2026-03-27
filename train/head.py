import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from torchmetrics import Accuracy, F1Score, AUROC, AveragePrecision
from torchvision import transforms

from utils.transforms import eval_transform
from config.constants import *
from models.vmamba_backbone import VisualMamba
from dataloader.idrid import IDRiDModule, compute_idrid_class_weights
from timm import create_model
from dataloader.aptos import APTOSModule
from optimizers.optimizer import warmup_cosine_optimizer

class VMambaHeadTask(pl.LightningModule):
    def __init__(self, backbone, lr, class_weights=None):
        super().__init__()
        self._final_eval = False
        self.save_hyperparameters(ignore=["backbone", "class_weights"])
        self.backbone = backbone

        # 1. Handle Freezing logic
        if FREEZE_BACKBONE:
            for p in self.backbone.parameters():
                p.requires_grad = False
            self.backbone.eval()
        
        # 2. Trainable head (Architecture kept identical to your original)
        self.head = nn.Sequential(
            nn.Linear(backbone.embed_dim, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(128, NUM_CLASSES)
        )

        # 3. Loss & Metrics
        self.loss_fn = nn.CrossEntropyLoss(weight=class_weights) if class_weights is not None else nn.CrossEntropyLoss()
        
        # Modularized metrics
        metric_args = {"task": "multiclass", "num_classes": NUM_CLASSES}
        self.train_metrics = nn.ModuleDict({
            "acc": Accuracy(**metric_args),
            "f1": F1Score(**metric_args, average="macro"),
            "auroc": AUROC(**metric_args),
            "aupr": AveragePrecision(**metric_args)
        })

        self.val_metrics = nn.ModuleDict({
            "acc": Accuracy(**metric_args),
            "f1": F1Score(**metric_args, average="macro"),
            "auroc": AUROC(**metric_args),
            "aupr": AveragePrecision(**metric_args)
        })

    def forward(self, x):
        # Apply no_grad only if frozen to save memory during training
        if FREEZE_BACKBONE and not self.training:
            with torch.no_grad():
                feats = self.backbone.forward_features(x)
        else:
            feats = self.backbone.forward_features(x)
        return self.head(feats)

    def shared_step(self, batch, stage):
        x, y, _ = batch
        logits = self(x)

        if y.ndim == 2:
            # Mixed labels from mixup / mosaic / copy-paste
            log_probs = F.log_softmax(logits, dim=1)
            loss = F.kl_div(log_probs, y, reduction='batchmean')
            y_for_metrics = y.argmax(dim=1)
        else:
            loss = self.loss_fn(logits, y)
            y_for_metrics = y

        probs = torch.softmax(logits, dim=1)
        preds = torch.argmax(probs, dim=1)

        metrics = self.train_metrics if stage == "train" else self.val_metrics

        metrics["acc"].update(preds, y_for_metrics)
        metrics["f1"].update(preds, y_for_metrics)
        try:
            metrics["auroc"].update(probs, y_for_metrics)
            metrics["aupr"].update(probs, y_for_metrics)
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
    
    def on_train_epoch_end(self): self._log_metrics("train")
    def on_validation_epoch_end(self): self._log_metrics("val")

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
        if not FREEZE_BACKBONE:
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
#  Training Entry Point
# -----------------------------------------------------------

def run_head_training(args):
    pl.seed_everything(args.seed or 42)
    print("\n=== PHASE II: VMAMBA HEAD TRAINING ===")

    # 1. Backbone Loading logic
    student_model = getattr(args, 'student_model', STUDENT_MODEL)
    if student_model == 'vmamba':
        backbone = VisualMamba(
            img_size=IMG_SIZE, patch_size=PATCH_SIZE, in_chans=IN_CHANS,
            embed_dim=VMAMBA_EMBED_DIM, depth=VMAMBA_DEPTH,
            learning_rate=0.0, mask_ratio=0.0, use_cls_token=False,
        )
    elif student_model == 'tinyvit':
        backbone = create_model('tinyvit_tiny_patch16_224', pretrained=False, num_classes=0)
    elif student_model == 'mobilenet_v3_small':
        backbone = create_model('mobilenetv3_small_100', pretrained=False, num_classes=0)
    elif student_model == 'efficientnet_b0':
        backbone = create_model('efficientnet_b0', pretrained=False, num_classes=0)
    else:
        raise ValueError(f"Unknown student_model: {student_model}")

    ckpt = torch.load(args.load_backbone, map_location="cpu")
    # Handle both Lightning state_dicts and raw state_dicts
    state_dict = ckpt.get("state_dict", ckpt)
    student_state = {k.replace("student.", ""): v for k, v in state_dict.items() if k.startswith("student.")}

    if student_state:
        backbone.load_state_dict(student_state, strict=False)
    else:
        # for direct backbone checkpoints or non-vmamba student
        try:
            # remove any prefix that might vary (e.g., backbone., encoder.)
            simplified = {k.split('.', 1)[-1]: v for k, v in state_dict.items()}
            backbone.load_state_dict(simplified, strict=False)
        except Exception:
            backbone.load_state_dict(state_dict, strict=False)

    # 2. Data & Weights
    tfm = eval_transform(IMG_SIZE)
    if args.dataset == "aptos":
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
        class_weights = None
    else:
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
        csv_path = os.path.join(IDRID_PATH, "2. Groundtruths", "a. IDRiD_Disease Grading_Training Labels.csv")
        class_weights = compute_idrid_class_weights(csv_path)

    dm.setup()

    # 3. Model & Trainer
    model = VMambaHeadTask(backbone, lr=args.lr or LR, class_weights=class_weights)
    
    ckpt_cb = ModelCheckpoint(monitor="val/f1", mode="max", save_top_k=1, filename="best_head")
    early_cb = EarlyStopping(monitor="val/f1", patience=100, mode="max")

    import time
    seed = args.seed or SEED
    mask_tag = "masked" if MASK_RATIO > 0.0 else "unmasked"
    timestamp = int(time.time())

    run_name = f"head_{args.dataset}_{mask_tag}_{seed}_{timestamp}"
    trainer = pl.Trainer(
        max_epochs=HEAD_EPOCHS,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        logger=WandbLogger(project="vmamba_head_training", name=run_name),
        precision="16-mixed",
        callbacks=[ckpt_cb, early_cb]
    )

    trainer.fit(model, dm)

    # 4. Final Save (Backbone + Head split)
    save_path = os.path.join(
        CHECKPOINT_DIR,
        f"head_{args.dataset}_{mask_tag}_{seed}_{timestamp}",
        "vmamba_final_head.pth"
    )
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    best_ckpt = torch.load(ckpt_cb.best_model_path)["state_dict"]

    final_dict = {
        "backbone": {k.replace("backbone.", ""): v for k, v in best_ckpt.items() if k.startswith("backbone.")},
        "head": {k.replace("head.", ""): v for k, v in best_ckpt.items() if k.startswith("head.")}
    }
    torch.save(final_dict, save_path)
    print(f"[✓] Model saved to {save_path}")

    # load best weights into model and compute metrics
    try:
        model.load_state_dict(best_ckpt, strict=False)
        dm.setup(stage="fit")

        train_dl = dm.train_dataloader()
        val_dl = dm.val_dataloader()

        # ---- ENABLE FINAL EVAL MODE (prevents reset) ----
        model._final_eval = True

        # Reset before starting
        for m in model.train_metrics.values():
            m.reset()
        for m in model.val_metrics.values():
            m.reset()

        # ---- VAL METRICS (real validation) ----
        trainer.validate(model, dataloaders=val_dl)

        val_metrics = {
            "val/acc": model.val_metrics["acc"].compute().item(),
            "val/f1": model.val_metrics["f1"].compute().item(),
            "val/auroc": model.val_metrics["auroc"].compute().item(),
            "val/aupr": model.val_metrics["aupr"].compute().item(),
        }

        # Reset again
        for m in model.train_metrics.values():
            m.reset()
        for m in model.val_metrics.values():
            m.reset()

        # ---- TRAIN METRICS (use TEST LOOP!) ----
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

    # append to results csv
    try:
        from utils.results import append_result
        import time
        row = {
            "timestamp": int(time.time()),
            "mode": "head",
            "dataset": args.dataset,
            "model_path": ckpt_cb.best_model_path,
            "run_name": run_name,
            "seed": args.seed,
            "monitor": ckpt_cb.monitor if hasattr(ckpt_cb, "monitor") else None,
            "monitor_value": float(ckpt_cb.best_model_score)
                if getattr(ckpt_cb, "best_model_score", None) is not None else None,
            "train_metrics": train_metrics if isinstance(train_metrics, dict) else {},
            "val_metrics": val_metrics if isinstance(val_metrics, dict) else {},
        }
        append_result(row)
    except Exception:
        pass