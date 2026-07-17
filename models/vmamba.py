import torch
import torch.nn as nn
import pytorch_lightning as pl
from torchmetrics import Accuracy, AUROC, F1Score, AveragePrecision

from config.constants import *
from optimizers.optimizer import warmup_cosine_optimizer
from models.vmamba_backbone import VisualMamba


class VMambaClassifier(pl.LightningModule):
    """
    End-to-end trainable VMamba classifier.
    Unlike the distillation setup, this trains the full model directly.
    """

    def __init__(
        self,
        num_classes=NUM_CLASSES,
        img_size=IMG_SIZE,
        patch_size=PATCH_SIZE,
        in_chans=IN_CHANS,
        embed_dim=VMAMBA_EMBED_DIM,
        depth=VMAMBA_DEPTH,
        ssm_dim=SSM_DIM,
        expand_dim=EXPAND_DIM,
        lr=LR,
        class_weights=None,
        pretrained=False,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["class_weights"])

        # Build the full VMamba backbone (encoder only)
        self.backbone = VisualMamba(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            num_classes=num_classes,  # Used only for initialization
            embed_dim=embed_dim,
            depth=depth,
            ssm_dim=ssm_dim,
            expand_dim=expand_dim,
            learning_rate=lr,
            mask_ratio=0.0,  # No masking for direct training
            use_cls_token=False,
        )
        self.embed_dim = embed_dim
        self.learning_rate = lr

        # Classification head (map from embed_dim to num_classes)
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(embed_dim // 2, num_classes),
        )

        self.loss_fn = nn.CrossEntropyLoss(weight=class_weights) if class_weights is not None else nn.CrossEntropyLoss()

        metric_args = {"task": "multiclass", "num_classes": num_classes}
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
        features = self.backbone.forward_features(x)
        return self.classifier(features)

    def forward_features(self, x):
        """Extract features before classification head."""
        return self.backbone.forward_features(x)

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
        return self.shared_step(batch, "val")

    def on_train_epoch_end(self):
        self._log_metrics("train")

    def on_validation_epoch_end(self):
        self._log_metrics("val")

    def _log_metrics(self, stage):
        metrics = self.train_metrics if stage == "train" else self.val_metrics
        for name, metric in metrics.items():
            try:
                self.log(f"{stage}/{name}", metric.compute(), prog_bar=(name == "f1"))
            except Exception:
                pass
            metric.reset()

    def configure_optimizers(self):
        optimizer, scheduler = warmup_cosine_optimizer(
            parameters=self.parameters(),
            max_epochs=self.trainer.max_epochs,
            lr=self.hparams.lr,
            warmup_epochs=WARMUP_EPOCHS,
            final_lr=FINAL_LR,
            weight_decay=WEIGHT_DECAY,
        )
        return {"optimizer": optimizer, "lr_scheduler": scheduler}
