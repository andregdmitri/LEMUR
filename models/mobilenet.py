import timm
import torch
import torch.nn as nn
import pytorch_lightning as pl
from torchmetrics import Accuracy, AUROC, F1Score, AveragePrecision

from config.constants import *
from optimizers.optimizer import warmup_cosine_optimizer


class MobileNetClassifier(pl.LightningModule):
    def __init__(
        self,
        num_classes=NUM_CLASSES,
        pretrained=False,
        lr=LR,
        class_weights=None,
        dropout=0.2,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["class_weights"])

        self.backbone = timm.create_model(
            "mobilenetv3_small_100",
            pretrained=pretrained,
            num_classes=num_classes,
        )
        self.embed_dim = self.backbone.num_features
        self.learning_rate = lr

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
        return self.backbone(x)

    def forward_features(self, x):
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
