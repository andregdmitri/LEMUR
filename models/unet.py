import torch
import torch.nn as nn
import pytorch_lightning as pl
from torchmetrics import Accuracy, AUROC, F1Score, AveragePrecision

from config.constants import *
from optimizers.optimizer import warmup_cosine_optimizer


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels),
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diff_h = x2.size(2) - x1.size(2)
        diff_w = x2.size(3) - x1.size(3)
        x1 = nn.functional.pad(x1, [diff_w // 2, diff_w - diff_w // 2, diff_h // 2, diff_h - diff_h // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class UNetEncoder(nn.Module):
    def __init__(self, in_channels=IN_CHANS, base_channels=32):
        super().__init__()
        self.inc = DoubleConv(in_channels, base_channels)
        self.down1 = Down(base_channels, base_channels * 2)
        self.down2 = Down(base_channels * 2, base_channels * 4)
        self.down3 = Down(base_channels * 4, base_channels * 8)
        self.down4 = Down(base_channels * 8, base_channels * 16)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        return x5


class UNetClassifier(pl.LightningModule):
    def __init__(
        self,
        num_classes=NUM_CLASSES,
        in_channels=IN_CHANS,
        base_channels=32,
        lr=LR,
        class_weights=None,
        pretrained=False,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["class_weights"])

        self.backbone = UNetEncoder(in_channels=in_channels, base_channels=base_channels)
        self.embed_dim = base_channels * 16
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(self.embed_dim, self.embed_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(self.embed_dim // 2, num_classes),
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
        features = self.backbone(x)
        return self.classifier(features)

    def forward_features(self, x):
        features = self.backbone(x)
        pooled = nn.functional.adaptive_avg_pool2d(features, 1).flatten(1)
        return pooled

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
