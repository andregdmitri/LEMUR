import timm
import torch
import torch.nn as nn
import pytorch_lightning as pl
from torchmetrics import Accuracy, AUROC, F1Score, AveragePrecision

from config.constants import *
from optimizers.optimizer import warmup_cosine_optimizer


class TinyViTClassifier(pl.LightningModule):
    """
    End-to-end trainable TinyViT classifier.

    Follows the same protocol as MobileNetClassifier / EfficientNetClassifier in
    this repo: a timm backbone + standard classification head, with the shared
    metric/logging/optimizer boilerplate used across the `train` folder.
    """

    def __init__(
        self,
        num_classes=NUM_CLASSES,
        model_name=TINYVIT_MODEL,
        pretrained=False,
        lr=LR,
        class_weights=None,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["class_weights"])

        self.backbone = timm.create_model(
            model_name,
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
        # TinyViT's forward_features returns a spatial map (B, C, H, W);
        # global average pool it to a single feature vector (B, C).
        feats = self.backbone.forward_features(x)
        return feats.mean(dim=(2, 3))

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


class TinyViTStudent(nn.Module):
    """
    Encoder-only TinyViT used as the *student* in RETFound feature distillation.

    TinyViT is a CNN/attention hybrid without an MAE-style mask token, so masking
    is disabled (mask_ratio=0). To be drop-in compatible with `DistillationModule`,
    it exposes the same API as `VisualMamba`:

        forward_features(x, return_pooled=False, apply_mask=False)
            -> (tokens, mask, ids_keep, ids_restore)

    where `tokens` is (B, N, embed_dim), `mask/ids_keep/ids_restore` are None.
    """

    def __init__(self, model_name=TINYVIT_MODEL, pretrained=False):
        super().__init__()
        self.model_name = model_name
        # num_classes=0 -> drop the classification head (encoder only)
        self.backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
        self.embed_dim = self.backbone.num_features
        # CNN-based student has no token masking
        self.mask_ratio = 0.0

    def forward_features(self, x, return_pooled=True, apply_mask=False):
        # TinyViT returns a spatial feature map (B, C, H, W)
        feats = self.backbone.forward_features(x)
        B, C, H, W = feats.shape
        # reshape to a token sequence (B, H*W, C)
        tokens = feats.flatten(2).transpose(1, 2)

        if return_pooled:
            return tokens.mean(dim=1)

        # return the 4-tuple expected by DistillationModule
        return tokens, None, None, None

    def forward(self, x):
        return self.forward_features(x, return_pooled=True, apply_mask=False)
