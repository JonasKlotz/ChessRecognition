import timm
import torch
import torch.nn as nn
import lightning as L
from torchmetrics import Accuracy, F1Score
from torchmetrics.classification import MulticlassConfusionMatrix

from .piece_datamodule import CLASSES

NUM_CLASSES = len(CLASSES)


class ChessPieceModule(L.LightningModule):
    """Per-square chess piece classifier using a DINO-pretrained ViT backbone.

    Default backbone vit_small_patch16_224.dino is the key novelty — no published
    paper uses ViT/DINOv2 features for chess piece recognition. Comparison baseline
    can be run by passing backbone='efficientnetv2_s'.

    Label smoothing regularises against mislabelled squares (piece colours can be
    ambiguous under unusual lighting).
    """

    def __init__(
        self,
        backbone: str = "vit_small_patch16_224.dino",
        lr: float = 1e-4,
        weight_decay: float = 1e-4,
        label_smoothing: float = 0.1,
        warmup_epochs: int = 5,
        max_epochs: int = 50,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.backbone = timm.create_model(backbone, pretrained=True, num_classes=0)
        with torch.no_grad():
            feat_dim = self.backbone(torch.zeros(1, 3, 224, 224)).shape[-1]
        self.head = nn.Sequential(
            nn.LayerNorm(feat_dim),
            nn.Linear(feat_dim, NUM_CLASSES),
        )

        self.loss_fn = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

        metric_kwargs = dict(task="multiclass", num_classes=NUM_CLASSES)
        self.train_acc = Accuracy(**metric_kwargs)
        self.val_acc = Accuracy(**metric_kwargs)
        self.val_f1 = F1Score(average="weighted", **metric_kwargs)
        self.test_acc = Accuracy(**metric_kwargs)
        self.test_f1 = F1Score(average="weighted", **metric_kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(x))

    def _shared_step(self, batch):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        preds = logits.argmax(dim=1)
        return loss, preds, y

    def training_step(self, batch, batch_idx):
        loss, preds, y = self._shared_step(batch)
        self.train_acc(preds, y)
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/acc", self.train_acc, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, preds, y = self._shared_step(batch)
        self.val_acc(preds, y)
        self.val_f1(preds, y)
        self.log("val/loss", loss, prog_bar=True)
        self.log("val/acc", self.val_acc, prog_bar=True)
        self.log("val/f1", self.val_f1)

    def test_step(self, batch, batch_idx):
        _, preds, y = self._shared_step(batch)
        self.test_acc(preds, y)
        self.test_f1(preds, y)
        self.log("test/acc", self.test_acc)
        self.log("test/f1", self.test_f1)

    def configure_optimizers(self):
        hp = self.hparams
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=hp.lr, weight_decay=hp.weight_decay
        )
        # Linear warmup then cosine decay
        def lr_lambda(epoch: int) -> float:
            if epoch < hp.warmup_epochs:
                return (epoch + 1) / hp.warmup_epochs
            progress = (epoch - hp.warmup_epochs) / max(1, hp.max_epochs - hp.warmup_epochs)
            return 0.5 * (1.0 + torch.cos(torch.tensor(torch.pi * progress)).item())

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"}}
