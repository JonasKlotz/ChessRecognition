import timm
import torch
import torch.nn as nn
import lightning as L
from torchmetrics import Accuracy, F1Score

from .orientation_datamodule import ORIENTATIONS

NUM_ORIENTATIONS = len(ORIENTATIONS)


class OrientationModule(L.LightningModule):
    """Board orientation classifier: 4-way (0°/90°/180°/270°).

    Runs once per image on the cropped board before the piece classifier.
    Replaces the fragile pixel-heuristic in calculate_fen/get_board_colors.py.

    MobileNetV3-small (2.5M params) is fast enough to add negligible latency.
    """

    def __init__(
        self,
        backbone: str = "mobilenetv3_small_100",
        lr: float = 3e-4,
        weight_decay: float = 1e-4,
        max_epochs: int = 30,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.backbone = timm.create_model(backbone, pretrained=True, num_classes=0)
        # Infer actual output dim via a dummy pass — num_features is unreliable for
        # some backbones (e.g. MobileNetV3 reports pre-expansion size, outputs 1024).
        with torch.no_grad():
            feat_dim = self.backbone(torch.zeros(1, 3, 224, 224)).shape[-1]
        self.head = nn.Linear(feat_dim, NUM_ORIENTATIONS)
        self.loss_fn = nn.CrossEntropyLoss()

        metric_kwargs = dict(task="multiclass", num_classes=NUM_ORIENTATIONS)
        self.train_acc = Accuracy(**metric_kwargs)
        self.val_acc = Accuracy(**metric_kwargs)
        self.test_acc = Accuracy(**metric_kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(x))

    def predict_orientation(self, x: torch.Tensor) -> int:
        """Inference helper: returns the rotation degrees (0/90/180/270)."""
        with torch.no_grad():
            logits = self(x.unsqueeze(0) if x.dim() == 3 else x)
            idx = logits.argmax(dim=1).item()
        return int(ORIENTATIONS[idx])

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
        self.log("val/loss", loss, prog_bar=True)
        self.log("val/acc", self.val_acc, prog_bar=True)

    def test_step(self, batch, batch_idx):
        _, preds, y = self._shared_step(batch)
        self.test_acc(preds, y)
        self.log("test/acc", self.test_acc)

    def configure_optimizers(self):
        hp = self.hparams
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=hp.lr, weight_decay=hp.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=hp.max_epochs)
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"}}
