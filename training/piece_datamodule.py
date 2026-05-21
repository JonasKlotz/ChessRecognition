from pathlib import Path

import cv2
import lightning as L
import numpy as np
from torch.utils.data import DataLoader, Dataset

from .transforms import piece_train_transforms, piece_val_transforms

# Must match the sorted class subdirectory names used in FEN assembly
CLASSES = ["bb", "bk", "bn", "bp", "bq", "br", "empty", "wb", "wk", "wn", "wp", "wq", "wr"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}


class PieceDataset(Dataset):
    """Loads square-crop images from a directory tree of the form:
        root/
          bb/  bk/  bn/  ...  wr/
            *.jpg  *.png  ...

    Multiple root directories can be merged (existing piece crops + ChessReD squares).
    """

    def __init__(self, roots: list[Path | str], transform=None):
        self.transform = transform
        self.samples: list[tuple[Path, int]] = []

        for root in roots:
            root = Path(root)
            if not root.exists():
                continue
            for class_dir in sorted(root.iterdir()):
                if not class_dir.is_dir():
                    continue
                label = CLASS_TO_IDX.get(class_dir.name)
                if label is None:
                    continue
                for ext in ("*.jpg", "*.jpeg", "*.png"):
                    self.samples.extend((p, label) for p in class_dir.glob(ext))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        img = cv2.imread(str(path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.transform:
            img = self.transform(image=img)["image"]
        return img, label


class ChessPieceDataModule(L.LightningDataModule):
    """LightningDataModule for per-square piece classification.

    Args:
        train_dirs: List of directories containing class subdirectories for training.
        val_dirs:   Same for validation.
        test_dirs:  Same for test.
        img_size:   Square input resolution fed to the backbone.
        batch_size: Per-GPU batch size.
        num_workers: DataLoader worker count.
    """

    def __init__(
        self,
        train_dirs: list[str],
        val_dirs: list[str],
        test_dirs: list[str],
        img_size: int = 224,
        batch_size: int = 64,
        num_workers: int = 4,
    ):
        super().__init__()
        self.save_hyperparameters()

    def setup(self, stage: str | None = None):
        hp = self.hparams
        self.train_ds = PieceDataset(hp.train_dirs, piece_train_transforms(hp.img_size))
        self.val_ds = PieceDataset(hp.val_dirs, piece_val_transforms(hp.img_size))
        self.test_ds = PieceDataset(hp.test_dirs, piece_val_transforms(hp.img_size))

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            num_workers=self.hparams.num_workers,
            pin_memory=True,
            drop_last=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=self.hparams.num_workers,
            pin_memory=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_ds,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=self.hparams.num_workers,
        )
