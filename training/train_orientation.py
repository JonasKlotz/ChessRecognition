"""Train the board orientation classifier.

Example:
    uv run python training/train_orientation.py \\
        --data_dir data/chessred_orientation \\
        --max_epochs 30
"""

import argparse

import lightning as L
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger

from .orientation_datamodule import OrientationDataModule
from .orientation_module import OrientationModule


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train board orientation classifier")
    p.add_argument("--data_dir", required=True,
                   help="Root with train/{0,90,180,270}/ val/ test/ subdirs")
    p.add_argument("--backbone", default="mobilenetv3_small_100",
                   help="timm model name (default: mobilenetv3_small_100)")
    p.add_argument("--img_size", type=int, default=224)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--max_epochs", type=int, default=30)
    p.add_argument("--precision", default="bf16-mixed")
    p.add_argument("--wandb_project", default="chess-orientation-classifier")
    p.add_argument("--fast_dev_run", action="store_true")
    p.add_argument("--ckpt_dir", default="checkpoints/orientation")
    return p.parse_args()


def main():
    args = parse_args()

    L.seed_everything(42, workers=True)

    datamodule = OrientationDataModule(
        data_dir=args.data_dir,
        img_size=args.img_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    model = OrientationModule(
        backbone=args.backbone,
        lr=args.lr,
        weight_decay=args.weight_decay,
        max_epochs=args.max_epochs,
    )

    callbacks = [
        ModelCheckpoint(
            dirpath=args.ckpt_dir,
            filename="orientation-{epoch:02d}-{val/acc:.4f}",
            monitor="val/acc",
            mode="max",
            save_top_k=2,
        ),
        EarlyStopping(monitor="val/acc", mode="max", patience=8),
        LearningRateMonitor(logging_interval="epoch"),
    ]

    logger = WandbLogger(project=args.wandb_project) if not args.fast_dev_run else None

    trainer = L.Trainer(
        max_epochs=args.max_epochs,
        precision=args.precision,
        callbacks=callbacks,
        logger=logger,
        fast_dev_run=args.fast_dev_run,
        deterministic=True,
    )

    trainer.fit(model, datamodule=datamodule)

    if not args.fast_dev_run:
        trainer.test(model, datamodule=datamodule, ckpt_path="best")


if __name__ == "__main__":
    main()
