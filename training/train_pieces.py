"""Train the chess piece classifier.

Example:
    uv run python training/train_pieces.py \\
        --train_dirs data/chessred_squares/train \\
        --val_dirs   data/chessred_squares/val \\
        --test_dirs  data/chessred_squares/test \\
        --backbone   vit_small_patch16_224.dino \\
        --max_epochs 50

Multiple --train_dirs can be provided to merge datasets:
    --train_dirs data/chessred_squares/train data/extra_pieces/train
"""

import argparse

import lightning as L
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger

from .piece_datamodule import ChessPieceDataModule
from .piece_module import ChessPieceModule


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train chess piece classifier")
    p.add_argument("--train_dirs", nargs="+", required=True, help="Training data directories (class subdirs)")
    p.add_argument("--val_dirs", nargs="+", required=True, help="Validation data directories")
    p.add_argument("--test_dirs", nargs="+", default=[], help="Test data directories")
    p.add_argument("--backbone", default="vit_small_patch16_224.dino",
                   help="timm model name (default: vit_small_patch16_224.dino; "
                        "baseline: efficientnetv2_s)")
    p.add_argument("--img_size", type=int, default=224)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--label_smoothing", type=float, default=0.1)
    p.add_argument("--warmup_epochs", type=int, default=5)
    p.add_argument("--max_epochs", type=int, default=50)
    p.add_argument("--precision", default="bf16-mixed", help="Trainer precision")
    p.add_argument("--wandb_project", default="chess-piece-classifier")
    p.add_argument("--fast_dev_run", action="store_true", help="1-batch smoke test")
    p.add_argument("--ckpt_dir", default="checkpoints/pieces", help="Checkpoint output directory")
    return p.parse_args()


def main():
    args = parse_args()

    L.seed_everything(42, workers=True)

    datamodule = ChessPieceDataModule(
        train_dirs=args.train_dirs,
        val_dirs=args.val_dirs,
        test_dirs=args.test_dirs,
        img_size=args.img_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    model = ChessPieceModule(
        backbone=args.backbone,
        lr=args.lr,
        weight_decay=args.weight_decay,
        label_smoothing=args.label_smoothing,
        warmup_epochs=args.warmup_epochs,
        max_epochs=args.max_epochs,
    )

    callbacks = [
        ModelCheckpoint(
            dirpath=args.ckpt_dir,
            filename=f"{args.backbone}-{{epoch:02d}}-{{val/acc:.4f}}",
            monitor="val/acc",
            mode="max",
            save_top_k=3,
        ),
        EarlyStopping(monitor="val/acc", mode="max", patience=10),
        LearningRateMonitor(logging_interval="epoch"),
    ]

    logger = WandbLogger(project=args.wandb_project, name=args.backbone) if not args.fast_dev_run else None

    trainer = L.Trainer(
        max_epochs=args.max_epochs,
        precision=args.precision,
        callbacks=callbacks,
        logger=logger,
        fast_dev_run=args.fast_dev_run,
        deterministic=True,
    )

    trainer.fit(model, datamodule=datamodule)

    if args.test_dirs and not args.fast_dev_run:
        trainer.test(model, datamodule=datamodule, ckpt_path="best")


if __name__ == "__main__":
    main()
