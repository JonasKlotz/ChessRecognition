"""Prepare ChessReD dataset for training the piece and orientation classifiers.

The main ChessReD dataset (10,800 images, annotations.json) has piece position
annotations but NO board corner data, so we cannot warp/crop individual squares
from it directly.

The ChessReD2K subset (~2,078 images) extends the annotations with board corner
coordinates and orientation labels, enabling perspective warp → square extraction.

This script processes ChessReD2K to produce:
  out_dir/chessred_squares/{train,val,test}/{class}/*.jpg   — piece classifier data
  out_dir/chessred_orientation/{train,val,test}/{0,90,180,270}/*.jpg — orientation data

Usage:
    uv run python training/prepare_chessred.py \\
        --chessred_dir  /path/to/chessred \\
        --chessred2k_annotations /path/to/chessred2k_annotations.json \\
        --out_dir data

ChessReD download: https://github.com/tmasouris/end-to-end-chess-recognition
ChessReD2K annotations: available from the 4TU dataset page linked in the repo.
"""

import argparse
import json
import logging
from pathlib import Path

import cv2
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

BOARD_SIZE = 1000  # output pixels for warped board

# Mapping from ChessReD category names → our 13-class directory names
# Category names in ChessReD use hyphens: "white-king", "black-pawn", etc.
CHESSRED_TO_CLASS = {
    "white-king": "wk", "white-queen": "wq", "white-rook": "wr",
    "white-bishop": "wb", "white-knight": "wn", "white-pawn": "wp",
    "black-king": "bk", "black-queen": "bq", "black-rook": "br",
    "black-bishop": "bb", "black-knight": "bn", "black-pawn": "bp",
    "empty": "empty",
}

# Algebraic notation → flat index (row 8 first, a-file first)
_COLS = "abcdefgh"
_ROWS = "87654321"


def alg_to_idx(pos: str) -> int:
    return 8 * _ROWS.index(pos[1]) + _COLS.index(pos[0])


def warp_board(img: np.ndarray, corners: dict) -> np.ndarray:
    """Perspective-warp the board to a BOARD_SIZE × BOARD_SIZE square.

    Args:
        img: BGR image.
        corners: dict with keys 'top-left', 'top-right', 'bottom-right', 'bottom-left'
                 each mapping to [x, y] pixel coordinates (from ChessReD2K annotations).
                 Keys are relative to white player's perspective (canonical orientation).

    Returns:
        Warped BGR board image of shape (BOARD_SIZE, BOARD_SIZE, 3).
    """
    src = np.float32([
        corners["top-left"],
        corners["top-right"],
        corners["bottom-right"],
        corners["bottom-left"],
    ])
    dst = np.float32([
        [0, 0],
        [BOARD_SIZE, 0],
        [BOARD_SIZE, BOARD_SIZE],
        [0, BOARD_SIZE],
    ])
    M = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(img, M, (BOARD_SIZE, BOARD_SIZE))


def extract_squares(board: np.ndarray) -> list[np.ndarray]:
    """Split a warped board into 64 squares, row-major from rank 8 (top)."""
    sq = BOARD_SIZE // 8
    squares = []
    for row in range(8):
        for col in range(8):
            y1, y2 = row * sq, (row + 1) * sq
            x1, x2 = col * sq, (col + 1) * sq
            squares.append(board[y1:y2, x1:x2])
    return squares


def save_image(img: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), img)


def generate_orientation_crops(board: np.ndarray, image_id: int, split: str, out_root: Path) -> None:
    """Save 4 rotated versions of the board for orientation classifier training.

    The canonical board (white at bottom, correctly warped) is saved as '0'.
    Rotations are applied to generate the other 3 labels.
    """
    rotations = {
        "0": board,
        "90": cv2.rotate(board, cv2.ROTATE_90_CLOCKWISE),
        "180": cv2.rotate(board, cv2.ROTATE_180),
        "270": cv2.rotate(board, cv2.ROTATE_90_COUNTERCLOCKWISE),
    }
    for label, rotated in rotations.items():
        path = out_root / "chessred_orientation" / split / label / f"{image_id}.jpg"
        save_image(rotated, path)


def process_chessred2k(
    chessred_dir: Path,
    chessred2k_annotations: Path,
    main_annotations: dict,
    out_root: Path,
) -> None:
    """Extract squares and orientation crops from the ChessReD2K subset.

    ChessReD2K annotation format (assumed from 4TU dataset description):
    {
      "images": [{"id": int, "path": str, "corners": {
                    "top-left": [x, y], "top-right": [x, y],
                    "bottom-right": [x, y], "bottom-left": [x, y]
                 }}, ...],
    }
    Corners are pixel coordinates in the white-player-canonical orientation.

    If the actual format differs, adapt the corner extraction below.
    """
    with open(chessred2k_annotations) as f:
        chessred2k = json.load(f)

    # Build lookup: image_id → split
    id_to_split = {}
    for split, data in main_annotations["splits"].items():
        for img_id in data["image_ids"]:
            id_to_split[img_id] = split

    # Build lookup: image_id → list of (square_idx, category_name)
    cat_id_to_name = {c["id"]: c["name"] for c in main_annotations["categories"]}
    id_to_pieces: dict[int, dict[int, str]] = {}
    for ann in main_annotations["annotations"]["pieces"]:
        img_id = ann["image_id"]
        sq_idx = alg_to_idx(ann["chessboard_position"])
        cat_name = cat_id_to_name[ann["category_id"]]
        id_to_pieces.setdefault(img_id, {})[sq_idx] = cat_name

    processed = 0
    for img_meta in chessred2k["images"]:
        image_id = img_meta["id"]
        split = id_to_split.get(image_id)
        if split is None:
            continue

        img_path = chessred_dir / img_meta["path"]
        if not img_path.exists():
            log.warning("Image not found: %s", img_path)
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            log.warning("Could not read: %s", img_path)
            continue

        # Extract corners — adapt this block if the actual format differs
        corners = img_meta.get("corners")
        if corners is None:
            log.warning("No corners for image %d, skipping", image_id)
            continue

        try:
            board = warp_board(img, corners)
        except Exception as e:
            log.warning("Warp failed for image %d: %s", image_id, e)
            continue

        # Orientation crops (all 4 rotations of the canonical board)
        generate_orientation_crops(board, image_id, split, out_root)

        # Square crops labeled by piece type
        squares = extract_squares(board)
        piece_map = id_to_pieces.get(image_id, {})
        for sq_idx, square in enumerate(squares):
            cat_name = piece_map.get(sq_idx, "empty")
            class_label = CHESSRED_TO_CLASS.get(cat_name, "empty")
            path = out_root / "chessred_squares" / split / class_label / f"{image_id}_{sq_idx:02d}.jpg"
            save_image(square, path)

        processed += 1
        if processed % 100 == 0:
            log.info("Processed %d images", processed)

    log.info("Done. Processed %d images total.", processed)


def main():
    p = argparse.ArgumentParser(description="Prepare ChessReD data for training")
    p.add_argument("--chessred_dir", required=True,
                   help="Root directory of ChessReD download (contains images/ and annotations.json)")
    p.add_argument("--chessred2k_annotations", required=True,
                   help="Path to ChessReD2K extended annotation file (with corner coordinates)")
    p.add_argument("--out_dir", default="data",
                   help="Output root; will create chessred_squares/ and chessred_orientation/ here")
    args = p.parse_args()

    chessred_dir = Path(args.chessred_dir)
    out_root = Path(args.out_dir)

    ann_path = chessred_dir / "annotations.json"
    log.info("Loading main annotations from %s", ann_path)
    with open(ann_path) as f:
        main_annotations = json.load(f)

    process_chessred2k(
        chessred_dir=chessred_dir,
        chessred2k_annotations=Path(args.chessred2k_annotations),
        main_annotations=main_annotations,
        out_root=out_root,
    )


if __name__ == "__main__":
    main()
