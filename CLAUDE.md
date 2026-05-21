# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Setup

1. Install dependencies: `uv sync`
2. Create a `debug/` directory in the project root (required even when debug is disabled)
3. Trained Keras model weights are in `data/13_classes/13_classes/{ModelName}/model.h5`
4. `config.yaml` paths are pre-configured for the local layout; update if data is elsewhere

## Running

```bash
# Single image prediction (outputs FEN string)
uv run python main.py testing_images/21.jpg -b CPS -m MobileNetV2

# Board algorithm choices: 'Mine' (custom Hough-based) or 'CPS' (Czyzewski)
# Model choices: MobileNetV2, NASNetMobile, InceptionResNetV2, Xception

# Batch evaluation across all models
uv run python evaluate.py
```

Logs are written to `output.log`. The result image is saved to `result_directory`.

## Architecture

The pipeline has three stages: **board detection → piece classification → FEN generation**.

### Board Detection (`detectboard/`)

Two interchangeable algorithms, selected via `-b` flag or `config.yaml:board_algorithm`:

- **CPS** (`get_slid.py` → `detect_board.py`): Wraps the external Czyzewski algorithm. Saves a cropped board to `<image_dir>/tmp/<filename>` and returns corner coordinates.
- **Mine** (`get_board.py`): Custom algorithm: CLAHE histogram equalization → Canny edge detection → Hough line transform → DBSCAN angle clustering to find horizontal/vertical lines → line intersection → convex hull + greedy quad-area maximization for corners → 4-point perspective warp.

Both return a square cropped board image (1816×1816 px by default).

### Piece Classification (`model.py`, `process_board.py`)

`process_board.py` splits the board into 64 squares using a sliding-window approach: each square is extracted twice — once at normal height and once with a 20% upward extension to capture tall pieces (pawns cut off at top). Predictions from both extractions are averaged.

`model.py` loads a `.h5` Keras model and runs inference. Model path is resolved as `{model_directory}/{num_classes}_classes/{model_name}/model.h5` unless `model_path` is set explicitly in `config.yaml`.

Supported models and their required input sizes:
- MobileNetV2: 224px
- NASNetMobile: 224px
- InceptionResNetV2: 150px
- Xception: 299px

### FEN Generation (`calculate_fen/`)

`get_fen.py` dispatches to either `get_fen_7_classes.py` (7 classes: piece type only) or `get_fen_13_classes.py` (13 classes: piece type + color) based on `config.yaml:num_of_classes`.

`get_board_colors.py` determines piece colors (for 7-class models) and board orientation by analyzing square pixel colors. If the bottom-left square is not white, the board is rotated 180°.

`fen_utility.py` converts a 64-element array of piece symbols to a FEN string.

### Configuration (`config.py`, `config.yaml`)

`config.py:configurator` reads `config.yaml` at init time. All runtime parameters (model path, image size, preprocessing function, board algorithm) flow from the configurator into `main.py`.

### Debugging (`debug.py`)

Set `DEBUG = True` and update `DEBUG_SAVE_DIR` in `debug.py` to save intermediate images at each pipeline step (edge maps, Hough lines, cluster points, cropped board, individual squares). Images are saved with a sequential counter prefix for step ordering.

### Training (`training/`)

New PyTorch Lightning pipeline with two learnable modules:

**Data preparation** (run once after downloading ChessReD2K):
```bash
uv run python training/prepare_chessred.py \
    --chessred_dir /path/to/chessred \
    --chessred2k_annotations /path/to/chessred2k_annotations.json \
    --out_dir data
# Produces: data/chessred_squares/{train,val,test}/{class}/*.jpg
#           data/chessred_orientation/{train,val,test}/{0,90,180,270}/*.jpg
```

**Orientation classifier** (`mobilenetv3_small_100`, 2.5M params):
```bash
uv run python training/train_orientation.py \
    --data_dir data/chessred_orientation --max_epochs 30
# Smoke test:
uv run python training/train_orientation.py --data_dir data/chessred_orientation --fast_dev_run
```

**Piece classifier** (`vit_small_patch16_224.dino`, 21M params — novelty: first ViT for chess piece recognition):
```bash
uv run python training/train_pieces.py \
    --train_dirs data/chessred_squares/train \
    --val_dirs   data/chessred_squares/val \
    --test_dirs  data/chessred_squares/test \
    --backbone vit_small_patch16_224.dino --max_epochs 50
# Multiple --train_dirs merges datasets. Baseline: --backbone efficientnetv2_s
# Smoke test:
uv run python training/train_pieces.py \
    --train_dirs data/chessred_squares/train --val_dirs data/chessred_squares/val --fast_dev_run
```

Checkpoints saved to `checkpoints/pieces/` and `checkpoints/orientation/`. The 13-class label order (`bb bk bn bp bq br empty wb wk wn wp wq wr`) must be preserved — it maps directly to the FEN assembly in `calculate_fen/`.

The old Keras training scripts (`generic_model.py`, `mobilenet_v2.py`, etc.) are superseded but kept for reference.
