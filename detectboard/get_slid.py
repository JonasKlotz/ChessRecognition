import argparse
import os

import chess.pgn
import cv2

import debug
from detectboard.detect_board import detect_input_board


def split_board(img):
    """
    Given a board image, returns an array of 64 smaller images.
    """
    arr = []
    sq_len = int(img.shape[0] / 8)
    for i in range(8):
        for j in range(8):
            arr.append(img[i * sq_len: (i + 1) * sq_len, j * sq_len: (j + 1) * sq_len])
    return arr


def get_board_slid(board_path):
    """

    :param board_path: path to a board image
    :return: squares images, cropped image and coordinates of the corners
    """
    corners = detect_input_board(board_path)
    head, tail = os.path.split(board_path)
    tmp_dir = os.path.join(head, "tmp/")
    cropped = cv2.imread(tmp_dir + tail)
    return cropped, corners


def pgn_to_fen_list(png_path):
    pgn = open(png_path)
    fen_list = []
    game = chess.pgn.read_game(pgn)
    board = game.board()

    for move in game.mainline_moves():
        board.push(move)
        fen_list.append(board.fen())
    pgn.close()
    return fen_list


if __name__ == '__main__':
    # Create the parser
    my_parser = argparse.ArgumentParser(prog='Chessy',
                                        description='Chessrecognition programm, evaluates a picture of a chess programm and ')

    # Add the arguments
    my_parser.add_argument('Path',
                           metavar='image_path',
                           type=str,
                           help='the path to the image')

    # Execute parse_args()
    args = my_parser.parse_args()

    input_path = args.Path
    img = cv2.imread(input_path, 1)
    print("Loading board from ", input_path)

    cropped, corners = get_board_slid(input_path)

    print(corners)
    debug.DebugImage(cropped) \
        .save("get_slid_cropped image")

    debug.DebugImage(img) \
        .points(corners, color=(0, 0, 255), size=10) \
        .save("get_points_final_corners")
