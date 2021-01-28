import glob
import os

import chess.pgn
import cv2

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
    corners = detect_input_board(board_path)
    head, tail = os.path.split(board_path)
    tmp_dir = os.path.join(head, "tmp/")
    cropped = cv2.imread(tmp_dir + tail)
    squares = split_board(cropped)
    return squares, cropped


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
    pgn_name = "DingvsCarlsen2019.pgn"
    dir_path = "/home/joking/Carlsen Ding Images"
    addrs = glob.glob(dir_path + "/*.png")

    fen_list = pgn_to_fen_list(dir_path + "/" + pgn_name)
    for i in range(len(addrs)):
        addr = addrs[i]
        img = cv2.imread(addr)
        # cv2.imshow("img", img)
        # cv2.waitKey(0)
        fen = fen_list[i]
        print(addr, fen)
        squares, cropped = get_board_slid(addr)
        break
