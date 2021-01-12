import os

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
