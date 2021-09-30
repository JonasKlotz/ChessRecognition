#!/usr/bin/env python
# coding: utf-8

import cv2
import matplotlib.pyplot as plt
import numpy as np


def find_field_colour(img, show=False, i=0):
    """ input field output color
    1 is white 0 is black
    @:returns 1 if field is white, 0 if black
    """
    gray_square = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray_square, (5, 5), 0)
    _, img_binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # remove noise
    morph_kernel = np.ones((15, 15), np.uint8)
    img_binary = cv2.morphologyEx(img_binary, cv2.MORPH_CLOSE, morph_kernel)

    rows, cols = img_binary.shape

    if show:
        fig = plt.figure(figsize=(3, 3))
        plt.imshow(img)
        plt.imsave(f"data/{i}.jpg", img)

    n_white_pix = cv2.countNonZero(img_binary)
    n_black_pix = rows * cols - n_white_pix

    if n_white_pix > n_black_pix:
        return 1
    return 0


def compare_board(board_to_compare):
    """
    compares board to the 2 possible boards
    @:returns board color array, true or false(if has to be turned or not
    """
    board_1 = [0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0,
               1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0]
    board_2 = [1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1,
               0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1]

    # Im Falle Board 2 muss das brett zusätzlich gedreht werden!!!

    if np.sum(board_to_compare == board_1) > np.sum(board_to_compare == board_2):
        return board_1, True
    return board_2, False


def find_board_colour(square_list):
    """
    gets quares as input
    collects their color
    @:returns board color array, True or false for turning of board
    """
    board_color = np.zeros(len(square_list), dtype=int)  # für jedes square eine liste
    i = 0
    for i in range(len(square_list)):
        board_color[i] = find_field_colour(square_list[i], i=i)
    return compare_board(board_color)


def get_piece_colors(square_list, board_color, empty_fields):
    piece_colors = [0] * 64

    for i in range(len(square_list)):
        img = square_list[i]  # grab img
        field_color = board_color[i]  # grab color of field

        if not empty_fields[i]:

            blur = cv2.GaussianBlur(img, (5, 5), 0)
            gray_square = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
            otsu_threshold, image_otsu = cv2.threshold(gray_square, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU, )
            rows, cols = image_otsu.shape
            img_otsu_inverted = cv2.bitwise_not(image_otsu)

            # remove noise
            morph_kernel = np.ones((15, 15), np.uint8)
            out = cv2.morphologyEx(img_otsu_inverted, cv2.MORPH_CLOSE, morph_kernel)
            # cv2.imshow('image',img)
            # cv2.waitKey(0)

            n_white_pix = cv2.countNonZero(out)
            n_black_pix = rows * cols - n_white_pix
            thresh = (rows * cols) / 3.5  # threshold 1/3.5 of img has to be of the other color (maybe play around)

            if field_color == 1:  # field is white
                if n_white_pix >= thresh:  # threshold einsetzen wie großer teil des bildes darf besetzt sein
                    piece_colors[i] = 'b'
                else:
                    piece_colors[i] = 'w'

            if field_color == 0:  # field is black

                if n_black_pix > thresh:  # threshold einsetzen wie großer teil des bildes darf besetzt sein 1/3
                    piece_colors[i] = 'w'
                else:
                    piece_colors[i] = 'b'

    return piece_colors


def get_colors(square_list, empty_fields):
    """
    main function to get Colors
    :param square_list:
    :param empty_fields:
    :return: board_color, piece_colors, turn(bool) if has to be turned
    """
    board_color, turn = find_board_colour(square_list)

    piece_colors = get_piece_colors(square_list, board_color, empty_fields)
    return board_color, piece_colors, turn


#


def rotate_board(board):
    """
    :todo unbedingt verschönern
    :param board: chessboard list
    :return:
    """
    rotation_list = [56, 48, 40, 32, 24, 16, 8, 0,
                     57, 49, 41, 33, 25, 17, 9, 1,
                     58, 50, 42, 34, 26, 18, 10, 2,
                     59, 51, 43, 35, 27, 19, 11, 3,
                     60, 52, 44, 36, 28, 20, 12, 4,
                     61, 53, 45, 37, 29, 21, 13, 5,
                     62, 54, 46, 38, 30, 22, 14, 6,
                     63, 55, 47, 39, 31, 23, 15, 7]
    result = []
    for i in range(64):
        result.append(board[rotation_list[i]])
    return result

    """
    if isinstance(board, list):
        print("LIST")
        matrix = np.asarray(board).reshape((8, 8))
        matrix = np.rot90(matrix, 3)
        return matrix.reshape(64, ).tolist()

    elif isinstance(board, np.ndarray):
        print("NP ARRAY")
        print(board.shape)
        #matrix = np.asarray(board).reshape((8, 8))
        matrix = np.rot90(board, 3)
        return matrix#.reshape(64, )"""
