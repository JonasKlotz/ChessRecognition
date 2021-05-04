#!/usr/bin/env python
# coding: utf-8


from collections import defaultdict

from calculate_fen.fen_utility import *

fen_gen = ["b", "1", "k", "n", "p", "q", "r", ]


def clear_pawns_back_rank(predictions):
    """
    remove all pawns from the back ranks
    :param predictions:
    :return:
    """
    for i in range(8):
        predictions[i][4] = 0
    for i in range(56, 64):
        predictions[i][4] = 0
    return predictions


def find_kings(index_array, piece_colors, board):
    """
    finds kings in predictions
    @returns updated board
    """
    king = index_array["k"]  # indices of most probable kings reverse
    wk_index, bk_index = -2, -2,

    for i in king:
        if wk_index >= 0 and bk_index >= 0: break  # both kings found

        if piece_colors[i] == 'w' and wk_index < 0:
            if bk_index == 1 and bk_index in get_neighbourhood(i):
                continue
            wk_index = i

        if piece_colors[i] == 'b' and bk_index < 0:
            if bk_index == 1 and wk_index in get_neighbourhood(i):
                continue
            bk_index = i
    board[wk_index] = "K"
    board[bk_index] = "k"
    index_array = remove_from_all_index_array(index_array, wk_index, fen_gen)
    index_array = remove_from_all_index_array(index_array, bk_index, fen_gen)

    return board, index_array


# index arrays - sort elements find highest probability of piece
def create_index_array(predictions):
    bishop = np.argsort(predictions[:, 0])
    empty = np.argsort(predictions[:, 1])
    king = np.argsort(-1 * predictions[:, 2])  # indices of most probable kings reverse
    knight = np.argsort(predictions[:, 3])
    pawn = np.argsort(predictions[:, 4])
    queen = np.argsort(predictions[:, 5])
    rook = np.argsort(predictions[:, 6])

    index_array = defaultdict()
    index_array["b"] = list(bishop)
    index_array["1"] = list(empty)
    index_array["k"] = list(king)
    index_array["n"] = list(knight)
    index_array["p"] = list(pawn)
    index_array["q"] = list(queen)
    index_array["r"] = list(rook)
    return index_array


def get_fen_array(predictions, piece_colors, field_colors, empty_fields):
    """

    :param predictions:
    :param piece_colors:
    :param field_colors:
    :param empty_fields:
    :return:
    """
    # index arrays - sort elements find highest probability of piece
    predictions = clear_pawns_back_rank(predictions)
    index_array = create_index_array(predictions)
    board = [1] * 64
    used_pieces = defaultdict(int)  # dictionary how many pieces where used
    board, index_array = find_kings(index_array, piece_colors, board)

    remaining_fields = 62  # kings are already in place
    fen_gen = ["b", "1", "n", "p", "q", "r", ]  # kings are not necessary

    # remove at least 32 empty fields as empty fields are easy to predict
    board, index_array, remaining_fields = remove_empty_fields(index_array, empty_fields, board, remaining_fields,
                                                               fen_gen)

    board = find_remaining(board, remaining_fields, used_pieces, index_array, field_colors, predictions, fen_gen,
                           piece_colors)

    return board
