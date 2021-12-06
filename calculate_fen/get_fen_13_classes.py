from collections import defaultdict

from calculate_fen.fen_utility import *

# FEN generator for 13 classes
fen_gen = ["bb", "bk", "bn", "bp", "bq", "br", "1", "wb", "wk", "wn", "wp", "wq", "wr"]


def clear_pawns_back_rank(predictions):
    """
    remove all pawns from the back ranks
    :param predictions:
    :return:
    """
    for i in range(8):
        predictions[i][3] = 0  # black pawn
        predictions[i][10] = 0  # white pawn
    for i in range(56, 64):
        predictions[i][3] = 0  # black pawn
        predictions[i][10] = 0  # white pawn
    return predictions


def create_index_array(predictions):
    """

    :param predictions:
    :return: index array
    """
    index_array = defaultdict()
    for i, piece_name in enumerate(fen_gen):
        index_array[piece_name] = list(np.argsort(predictions[:, i]))
    return index_array


def find_kings(index_array, board, predictions):
    """

    :param index_array:
    :param board:
    :return:
    """
    wk_index, bk_index = -1, -1,
    # reverse lists for easier access

    index_array["bk"].reverse()
    black_king = index_array["bk"]
    index_array["wk"].reverse()
    white_king = index_array["wk"]

    i, k = 0, 0
    while wk_index < 0 or bk_index < 0:
        bk_index = black_king[i]
        wk_index = white_king[k]
        if bk_index == wk_index or bk_index in get_neighbourhood(wk_index):  # king not valid
            # find the one with bigger probability
            if predictions[wk_index][8] > predictions[bk_index][1]:
                bk_index = -1
                i += 1
            else:
                wk_index = -1
                k += 1

    board[wk_index] = "K"
    board[bk_index] = "k"

    index_array = remove_from_all_index_array(index_array, wk_index, fen_gen)
    index_array = remove_from_all_index_array(index_array, bk_index, fen_gen)

    return board, index_array


def get_fen_array(predictions, piece_colors, field_colors, empty_fields):
    """

    :param predictions:
    :param piece_colors:
    :param field_colors:
    :param empty_fields:
    :return:
    """
    predictions = clear_pawns_back_rank(predictions)
    index_array = create_index_array(predictions)
    board = ["1"] * 64  # 64 empty squares
    used_pieces = defaultdict(int)  # dictionary how many pieces where used

    board, index_array = find_kings(index_array, board, predictions)

    remaining_fields = 62  # kings are already in place
    fen_gen = ["bb", "bn", "bp", "bq", "br", "1", "wb", "wn", "wp", "wq", "wr"]  # kings are not necessary

    # remove at least 32 empty fields as empty fields are easy to predict
    board, index_array, remaining_fields = remove_empty_fields(index_array, empty_fields, board, remaining_fields,
                                                               fen_gen)

    board = find_remaining(board, remaining_fields, used_pieces, index_array, field_colors, predictions, fen_gen,
                           piece_colors)

    return board
