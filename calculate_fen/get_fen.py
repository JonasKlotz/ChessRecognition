import numpy as np

from calculate_fen.fen_utility import get_fen_from_array
from calculate_fen.get_board_colors import get_colors, rotate_board


def get_fen_from_predictions(predictions, squares, num_of_classes):
    """

    :param predictions:
    :param squares:
    :param num_of_classes:
    :return:
    """

    if num_of_classes == 7:
        from calculate_fen.get_fen_7_classes import collect_emtpy_squares, get_fen_array
    else:
        from calculate_fen.get_fen_13_classes import collect_emtpy_squares, get_fen_array

    empty_fields = collect_emtpy_squares(predictions)
    field_colors, piece_colors, turn = get_colors(squares, empty_fields)

    # wenn unteres linkes Feld nicht weiß ist, muss das brett gedreht werden
    if turn:
        empty_fields = rotate_board(empty_fields)
        piece_colors = rotate_board(piece_colors)
        squares = rotate_board(squares)
        predictions = np.asarray(rotate_board(predictions))

    # dazu finden was unteres rechtes feld ist, funktion für "drehen"
    fen_array = get_fen_array(predictions, piece_colors, field_colors, empty_fields)
    fen = get_fen_from_array(fen_array)

    # print(fen)
    # display_fen_board(fen, save=False)

    return fen


def fen_max_prob(predictions, num_of_classes=7):
    """
    Always choose piece with maximum probability
    :return:
    """
    board = []
    fen_gen = ["b", "k", "n", "p", "q", "r", "1", "B", "K", "N", "P", "Q", "R"]
    for pred in predictions:
        piece_index = np.argmax(pred)  # which piece is it?
        piece_name = fen_gen[piece_index]  # namestring in chess notation

        board.append(piece_name)
    fen = get_fen_from_array(board)
    return fen
