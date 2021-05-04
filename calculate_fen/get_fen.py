from calculate_fen.fen_utility import get_fen_from_array
from calculate_fen.get_board_colors import get_colors
from utility import display_fen_board


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

    field_colors, piece_colors = get_colors(squares, empty_fields)

    fen_array = get_fen_array(predictions, piece_colors, field_colors, empty_fields)
    fen = get_fen_from_array(fen_array)

    print(fen)
    display_fen_board(fen, save=False)

    return fen
