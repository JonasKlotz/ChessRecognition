import numpy as np


def collect_emtpy_squares(predictions, empty_index=6):
    """
    used to evaluate piece color
    takes fields where highest probability is that it is empty
    :param empty_index either 1 for 7 classes or 6 for 13 classes
    :param predictions
    :return: list that contains true if field is empty
    """
    empty = np.argsort(-1 * predictions[:, empty_index])  # sort empty fields after predictions
    empty_fields = [False] * 64
    empty_count = 0
    for i in range(64):
        field_index = empty[i]
        if np.argmax(predictions[field_index]) == 1 or empty_count < 32:  # if field has highest prob to be empty
            empty_fields[field_index] = True
            empty_count += 1
    return empty_fields


def get_neighbourhood(index):
    """
    calculates the neighbouring indices
    :param index: given index
    :return: list of indices, if not on board = -1
    """

    left, right = -10, -10
    top, top_left, top_right = -10, -10, -10
    bot, bot_left, bot_right = -10, -10, -10

    if index - 8 >= 0:  # calc field above
        top = index - 8

    if index + 8 < 63:  # calc field below
        bot = index + 8

    if not index % 8 == 0:
        left = index - 1
        if top >= 0: top_left = index - 8 - 1
        if bot >= 0: bot_left = index + 8 - 1

    if not index % 8 - 1 == 0:  #
        right = index + 1
        if top >= 0: top_right = index - 8 + 1
        if bot >= 0: bot_right = index + 8 + 1

    return [top, top_left, top_right, bot, bot_left, bot_right, left, right]


def remove_from_single_index_array(dictionary, piece_name, index):
    """
    # remove value from array in dictionary - used when piece is not found
    :param dictionary: dict of arrays
    :param piece_name: name of piece
    :param index: on board
    :return: dict of arrays
    """
    dictionary[piece_name].remove(index)
    return dictionary


def remove_from_all_index_array(dictionary, index, fen_gen):
    """
    remove value from all arrays in dictionary - used when piece is found
    :param dictionary: dict of arrays
    :param index: on board
    :return: dict of arrays
    """
    for piece_name in fen_gen:
        try:
            dictionary[piece_name].remove(index)
        except:
            continue
    return dictionary


def remove_empty_fields(index_array, empty_fields, board, remaining_fields, fen_gen):
    """

    :param index_array:
    :param empty_fields:
    :param board:
    :param remaining_fields:
    :return:
    """
    for i in range(len(empty_fields)):
        if empty_fields[i] and board[i] == "1":  # if no kings was placed on field
            index_array = remove_from_all_index_array(index_array, i, fen_gen)
            remaining_fields -= 1

    return board, index_array, remaining_fields


def get_tops(index_array, predictions, fen_gen):
    """
    collects the indices and probabilities of the top next piece
    :param index_array:
    :param predictions:
    :return:
    """
    top_index = []  # contains index of most probable piece
    top_predictions = []

    if len(fen_gen) > 7:  # if 13 classes
        iter_list = [0, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12]  # is used to skip kings
    else:  # 7 classes
        iter_list = [0, 1, 3, 4, 5, 6]  # is used to skip kings

    for i in range(len(iter_list)):
        piece = fen_gen[i]
        index = iter_list[i]

        if index_array[piece]:
            elem = index_array[piece][-1]  # take highest elem from sorted list
            top_index.append(elem)
            top_predictions.append(predictions[top_index[i], index])  # todo Check if correct
        else:
            top_index.append(-1)
            top_predictions.append(-1)

    return top_index, top_predictions


def check_new_piece_found(piece, field_color, used_pieces):
    """
    uses a hashmap to check if max capacity on board is reached, checks for bishop color
    @:returns true if new piece was found
    """
    max_pieces = {"r": 2, "R": 2, "n": 2, "N": 2, "p": 8, "P": 8, "q": 1, "Q": 1, "b_0": 1, "b_1": 1, "B_0": 1,
                  "B_1": 1}
    if piece == "1":  # Empty field found - all empty fields are accepted
        return True

    if piece == "b" or piece == "B":  # bishop found
        piece += "_" + str(field_color)  # 1 black and 1 white bishop

    if used_pieces[piece] > max_pieces[piece] - 1:
        return False
    used_pieces[piece] += 1  # one more piece on board
    return True


def find_remaining(board, remaining_fields, used_pieces, index_array, field_colors, predictions, fen_gen, piece_colors):
    """
    finds the remaining pieces by using TOPS algorithm
    :param board:
    :param remaining_fields:
    :param used_pieces:
    :param index_array:
    :param field_colors:
    :param predictions:
    :param fen_gen:
    :param piece_colors:
    :return:
    """

    while remaining_fields > 0:
        # get the board position of the most probable pieces
        top_index, top_predictions = get_tops(index_array, predictions, fen_gen)

        # which is the most likely piece
        piece_index = np.argmax(top_predictions)  # which piece is it?
        piece_name = fen_gen[piece_index]  # namestring in chess notation

        field_index = top_index[piece_index]  # where on the board is the piece
        field_color = field_colors[field_index]  # what color is the field

        piece = get_piece_fen(piece_name, piece_colors, field_index)
        # if on board
        if check_new_piece_found(piece, field_color, used_pieces):
            # add piece to board
            board[field_index] = piece
            # remove index from index array all arrays
            index_array = remove_from_all_index_array(index_array, field_index, fen_gen)
            remaining_fields -= 1
        else:
            index_array = remove_from_single_index_array(index_array, piece_name, field_index)

    return board


def get_piece_fen(piece_name, piece_colors, field_index):
    """
    determines color of the piece and gets the piece in FEN notation
    :param piece_name:
    :param piece_colors:
    :param field_index:
    :return:
    """
    if len(piece_name) == 1:  # 7 classes
        # check color
        if piece_colors[field_index] == 'w':
            return piece_name.upper()
        else:
            return piece_name
    else:
        if piece_name[0] == 'b':
            return piece_name[1]
        else:
            return piece_name[1].upper()


def get_fen_from_array(fen_array):
    """
    Gets array with FEN content
    combines it to  FEN string
    """
    fen = ""
    empty = 0
    for i in range(len(fen_array) + 1):
        if i == len(fen_array):
            if empty != 0:  # if last field and still empties
                fen += str(empty)
            break

        if i > 0 and i % 8 == 0:  # if new line achieved
            if empty > 0:
                fen += str(empty) + '/'
                empty = 0
            else:
                fen += '/'

        if fen_array[i] == '1':
            empty += 1

        elif empty > 0:  # piece found and empty before that
            fen += str(empty) + str(fen_array[i])
            empty = 0
        else:  # piece found an piece before that
            fen += str(fen_array[i])

    fen = fen + " w - - 0 0"
    return fen
