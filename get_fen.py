#!/usr/bin/env python
# coding: utf-8

import get_board_colors
import model
import utility

fen_gen_small = ["b", "1", "k", "n", "p", "q", "r", ]
# 1 is empty


"""
@returns list that contains true if field is empty 
used to evaluate piece color
takes fields where highest probability is that it is empty
should work out at beginning as emtpy fields can be recognized quite good
"""


def collect_emtpy_squares(predictions):
    empty = np.argsort(-1 * predictions[:, 1])  # sort empty fields after predictions
    empty_fields = [False] * 64
    empty_count = 0
    for i in range(64):
        field_index = empty[i]
        if np.argmax(predictions[field_index]) == 1 or empty_count < 32:  # if field has highest prob to be empty
            empty_fields[field_index] = True
            empty_count += 1
    return empty_fields


# !/usr/bin/env python
# coding: utf-8

# In[335]:


from collections import defaultdict

import numpy as np

fen_gen = ["b", "1", "k", "n", "p", "q", "r", ]


def get_fen_from_array(fen_array):
    """
    Gets array with FEN content
    combines it to valid FEN string
    """
    fen = ""
    empty = 0
    for i in range(len(fen_array)):
        if i > 0 and i % 8 == 0:
            if empty > 0:
                fen += str(empty) + '/'
                empty = 0
            else:
                fen += '/'
        if fen_array[i] == '1':
            empty += 1
        elif empty > 0:
            fen += str(empty) + fen_array[i]
            empty = 0
        else:
            fen += fen_array[i]
    fen = fen + " w - - 0 0"
    print(fen)
    return fen


"""
calculates the neighbouring indices
"""


def get_neighbourhood(wk_index):
    left, right = -1, -1
    top, top_left, top_right = -1, -1, -1
    bot, bot_left, bot_right = -1, -1, -1

    if wk_index - 8 >= 0:  # calc field above
        top = wk_index - 8

    if wk_index + 8 < 63:  # calc field below
        bot = wk_index + 8

    if not wk_index % 8 == 0:
        left = wk_index - 1
        if top >= 0: top_left = wk_index - 8 - 1
        if bot >= 0: bot_left = wk_index + 8 - 1

    if not wk_index % 8 - 1 == 0:  #

        right = wk_index + 1
        if top >= 0: top_right = wk_index - 8 + 1
        if bot >= 0: bot_right = wk_index + 8 + 1
    return [top, top_left, top_right, bot, bot_left, bot_right, left, right]


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


# remove value from array in dictionary - used when piece is not found
def remove_from_single_index_array(dictionary, piece_name, index):
    dictionary[piece_name].remove(index)
    return dictionary


# remove value from all arrays in dictionary - used when piece is found
def remove_from_all_index_array(dictionary, index):
    for piece_name in fen_gen:
        try:
            dictionary[piece_name].remove(index)
        except:
            continue

    return dictionary


"""
top index contains index for most probable next piece eG ["b"][-1] is the most probable next bishop
"""


def get_tops(index_array, predictions):
    top_index = []  # contains index of most probable piece
    top_predictions = []
    fen_gen = ["b", "1", "n", "p", "q", "r"]  # no king as they are already placed
    iter_list = [0, 1, 3, 4, 5, 6]  # is used to skip kings

    for i in range(6):
        piece = fen_gen[i]
        index = iter_list[i]

        if index_array[piece]:
            elem = index_array[piece][-1]  # take highest elem from sorted list
            top_index.append(elem)
            top_predictions.append(predictions[top_index[i], index])
        else:
            top_index.append(-1)
            top_predictions.append(-1)

    return top_index, top_predictions


"""
finds kings in predictions 
@returns updated board
"""


def find_kings_2(index_array, piece_colors, board):
    king = index_array["k"]  # indices of most probable kings reverse
    wk_index, bk_index = -1, -1,

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
    index_array = remove_from_all_index_array(index_array, wk_index)
    index_array = remove_from_all_index_array(index_array, bk_index)

    return board, index_array


# In[340]:


"""
pawns can't be on the back rank
"""


def clear_pawns_back_rank(predictions):
    for i in range(8):
        predictions[i][4] = 0
    for i in range(56, 64):
        predictions[i][4] = 0
    return predictions


"""
uses a hashmap to check if max capacity on board is reached, checks for bishop color
@:returns true if new piece was found
"""


def check_new_piece_found(piece, field_color, used_pieces):
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


def remove_empty_fields(index_array, empty_fields, board, remaining_fields):
    for i in range(64):
        if empty_fields[i]:
            board[i] = "1"
            index_array = remove_from_all_index_array(index_array, i)
            remaining_fields -= 1

    return board, index_array, remaining_fields


def get_fen_array(predictions, piece_colors, field_colors, empty_fields):
    ########## Setup ###########
    fen_gen = ["b", "1", "n", "p", "q", "r"]  # no king as they are already placed
    # index arrays - sort elements find highest probability of piece
    predictions = clear_pawns_back_rank(predictions)
    index_array = create_index_array(predictions)
    board = [1] * 64
    used_pieces = defaultdict(int)
    board, index_array = find_kings_2(index_array, piece_colors, board)
    remaining_fields = 62  # kings are already in place
    # remove at least 32 empty fields as empty fields are easy to predict
    board, index_array, remaining_fields = remove_empty_fields(index_array, empty_fields, board, remaining_fields)

    while remaining_fields > 0:
        # get the board position of the most probable pieces
        top_index, top_predictions = get_tops(index_array, predictions)

        # which is the most likely piece
        piece_index = np.argmax(top_predictions)  # which piece is it?
        piece_name = fen_gen[piece_index]  # namestring in chess notation

        field_index = top_index[piece_index]  # where on the board is the piece
        field_color = field_colors[field_index]  # what color is the field

        # check color
        if piece_colors[field_index] == 'w':
            piece = piece_name.upper()
        else:
            piece = piece_name

        # if on board
        if check_new_piece_found(piece, field_color, used_pieces):

            # add piece to board
            board[field_index] = piece
            # remove index from index array all arrays
            index_array = remove_from_all_index_array(index_array, field_index)
            remaining_fields -= 1
        else:
            index_array = remove_from_single_index_array(index_array, piece_name, field_index)

    return board


def get_fen_from_predictions(predictions, square_list):
    empty_fields = collect_emtpy_squares(predictions)

    field_colors, piece_colors = get_board_colors.get_colors(square_list, empty_fields)

    fen_array = get_fen_array(predictions, piece_colors, field_colors, empty_fields)
    fen = get_fen_from_array(fen_array)

    utility.display_fen_board(fen, save=False)
    return fen


if __name__ == '__main__':
    dir_path = '/home/joking/Projects/Chessrecognition/data/chessboards/squares/1/'
    model_path = '/home/joking/Projects/Chessrecognition/models/trained_models/best_model.h5'

    # load ima
    tensor_list, square_list = utility.load_square_lists_from_dir(dir_path)

    # model = tf.keras.models.load_model(model_path)
    chess_model = model.load_model(model_path)
    predictions = model.get_predictions(chess_model, tensor_list)

    empty_fields = collect_emtpy_squares(predictions)
    board_colors, piece_colors = get_board_colors.get_colors(square_list, empty_fields)

    fen_array = get_fen_array(predictions, piece_colors, board_colors)
    fen = get_fen_from_array(fen_array)

    utility.display_fen_board(fen, save=False)


def load_single_image():
    img_path = '/home/joking/Projects/Chessrecognition/Data/output_test/bk/0760_4.jpg'

    #
    # load a single image
    # new_image = utility.load_image(img_path, show=False)
    # check prediction
    # pred = reloaded_keras_model.predict(new_image)
    # print(np.squeeze(pred).shape)
    # index = np.argmax(pred[0])
    # print(pieces[index])

    # plot_prob(pred[0], pieces, new_image)
