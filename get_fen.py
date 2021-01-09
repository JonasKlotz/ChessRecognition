#!/usr/bin/env python
# coding: utf-8
from collections import defaultdict

import numpy as np

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
    # TODO sort and take first 32 field safe, continue with fields that have highest probability
    empty_fields = [False] * 64
    for field_index in range(len(predictions)):
        if np.argmax(predictions[field_index]) == 1:  # if field has highest prob to be empty
            empty_fields[field_index] = True

    return empty_fields


"""
takes information about field colour and content of field to evaluate the color of a piece
"""

# ## Extract FEN

# ## Helper functions
# * Fen to Array
# * lazy fen calculation
#

"""
Gets array with FEN content
combines it to valid FEN string
"""


def get_fen_from_array(fen_array):
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
lazy fen calculation takes always the most probable result
useable to test accuracy of model
"""


def most_probable_array(predictions):
    fen_array = []
    for pred in predictions:
        index = np.argmax(pred)  # get highest probable index
        fen_array.append(fen_gen_small[index])
    return fen_array


"""
splits probability of predicted piece to all other pieces (used to null probabilitys)
"""


def split_probability(predictions, field_index, piece_index):
    # dont divide 0
    if predictions[field_index][piece_index] > 0:
        add_value = predictions[field_index][piece_index] / 5
    else:
        add_value = 0

    # Todo: count non zero value for more eff
    for i in range(len(predictions[field_index])):  # for every possible piece
        if i == piece_index:
            predictions[field_index][piece_index] = 0  # own probability set to zero
        elif i == 2 or i == 1:
            continue  # index 2 is King, only one is possible 2 is empty- empty fields stay empty
        else:
            predictions[field_index][i] += add_value  # add splitted prob

    return predictions


def new_piece_found(count, predictions, index_list, field, piece_index, piece, number=2):
    """ used if piece found
    checks if there is already a maximum amount of this piece on the board
    does not consider boundarys

    """
    if len(count[piece]) > number - 1:  # already a queen #TODO Momentan nicht möglich das mehr als 1 queen
        for k in count[piece]:

            if predictions[field][piece_index] > predictions[k][piece_index]:
                count[piece].remove(k)
                index_list.append(k)  # remove old piece from count and put into index list
                # half original distribution, rest equally distributed for all other pieces
                predictions = split_probability(predictions, k, piece_index)
                count[piece].append(field)
                break

        # no new piece found still append
        if field not in count[piece]:
            index_list.append(field)
            predictions = split_probability(predictions, field, piece_index)


    else:
        count[piece].append(field)
    return predictions


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


def find_king(predictions, count, index_list):
    """
    todo vlt king and queen together
    @binds: 2 kings only, not possible to be faced close to each other
    idea take most probable king for each color
    updates count dict
    """
    not_found = True
    white_king, wk_index, black_king, bk_index = 0, 0, 0, 0
    while not_found:
        # iterate through predictions find 2 most probable kings
        for i in range(len(predictions)):
            if piece_colors[i] == 'w':
                if predictions[i][2] > white_king:
                    white_king = predictions[i][2]
                    wk_index = i

            elif piece_colors[i] == 'b':
                if predictions[i][2] > black_king:
                    black_king = predictions[i][2]
                    bk_index = i

        # calculate neighbourhood only whiteking
        neighbourhood = get_neighbourhood(wk_index)

        if bk_index in neighbourhood:  # kings are close to each other
            if predictions[wk_index][2] > predictions[bk_index][2]:
                predictions = split_probability(predictions, bk_index, 2)

            else:
                predictions = split_probability(predictions, wk_index, 2)

        else:
            break

    count["K"].append(wk_index)
    index_list.remove(wk_index)

    count["k"].append(bk_index)  # 60
    index_list.remove(bk_index)
    print(type(predictions))
    return predictions


# first find king for each side to ensure that there is one
def get_fen_array(predictions, piece_colors):
    fen_gen = ["b", "1", "k", "n", "p", "q", "r"]

    predictions = np.asarray(predictions)
    count = defaultdict(list)  # count dict - how many pieces of each class are on board
    index_list = list(np.arange(64))  # field indizes

    # find most probable king
    predictions = find_king(predictions, count, index_list)

    # no pawns on back rank
    for field in range(8):
        predictions = split_probability(predictions, field, 4)
    for field in range(56, 64):
        predictions = split_probability(predictions, field, 4)

    i = 0
    while index_list:
        # i += 1
        field_index = index_list.pop(0)

        pred = predictions[field_index]
        piece_index = np.argmax(pred)  # get highest probable index in piece array
        piece = fen_gen[piece_index]

        # f piece is white, piece num gets changed to uppercase
        if piece_colors[field_index] == 'w':
            piece = piece.upper()
        # print("most probable piece= ", piece)

        if piece == "k" or piece == "K":  # king found
            predictions = new_piece_found(count, predictions, index_list, field_index, piece_index, piece, number=1)

        elif piece == "n" or piece == "N":  # knight found
            predictions = new_piece_found(count, predictions, index_list, field_index, piece_index, piece, number=2)

        elif piece == "q" or piece == "Q":  # queen found
            predictions = new_piece_found(count, predictions, index_list, field_index, piece_index, piece, number=1)

        elif piece == "b" or piece == "B":  # bishop found
            predictions = new_piece_found(count, predictions, index_list, field_index, piece_index, piece, number=2)

        elif piece == "r" or piece == "R":  # rook found
            predictions = new_piece_found(count, predictions, index_list, field_index, piece_index, piece, number=2)

        elif piece == "p" or piece == "P":  # pawn found
            predictions = new_piece_found(count, predictions, index_list, field_index, piece_index, piece, number=8)
        else:  # empty
            count[piece].append(field_index)

    print("Final Count")
    print(dict(count))
    print()

    fen_array = ["1"] * 64

    for key, values in count.items():
        for value in values:
            try:
                fen_array[value] = key
                # print("Field: ", value, " = ", key)
            except:
                print(key, value)

    print(len(fen_array))
    print(fen_array)
    return fen_array


if __name__ == '__main__':
    # todo args machen
    dir_path = '/home/joking/Projects/Chessrecognition/Data/chessboards/squares/1/'
    model_path = '/home/joking/Projects/Chessrecognition/models/trained_models/best_model.h5'

    # load ima
    tensor_list, square_list = utility.load_square_lists_from_dir(dir_path)

    # model = tf.keras.models.load_model(model_path)
    chess_model = model.load_model("1609865413_small_Model.h5")
    predictions = model.get_predictions(chess_model, tensor_list)
    empty_fields = collect_emtpy_squares(predictions)

    board_colors, piece_colors = get_board_colors.get_colors(square_list, empty_fields)

    fen_array = get_fen_array(predictions, piece_colors)
    fen = get_fen_from_array(fen_array)

    print(fen)
    utility.display_fen_board(fen, save=False)

"""
# ## Alternative approach to evaluate FEN Position
# * Group the figures into their color
# * Group the figures into their class:
# * First find Kings take most probable king for both sides
# * in this step modifications of probability can be used
# * take at least 32 empty fields and all with very high probability of beeing empty ( can be sorted beforehand) empty fields can be recognized quite easily
# * remove pawns from back rank distribute probability
# * Group all pieces regarding the highest probability of beeing that piece and take top tier candidates (number depends on class(minor and rook = 2, queen = 1)
# * - eG for white bishop take the 2 most likely bishops - apply constraints one white one black bishop - even tho possible it is very uncommon to have two bishops of same color, as when a pawn is promoted there aren't much piece on board and queen is always prefered.
#
# * Then group other pieces according to that scheme

###################################### How to find FEN
#


    # todo Ensure 32 empty fields
    # todo max 16 pieces of color
    # todo pawns + queen <= 9
    # todo pawns + minor piece <= 10
# 2B5/1p6/4kQk1/2k1Q3/3kN3/6N1/NN4kP/3RQN1R
#
#

#
# pawns + queen <= 9
#
# pawns + minor piece <= 10
#
# 
#
# sliding windows with stockfish https://arxiv.org/pdf/1607.04186.pdf
#
# Clustering  into  groups:
#     After  clustering  similar  figuresinto groups,
#     we can reject certain trivial cases based on thecardinality of clusters.
#     For example, having two groups ofsimilar figures with cardinalities of{5,1}
#     and candidatesof{bishop,pawn},  it  can  be  deduced  that  there  are
#     fivepawns and one bishop.
# 
#
#      https://chess.stackexchange.com/questions/1482/how-to-know-when-a-fen-position-is-legal?rq=1
#     approach linear programming
#

"""


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
