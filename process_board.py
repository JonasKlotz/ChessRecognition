# Mobile Net V2

import model
import utility


def split_board_scaling(img, scaling_factor=0.):
    """
    Given a board image, returns an array of 64 smaller images.
    scaled by scaling factor to evaluate the full piece
    """
    height, width, channels = img.shape
    arr = []
    sq_len = int(img.shape[0] / 8)
    for i in range(8):
        for j in range(8):
            # use scaling_factor % from square above, except in first row
            y1 = int(max(0, (i - scaling_factor) * sq_len))  # do not leave image dimensions
            arr.append(img[y1: (i + 1) * sq_len, j * sq_len: (j + 1) * sq_len])
    return arr


def double_split_board(img):
    """

    :param img:
    :return: 2 np arrays each containg 64 squares with different scaling factors
    """
    squares = []
    squares.append(split_board_scaling(img))
    squares.append(split_board_scaling(img, scaling_factor=0.3))
    return squares


def merge_predictions(preds_list):
    """
    simple merge not considering any pieces
    :param preds_list:
    :return:
    """

    predictions = (preds_list[0] + preds_list[1]) / 2
    return predictions


def process_board(board_img, reloaded_model, img_size, preprocess_input):
    squares_list = double_split_board(board_img)
    preds_list = []

    for squares in squares_list:
        tensor_list = utility.load_tensor_list_from_squares(squares, img_size, preprocess_input)
        # get Predictions
        predictions = model.get_predictions(reloaded_model, tensor_list)
        preds_list.append(predictions)

    predictions = merge_predictions(preds_list)

    return predictions, squares_list[0]
