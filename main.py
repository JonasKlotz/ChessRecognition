# main.py
# Import the argparse library
import argparse
import logging
import time

import config
import model
import utility
from calculate_fen.get_fen import get_fen_from_predictions
from detectboard import get_slid, get_board
from process_board import process_board

logging.basicConfig(filename='output.log', level=logging.INFO)


def process_image(path, save=True, model_name="", board_algorithm=""):
    """
    processes an image given by a path.
    serves as main function for a prediction
    the output is logged to the file "output.log" the result is saved in the result file.
    :param path:
    :param save:
    :param model_name:
    :param board_algorithm:
    :return:
    """
    # get configuration for the run
    configurator = config.configurator(model_name)
    num_of_classes = configurator.get_num_of_classes()
    model_path = configurator.get_model_path()
    img_size = configurator.get_model_img_size()
    preprocess_input = configurator.get_model_preprocess()

    if board_algorithm == "":
        board_algorithm = configurator.board_algorithm

    # Get the cropped chessboard image
    start_time = time.process_time()
    logging.info("Starting")
    if board_algorithm == "CPS":
        board_img, corners = get_slid.get_board_cps(path)  # CPS Implementation
    else:
        board_img = get_board.get_board(path)  # my algorithm

    elapsed_time = time.process_time() - start_time
    logging.info("Get Board took " + str(elapsed_time) + "seconds...")

    # split it into squares and predict the pieces
    start_time = time.process_time()
    logging.info("load Model from " + model_path)
    reloaded_model = model.load_compiled_model(model_path)
    logging.info("process board")

    predictions, squares = process_board(board_img, reloaded_model, img_size, preprocess_input)
    # predictions, squares = process_board_no_scaling(board_img, reloaded_model, img_size, preprocess_input)
    elapsed_time = time.process_time() - start_time
    logging.info("Model took " + str(elapsed_time) + "seconds...")

    # Evaluate Predictions
    start_time = time.process_time()
    fen = get_fen_from_predictions(predictions, squares, num_of_classes=num_of_classes)
    # fen = fen_max_prob(predictions) # max search algorithm
    elapsed_time = time.process_time() - start_time
    logging.info("Get Fen took " + str(elapsed_time) + "seconds...")
    print(fen)

    utility.display_fen_board(fen, save=save)

    return fen


if __name__ == '__main__':
    # Create the parser
    my_parser = argparse.ArgumentParser(prog='Single Image Chess Recognition',
                                        description='Chessrecognition programm, evaluates a picture of a chess programm and ')

    # Add the arguments
    my_parser.add_argument('Path',
                           metavar='image_path',
                           type=str,
                           help='the path to the image')

    my_parser.add_argument('-m',
                           '--model_name',
                           type=str,
                           default="MobileNetV2",
                           help='the model name')

    my_parser.add_argument('-b',
                           '--board',
                           type=str,
                           choices=["Mine", "CPS"],
                           default="Mine",
                           help='board recognition algorithm to choose')

    my_parser.add_argument('-s',
                           '--save',
                           action='store_true',
                           default=True,
                           help='store the image')

    # parse the arguments
    args = my_parser.parse_args()
    input_path = args.Path
    save = args.save
    model_name = args.model_name
    board_recognition = args.board

    logging.info("Loading board from " + input_path)
    fen = process_image(input_path, save=save, model_name=model_name, board_algorithm=board_recognition)
