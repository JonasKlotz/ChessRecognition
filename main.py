# main.py
# Import the argparse library
import argparse
import logging
import time

import model
import utility
from calculate_fen.get_fen import get_fen_from_predictions
from detectboard import get_slid
from process_board import process_board

logging.basicConfig(filename='output.log', level=logging.INFO)


def process_image(path, save=False, model_name='InceptionResNetV2'):
    configurator = utility.configurator(model_name)
    num_of_classes = configurator.get_num_of_classes()
    model_path = configurator.get_model_path()
    img_size = configurator.get_model_img_size()
    preprocess_input = configurator.get_model_preprocess()

    # cropped = get_board.get_board(path, show=False)
    # squares, board_img = get_board.get_squares(cropped, show=False)  #

    start_time = time.process_time()
    logging.info("Starting")
    board_img, corners = get_slid.get_board_slid(path)
    elapsed_time = time.process_time() - start_time
    logging.info("Get Board took " + str(elapsed_time) + "seconds...")

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
    # fen = fen_max_prob(predictions)
    elapsed_time = time.process_time() - start_time
    logging.info("Get Fen took " + str(elapsed_time) + "seconds...")
    print(fen)

    utility.display_fen_board(fen, save=True)

    return fen


if __name__ == '__main__':
    # Create the parser
    my_parser = argparse.ArgumentParser(prog='Chessy',
                                        description='Chessrecognition programm, evaluates a picture of a chess programm and ')

    # Add the arguments
    my_parser.add_argument('Path',
                           metavar='image_path',
                           type=str,
                           help='the path to the image')

    my_parser.add_argument('-m',
                           '--model_name',
                           type=str,
                           help='the modelName')

    my_parser.add_argument('-s',
                           '--save',
                           action='store_true',
                           help='store the image')

    # Execute parse_args()
    args = my_parser.parse_args()

    input_path = args.Path
    save = args.save
    model_name = args.model_name
    logging.info("Loading board from " + input_path)

    if model_name:
        fen = process_image(input_path, save=save, model_name=model_name)
    # print(vars(args))
    else:
        fen = process_image(input_path, save=save)
