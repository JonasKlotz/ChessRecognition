# main.py
# Import the argparse library
import argparse
import logging
import time

import get_slid
import model
# do some stuff
import utility
from calculate_fen.get_fen import get_fen_from_predictions

##### Model Specific Parameters
# from keras.applications.inception_resnet_v2 import preprocess_input
# from keras.applications.resnet_v2 import preprocess_input

"""
# Mobile Net V2
from keras.applications.mobilenet_v2 import preprocess_input
num_of_classes=13
img_size = 224
model_path = '/home/joking/PycharmProjects/Chess_Recognition/models/13_class_MobileNetV2/model.h5'
"""
# Resnet V2
from keras.applications.inception_resnet_v2 import preprocess_input

num_of_classes = 7
img_size = 150
model_path = '/home/joking/PycharmProjects/Chess_Recognition/models/InceptionResNetV2/model.h5'

logging.getLogger("tensorflow").setLevel(logging.ERROR)


def process_image(path, save=False):
    start_time = time.process_time()

    # cropped = get_board.get_board(path, show=False)
    # squares, board_img = get_board.get_squares(cropped, show=False)  #
    # print("Anzahl gefundene Squares ", len(squares))

    squares, board_img, corners = get_slid.get_board_slid(path)


    if save:
        save_path = utility.fill_dir_with_squares(path, squares)
        print("Saved to '%s'" % save_path)
        # tensor_list, square_list = utility.load_square_lists_from_dir(save_path)

    tensor_list = utility.load_tensor_list_from_squares(squares, img_size, preprocess_input)

    elapsed_time = time.process_time() - start_time
    print("Processing took ", elapsed_time, "seconds...")

    # get Predictions
    start_time = time.process_time()
    reloaded_model = model.load_compiled_model(model_path)
    predictions = model.get_predictions(reloaded_model, tensor_list)

    elapsed_time = time.process_time() - start_time
    print("Model took ", elapsed_time, "seconds...")

    # Evaluate Predictions
    start_time = time.process_time()
    fen = get_fen_from_predictions(predictions, squares, num_of_classes=num_of_classes)
    elapsed_time = time.process_time() - start_time
    print("Get Fen took ", elapsed_time, "seconds...")
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
    my_parser.add_argument('-s',
                           '--save',
                           action='store_true',
                           help='store the image')

    # Execute parse_args()
    args = my_parser.parse_args()

    input_path = args.Path
    save = args.save

    # print(vars(args))
    path = "data/chessboards/32.jpg"
    print("Loading board from ", path)
    process_image(path, save=save)
