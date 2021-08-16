import copy
import glob
import time
from collections import defaultdict

import cv2
import natsort
import numpy as np
import pandas as pd
from keras.applications.inception_resnet_v2 import preprocess_input as resnet_input
from keras.applications.mobilenet_v2 import preprocess_input as mobilenet_input
from keras.applications.nasnet import preprocess_input as nasnet_input
from keras.applications.xception import preprocess_input as xception_input

import get_points
import model
import utility
from calculate_fen.get_fen import fen_max_prob
from get_slid import get_board_slid

# from training.generic_model import get_y_pred, generate_generators, evaluate_model

res_dir = "/home/joking/Projects/Chessrecognition/Data/Results/Board Recognition/trash/"
images_path = "/home/joking/Projects/Chessrecognition/Data/Results/pieces/cropped/"
fens_path = "/home/joking/Projects/Chessrecognition/Data/Results/pieces/string.fen"

model_names = ["InceptionResNetV2", "MobileNetV2", "NASNetMobile", "Xception"]
num_of_classes = 13


def draw_and_save(index, img, points, path, slid=True):
    img2 = copy.copy(img)
    for point in points:
        cv2.circle(img2, (int(point[0]), int(point[1])), 10, (0, 0, 255), -1)

    if slid:
        save_path = path + str(index + 1) + "_slid.jpg"
    else:
        save_path = path + str(index + 1) + "_points.jpg"
    cv2.imwrite(save_path, img2)
    print("printed to ", save_path)


def evaluate_board(index, path):
    img = cv2.imread(path, 1)
    elapsed_time_slid, elapsed_time_points = -1, -1
    try:
        # Evaluate SLID
        start_time = time.time()
        squares, board_img, corners = get_board_slid(path)
        elapsed_time_slid = time.time() - start_time
        print("Processing SLID took ", elapsed_time_slid, "seconds...")
        draw_and_save(index, img, corners, res_dir, slid=True)

    except:
        print("SLID failed")

    try:
        # Evaluate GETPOINTS
        start_time = time.time()
        corners2 = get_points.get_points(img=img)
        elapsed_time_points = time.time() - start_time
        print("Processing GETPOINTS took ", elapsed_time_points, "seconds...")
        draw_and_save(index, img, corners2, res_dir, slid=False)
    except:
        print("points failed")

    return elapsed_time_slid, elapsed_time_points


def read_corner_file(path="strings.corners"):
    results = []
    with open(path) as file:
        for line in file:
            line = line.replace('(', "").replace(')', "").replace('\n', "").replace(' ', '')
            line = line.split(',')
            r = list(map(int, line))
            tl, tr, br, bl = (r[0], r[1]), (r[2], r[3]), (r[4], r[5]), (r[6], r[7])
            r = [tl, tr, br, bl]
            results.append(r)
    return results


def eval_boards():
    """
    evaluates accuracy of board detection
    :return:
    """
    data_dir = "/home/joking/Projects/Chessrecognition/Data/chessboards/board_recog/"
    res_slid_list, res_points_list = [], []

    filenames = natsort.natsorted(glob.glob(data_dir + '*.jpg'))

    for i, filename in enumerate(filenames):
        print(filename)
        res_slid, res_points = evaluate_board(i, filename)
        res_points_list.append(res_points)
        res_slid_list.append(res_slid)

    data = np.column_stack((res_points_list, res_slid_list))
    print(data)
    pd.DataFrame(data, columns=["Ours", "CPS"]).to_csv(res_dir + '/results.csv', index=False)


#############################################################################################

def read_fen_file(path="string.fen"):
    results = []
    with open(path) as file:
        for fen in file:
            results.append(fen)
    return results


def read_images(path, n):
    results = []
    for i in range(n):
        img_path = path + "{}.jpg".format(i + 1)
        print("Read ", img_path)
        img = cv2.imread(img_path, 1)
        results.append(img)
    return results


def load_models():
    # modelname is key, Model, preprocessing function, image size
    models = {"InceptionResNetV2": [None, resnet_input, 150],
              "MobileNetV2": [None, mobilenet_input, 224],
              "NASNetMobile": [None, nasnet_input, 224],
              "Xception": [None, xception_input, 299],
              }
    for model_name in model_names:
        model_path = '/home/joking/PycharmProjects/Chess_Recognition/models/{}_classes/{}/model.h5' \
            .format(num_of_classes, model_name)
        models[model_name][0] = model.load_compiled_model(model_path)

    return models


def eval_pieces():
    images_path = "/home/joking/Projects/Chessrecognition/Data/Results/pieces/cropped/"
    fens_path = "/home/joking/Projects/Chessrecognition/Data/Results/pieces/string.fen"

    # prepare data
    data = defaultdict()
    for key in model_names:
        data["{}_pred".format(key)] = []
        data["{}_time".format(key)] = []

    # read fen File for true FEN String
    data["True"] = read_fen_file(fens_path)

    # Load all models
    models = load_models()
    print("Models loaded")

    n = 30  # n images
    # read images
    images = read_images(images_path, n)
    # for every image
    for i in range(n):
        # load squares
        squares = get_points.split_board(images[i])
        # evaluate with every model
        for model_name in model_names:
            # start preprocessing
            preprocess_fct = models[model_name][1]
            img_size = models[model_name][2]
            tensor_list = utility.load_tensor_list_from_squares(squares, img_size, preprocess_fct)

            # evaluate
            start_time = time.process_time()
            predictions = model.get_predictions(models[model_name][0], tensor_list)
            fen = fen_max_prob(predictions)
            # fen = get_fen_from_predictions(predictions, squares, num_of_classes=num_of_classes)
            fen = fen.replace(" w - - 0 0", "")
            elapsed_time = time.process_time() - start_time
            # save result
            data["{}_pred".format(model_name)].append(fen)
            data["{}_time".format(model_name)].append(elapsed_time)
        print(i, " Done")
    print(data)
    pd.DataFrame(data).to_csv("pieces_data.csv", index=False)
    print("Saved to pieces_data.csv")


"""
def eval_pieces_testset():
    models = load_models()

    i = 0
    for model_name in models.keys():

        reloaded_model = models[model_name][0]
        preprocess_input = models[model_name][1]
        img_size = models[model_name][2]
        train_dataset, validation_dataset, test_dataset =   generate_generators(preprocess_input, img_size)

        evaluate_model(reloaded_model, test_dataset, "", history=None)

        y_pred = get_y_pred(reloaded_model, test_dataset)
        print(model_name)
        print(classification_report(test_dataset.labels, y_pred, target_names=models.keys()))

"""

if __name__ == '__main__':
    modelx = load_models()["InceptionResNetV2"]

    img = cv2.imread("/home/joking/PycharmProjects/Chess_Recognition/tmp/59.jpg")
    squares = [img]
    reloaded_model = modelx[0]
    preprocess_fct = modelx[1]
    img_size = modelx[2]
    tensor_list = utility.load_tensor_list_from_squares(squares, img_size, preprocess_fct)

    predictions = model.get_predictions(reloaded_model, tensor_list)
    np.set_printoptions(suppress=True)
    print(predictions)
