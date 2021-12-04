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
from sklearn.metrics import f1_score

import model
from calculate_fen.get_fen import get_fen_from_predictions, fen_max_prob
from detectboard import get_board
from detectboard.get_slid import get_board_slid
# from training.generic_model import get_y_pred, generate_generators, evaluate_model
from process_board import process_board

res_dir = "/home/joking/Projects/Chessrecognition/Data/Results/Board Recognition/trash/"
images_path = "/home/joking/Projects/Chessrecognition/Data/Results/pieces/cropped/"
fens_path = "/home/joking/Projects/Chessrecognition/Data/Results/pieces/string.fen"

model_names = ["InceptionResNetV2", "MobileNetV2", "NASNetMobile", "Xception"]
num_of_classes = 7


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
        corners2 = get_board.get_points(img=img)
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


def convert_fen_to_array(fen):
    """
    :param fen: FEN in string format
    :return: FEN in FEN array format
    """
    fen = fen.replace('/', "").replace('\n', "")  # [::-1]

    # convert FEN to Array
    fen_array = [] * 64
    fen = [char for char in fen]  # splot string into chars
    i = 0
    while fen:
        elem = fen.pop(0)
        if elem.isdigit():
            empty = int(elem)
            while empty > 0:
                fen_array.append('1')
                empty -= 1

        else:
            fen_array.append(elem)
        i += 1
    return fen_array


def read_images(path, n):
    results = []
    for i in range(n):
        img_path = path + "{}.jpg".format(i + 1)
        print("Read ", img_path)
        img = cv2.imread(img_path, 1)
        results.append(img)
    return results


def load_models():
    """
    loads models
    :return: dict of models with preprocess images size and model
    """
    # modelname is key, Model, preprocessing function, image size
    models = {"InceptionResNetV2": [None, resnet_input, 150],
              "MobileNetV2": [None, mobilenet_input, 224],
              "NASNetMobile": [None, nasnet_input, 224],
              "Xception": [None, xception_input, 299],
              }
    for model_name in model_names:
        model_path = '/home/joking/PycharmProjects/Chess_Recognition/models/{}_classes/new/{}/model.h5' \
            .format(num_of_classes, model_name)
        models[model_name][0] = model.load_compiled_model(model_path)

    return models

def compare_fen_files(true_path, fens_path):
    """
    Compars two FEN files and saves accuracy
    :return:
    """
    #true_path = "/home/joking/Projects/Chessrecognition/Data/Results/pieces/string.fen"
    #fens_path = "/home/joking/Projects/Chessrecognition/Data/Results/pieces/wolf.fen"

    true = read_fen_file(true_path)
    fens = read_fen_file(fens_path)

    true_array = list(map(convert_fen_to_array, true))
    fens_array = list(map(convert_fen_to_array, fens))
    data = defaultdict()
    data["acc"] = []
    for i, true in enumerate(true_array):
        true = np.array(true)
        res = np.array(fens_array[i])

        acc = np.count_nonzero(true == res) / 64
        print(acc)
        data["acc"].append(acc)

    name = "fen_eval.csv"
    pd.DataFrame(data).to_csv(name, index=False)
    print("Saved to " + name)


def eval_pieces():
    """
    Evaluates the piece recognition
    loads model with n of classes, images and true fen, then calculates acc f1 and runtime and stores it
    :return:
    """
    images_path = "/home/joking/Projects/Chessrecognition/Data/Results/pieces/cropped/"
    fens_path = "/home/joking/Projects/Chessrecognition/Data/Results/pieces/string.fen"

    # prepare data
    data = defaultdict()
    for key in model_names:
        data["{}_pred".format(key)] = []
        data["{}_time".format(key)] = []
        data["{}_acc".format(key)] = []
        data["{}_f1".format(key)] = []

    # read fen File for true FEN String
    results = read_fen_file(fens_path)
    data["True"] = results
    results_array = list(map(convert_fen_to_array, results))

    # Load all models
    models = load_models()
    print("Models loaded")

    n = 30  # n images
    # read images
    images = read_images(images_path, n)
    # for every image
    for i in range(n):

        # evaluate with every model
        for model_name in model_names:
            # start preprocessing
            preprocess_fct = models[model_name][1]
            img_size = models[model_name][2]
            reloaded_model = models[model_name][0]

            # evaluate
            start_time = time.process_time()
            predictions, squares = process_board(images[i], reloaded_model, img_size, preprocess_fct)

            #fen = fen_max_prob(predictions) # maximum search algorithm
            fen = get_fen_from_predictions(predictions, squares, num_of_classes=num_of_classes)
            fen = fen.replace(" w - - 0 0", "")

            elapsed_time = time.process_time() - start_time

            # calculate accuracy
            true = np.array(results_array[i])
            Y = convert_fen_to_array(fen)
            acc = np.count_nonzero(true == Y) / 64
            try:
                f1 = f1_score(true, Y, average='weighted')
            except:
                f1 = -1
                print("len true ", len(true), " len Y ", len(Y))
                print(true)
                print(Y)

            print("accuracy is: ", acc, " for model ", model_name)
            print("f1 is: ", f1, " for model ", model_name)
            # save result
            data["{}_pred".format(model_name)].append(fen)
            data["{}_time".format(model_name)].append(elapsed_time)
            data["{}_acc".format(model_name)].append(acc)
            data["{}_f1".format(model_name)].append(f1)

        print(i, " Done")
    print(data)
    filename= "{}_f1_data.csv".format(num_of_classes)
    pd.DataFrame(data).to_csv(filename, index=False)
    print("Saved to " + filename)


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
    eval_pieces()
