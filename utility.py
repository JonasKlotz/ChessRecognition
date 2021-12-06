import glob
import io
import logging
import os
import time

import chess
import chess.svg
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from cairosvg import svg2png
from natsort import natsorted
from tensorflow.keras.preprocessing import image


def display_fen_board(fen, save=False):
    board = chess.Board(fen)
    svg = chess.svg.board(board=board)

    if save:
        svg2png(bytestring=bytes(svg, 'UTF-8'), write_to="result.png")
    else:
        img = io.BytesIO()
        svg2png(bytestring=bytes(svg, 'UTF-8'), write_to=img)
        img = Image.open(img)
        img.show()
        img.close()


def print_points(plist, img):
    """
    Plots list of points as red circles on given image
    """
    circled = img.copy()
    for point in plist:
        cx, cy = point
        cx = int(cx)
        cy = int(cy)
        cv2.circle(circled, (cx, cy), 20, (255, 0, 0), -1)  # red (255,0,0), black 1
    fig = plt.figure(figsize=(10, 10))
    plt.imshow(circled)


def chunks(l, n):
    n = max(1, n)
    return (l[i:i + n] for i in range(0, len(l), n))


def save_history(history):
    t = time.time()
    df = pd.DataFrame.from_dict(history)
    history_path = '/history/{}_history.csv'.format(int(t))
    df.to_csv(history_path)
    logging.info("History saved to " + history_path)


#########################################################  Plotting ###################################################


def plot_history(history):
    training_accuracy = history.history['accuracy']
    validation_accuracy = history.history['val_accuracy']

    training_loss = history.history['loss']
    validation_loss = history.history['val_loss']

    epochs_range = range(len(training_accuracy))

    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, training_accuracy, label='Training Accuracy')
    plt.plot(epochs_range, validation_accuracy, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, training_loss, label='Training Loss')
    plt.plot(epochs_range, validation_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training')
    plt.savefig("history.png")

    plt.show()


def plot_prob(pred, class_names, img=None, label=""):
    # fig = plt.figure(figsize=(10,10))
    # plt.imshow( np.squeeze(img))
    # plt.show()

    if len(img.shape) > 3:
        img = np.squeeze(img)
    fig, (ax1, ax2) = plt.subplots(figsize=(6, 9), ncols=2)

    ax1.imshow(img)
    ax1.axis('off')
    ax1.set_title(label)
    ax2.barh(np.arange(len(class_names)), pred, align='center', alpha=0.6, )
    ax2.set_aspect(0.1)
    ax2.set_yticks(np.arange(len(class_names)))
    ax2.set_yticklabels(class_names, size='small')
    ax2.set_title('Class Probability')
    ax2.set_xlim(0, 1.1)
    plt.tight_layout()
    plt.show()


def plotImages(images_arr):
    fig, axes = plt.subplots(1, 5, figsize=(20, 20))
    axes = axes.flatten()
    for img, ax in zip(images_arr, axes):
        ax.imshow(img)
    plt.tight_layout()
    plt.show()


####################################################### Load Functions ####################################################
#

# In[49]:
def load_image_to_tensor(img, image_size, preprocess_input):
    """img_tensor = image.img_to_array(img)  # (height, width, channels)
    img_tensor = np.expand_dims(img_tensor,
                                axis=0)  # (1, height, width, channels), add a dimension because the model expects this shape: (batch_size, height, width, channels)
    img_tensor /= 255.  """
    img = cv2.resize(img, (image_size, image_size))

    img_tensor = preprocess_input(img)
    img_tensor = np.expand_dims(img_tensor,
                                axis=0)
    return img_tensor


def load_tensor_list_from_squares(square_list, image_size, preprocess_input):
    """
    @input: a list of images of squares
    @returns: tensor list of the given squares
    """
    tensor_list = []
    for square in square_list:
        tensor_list.append(load_image_to_tensor(square, image_size, preprocess_input))

    return tensor_list


def load_image_path_to_tensor(img_path, image_size, show=False):
    """load image in tensorformat
    """

    img = image.load_img(img_path, target_size=(image_size, image_size))

    img_tensor = image.img_to_array(img)  # (height, width, channels)
    img_tensor = np.expand_dims(img_tensor,
                                axis=0)  # (1, height, width, channels), add a dimension because the model expects this shape: (batch_size, height, width, channels)
    img_tensor /= 255.  # imshow expects values in the range [0, 1]

    if show:
        plt.imshow(img_tensor[0])
        plt.axis('off')
        plt.show()

    return img_tensor


def load_square(img_path, show=False):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (150, 150))
    return img


def load_square_lists_from_dir(dir_path):
    """
    loads squares from a directory
    @:arg path of dir
    returns images in tensors and img (150, 150)
    """
    addrs = glob.glob(dir_path + "/*.jpg")
    addrs = natsorted(addrs)

    tensor_list, square_list = [], []
    for addr in addrs:
        img = load_image_path_to_tensor(addr, image_size=150)
        tensor_list.append((img))
        square_list.append(load_square(addr))

    return tensor_list, square_list


def combine_squares_board_image(squares):
    """
    takes array of squares and recombines them to board
    for testing purposes
    needs 64 squares
    """
    n_squares = len(squares)
    assert n_squares == 64

    first_col = squares[0]
    for k in range(1, 8):  # each col
        first_col = cv2.vconcat([first_col, squares[k]])

    for i in range(1, 8):
        temp_col = squares[i * 8]  # start der spalte
        for k in range(1, 8):
            temp_col = cv2.vconcat([temp_col, squares[(i * 8) + k]])

        first_col = cv2.hconcat([first_col, temp_col])
    fig = plt.figure(figsize=(10, 10))
    plt.imshow(first_col)
    plt.show()

    return first_col


def fill_dir_with_squares(save_path, squares):
    try:
        os.mkdir(save_path)
        logging.info("Directory '%s' created" % save_path)
    except:
        logging.warning("Directory  already exists")
    try:
        for k, square in enumerate(squares):
            cv2.imwrite(save_path + "/" + str(k) + '.jpg', square)  # './data/chessboards/squares/' + str(i)
            k += 1
    except:
        logging.warning("Couldnt save the image")


def create_dir(parent_dir, dir_name):
    path = os.path.join(parent_dir, dir_name)
    try:
        os.mkdir(path)
        logging.info("Directory '%s' created" % path)
    except:
        logging.warning("Directory  already exists")
    return path


def read_images(path, n):
    """
    read n images from path
    :param path:
    :param n:
    :return:
    """
    results = []
    for i in range(n):
        img_path = path + "{}.jpg".format(i + 1)
        logging.info("Read ", img_path)
        img = cv2.imread(img_path, 1)
        results.append(img)
    return results


if __name__ == '__main__':
    pass
