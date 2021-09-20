#!/usr/bin/env python
# coding: utf-8

import numpy as np
import tensorflow as tf
from keras.optimizers import Adam

import utility

labels = ["bishop", "empty", "king", "knight", "pawn", "queen", "rook"]
pieces = ["b", "empty", "k", "n", "p", "q", "r"]

import logging

logging.getLogger("tensorflow").setLevel(logging.ERROR)


# ## Creating ImageDataGenerators


# ## Load Model
def load_model(file_path):
    return tf.keras.models.load_model(file_path)


def load_compiled_model(path):
    """
    load the already compiled model to retrain it
    """
    model = tf.keras.models.load_model(path, compile=False)
    adam = Adam(lr=0.0001)

    model.compile(adam, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model


def get_predictions(model, tensor_list, show=False):
    predictions = []
    # model = load_model(model_path)
    for img in tensor_list:
        pred = np.squeeze(model.predict(img))
        if show:
            utility.plot_prob(pred, pieces, img)
        predictions.append(pred)

    return np.asarray(predictions)


if __name__ == '__main__':
    pass
