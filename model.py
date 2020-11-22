#!/usr/bin/env python
# coding: utf-8
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
from collections import defaultdict

from matplotlib.image import imread
import matplotlib.pyplot as plt
from copy import deepcopy
from natsort import natsorted, ns
import tensorflow as tf
import numpy as np

import glob
import time
import sys
import cv2
import os

import keras
from keras import layers
from keras import models
from keras.optimizers import Adam
from keras.applications.inception_resnet_v2 import InceptionResNetV2

import utility

train_path = "/home/joking/Projects/Chessrecognition/Data/smaller_model/train/"
test_path = "/home/joking/Projects/Chessrecognition/Data/smaller_model/test/"
model_path = './model/1600679265_small_Model.h5'

labels = ["bishop", "empty", "king", "knight", "pawn", "queen", "rook"]
pieces = ["b", "empty", "k", "n", "p", "q", "r"]

fen_real = "r1b5/pp3p1p/4kQp1/2pNN3/2PnP3/6P1/PP4qP/3RK2R"  # r1b5/pp3p1p/4kQp1/2pNN3/2PnP3/6P1/PP4qP/3RK2R w - - 0 1


# ## Creating ImageDataGenerators


def load_training_dataset(training_dir, test_dir):
    train = ImageDataGenerator(rescale=1. / 255,
                               horizontal_flip=True,
                               width_shift_range=0.2,
                               height_shift_range=0.2,
                               rotation_range=45,
                               shear_range=0.5,
                               zoom_range=0.5,
                               fill_mode='nearest',
                               )
    val = ImageDataGenerator(rescale=1 / 255)

    img_shape = 150

    train_dataset = train.flow_from_directory(training_dir,
                                              target_size=(img_shape, img_shape),
                                              class_mode="sparse",
                                              color_mode="rgb",
                                              batch_size=32,
                                              shuffle=True,
                                              seed=42
                                              )
    validation_dataset = val.flow_from_directory(test_dir,
                                                 target_size=(img_shape, img_shape),
                                                 class_mode="sparse",
                                                 color_mode="rgb",
                                                 batch_size=32,
                                                 shuffle=True,
                                                 seed=42)
    return train_dataset, validation_dataset


def create_model(output_size=7):
    base_model = InceptionResNetV2(include_top=False, weights='imagenet', input_shape=(IMG_SHAPE, IMG_SHAPE, 3))
    x = base_model.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(1024, activation='relu')(x)
    predictions = layers.Dense(output_size, activation='softmax')(x)  # New softmax layer

    model = models.Model(inputs=base_model.input, outputs=predictions)

    # we chose to train the top 2 inception blocks
    # we will freeze the first 249 layers and unfreeze the rest
    for layer in model.layers[:249]:
        layer.trainable = False
    for layer in model.layers[249:]:
        layer.trainable = True

    adam = Adam(lr=0.0001)
    model.compile(adam, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model





# ## Train Model
def train_model(model, train_dataset, validation_dataset, epochs=1):
    steps_p_epoch = 10000 / 32  # num_samples // batch_size,
    early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)

    # Save the Model with the lowest validation loss
    save_best = keras.callbacks.ModelCheckpoint('./best_model.h5',
                                                monitor='val_loss',
                                                save_best_only=True)

    history = model.fit(train_dataset,
                        epochs=epochs,
                        steps_per_epoch=steps_p_epoch,
                        validation_data=validation_dataset,
                        callbacks=[early_stopping, save_best])

    t = time.time()

    saved_keras_model_filepath = './model/{}_small_Model.h5'.format(int(t))

    model.save(saved_keras_model_filepath)
    return model, history
    
# ## Load Model
def load_model(file_path):
    return tf.keras.models.load_model(file_path)



def get_predictions(tensor_list, show=False):
    
    predictions = []
    model = load_model(model_path)
    for img in tensor_list:
        pred = np.squeeze(model.predict(img))
        if show:
            utility.plot_prob(pred, pieces, img)
        predictions.append(pred)

    return predictions

if __name__ == '__main__':
    #TODO setup args, define which operations done
    pass



