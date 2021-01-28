#!/usr/bin/env python
# coding: utf-8
import time

import keras
import numpy as np
import tensorflow as tf
from keras import layers
from keras import models
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import utility

train_path = "/home/joking/Projects/Chessrecognition/data/7_classes/train/"
test_path = "/home/joking/Projects/Chessrecognition/data/7_classes/test/"

model_path = '/home/joking/PycharmProjects/Chess_Recognition/models/best_model.h5'
empty_model_path = '/home/joking/PycharmProjects/Chess_Recognition/models/empty_small_model.h5'

labels = ["bishop", "empty", "king", "knight", "pawn", "queen", "rook"]
pieces = ["b", "empty", "k", "n", "p", "q", "r"]

fen_real = "r1b5/pp3p1p/4kQp1/2pNN3/2PnP3/6P1/PP4qP/3RK2R"  # r1b5/pp3p1p/4kQp1/2pNN3/2PnP3/6P1/PP4qP/3RK2R w - - 0 1

import logging

logging.getLogger("tensorflow").setLevel(logging.ERROR)


# ## Creating ImageDataGenerators


def load_training_dataset(training_dir, test_dir):
    train = ImageDataGenerator(rescale=1. / 255,
                               featurewise_center=True,
                               horizontal_flip=True,
                               width_shift_range=0.2,
                               height_shift_range=0.2,
                               rotation_range=45,
                               shear_range=0.5,
                               zoom_range=0.5,
                               fill_mode='nearest',
                               )
    val = ImageDataGenerator(rescale=1 / 255, featurewise_center=True, )

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
    IMG_SHAPE = 150

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
    steps_p_epoch = 50000 / 32  # num_samples // batch_size,
    early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)

    # Save the Model with the lowest validation loss
    save_best = keras.callbacks.ModelCheckpoint('models/best_model.h5',
                                                monitor='val_loss',
                                                save_best_only=True)

    history = model.fit(train_dataset,
                        epochs=epochs,
                        steps_per_epoch=steps_p_epoch,
                        validation_data=validation_dataset,
                        callbacks=[early_stopping, save_best])

    t = time.time()

    saved_keras_model_filepath = '{}_small_Model.h5'.format(int(t))

    model.save(saved_keras_model_filepath)
    print("Model saved to: " + saved_keras_model_filepath)
    return model, history


# ## Load Model
def load_model(file_path):
    return tf.keras.models.load_model(file_path)


"""
load the already compiled model to retrain it
"""


def load_compiled_model(path):
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
    # TODO setup args, define which operations done
    train_dataset, validation_dataset = load_training_dataset(train_path, test_path)

    # model = create_model()
    model = load_compiled_model(empty_model_path)
    model, history = train_model(model, train_dataset, validation_dataset, epochs=1)

    utility.plot_history(history)
    # utility.save_history(history)

    print("Success")
