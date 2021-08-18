import sys

# set base path
sys.path.append("/home/users/j/jonasklotz/ChessRecognition")

import datetime
import itertools
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from utility import create_dir

classes = 13
# data_dir = "/home/ubuntu/data/{}_classes".format(classes)
# data_dir = "/home/joking/Projects/Chessrecognition/Data/{}_classes".format(classes)
data_dir = "/home/users/j/jonasklotz/TrainingCluster/Data/{}_classes".format(classes)  # hpc cluster

class_names_7 = ["bishop", "empty", "king", "knight", "pawn", "queen", "rook"]
class_names_13 = ["bb", "bk", "bn", "bp", "bq", "br", "empty", "wb", "wk", "wn", "wp", "wq", "wr"]

if classes == 13:
    class_names = class_names_13
else:
    class_names = class_names_7


def create_model(base_model, trainable):
    """
    Creates and compiles model based on given model
    :param base_model: base model for transfer learning
    :param trainable: amount of trainable layers
    :return: compiled model
    """
    # First train only the top layers
    for layer in base_model.layers:
        layer.trainable = False

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    predictions = Dense(classes, activation='softmax')(x)  # 7 classes for no color detection

    model = Model(inputs=base_model.input, outputs=predictions)


    adam = Adam(lr=0.0001)
    model.compile(adam, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    return model


def load_compiled_model(path):
    """
    loads and compiles model for retraining
    :param path: path of the model
    :return: model
    """
    model = load_model(path, compile=False)
    adam = Adam(lr=0.0001)

    model.compile(adam, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model


def save_model(model, model_name, parent_dir):
    """
    saves model
    :param model:
    :param parent_dir:
    """
    saved_keras_model_filepath = os.path.join(parent_dir, '{}.h5'.format(model_name))

    model.save(saved_keras_model_filepath)
    print("Model saved to: " + saved_keras_model_filepath)


def generate_generators(preprocess_input, img_shape):
    """

    :param preprocess_input:
    :param img_shape:
    :param data_dir:
    :return: all 3 datasets
    """

    train_dir = os.path.join(data_dir, 'train')
    val_dir = os.path.join(data_dir, 'val')
    test_dir = os.path.join(data_dir, 'test')

    train = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        horizontal_flip=True,
        width_shift_range=0.2,
        height_shift_range=0.2,
        rotation_range=45,
        shear_range=0.5,
        zoom_range=0.5,
        fill_mode='nearest',
    )

    val_and_test = ImageDataGenerator(
        preprocessing_function=preprocess_input,
    )

    train_dataset = train.flow_from_directory(train_dir,
                                              target_size=(img_shape, img_shape),
                                              class_mode="sparse",
                                              color_mode="rgb",
                                              batch_size=32,
                                              shuffle=True,
                                              seed=42
                                              )

    validation_dataset = val_and_test.flow_from_directory(val_dir,
                                                          target_size=(img_shape, img_shape),
                                                          class_mode="sparse",
                                                          color_mode="rgb",
                                                          batch_size=32,
                                                          shuffle=True,
                                                          seed=42)

    test_dataset = val_and_test.flow_from_directory(test_dir,
                                                    target_size=(img_shape, img_shape),
                                                    class_mode="sparse",
                                                    color_mode="rgb",
                                                    batch_size=32,
                                                    shuffle=False,
                                                    seed=42)

    return train_dataset, validation_dataset, test_dataset


def train_model(model, train_dataset, validation_dataset, test_dataset, model_name, base_path):
    """

    :param model:
    :param train_dataset:
    :param validation_dataset:
    :param model_name: name of the base model
    :return:
    """
    EPOCHS = 20
    steps_p_epoch = train_dataset.samples // train_dataset.batch_size + 1
    validation_steps = validation_dataset.samples // validation_dataset.batch_size + 1

    time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    print("Start at ", time)

    # Setup folder for evaluation process logs etc
    parent_dir = create_dir(base_path, time + "_" + model_name)

    callbacks = generate_callbacks(parent_dir, 0.1, 10)
    history = model.fit(train_dataset,
                        epochs=EPOCHS,
                        steps_per_epoch=steps_p_epoch,
                        validation_data=validation_dataset,
                        validation_steps=validation_steps,
                        callbacks=callbacks)

    time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    print("Finished Training at ", time)

    evaluate_model(model, test_dataset, parent_dir, history)
    time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    print("Finished Evaluation at ", time)


def generate_callbacks(parent_dir, reduce_lr_factor, reduce_lr_patience):
    """
    :param reducelr_patience:
    :param reducelr_factor:
    :param parent_dir: location to save
    :return: list containing callbacks
    """
    best_save_string = os.path.join(parent_dir, 'model.h5')
    log_folder = os.path.join(parent_dir, 'logs/fit')  # parent_dir + "/logs/fit/"

    early_stopping = EarlyStopping(monitor='val_loss', patience=5)

    save_best = ModelCheckpoint(best_save_string,
                                monitor='val_loss',
                                save_best_only=True)

    tb = TensorBoard(log_dir=log_folder,
                     histogram_freq=1,
                     write_graph=True,
                     write_images=True,
                     update_freq='epoch',
                     profile_batch=2,
                     embeddings_freq=1),

    reduce_lr = ReduceLROnPlateau(monitor='val_accuracy',
                                  mode='max',
                                  factor=reduce_lr_factor,
                                  patience=reduce_lr_patience,
                                  verbose=1)

    # return [early_stopping, save_best, tb, LambdaCallback(on_epoch_end=log_confusion_matrix)]
    return [early_stopping, save_best]  # , tb]

def evaluate_model(model, test_dataset, parent_dir, history=None):
    """

    :param model:
    :param test_dataset:
    :param parent_dir:
    :param history:
    :return:
    """
    y_pred = get_y_pred(model, test_dataset)

    cm = confusion_matrix(test_dataset.classes, y_pred)
    cm_path = os.path.join(parent_dir, 'final_cm.jpg')
    print("Save Matrix to ", cm_path)
    figure = plot_confusion_matrix(cm, save_path=cm_path)

    # setup right classnames

    print(classification_report(test_dataset.labels, y_pred, target_names=class_names))
    if history:
        save_history(history, parent_dir)


def get_y_pred(model, test_dataset):
    """

    :param model:
    :param test_dataset:
    :return:
    """
    test_steps_per_epoch = test_dataset.samples // test_dataset.batch_size + 1
    y_pred = model.predict(test_dataset, test_steps_per_epoch)
    y_pred = np.argmax(y_pred, axis=1)
    return y_pred


def plot_confusion_matrix(cm, save_path=None, epoch=0):
    """

    :param cm:
    :param save_path:
    :param epoch:
    :return:
    """
    figure = plt.figure(figsize=(8, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Accent)
    plt.title("Confusion matrix")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    cm = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=2)
    threshold = cm.max() / 2.

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        color = "white" if cm[i, j] > threshold else "black"
        plt.text(j, i, cm[i, j], horizontalalignment="center", color=color)

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
    return figure


def save_history(history, parent_dir):
    """

    :param history:
    :param parent_dir:
    :return:
    """
    history_path = os.path.join(parent_dir, 'history.csv')
    pd.DataFrame.from_dict(history.history).to_csv(history_path, index=False)


def change_trainable(model, trainable, lr):
    if trainable > 0:
        t = len(model.layers) - trainable
        for layer in model.layers[:t]:
            layer.trainable = False
        for layer in model.layers[t:]:
            layer.trainable = True

    adam = Adam(lr=lr)
    model.compile(adam, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    return model


if __name__ == '__main__':
    pass
