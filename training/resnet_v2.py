"""import sys
# set base path
sys.path.append("/home/ubuntu/ChessRecognition/")"""
import sys
sys.path.append("/home/users/j/jonasklotz/ChessRecognition")


# set base path

sys.path.append("/home/users/j/jonasklotz/ChessRecognition")

import logging

from keras.applications.inception_resnet_v2 import preprocess_input

from training.generic_model import train_model, change_trainable, generate_generators, save_model, \
    load_compiled_model

model_name = 'InceptionResNetV2'
img_shape = 150

""" model_path = ""
data_dir = "/home/ubuntu/data/13_classes"
base_path = "./" """

model_path = "/home/users/j/jonasklotz/TrainingCluster/training/empty/empty_InceptionResNetV2.h5"
base_path = "/home/users/j/jonasklotz/TrainingCluster/training/results"

logging.getLogger("tensorflow").setLevel(logging.ERROR)


def start_training():
    # base_model = InceptionResNetV2(include_top=False, weights='imagenet',
    # input_shape=(img_shape, img_shape, 3))
    # model = create_model(base_model, trainable=1)

    # save_model(model, model_name, base_path)
    # on cluster

    model = load_compiled_model(model_path)
    model = change_trainable(model, trainable=1, lr=0.0001)

    train_dataset, validation_dataset, test_dataset = generate_generators(preprocess_input, img_shape)
    train_model(model, train_dataset, validation_dataset, test_dataset, model_name, base_path)

    trainable = int(len(model.layers) / 3)
    model = change_trainable(model, trainable=trainable, lr=0.00001)

    train_model(model, train_dataset, validation_dataset, test_dataset, model_name, base_path)

    model = change_trainable(model, trainable=len(model.layers), lr=0.00001)
    train_model(model, train_dataset, validation_dataset, test_dataset, model_name, base_path)

    save_model(model, "final_" + model_name, base_path)
    print("Done")


if __name__ == '__main__':
    start_training()
