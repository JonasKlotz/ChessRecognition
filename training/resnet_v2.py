from keras.applications import InceptionResNetV2
from keras.applications.inception_resnet_v2 import preprocess_input

from training.generic_model import create_model, generate_generators, train_model

img_shape = 150
model_name = 'InceptionResNetV2'
data_dir = "/home/joking/Projects/Chessrecognition/Data/7_classes"
base_path = "./"

import logging

logging.getLogger("tensorflow").setLevel(logging.ERROR)


def start_training():
    base_model = InceptionResNetV2(include_top=False, weights='imagenet', input_shape=(img_shape, img_shape, 3))
    model = create_model(base_model, trainable=249)

    # on cluster
    # load_compiled_model(model_path)

    train_dataset, validation_dataset, test_dataset = generate_generators(preprocess_input, img_shape, data_dir)
    train_model(model, train_dataset, validation_dataset, test_dataset, model_name, base_path)


if __name__ == '__main__':
    start_training()
