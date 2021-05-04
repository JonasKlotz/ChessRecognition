import logging

from keras.applications import MobileNetV2
from keras.applications.mobilenet_v2 import preprocess_input

from training.generic_model import create_model, generate_generators, train_model

model_name = 'MobileNetV2'
img_shape = 224
model_path = ""
data_dir = "/home/joking/Projects/Chessrecognition/Data/13_classes"
base_path = "./"

logging.getLogger("tensorflow").setLevel(logging.ERROR)


def start_training():
    base_model = MobileNetV2(input_shape=(img_shape, img_shape, 3), include_top=False,
                             weights='imagenet', alpha=0.5)
    model = create_model(base_model, trainable=126, classes=13)

    # save_model(model, model_name, base_path)
    # on cluster
    # model =  load_compiled_model(model_path)

    train_dataset, validation_dataset, test_dataset = generate_generators(preprocess_input, img_shape, data_dir)
    train_model(model, train_dataset, validation_dataset, test_dataset, model_name, base_path)


if __name__ == '__main__':
    start_training()
