
import sys
# set base path
sys.path.append("/home/ubuntu/ChessRecognition")

import logging

from keras.applications import Xception
from keras.applications.xception import preprocess_input

from training.generic_model import create_model, train_model, generate_generators, change_trainable

model_name = 'Xception'
img_shape = 299
model_path = ""
data_dir = "/home/joking/Projects/Chessrecognition/Data/13_classes" # /home/ubuntu/data/13_classes"

base_path = "./"

logging.getLogger("tensorflow").setLevel(logging.ERROR)


def start_training():
    base_model = Xception(input_shape=(img_shape, img_shape, 3), include_top=False,
                          weights='imagenet')
    model = create_model(base_model, trainable=1)

    save_model(model, model_name, base_path)
    # on cluster
    # model =  load_compiled_model(model_path)

    train_dataset, validation_dataset, test_dataset = generate_generators(preprocess_input, img_shape)
    train_model(model, train_dataset, validation_dataset, test_dataset, model_name, base_path)

    trainable = int(len(model.layers) / 3)
    model = change_trainable(model, trainable=trainable, lr=0.00001)
    train_model(model, train_dataset, validation_dataset, test_dataset, model_name, base_path)
    save_model(model, model_name, base_path)
    print("Done")

if __name__ == '__main__':
    start_training()
