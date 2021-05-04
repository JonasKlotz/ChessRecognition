import logging

from keras.applications import NASNetMobile

from training.generic_model import create_model, save_model

model_name = 'NASNetMobile'
img_shape = 224
model_path = ""
data_dir = "/home/joking/Projects/Chessrecognition/Data/7_classes"
base_path = "./"

logging.getLogger("tensorflow").setLevel(logging.ERROR)


def start_training():
    base_model = NASNetMobile(input_shape=(img_shape, img_shape, 3), include_top=False,
                              weights='imagenet')

    model = create_model(base_model, trainable=250)

    save_model(model, model_name, base_path)
    # on cluster
    # model =  load_compiled_model(model_path)

    # train_dataset, validation_dataset, test_dataset = generate_generators(preprocess_input, img_shape, data_dir)
    # train_model(model, train_dataset, validation_dataset, test_dataset, model_name, base_path)


if __name__ == '__main__':
    start_training()
