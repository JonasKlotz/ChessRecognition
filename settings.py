import logging

global model_name, num_of_classes


def initialize(model_name, num_of_classes):
    logging.getLogger("tensorflow").setLevel(logging.ERROR)
