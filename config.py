import yaml


class configurator:

    def __init__(self, model_name="InceptionResNetV2", config_path="config.yaml"):
        self.config = parse_config(config_path)

        self.num_of_classes = self.config["num_of_classes"]
        self.model_config = self.config[model_name]

        if self.config["model_path"] == "":
            self.model_path = str(
                self.config["model_directory"] + str(self.num_of_classes) + "_classes/" + model_name + "/model.h5")
        else:
            self.model_path = self.config["model_path"]

        from model import get_preprocess_function
        self.model_config["preprocess_input"] = get_preprocess_function(model_name)

        self.data_directory = self.config["data_directory"]
        self.result_directory = self.config["result_directory"]
        self.model_directory = self.config["model_directory"]
        self.board_algorithm = self.config["board_algorithm"]

    def get_num_of_classes(self):
        return self.num_of_classes

    def get_model_config(self):
        return self.model_config

    def get_model_path(self):
        return self.model_path

    def get_model_img_size(self):
        return self.model_config["img_size"]

    def get_model_preprocess(self):
        return self.model_config["preprocess_input"]


def parse_config(config_path):
    with open(config_path, "r") as stream:
        try:
            return yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            RuntimeError("No Config File, ", exc)
