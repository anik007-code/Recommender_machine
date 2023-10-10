# data_loader.py
import json
from configs.config_folder import DATA_PATH


def load_data_from_json():
    with open(DATA_PATH, "r") as json_file:
        data = json.load(json_file)
    return data
