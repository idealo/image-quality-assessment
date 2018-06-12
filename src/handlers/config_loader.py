
from utils.utils import load_json


def load_config(config_file):
    config = load_json(config_file)
    return config
