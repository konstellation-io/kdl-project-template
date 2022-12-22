"""
ML pipeline for breast cancer classification
Part 1: Data preparation
"""

import yaml
from yaml.loader import SafeLoader

from cancer_data import prepare_cancer_data

with open("params.yaml", "r") as f:
    config = yaml.load(f, Loader=SafeLoader)

DIR_DATA_PROCESSED = config["paths"]["dir_processed"]


if __name__ == "__main__":

    prepare_cancer_data(dir_output=DIR_DATA_PROCESSED)
