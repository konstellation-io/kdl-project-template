"""
ML pipeline for breast cancer classification
Part 1: Data preparation
"""

from lib.utils import load_params

# You may also use relative imports
from cancer_data import prepare_cancer_data

config = load_params("params.yaml")

DIR_DATA_PROCESSED = config.paths.dir_processed


if __name__ == "__main__":

    prepare_cancer_data(dir_output=DIR_DATA_PROCESSED)
