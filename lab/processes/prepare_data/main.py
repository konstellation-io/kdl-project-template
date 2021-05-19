"""
PyTorch usage example in KDL, 1/3 
Demonstrating the usage of PyTorch within KDL, solving a standard classification problem.
Part 1: Data preparation
"""

import configparser
from pathlib import Path

from processes.prepare_data.cancer_data import prepare_cancer_data


PATH_CONFIG = "/drone/src/lab/processes/pytorch_example/config.ini"
config = configparser.ConfigParser()
config.read(PATH_CONFIG)

DIR_DATA_PROCESSED = config['paths']['dir_processed']


if __name__ == "__main__":

    Path(DIR_DATA_PROCESSED).mkdir(exist_ok=True)
    prepare_cancer_data(dir_output=DIR_DATA_PROCESSED)
