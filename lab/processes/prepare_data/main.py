"""
ML pipeline for breast cancer classification
Part 1: Data preparation
"""

import os

import dvc.api

from cancer_data import prepare_cancer_data


DIR_DATA_PROCESSED = dvc.api.params_show()["prepare_data"]["dir_processed"]


if __name__ == "__main__":

    prepare_cancer_data(dir_output=DIR_DATA_PROCESSED)
