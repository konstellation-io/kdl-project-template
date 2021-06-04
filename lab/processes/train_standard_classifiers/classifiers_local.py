"""
Setup for local execution and debugging of scikit-learn classifiers training
"""

import configparser

from lib.testing import get_mlflow_stub
from processes.prepare_data.cancer_data import prepare_cancer_data
from processes.train_standard_classifiers.classifiers import train_classifiers

vscode_config = configparser.ConfigParser()
vscode_config.read("lab/processes/config_local.ini")


if __name__ == "__main__":

    prepare_cancer_data(dir_output=vscode_config["paths"]["dir_processed"])
    train_classifiers(mlflow=get_mlflow_stub(), config=vscode_config, mlflow_url=None, mlflow_tags=None)
