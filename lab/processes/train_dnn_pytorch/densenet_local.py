"""
Setup for local execution and debugging of pytorch densenet training.
"""

import configparser

from lib.testing import get_mlflow_stub
from processes.prepare_data.cancer_data import prepare_cancer_data
from processes.train_dnn_pytorch.densenet import train_densenet

vscode_config = configparser.ConfigParser()
vscode_config.read("lab/processes/config_local.ini")


if __name__ == "__main__":

    prepare_cancer_data(dir_output=vscode_config["paths"]["dir_processed"])
    train_densenet(
        mlflow=get_mlflow_stub(),
        config=vscode_config,
        mlflow_url=None,
        mlflow_tags=None,
    )
