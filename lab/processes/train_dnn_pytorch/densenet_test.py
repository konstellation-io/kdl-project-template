"""
Integration test for train_dnn_pytorch
"""

import configparser

import pytest

from lib.testing import get_mlflow_stub
from processes.train_dnn_pytorch.densenet import train_densenet

vscode_config = configparser.ConfigParser()
vscode_config.read("lab/processes/config_test.ini")


@pytest.mark.integration
def test_train_densenet_without_errors(temp_data_dir):
    """
    Runs train_densenet with a mock mlflow instance and fails the test if the run raises any exceptions
    Uses test fixture temp_data_dir to create a temporary dataset required by train_densenet (see conftest.py)
    """
    vscode_config["paths"]["dir_processed"] = temp_data_dir
    vscode_config["training"]["epochs"] = "1"
    train_densenet(
        mlflow=get_mlflow_stub(),
        config=vscode_config,
        mlflow_url=None,
        mlflow_tags=None,
    )
    # TODO Created the temporary files (.csv y .png)
    # TODO Assert the mlflow mock has been called a number of times
    # 
