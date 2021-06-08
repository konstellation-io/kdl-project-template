"""
Integration test for train_dnn_pytorch
"""

import configparser
import os

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
        mlflow_tags=None
        )

    # Check the resulting artifact files (.csv y .png) have been created
    dir_artifacts = vscode_config["paths"]["artifacts_temp"]
    artifacts_contents = os.listdir(dir_artifacts)
    fname_model = vscode_config["filenames"]["fname_model"]
    fname_conf_matrix = vscode_config["filenames"]["fname_conf_mat"]
    fname_training_history = vscode_config["filenames"]["fname_training_history"]
    fname_training_history_csv = vscode_config["filenames"]["fname_training_history_csv"]

    for fname in [fname_model, fname_conf_matrix, fname_training_history, fname_training_history_csv]:
        assert fname in artifacts_contents

    # TODO Assert that mlflow mock has been called a number of times
