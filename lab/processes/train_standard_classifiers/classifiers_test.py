"""
Integration test for train_standard_classifiers
"""

import configparser
import os

import pytest

from lib.testing import get_mlflow_stub
from processes.train_standard_classifiers.classifiers import train_classifiers

vscode_config = configparser.ConfigParser()
vscode_config.read("lab/processes/config_test.ini")


@pytest.mark.integration
def test_train_classifiers_runs_without_errors(temp_data_dir):
    """
    Runs train_classifiers with a mock mlflow instance, failing the test if the run raises any errors
    Uses test fixture temp_data_dir to create a temporary dataset required by train_classifiers (see conftest.py)
    """
    vscode_config["paths"]["dir_processed"] = temp_data_dir
    
    train_classifiers(
        mlflow=get_mlflow_stub(),
        config=vscode_config,
        mlflow_url=None,
        mlflow_tags=None,
    )

    # Verify that the resulting files have been created in the temporary artifacts directory:
    dir_artifacts = vscode_config["paths"]["artifacts_temp"]
    filename_conf_matrix = vscode_config["filenames"]["fname_conf_mat"]

    assert filename_conf_matrix in os.listdir(dir_artifacts)
