"""
Integration test for train_standard_classifiers
"""

import os
from pathlib import Path

import pytest
import yaml
from yaml.loader import SafeLoader

from lab.processes.train_standard_classifiers.classifiers import train_classifiers
from lib.testing import get_mlflow_stub


@pytest.mark.integration
def test_train_classifiers_runs_without_errors(config_test, temp_data_dir):
    """
    Runs train_classifiers with a mock mlflow instance. Verifies that:
    - train_classifiers runs without any errors
    - artifacts are created in the temporary artifacts directory (provided with config_test)
    - the mlflow provided to the function is called to log the metrics, params and artifacts

    Uses test fixture temp_data_dir to create a temporary dataset required by train_classifiers (see conftest.py)
    as well as load_test_config to load the testing config file
    """

    mlflow_stub = get_mlflow_stub()

    train_classifiers(
        mlflow=mlflow_stub,
        config=config_test,
        mlflow_url=None,
        mlflow_tags=None,
    )

    # Verify that the resulting files have been created in the temporary artifacts directory:
    dir_artifacts = config_test.paths.dir_artifacts_standard
    filename_conf_matrix = "confusion_matrix.png"
    assert filename_conf_matrix in os.listdir(dir_artifacts)

    # Check that mlflow has been called to log artifacts, metrics, and params
    mlflow_stub.start_run.assert_called()
    mlflow_stub.log_artifacts.assert_called_with(Path(dir_artifacts))
    mlflow_stub.log_params.assert_called()
    mlflow_stub.log_metrics.assert_called()
