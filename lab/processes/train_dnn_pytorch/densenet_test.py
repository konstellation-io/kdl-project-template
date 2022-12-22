"""
Integration test for train_dnn_pytorch
"""

import os
from pathlib import Path

import pytest
import yaml
from yaml.loader import SafeLoader

from lab.processes.train_dnn_pytorch.densenet import train_densenet
from lib.testing import get_mlflow_stub


@pytest.mark.integration
@pytest.mark.filterwarnings("ignore:CUDA initialization")
def test_train_densenet_without_errors(temp_data_dir):
    """
    Runs train_densenet with a mock mlflow instance.

    Verifies that:
    - train_densenet runs without any errors
    - artifacts are created in the temporary artifacts directory (provided with config)
    - the mlflow provided to the function is called to log the metrics, params and artifacts

    Uses test fixture temp_data_dir to create a temporary dataset required by train_densenet (see conftest.py)
    """

    with open("params.yaml", "r") as f:
        config = yaml.load(f, Loader=SafeLoader)

    config = config["test"]
    mlflow_stub = get_mlflow_stub()

    train_densenet(mlflow=mlflow_stub, config=config, mlflow_url=None, mlflow_tags=None)

    # Check the resulting artifact files (.csv y .png) have been created
    dir_artifacts = config["paths"]["dir_artifacts_dnn"]
    artifacts_contents = os.listdir(dir_artifacts)
    fname_model = "densenet.pt"
    fname_conf_matrix = "confusion_matrix.png"
    fname_training_history = "training_history.png"
    fname_training_history_csv = "training_history.csv"

    for fname in [
        fname_model,
        fname_conf_matrix,
        fname_training_history,
        fname_training_history_csv,
    ]:
        assert fname in artifacts_contents

    # Assert that mlflow has been called to log artifacts, metrics, and params
    mlflow_stub.start_run.assert_called()
    mlflow_stub.log_artifacts.assert_called_with(Path(dir_artifacts))
    mlflow_stub.log_params.assert_called()
    mlflow_stub.log_metrics.assert_called()
