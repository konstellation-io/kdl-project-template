"""
Setup for local execution and debugging of the sklearn classifiers training
"""

from lib.testing import get_mlflow_stub
from processes.prepare_data.cancer_data import prepare_cancer_data
from processes.train_standard_classifiers.classifiers import train_classifiers


# TODO Move to separate config for VScode executions
vscode_config = {
    "paths": {
        "artifacts_temp": "temp_artifacts",
        "dir_processed": "temp_data"
    },
    "mlflow": {
        "mlflow_experiment": "",
    },
    "training": {
        "random_seed": 42
    }
}


if __name__ == "__main__":

    prepare_cancer_data(dir_output=vscode_config["paths"]["dir_processed"])
    train_classifiers(mlflow=get_mlflow_stub(), config=vscode_config, mlflow_url=None, mlflow_tags=None)
