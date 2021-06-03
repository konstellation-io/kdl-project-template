"""
Setup for local execution and debugging of pytorch densenet training.
"""

from lib.testing import get_mlflow_stub
from processes.prepare_data.cancer_data import prepare_cancer_data
from processes.train_dnn_pytorch.densenet import train_densenet

vscode_config = {
    "paths": {
        "artifacts_temp": "temp_artifacts",
        "dir_processed": "temp_data"
    },
    "filenames": {
        "fname_model": "dnn.pt",
        "fname_conf_mat": "confusion_matrix.png",
        "fname_training_history": "history.png",
        "fname_training_history_csv": "history.csv"
    },
    "mlflow": {
        "mlflow_experiment": "",
    },
    "training": {
        "random_seed": 42,
        "batch_size": 30,
        "n_workers": 1,
        "epochs": 5,
        "lr": 0.001
    }
}


if __name__ == "__main__":

    prepare_cancer_data(dir_output=vscode_config["paths"]["dir_processed"])
    train_densenet(mlflow=get_mlflow_stub(), config=vscode_config, mlflow_url=None, mlflow_tags=None)
