"""
PyTorch usage example in KDL, 3/3 
Demonstrating the usage of PyTorch within KDL, solving a simple digit image classification problem.
Part 3: Model testing
"""

import configparser
import os
from pathlib import Path

import mlflow
from sklearn.metrics import confusion_matrix
import torch
import torch.nn as nn

from processes.pytorch_example.train_model import load_data_splits, Net, val_loop  # TODO: Move to shared directory
from lib.viz import plot_confusion_matrix


PATH_CONFIG = "/drone/src/lab/processes/pytorch_example/config.ini"
config = configparser.ConfigParser()
config.read(PATH_CONFIG)


MLFLOW_URL = os.getenv("MLFLOW_URL")
MLFLOW_EXPERIMENT = os.getenv("MLFLOW_EXPERIMENT")

MLFLOW_RUN_NAME = "pytorch_example_test"

DIR_DATA_PROCESSED = config['paths']['dir_processed']
# DIR_MLFLOW_ARTIFACTS = config['paths']['artifacts_mlflow']
DIR_MLFLOW_ARTIFACTS = Path(DIR_DATA_PROCESSED).parent.parent / "mlflow-artifacts"
DIR_ARTIFACTS = config['paths']['artifacts_temp']  # Path for temporarily hosting artifacts before logging to MLflow

FILEPATH_MODEL = DIR_MLFLOW_ARTIFACTS / "d8a35d1dfdb6407b89dc851ffac61b97" / "artifacts" / "convnet.pt"

# RUN_ID = config['testing']['run_id']

# FNAME_MODEL = config['filenames']['fname_model']
FNAME_CONF_MAT = config['filenames']['fname_conf_mat']

# FILEPATH_MODEL = Path(DIR_MLFLOW_ARTIFACTS) / RUN_ID / "artifacts" / FNAME_MODEL
FILEPATH_CONF_MATRIX = Path(DIR_ARTIFACTS) / FNAME_CONF_MAT

# "kdl-project-template/mlflow-artifacts/d8a35d1dfdb6407b89dc851ffac61b97"
# "kdl-project-template/mlflow-artifacts/d8a35d1dfdb6407b89dc851ffac61b97"
# "kdl-project-template/mlflow-artifacts/b8caaa8a3ec24ebca12d7c6e22054c45/"

def main():
    """
    The main function of the example Pytorch model testing script

    - Loads a ConvNet model trained and saved by train_model.py
    - Applies the trained model on test data loaded from Minio
    - Logs test metrics to MLflow
    """

    Path(DIR_ARTIFACTS).mkdir(exist_ok=True)

    mlflow.set_tracking_uri(MLFLOW_URL)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    with mlflow.start_run(run_name=MLFLOW_RUN_NAME):

        # Load test data
        _, _, test_loader = load_data_splits()

        # Load the saved model
        net = Net()
        net.load_state_dict(torch.load(FILEPATH_MODEL))
        loss_fn = nn.CrossEntropyLoss()

        # Make and score predictions on test data
        test_loss, test_acc, (y_test_true, y_test_pred) = val_loop(dataloader=test_loader, model=net, loss_fn=loss_fn)
        cm = confusion_matrix(y_test_true, y_test_pred)
        plot_confusion_matrix(
            cm, normalize=False, title="Confusion matrix (test set)", savepath=FILEPATH_CONF_MATRIX)

        # Log to MLflow:
        mlflow.log_artifacts(DIR_ARTIFACTS)
        mlflow.log_metrics(dict(
            test_loss=test_loss,
            test_acc=test_acc))
        print("Test run completed")


if __name__ == "__main__":
    main()
