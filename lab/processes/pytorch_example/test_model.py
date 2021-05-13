"""
PyTorch usage example in KDL, 3/3 
Demonstrating the usage of PyTorch within KDL, solving a simple digit image classification problem.
Part 3: Model testing
"""

import os
from pathlib import Path

import mlflow
from sklearn.metrics import confusion_matrix
import torch
import torch.nn as nn

from processes.pytorch_example.train_model import load_data_splits, Net, val_loop  # TODO: Move to shared directory
from lib.viz import plot_confusion_matrix


MLFLOW_URL = os.getenv("MLFLOW_URL")
MLFLOW_EXPERIMENT = os.getenv("MLFLOW_EXPERIMENT")
MLFLOW_RUN_NAME = "pytorch_example_test"

DIR_DATA = Path(os.getenv("MINIO_DATA_FOLDER"))
DIR_DATA_PROCESSED = DIR_DATA / "processed"
DIR_MLFLOW_ARTIFACTS = DIR_DATA.parent / "mlflow-artifacts"
FILEPATH_MODEL = DIR_MLFLOW_ARTIFACTS / "d8a35d1dfdb6407b89dc851ffac61b97" / "artifacts" / "convnet.pt"
DIR_ARTIFACTS = Path("artifacts")  # Path for temporarily hosting artifacts before logging to MLflow
FILEPATH_CONF_MATRIX = DIR_ARTIFACTS / "confusion_matrix.png"


def main():

    # TODO: Get the best-performing model from MLflow tracking
    # and register the model to MLflow registry

    # Load the saved model
    net = Net()
    net.load_state_dict(torch.load(FILEPATH_MODEL))

    # Load test data
    _, _, test_loader = load_data_splits()

    # val_loop with test_loader
    loss_fn = nn.CrossEntropyLoss()
    test_loss, test_acc, (y_test_true, y_test_pred) = val_loop(dataloader=test_loader, model=net, loss_fn=loss_fn)
    cm = confusion_matrix(y_test_true, y_test_pred)

    plot_confusion_matrix(
        cm, normalize=False, title="Confusion matrix (validation set)", savepath=FILEPATH_CONF_MATRIX)

    # Log to MLflow:
    mlflow.log_artifacts(DIR_ARTIFACTS)
    mlflow.log_metrics(dict(
        test_loss=test_loss,
        test_acc=test_acc))
    print("Test run completed")


if __name__ == "__main__":
    main()