"""
PyTorch usage example in KDL, 3/3 
Demonstrating the usage of PyTorch within KDL, solving a simple digit image classification problem.
Part 3: Model testing
"""

import os
from pathlib import Path

import torch
import torch.nn as nn

from processes.pytorch_example.train_model import (
    load_data_splits, Net, val_loop)  # TODO: Move to a shared directory


DIR_DATA = Path(os.getenv("MINIO_DATA_FOLDER"))  # From Drone
DIR_DATA_PROCESSED = DIR_DATA / "processed"
DIR_MLFLOW_ARTIFACTS = DIR_DATA.parent / "mlflow-artifacts"
FILEPATH_MODEL = DIR_MLFLOW_ARTIFACTS / "d8a35d1dfdb6407b89dc851ffac61b97" / "artifacts" / "convnet.pt"


def main():

    # TODO: Get the best-performing model from MLflow tracking
    # and register the model to MLflow registry

    # Load the saved model
    net = Net()
    net.load_state_dict(torch.load(FILEPATH_MODEL))

    # Load test data
    _, _, test_loader = load_data_splits()

    # val_loop with test_loader
    loss_fn = nn.CrossEntropyLoss()  # TODO: Get from model object directly
    test_loss, test_acc, _ = val_loop(dataloader=test_loader, model=net, loss_fn=loss_fn)

    print(test_loss)
    print(test_acc)
    print("Test run completed")


if __name__ == "__main__":
    main()