"""
ML pipeline for breast cancer classification
Part 3: Training NN models in PyTroch
"""

import dvc.api
import os

import mlflow

from lab.processes.train_dnn_pytorch.densenet import train_densenet

PATH_CONFIG = "params.yaml"
config = dvc.api.params_show(PATH_CONFIG)
config = config["train_dnn_pytorch"]

MLFLOW_URL = os.getenv("MLFLOW_URL")
MLFLOW_TAGS = {"git_tag": os.getenv("DRONE_TAG")}


if __name__ == "__main__":

    train_densenet(mlflow=mlflow, config=config, mlflow_url=MLFLOW_URL, mlflow_tags=MLFLOW_TAGS)
