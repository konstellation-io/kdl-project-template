"""
ML pipeline for breast cancer classification
Part 3: Training NN models in PyTroch
"""

import os

import mlflow
import yaml
from yaml.loader import SafeLoader

from lab.processes.train_dnn_pytorch.densenet import train_densenet

with open("params.yaml", "rb") as config_file:
    config = yaml.load(config_file, Loader=SafeLoader)

MLFLOW_URL = os.getenv("MLFLOW_URL")
MLFLOW_TAGS = {"git_tag": os.getenv("GIT_TAG")}


if __name__ == "__main__":

    train_densenet(mlflow=mlflow, config=config, mlflow_url=MLFLOW_URL, mlflow_tags=MLFLOW_TAGS)
