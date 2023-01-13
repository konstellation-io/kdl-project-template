"""
ML pipeline for breast cancer classification
Part 2: Training traditional ML models
"""


import os
import mlflow

from utils.lib import load_params
from lab.processes.train_standard_classifiers.classifiers import train_classifiers

config = load_params("params.yaml")

MLFLOW_URL = os.getenv("MLFLOW_URL")
MLFLOW_TAGS = {"git_tag": os.getenv("GIT_TAG")}


if __name__ == "__main__":

    train_classifiers(
        mlflow=mlflow, config=config, mlflow_url=MLFLOW_URL, mlflow_tags=MLFLOW_TAGS
    )
