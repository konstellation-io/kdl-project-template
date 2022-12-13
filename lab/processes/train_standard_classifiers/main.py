"""
ML pipeline for breast cancer classification
Part 2: Training traditional ML models
"""


import configparser
import os
import dvc.api
import mlflow

from lab.processes.train_standard_classifiers.classifiers import train_classifiers

config = dvc.api.params_show()["train_standard_classifiers"]

MLFLOW_URL = os.getenv("MLFLOW_URL")
MLFLOW_TAGS = {"git_tag": os.getenv("DRONE_TAG")}


if __name__ == "__main__":

    train_classifiers(mlflow=mlflow, config=config, mlflow_url=MLFLOW_URL, mlflow_tags=MLFLOW_TAGS)
