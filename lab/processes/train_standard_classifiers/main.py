"""
ML pipeline for breast cancer classification
Part 2: Training traditional ML models
"""

import configparser
import os

import mlflow
from processes.train_standard_classifiers.classifiers import train_classifiers

PATH_CONFIG = os.getenv("PATH_CONFIG")
config = configparser.ConfigParser()
config.read(PATH_CONFIG)

config["mlflow"]["mlflow_tags"] = {"git_tag": os.getenv('DRONE_TAG')}
config["mlflow"]["mlflow_url"] = os.getenv("MLFLOW_URL")


if __name__ == "__main__":

    train_classifiers(mlflow=mlflow, config=config)
