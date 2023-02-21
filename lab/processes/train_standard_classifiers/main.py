"""
ML pipeline for breast cancer classification
Part 2: Training traditional ML models
"""

import os
import mlflow

from lib.utils import load_params
from lab.processes.train_standard_classifiers.classifiers import train_classifiers

config = load_params("params.yaml")

MLFLOW_URL = os.getenv("MLFLOW_URL")

# When running with dvc exp, dvc can give us the experiment name
dvc_exp_name = os.getenv("DVC_EXP_NAME")

# If not run within an experiment, we just use the commit's SHA
MLFLOW_TAGS = {"run_name": dvc_exp_name if dvc_exp_name else os.getenv("GIT_SHA")}


if __name__ == "__main__":

    train_classifiers(mlflow=mlflow, config=config, mlflow_url=MLFLOW_URL, mlflow_tags=MLFLOW_TAGS)
