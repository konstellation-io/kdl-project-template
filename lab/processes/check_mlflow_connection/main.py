""" 
A simple script to check the connection from CI/CD runner (Drone) to MLflow.
It establishes the connection and stores a sample test file as a logged artifact 
and a sample metric as a logged metric in the MLflow tracked experiment.
"""

import os
from pathlib import Path

import mlflow


MLFLOW_URL = os.getenv("MLFLOW_URL")
MLFLOW_EXPERIMENT = os.getenv("MLFLOW_EXPERIMENT")
MLFLOW_RUN_NAME = "check-mlflow-connection"

PATH_MINIO_DATA = os.getenv("MINIO_DATA_FOLDER")  # from Drone
FILEPATH_ARTIFACT = Path(PATH_MINIO_DATA) / "raw" / "test.txt"


if __name__ == "__main__":
    
    mlflow.set_tracking_uri(MLFLOW_URL)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)
    
    with mlflow.start_run(run_name=MLFLOW_RUN_NAME):
        mlflow.log_metrics({'test_metric': 0})
        mlflow.log_artifact(str(FILEPATH_ARTIFACT))
