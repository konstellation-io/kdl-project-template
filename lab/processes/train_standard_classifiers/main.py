"""
This script demonstrates the usage of sklearn within KDL using a simple wine-classification challenge
"""
import configparser
from pathlib import Path
import os

from matplotlib import pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.datasets import load_wine
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from lib.viz import plot_confusion_matrix
from processes.prepare_data.cancer_data import load_data_splits


PATH_CONFIG = "/drone/src/lab/processes/config.ini"
config = configparser.ConfigParser()
config.read(PATH_CONFIG)

MLFLOW_URL = os.getenv("MLFLOW_URL")
MLFLOW_EXPERIMENT = os.getenv("MLFLOW_EXPERIMENT")
MLFLOW_RUN_NAME = "sklearn_example_train"

DIR_DATA_PROCESSED = config['paths']['dir_processed']
DIR_ARTIFACTS = Path(config['paths']['artifacts_temp'])  # Path for temporarily hosting artifacts before logging to MLflow
FILEPATH_CONF_MATRIX = DIR_ARTIFACTS / "confusion_matrix.png"

RANDOM_SEED = int(config['training']['random_seed'])


if __name__ == "__main__":
    
    np.random.seed(RANDOM_SEED)

    DIR_ARTIFACTS.mkdir(exist_ok=True)

    mlflow.set_tracking_uri(MLFLOW_URL)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)
    
    with mlflow.start_run(run_name=MLFLOW_RUN_NAME):
        
        # Load training and validation data
        X_train, X_val, _, y_train, y_val, _ = load_data_splits(dir_processed=DIR_DATA_PROCESSED, as_type="array")
        
        # Define a number of classifiers
        models = {
            "Logistic regression": LogisticRegression(),
            "Naive Bayes": GaussianNB(),
            "K-nearest neighbour": KNeighborsClassifier(),
            "Random forest": RandomForestClassifier(),
            "Linear SVM": SVC(kernel="linear"),
            "GradientBoost": GradientBoostingClassifier(),
            "AdaBoost": AdaBoostClassifier(),
        }

        # Iterate fitting and validation through all model types, logging results to MLflow:
        for model_name, model in models.items():

            with mlflow.start_run(run_name=model_name, nested=True):

                model.fit(X_train, y_train)
                y_pred = model.predict(X_val)
                val_accuracy = accuracy_score(y_pred, y_val)
                cm = confusion_matrix(y_val, y_pred)
                plot_confusion_matrix(
                    cm, normalize=False, 
                    title="Confusion matrix (validation set)", 
                    savepath=FILEPATH_CONF_MATRIX)
                
                mlflow.log_artifacts(DIR_ARTIFACTS)
                mlflow.log_param("classifier", model_name)
                mlflow.log_metric("validation_accuracy", val_accuracy)
