from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from pandas import DataFrame, Series
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch


RANDOM_STATE = 42


def load_cancer_data() -> Tuple[DataFrame, Series]:
    """
    Loads breast cancer data as pandas DataFrame (features) and Series (target)
    """
    X, y = load_breast_cancer(return_X_y=True, as_frame=True)
    return X, y


def split_data(X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray]:
    """
    Splits the data into train/val/test sets
    """
    X_trainval, X_test, y_trainval, y_test = train_test_split(X, y, test_size=0.15, random_state=RANDOM_STATE)
    X_train, X_val, y_train, y_val = train_test_split(X_trainval, y_trainval, test_size=0.2, random_state=RANDOM_STATE)
    return X_train, X_val, X_test, y_train, y_val, y_test


def prepare_cancer_data(dir_output: str) -> None:
    """
    Conducts a series of steps necessary to prepare the digit data for training and validation:
    - Loads digit image data from sklearn
    - Splits the data into train, val, and test sets
    - Applies transformations as defined in transform_data
    - Saves output tensors to the destination path provided

    Args:
        dir_output: (str) destination filepath, must be str not PosixPath

    Returns:
        (None)
    """
    # Load digit data
    imgs, y = load_cancer_data()

    # Split into train/test/val
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X=imgs, y=y)

    # Normalize input features:
    scaler = StandardScaler()
    X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
    X_val = pd.DataFrame(scaler.transform(X_val), columns=X_val.columns)
    X_test = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

    # Save processed data
    torch.save(X_train.to_numpy(), Path(dir_output) / "X_train.pt")
    torch.save(y_train.to_numpy(), Path(dir_output) / "y_train.pt")
    torch.save(X_val.to_numpy(), Path(dir_output) / "X_val.pt")
    torch.save(y_val.to_numpy(), Path(dir_output) / "y_val.pt")
    torch.save(X_test.to_numpy(), Path(dir_output) / "X_test.pt")
    torch.save(y_test.to_numpy(), Path(dir_output) / "y_test.pt")