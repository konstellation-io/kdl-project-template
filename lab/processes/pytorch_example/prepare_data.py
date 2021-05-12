"""
PyTorch usage example in KDL, 1/3 
Demonstrating the usage of PyTorch within KDL, solving a simple digit image classification problem.
Part 1: Data preparation
"""

import os
from pathlib import Path
from typing import Tuple

import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
import torch
import torchvision


DIR_ARTIFACTS = Path("artifacts")

# From Drone:
DIR_DATA = os.getenv("MINIO_DATA_FOLDER")  # From Drone
DIR_DATA_PROCESSED = Path(DIR_DATA) / "processed"

# From Vscode:
# DIR_DATA_PROCESSED = Path("temp_processed")
SAVEPATH_TENSORS = str(DIR_DATA_PROCESSED / "{}.pt")

RANDOM_STATE = 42


def load_digit_data() -> None:
    """
    Loads MNIST image data as normalized numpy arrays
    """
    imgs, y = load_digits(return_X_y=True)
    imgs = imgs.reshape(imgs.shape[0], 1, 8, 8)
    imgs = imgs / imgs.max()
    return imgs, y


def split_data(X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray]:
    """
    Splits the data into train/val/test sets
    """
    X_trainval, X_test, y_trainval, y_test = train_test_split(X, y, test_size=0.15, random_state=RANDOM_STATE)
    X_train, X_val, y_train, y_val = train_test_split(X_trainval, y_trainval, test_size=0.2, random_state=RANDOM_STATE)
    return X_train, X_val, X_test, y_train, y_val, y_test


def transform_data(X: np.ndarray, y: np.ndarray, image_size: Tuple = (32, 32)) -> Tuple[torch.Tensor]:
    """
    Transforms image data as necessary to match the image size specified, and converts arrays to tensors
    """
    # Convert numpy array to torch tensor
    X = torch.tensor(X).float()
    y = torch.tensor(y).float()

    # Apply transformation
    transf = torchvision.transforms.Resize(size=image_size)
    X = transf(X)

    return X, y


def prepare_digit_data(savepath: str) -> None:
    """
    Conducts a series of steps necessary to prepare the digit data for training and validation:
    - Loads digit image data from sklearn
    - Splits the data into train, val, and test sets
    - Applies transformations as defined in transform_data
    - Saves output tensors to the destination path provided

    Args:
        savepath: (str) destination filepath, must be str not PosixPath

    Returns:
        (None)
    """
    # Load digit data
    imgs, y = load_digit_data()

    # Split into train/test/val
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X=imgs, y=y)

    # Apply transformations
    X_train, y_train = transform_data(X_train, y_train)
    X_val, y_val = transform_data(X_val, y_val)
    X_test, y_test = transform_data(X_test, y_test)

    # Save processed data
    torch.save(X_train, savepath.format("X_train"))
    torch.save(y_train, savepath.format("y_train"))
    torch.save(X_val, savepath.format("X_val"))
    torch.save(y_val, savepath.format("y_val"))
    torch.save(X_test, savepath.format("X_test"))
    torch.save(y_test, savepath.format("y_test"))


if __name__ == "__main__":

    DIR_DATA_PROCESSED.mkdir(exist_ok=True)
    prepare_digit_data(savepath=SAVEPATH_TENSORS)
