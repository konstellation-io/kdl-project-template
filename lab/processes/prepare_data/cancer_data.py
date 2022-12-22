"""
Functions for preparing the breast cancer dataset for training and validating ML algorithms
"""

from pathlib import Path
from typing import Tuple, Union

import numpy as np
import pandas as pd
import torch
from pandas import DataFrame, Series
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

from lib.pytorch import create_dataloader

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
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=0.15, random_state=RANDOM_STATE, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval,
        y_trainval,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y_trainval,
    )
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
    Path(dir_output).mkdir(exist_ok=True, parents=True)

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
    X_train.to_parquet(Path(dir_output) / "X_train.gzip", compression="gzip")
    y_train.to_frame().to_parquet(Path(dir_output) / "y_train.gzip", compression="gzip")

    X_val.to_parquet(Path(dir_output) / "X_val.gzip", compression="gzip")
    y_val.to_frame().to_parquet(Path(dir_output) / "y_val.gzip", compression="gzip")

    X_test.to_parquet(Path(dir_output) / "X_test.gzip", compression="gzip")
    y_test.to_frame().to_parquet(Path(dir_output) / "y_test.gzip", compression="gzip")


def load_data_splits(
    dir_processed: Union[str, Path], as_type: str
) -> Tuple[Union[np.ndarray, torch.Tensor]]:
    """
    Loads train/val/test files for X and y (named 'X_train.npy', 'y_train.npy', etc.)
    from the location specified and returns as numpy arrays.

    Args:
        dir_processed: (str or Path) directory containing processed data files
        as_type: (str) type of outputs; one of 'array' (returns as numpy ndarray)
            or 'tensor' (returns as pytorch tensor)

    Returns:
        (tuple) of numpy arrays or torch tensors for
            X_train, X_val, X_test, y_train, y_val, y_test
    """
    X_train = pd.read_parquet(Path(dir_processed) / "X_train.gzip").to_numpy()
    y_train = pd.read_parquet(Path(dir_processed) / "y_train.gzip").squeeze().to_numpy()
    X_val = pd.read_parquet(Path(dir_processed) / "X_val.gzip").to_numpy()
    y_val = pd.read_parquet(Path(dir_processed) / "y_val.gzip").squeeze().to_numpy()
    X_test = pd.read_parquet(Path(dir_processed) / "X_test.gzip").to_numpy()
    y_test = pd.read_parquet(Path(dir_processed) / "y_test.gzip").squeeze().to_numpy()

    if as_type == "array":
        return X_train, X_val, X_test, y_train, y_val, y_test

    elif as_type == "tensor":
        # pylint: disable=not-callable
        X_train = torch.Tensor(X_train).float()
        y_train = torch.Tensor(y_train).float()
        X_val = torch.Tensor(X_val).float()
        y_val = torch.Tensor(y_val).float()
        X_test = torch.Tensor(X_test).float()
        y_test = torch.Tensor(y_test).float()

        return X_train, X_val, X_test, y_train, y_val, y_test

    else:
        raise ValueError(
            "Please specify as_type argument as one of 'array' or 'tensor'"
        )


def load_data_splits_as_dataloader(
    dir_processed: str, batch_size: int, n_workers: int
) -> Tuple[DataLoader]:
    """
    Loads data tensors saved in processed data directory and returns as dataloaders.
    """
    X_train, X_val, X_test, y_train, y_val, y_test = load_data_splits(
        dir_processed, as_type="tensor"
    )

    # Convert tensors to dataloaders
    dataloader_args = dict(batch_size=batch_size, num_workers=n_workers, shuffle=True)

    train_loader = create_dataloader(X_train, y_train, dataloader_args)
    val_loader = create_dataloader(X_val, y_val, dataloader_args)
    test_loader = create_dataloader(X_test, y_test, dataloader_args)

    return train_loader, val_loader, test_loader
