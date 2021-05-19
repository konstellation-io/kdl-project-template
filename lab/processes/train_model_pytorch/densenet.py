from typing import Tuple

import numpy as np
import pandas as pd
from pandas import Series, DataFrame
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

from lib.utils import flatten_list
from lib.viz import plot_confusion_matrix, plot_training_history


def create_dataloader(X: torch.Tensor, y: torch.Tensor, dataloader_args: dict) -> DataLoader:
    """
    Converts input torch tensors X and y into a DataLoader object.

    Args:
        X: (torch Tensor) a tensor containing input features
        y: (torch Tensor) a tensor containing labels
        dataloader_args: (dict) keyword arguments for torch DataLoader
            (e.g. batch_size: int, num_workers: int, shuffle: bool)
    
    Returns:
        (torch DataLoader)
    """
    dataset = TensorDataset(X, y)
    dataloader = DataLoader(dataset, **dataloader_args)
    return dataloader


def load_data_splits(dir_processed: str, batch_size: int, n_workers: int) -> Tuple[DataLoader]:
    """
    Loads data tensors saved in processed data directory and returns as dataloaders.
    """
    # Load tensors from preprocessed data directory on Minio
    data = dict()
    for fname in ["X_train", "X_val", "X_test", "y_train", "y_val", "y_test"]:
        fpath = f"{dir_processed}/{fname}.pt"
        data[fname] = torch.tensor(torch.load(fpath)).float()

    # Convert tensors to dataloaders
    dataloader_args = dict(batch_size=batch_size, num_workers=n_workers, shuffle=True)
    train_loader = create_dataloader(data['X_train'], data['y_train'], dataloader_args)
    val_loader = create_dataloader(data['X_val'], data['y_val'], dataloader_args)
    test_loader = create_dataloader(data['X_test'], data['y_test'], dataloader_args)
    
    return train_loader, val_loader, test_loader


class DenseNN(nn.Module):
    
    def __init__(self):
        super(DenseNN, self).__init__()
        
        self.dense1 = nn.Linear(30, 200)
        self.bn1 = nn.BatchNorm1d(200)
        self.dense2 = nn.Linear(200,100)
        self.bn2 = nn.BatchNorm1d(100)
        
        self.dense3 = nn.Linear(100,100)
        self.bn3 = nn.BatchNorm1d(100)

        self.output_layer = nn.Linear(100, 1)
    
    def forward(self, x):
        
        x = self.dense1(x)
        x = self.bn1(x)
        x = F.relu(x)
        
        x = self.dense2(x)
        x = self.bn2(x)
        x = F.relu(x)
        
        x = self.dense3(x)
        x = self.bn3(x)
        x = F.relu(x)
        
        x = self.output_layer(x)
        x = torch.sigmoid(x)
    
        return x
