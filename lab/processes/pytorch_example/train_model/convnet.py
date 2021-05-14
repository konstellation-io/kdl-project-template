from typing import Tuple

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
        data[fname] = torch.load(fpath)
    
    # Convert tensors to dataloaders
    dataloader_args = dict(batch_size=batch_size, num_workers=n_workers, shuffle=True)
    train_loader = create_dataloader(data['X_train'], data['y_train'], dataloader_args)
    val_loader = create_dataloader(data['X_val'], data['y_val'], dataloader_args)
    test_loader = create_dataloader(data['X_test'], data['y_test'], dataloader_args)
    
    return train_loader, val_loader, test_loader


class Net(nn.Module):
    """
    A simple ConvNet based on https://pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html
    """
    def __init__(self):
        super(Net, self).__init__()

        # 1 input image channel, 6 output channels, 5x5 square convolution kernel
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5)
        self.conv1_bn = nn.BatchNorm2d(6)

        # 6 inputs, 16 outputs, 5x5 kernel
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)
        self.conv2_bn = nn.BatchNorm2d(16)

        # Fully connected layers
        self.fc1 = nn.Linear(16 * 5 * 5, 120)  # 5*5 from kernel dimensions
        self.fc1_bn = nn.BatchNorm1d(num_features=120)
        self.fc2 = nn.Linear(120, 84)
        self.fc2_bn = nn.BatchNorm1d(num_features=84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Shape input: torch.Size([16, 1, 32, 32])
        Shape Conv1: torch.Size([16, 6, 28, 28])
        Shape Pool1: torch.Size([16, 6, 14, 14])
        Shape Conv2: torch.Size([16, 16, 10, 10])
        Shape Pool2: torch.Size([16, 16, 5, 5])
        Shape Flatten: torch.Size([16, 400])
        Shape Fc1: torch.Size([16, 120])
        Shape Fc2: torch.Size([16, 84])
        Shape Fc3: torch.Size([16, 10])
        """
        # Convolutional layers:
        x = self.conv1(x)
        x = self.conv1_bn(x)
        x = F.relu(x)
        x = F.max_pool2d(x, (2, 2))
        x = self.conv2(x)
        x = self.conv2_bn(x)
        x = F.relu(x)
        x = F.max_pool2d(x, (2, 2))

        # Flatten:
        x = x.view(-1, self.num_flat_features(x))

        # Classification (fully-connected) layers:
        x = self.fc1(x)
        x = self.fc1_bn(x)
        x = F.relu(x)

        x = self.fc2(x)
        x = self.fc2_bn(x)
        x = F.relu(x)

        x = self.fc3(x)

        return x

    def num_flat_features(self, x: torch.Tensor) -> int:
        """
        Returns the number of features per example in input tensor x
        (assuming that x contains a batch of samples where the first
        dimension is the sample)
        """
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
