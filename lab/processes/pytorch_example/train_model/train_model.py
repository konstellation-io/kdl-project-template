import os
from pathlib import Path
from typing import Union, Tuple

import mlflow
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
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


def load_data_splits(dir_processed: str, batch_size: int, n_workers: int) -> Tuple(DataLoader):
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


def train_loop(
    dataloader: DataLoader,
    model: nn.Module,
    loss_fn: nn.modules.loss._Loss,
    optimizer: torch.optim.Optimizer
        ) -> tuple:
    """
    Training loop through the dataset for a single epoch of training.
    Side effect: modifies input objects (model, loss_fn and optimizer) without returning.

    Args:
        dataloader: (Dataloader) a torch DataLoader containing training samples (X) and labels (y)
        model: (nn.Module) a torch neural network object to train
        loss_fn: (torch.nn.modules.loss._Loss) a torch loss function object
        optimizer: (torch.optim.Optimizer) torch optimizer object, e.g. Adam or SGD

    Returns:
        (tuple):
            (float) train_loss: the value of loss function on all training data provided with the dataloader
            (float) correct: training set accuracy
    """
    size = len(dataloader.dataset)
    model.train()

    train_loss, correct = 0, 0

    for X, y in dataloader:
        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y.long())
        train_loss += loss.item()
        correct += (pred.argmax(1) == y).type(torch.float).sum().item()

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    train_loss /= size
    correct /= size

    return train_loss, correct


def val_loop(dataloader: DataLoader, model: nn.Module, loss_fn: nn.modules.loss._Loss) -> tuple:
    """
    Validation loop through the dataset.

    Args:
        dataloader: (Dataloader) a torch DataLoader containing validation samples (X) and labels (y)
        model: (nn.Module) a torch neural network object to validate
        loss_fn: (torch.nn.modules.loss._Loss) a torch loss function object to compute validation loss

    Returns:
        (tuple):
            (float) val_loss: the value of loss function on all validation data provided with the dataloader
            (float) correct: validation set accuracy
            (tuple[list]): (y_true, y_pred): lists containing true labels and the labels as predicted by the model
    """
    size = len(dataloader.dataset)
    model.eval()

    y_true = []
    y_pred = []
    val_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            logits = model(X)
            val_loss += loss_fn(logits, y.long()).item()
            correct += (logits.argmax(1) == y).type(torch.float).sum().item()

            y_proba = nn.Softmax()(logits)
            y_preds_batch = y_proba.argmax(axis=1)
            y_pred.append(y_preds_batch.tolist())
            y_true.append(y.tolist())

    val_loss /= size
    correct /= size

    y_pred = flatten_list(y_pred)
    y_true = flatten_list(y_true)

    return val_loss, correct, (y_true, y_pred)


def train_and_validate(
    model: nn.Module,
    loss_fn: nn.modules.loss._Loss,
    optimizer: torch.optim.Optimizer,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int,
    filepath_model: Union[str, Path]
        ) -> tuple:
    """
    Runs model training and validation using the dataloaders provided for the number of epochs specified,
    saving the best version of the model to specified location.

    Args:
        model: (torch.nn.Module) torch model object to train and validate
        loss_fn: (torch.nn.modules.loss._Loss) torch loss function object to use in training and validation
        optimizer: (torch.optim.Optimizer) torch optimizer object to use in training
        train_loader: (DataLoader) the dataloader containing training data
        val_loader: (DataLoader) the dataloader containing validation data
        epochs: (int) the number of epochs
        filepath_model: (str or Path) the location at which to save the best model

    Returns:
        (tuple):
            (torch.nn.Module): trained model
            (pandas DataFrame): training history metrics
            (tuple[list]): (y_true, y_pred): lists containing true labels and the labels as predicted by
                the model for the validation set in last iteration
    """
    df_history = pd.DataFrame([], columns=['epoch', 'loss', 'val_loss', 'acc', 'val_acc'])

    best_acc = 0

    # Loop through epochs
    for epoch in range(1, epochs+1):
        print(f"Epoch {epoch}\n-------------------------------")

        train_loss, train_acc = train_loop(dataloader=train_loader, model=model, loss_fn=loss_fn, optimizer=optimizer)
        print(f"Training set: Accuracy: {(100*train_acc):>0.1f}%, Avg loss: {train_loss:>7f}")

        val_loss, val_acc, (y_true, y_pred) = val_loop(dataloader=val_loader, model=model, loss_fn=loss_fn)
        print(f"Validation set: Accuracy: {(100*val_acc):>0.1f}%, Avg loss: {val_loss:>7f} \n")

        df_history = df_history.append(
            dict(epoch=epoch, loss=train_loss, val_loss=val_loss, acc=train_acc, val_acc=val_acc), ignore_index=True)

        if val_acc > best_acc:
            torch.save(model.state_dict(), filepath_model)
            best_acc = val_acc

    return model, df_history, (y_true, y_pred)
