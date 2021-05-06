"""
This script demonstrates the usage of pytorch within KDL using a simple digit classification (MNIST)
"""

import itertools
import os
from pathlib import Path
from typing import Tuple

from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator
import mlflow
import numpy as np
import pandas as pd
from sklearn.datasets import load_digits
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from torchsummary import summary
from torchvision.utils import make_grid
import torchvision
import torchvision.transforms as transforms


MLFLOW_URL = os.getenv("MLFLOW_URL")
MLFLOW_EXPERIMENT = os.getenv("MLFLOW_EXPERIMENT")
MLFLOW_RUN_NAME = "pytorch_example"

RANDOM_SEED = 0
BATCH_SIZE = 16
N_WORKERS = 2

EPOCHS = 20
LR = 0.01

DIR_ARTIFACTS = Path("artifacts")
FILEPATH_MODEL = DIR_ARTIFACTS / "convnet.pt"
FILEPATH_TRAINING_HISTORY = DIR_ARTIFACTS / "training_history.png"
FILEPATH_TRAINING_HISTORY_CSV = DIR_ARTIFACTS / "training_history.csv"
FILEPATH_CONF_MATRIX = DIR_ARTIFACTS / "confusion_matrix.png"


def load_mnist_data():
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
    X_trainval, X_test, y_trainval, y_test = train_test_split(X, y, test_size=0.15, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_trainval, y_trainval, test_size=0.2, random_state=42)
    return X_train, X_val, X_test, y_train, y_val, y_test


def transform_data(X: np.ndarray, y: np.ndarray, image_size: Tuple=(32, 32)) -> Tuple[torch.Tensor]:
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


def create_dataloader(X: torch.Tensor, y: torch.Tensor, dataloader_args: dict) -> DataLoader:
    """
    Converts torch tensors X and y into a DataLoader object
    """
    dataset = TensorDataset(X, y)
    dataloader = DataLoader(dataset, **dataloader_args)
    return dataloader


def prepare_mnist_data() -> Tuple[DataLoader]:
    """ 
    Conducts a series of steps necessary to prepare the MNIST data for training:
    - Loads MNIST image data from sklearn
    - Splits the data into train, val, and test sets
    - Applies transformations as defined in transform_data
    - Creates DataLoader objects 

    Args: 
        (None)

    Returns:
        (tuple of DataLoader): train, val and test DataLoaders for consumption by a PyTorch network
    """
    imgs, y = load_mnist_data()
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X=imgs, y=y)
    
    X_train, y_train = transform_data(X_train, y_train)
    X_val, y_val = transform_data(X_val, y_val)
    X_test, y_test = transform_data(X_test, y_test)

    dataloader_args = dict(batch_size=BATCH_SIZE, num_workers=N_WORKERS, shuffle=True)
    train_loader = create_dataloader(X_train, y_train, dataloader_args)
    val_loader = create_dataloader(X_val, y_val, dataloader_args)
    test_loader = create_dataloader(X_test, y_test, dataloader_args)
    
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

    def forward(self, x):
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
        ## Convolutional layers:
        x = self.conv1(x)
        x = self.conv1_bn(x)
        x = F.relu(x)
        x = F.max_pool2d(x, (2, 2))
        x = self.conv2(x)
        x = self.conv2_bn(x)
        x = F.relu(x)
        x = F.max_pool2d(x, (2, 2))
                
        ## Flatten:
        x = x.view(-1, self.num_flat_features(x))

        ## Classification (fully-connected) layers:
        x = self.fc1(x)
        x = self.fc1_bn(x)
        x = F.relu(x)
        
        x = self.fc2(x)
        x = self.fc2_bn(x)
        x = F.relu(x)
        
        x = self.fc3(x)
    
        return x
    
    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


def train_loop(dataloader, model, loss_fn, optimizer):
    """
    Training loop through the dataset for a single epoch of training
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


def val_loop(dataloader, model, loss_fn):
    """
    Validation loop through the dataset
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


def flatten_list(input_list):
    """
    Flattens a list containing lists as its elements (only one level deep).
    """
    return [item for sublist in input_list for item in sublist]


def train_and_validate(net, loss_fn, optimizer, train_loader, val_loader, epochs, filepath_model):
    """
    TODO: Docstring
    """
    df_history = pd.DataFrame([], columns=['epoch', 'loss', 'val_loss', 'acc', 'val_acc'])

    best_acc = 0

    # Loop through
    for epoch in range(1, epochs+1):
        print(f"Epoch {epoch}\n-------------------------------")
        
        train_loss, train_acc = train_loop(dataloader=train_loader, model=net, loss_fn=loss_fn, optimizer=optimizer)
        print(f"Training set: Accuracy: {(100*train_acc):>0.1f}%, Avg loss: {train_loss:>7f}")
        
        val_loss, val_acc, (y_true, y_pred) = val_loop(dataloader=val_loader, model=net, loss_fn=loss_fn)
        print(f"Validation set: Accuracy: {(100*val_acc):>0.1f}%, Avg loss: {val_loss:>7f} \n")
        
        df_history = df_history.append(
            dict(epoch=epoch, loss=train_loss, val_loss=val_loss, acc=train_acc, val_acc=val_acc), ignore_index=True)

        if val_acc > best_acc:
            torch.save(net.state_dict(), filepath_model)
            best_acc = val_acc

    return net, df_history, (y_true, y_pred)


def plot_training_history(history, show=True, title='', savepath=None, accuracy_metric='acc'):
    """
    Plots training history (validation and loss) for a model, given a table of metrics per epoch.

    Args:
        history: (np.array or pd.DataFrame) an array containing data for each epoch for accuracy ('acc'), loss ('loss'),
            validation accuracy ('val_acc'), and validation loss ('val_loss'), with names as indicated (requires
            indexing by these names).
        show: (bool) show plot if True; returns Figure and Axes objects if False
        title: (str) Title of the plot
        savepath: (str) file path to save resulting plot
        accuracy_metric: (str) name of accuracy metric in the model history

    Returns:
        (None or tuple of (fig, axes))
    """
    loss = history['loss']
    val_loss = history['val_loss']

    acc = history[accuracy_metric]
    val_acc = history[f"val_{accuracy_metric}"]

    assert len(acc) == len(val_acc) == len(loss) == len(
        val_loss), "All metrics should have the same number of measurements (one for each epoch)."

    epochs = [int(n) for n in range(len(acc))]

    fig, ax = plt.subplots(1, 1)

    ax.plot(epochs, acc, c='r', lw=1, label='Train accuracy')
    ax.plot(epochs, val_acc, c='r', lw=2, label='Validation accuracy')
    ax.set_xlabel('Epoch')
    plt.xticks(epochs)

    ax.set_ylim(0.5, 1.01)
    ax.set_ylabel('Accuracy', color='r')
    ax.tick_params(axis='y', labelcolor='r')

    ax2 = ax.twinx()
    ax2.plot(epochs, loss, c='b', lw=1, label='Train loss')
    ax2.plot(epochs, val_loss, c='b', lw=2, label='Validation loss')
    ax2.set_ylim(0, 1.05 * max(max(loss), max(val_loss)))
    ax2.set_ylabel('Loss', color='b')
    ax2.tick_params(axis='y', labelcolor='b')

    ax.legend(loc=6)
    ax2.legend(loc=5)
    plt.title(title)

    if savepath:
        plt.savefig(savepath)

    if show:
        plt.show()

    plt.close()
    

def plot_confusion_matrix(cm, normalize=True, title='Confusion matrix', cmap=plt.cm.Blues, show=True,
                          class_names=None, savepath=None):
    """
    Prints and plots the confusion matrix.

    Args:
        cm: (np.array) of shape (2, 2) containing elements of the confusion matrix (order: tn, fp, fn, tp)
        normalize: (bool) Normalization can be applied by setting normalize = True
        title: (string) plot title
        cmap: (pyplot colormap)
        show: (bool) display resulting plot
        savepath: (str) destination to save results on disk

    Returns:
        (Figure) or None
    """
    assert cm.shape[0] == cm.shape[1], "Confusion matrix not square!"

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    vmax = 1 if normalize else np.sum(cm, axis=1).max()

    fig, ax = plt.subplots(figsize=(6, 6))

    ax.imshow(cm, interpolation='nearest', cmap=cmap, vmin=0, vmax=vmax)
    ax.set_title(title)
    tick_marks = np.arange(cm.shape[0])
    if not class_names:
        class_names = tick_marks

    ax.set_xticks(ticks=tick_marks)
    ax.set_xticklabels(class_names, rotation=0)
    ax.set_yticks(ticks=tick_marks)
    ax.set_yticklabels(class_names, rotation=0)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()

    if savepath:
        fig.savefig(savepath)
        print(f"Saved confusion matrix plot to {savepath}")
    if show:
        plt.show()
    else:
        return fig
    

def main():

    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    
    DIR_ARTIFACTS.mkdir(exist_ok=True)
    
    mlflow.set_tracking_uri(MLFLOW_URL)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)
    
    with mlflow.start_run(run_name=MLFLOW_RUN_NAME):

        # Load the data splits
        train_loader, val_loader, test_loader = prepare_mnist_data()

        # Instantiate the ConvNet, loss function and optimizer 
        net = Net()
        loss_fn = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(net.parameters(), lr=LR)

        # Train and validate
        net, df_history, _ = train_and_validate(
            net=net, loss_fn=loss_fn, optimizer=optimizer, 
            train_loader=train_loader, val_loader=val_loader,  
            epochs=EPOCHS, filepath_model=FILEPATH_MODEL)

        # Load best version:
        net = Net()
        net.load_state_dict(torch.load(FILEPATH_MODEL))

        # Get metrics on best model
        train_loss, train_acc, _ = val_loop(dataloader=train_loader, model=net, loss_fn=loss_fn)
        val_loss, val_acc, (y_val_true, y_val_pred) = val_loop(dataloader=val_loader, model=net, loss_fn=loss_fn)
        cm = confusion_matrix(y_val_true, y_val_pred)

        # Save artifacts
        plot_confusion_matrix(cm, normalize=False, title="Confusion matrix (validation set)", savepath=FILEPATH_CONF_MATRIX)
        plot_training_history(df_history, title="Training history", savepath=FILEPATH_TRAINING_HISTORY)
        df_history.to_csv(FILEPATH_TRAINING_HISTORY_CSV)

        # Log to MLflow:
        mlflow.log_artifacts(DIR_ARTIFACTS)
        mlflow.log_metrics(dict(
            val_loss=val_loss,
            val_acc=val_acc,
            train_loss=train_loss,
            train_acc=train_acc))
        mlflow.log_params(dict(
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            learning_rate=LR
        ))

        print("Done!")


if __name__ == "__main__":

    main()
