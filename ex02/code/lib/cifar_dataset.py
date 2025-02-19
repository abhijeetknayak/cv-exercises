"""CIFAR10 dataset."""

import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms

import os
import shutil
import sys
import json
from typing import Dict, List, Optional, Any, Iterable, Mapping, Tuple, Union, Callable, BinaryIO
from pathlib import Path

import torch as th
from torch import nn
from pprint import pprint
import time
from timeit import default_timer as timer
from enum import Enum


def create_cifar_datasets(path: str = "./data") -> Tuple[Dataset, Dataset]:
    """
    Setup CIFAR10 train and test set.

    Args:
        path: Target path to store the downloaded data

    Returns:
        Tuple of train and test dataset
    """
    train_set = datasets.CIFAR10(path, train=True, download=True, transform=transforms.ToTensor())
    test_set = datasets.CIFAR10(path, train=False, download=True, transform=transforms.ToTensor())

    return train_set, test_set


def create_dataloader(dataset: Dataset, batch_size: int, is_train: bool = True, num_workers: int = 0) -> DataLoader:
    """
    Given a dataset, create the dataloader.

    Args:
        dataset: Input dataset
        batch_size: Batch size
        is_train: Whether this is a dataloader for training or for testing
        num_workers: How many processes to use for dataloading

    Returns:
        dataloader
    """
    # START TODO #################
    # Create an instance of the DataLoader class given the dataset, batch_size and num_workers.
    # Set the shuffle parameter to True if is_train is True, otherwise set it to False.

    if is_train:
        return DataLoader(dataset=dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True)
    else:
        return DataLoader(dataset=dataset, batch_size=batch_size, num_workers=num_workers)

    # END TODO ###################
