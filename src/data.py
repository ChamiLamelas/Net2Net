"""
NEEDSWORK document
"""

import torch
from torchvision import datasets, transforms
import os

DATA_FOLDER = os.path.join("..", "data")


def load_mnist(train, batch_size):
    return torch.utils.data.DataLoader(
        datasets.MNIST(DATA_FOLDER, train=train, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=batch_size, shuffle=train)


def load_cifar10(train, batch_size):
    return torch.utils.data.DataLoader(
        datasets.CIFAR10(DATA_FOLDER, train=True, download=True,
                         transform=transforms.Compose([
                             transforms.ToTensor(),
                             transforms.Resize((299, 299), antialias=True),
                             transforms.Normalize(
                                 (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                         ])),
        batch_size=batch_size, shuffle=train)
