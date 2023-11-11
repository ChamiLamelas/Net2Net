"""
NEEDSWORK document
"""

import torch
from torchvision import datasets, transforms
import os

DATA_FOLDER = os.path.join("..", "data")


def load_mnist(train, batch_size):
    return torch.utils.data.DataLoader(
        datasets.MNIST(
            DATA_FOLDER,
            train=train,
            download=True,
            transform=transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
            ),
        ),
        batch_size=batch_size,
        shuffle=train,
    )


def load_cifar10(train, batch_size):
    return torch.utils.data.DataLoader(
        datasets.CIFAR10(
            DATA_FOLDER,
            train=train,
            download=True,
            transform=transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Resize((299, 299), antialias=True),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ]
            ),
        ),
        batch_size=batch_size,
        shuffle=train,
    )


def load_imagenet(train, batch_size):
    """
    Following:
    https://moiseevigor.github.io/software/2022/12/18/one-pager-training-resnet-on-imagenet/
    """

    split = 'train' if train else 'val'
    return torch.utils.data.DataLoader(
        datasets.ImageFolder(
            os.path.join(DATA_FOLDER, "ImageNet", "ILSVRC", "Data", "CLS-LOC", split),
            transform=transforms.Compose(
                [
                    transforms.Resize(299),
                    transforms.CenterCrop(299),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ]
            ),
        ),
        batch_size=batch_size,
        shuffle=split == "train",
    )
