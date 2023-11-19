"""
NEEDSWORK document
"""

import torch
from torchvision import datasets, transforms
from torch.utils.data import Dataset
from datasets import load_dataset
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
        pin_memory=True,
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
        pin_memory=True,
    )


def load_imagenet(train, batch_size):
    """
    Following:
    https://moiseevigor.github.io/software/2022/12/18/one-pager-training-resnet-on-imagenet/
    """

    split = "train" if train else "val"
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
        shuffle=train,
        pin_memory=True,
    )


class TinyImageNetDataset(Dataset):
    def __init__(self, train):
        self.huggingface_dataset = load_dataset(
            "Maysee/tiny-imagenet", split="train" if train else "valid"
        )
        self.transforms = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize(299, antialias=True),
                transforms.CenterCrop(299),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

    def __len__(self):
        return len(self.huggingface_dataset)

    def __getitem__(self, index):
        huggingface_data = self.huggingface_dataset[index]
        return (
            self.transforms(huggingface_data["image"].convert("RGB")),
            huggingface_data["label"],
        )


def load_tiny_imagenet(train, batch_size):
    """
    Following:
    https://paperswithcode.com/dataset/tiny-imagenet (Dataset loaders)
    https://huggingface.co/datasets/zh-plus/tiny-imagenet
    https://pytorch.org/tutorials/beginner/basics/data_tutorial.html#creating-a-custom-dataset-for-your-files
    """

    return torch.utils.data.DataLoader(
        TinyImageNetDataset(train),
        batch_size=batch_size,
        shuffle=train,
        pin_memory=True,
    )
