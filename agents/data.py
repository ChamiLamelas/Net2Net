import torch
from torchvision import datasets, transforms
import os

DATA_FOLDER = os.path.join("..", "data")


def cifar10(train, batch_size):
    return torch.utils.data.DataLoader(
        datasets.CIFAR10(
            DATA_FOLDER,
            train=train,
            download=True,
            transform=transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ]
            ),
        ),
        batch_size=batch_size,
        shuffle=train,
        pin_memory=True,
    )
