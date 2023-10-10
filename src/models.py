import torchvision.models as models
import torch.nn as nn
import torch


class SmallFeedForward(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.fc1 = nn.Linear(in_features, 1024)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(1024, out_features)

    def forward(self, x):
        assert x.dim() >= 2
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x


class TinyConvolution(nn.Module):
    def __init__(self, hin, win, in_channels, out_features):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels,
                               out_channels=32, kernel_size=3, stride=1)
        self.relu1 = nn.ReLU()
        # taken from here: https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
        hout = hin - 2
        wout = win - 2
        self.fc = nn.Linear(32 * hout * wout, out_features)

    def forward(self, x):
        assert x.dim() == 4
        x = self.conv1(x)
        x = self.relu1(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


class SmallConvolution(nn.Module):
    def __init__(self, hin, win, in_channels, out_features):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels,
                               out_channels=32, kernel_size=3, stride=1)
        self.conv2 = nn.Conv2d(
            in_channels=32, out_channels=64, kernel_size=3, stride=1)
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        # taken from here: https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
        hout = (hin - 2) - 2
        wout = (win - 2) - 2
        self.fc = nn.Linear(64 * hout * wout, out_features)

    def forward(self, x):
        assert x.dim() == 4
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


def cifar10_resnet18():
    """11 million parameters"""

    model = models.resnet18()
    model.fc = nn.Linear(in_features=512, out_features=10, bias=True)
    return model


def cifar10_resnet34():
    """21 million parameters"""

    model = models.resnet34()
    model.fc = nn.Linear(in_features=512, out_features=10, bias=True)
    return model


def cifar10_resnet50():
    """25 million parameters"""

    model = models.resnet50()
    model.fc = nn.Linear(in_features=2048, out_features=10, bias=True)
    return model


def cifar10_resnet101():
    """45 million parameters"""

    model = models.resnet101()
    model.fc = nn.Linear(in_features=2048, out_features=10, bias=True)
    return model


def cifar10_resnet152():
    """60 million parameters"""

    model = models.resnet152()
    model.fc = nn.Linear(in_features=2048, out_features=10, bias=True)
    return model


def cifar10_squeezenet1_0():
    """1.2 million parameters"""

    model = models.squeezenet1_0()
    model.classifier[1] = nn.Conv2d(512, 10, kernel_size=(1, 1), stride=(1, 1))
    return model


def cifar10_squeezenet1_1():
    """1.2 million parameters"""

    model = models.squeezenet1_1()
    model.classifier[1] = nn.Conv2d(512, 10, kernel_size=(1, 1), stride=(1, 1))
    return model
