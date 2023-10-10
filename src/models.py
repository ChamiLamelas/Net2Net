import torchvision.models as models
import torch.nn as nn
import torch
import net2net


class LayerTable:
    def _helper(self, parent, name, curr):
        if len(list(curr.children())) == 0:
            self.table.append((parent, name))
        for n, child in curr.named_children():
            self._helper(curr, n, child)

    def __init__(self, model):
        self.table = list()
        self._helper(None, None, model)

    def __iter__(self):
        yield from self.table

    def get(self, parent, name):
        return (
            parent[int(name)]
            if isinstance(parent, nn.Sequential)
            else getattr(parent, name)
        )

    def set(self, parent, name, value):
        if isinstance(parent, nn.Sequential):
            parent[int(name)] = value
        else:
            setattr(parent, name, value)


def is_conv(layer):
    return "Conv2d" in layer.__class__.__name__


def is_linear(layer):
    return "Linear" in layer.__class__.__name__


def is_batchnorm(layer):
    return "BatchNorm" in layer.__class__.__name__


def get_out_size(layer):
    return layer.out_features if isinstance(layer, nn.Linear) else layer.out_channels


def widen(model, scale):
    table = LayerTable(model)
    prev = None
    between_batchnorm = None
    for p, n in table:
        curr = table.get(p, n)
        if is_batchnorm(curr):
            between_batchnorm = (p, n)
        elif is_conv(curr) or is_linear(curr):
            if prev is not None:
                old_layer1 = table.get(*prev)
                old_layer2 = curr
                old_batchnorm = (
                    table.get(*between_batchnorm)
                    if between_batchnorm is not None
                    else None
                )
                new_out_size = round(scale * get_out_size(old_layer1))
                new_layer1, new_layer2, new_batchnorm = net2net.wider(
                    old_layer1, old_layer2, new_out_size, old_batchnorm
                )
                table.set(*prev, new_layer1)
                table.set(p, n, new_layer2)
                if between_batchnorm is not None:
                    table.set(*between_batchnorm, new_batchnorm)
            prev = (p, n)
            between_batchnorm = None


class SmallFeedForward(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.fc1 = nn.Linear(in_features, 1024)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(1024, out_features)

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x


class SmallConvolution(nn.Module):
    def __init__(self, in_channels, out_features):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.fc = nn.Linear(9216, out_features)

    def forward(self, x):
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


if __name__ == "__main__":
    model = SmallFeedForward(in_features=784, out_features=10)
    print(model)
    widen(model, 1.5)
    print(model)
