"""
NEEDSWORK document
"""

import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F
import torch
import tracing


def _count_layers(model, layertype):
    table = tracing.LayerTable(model)
    return sum(
        1
        for e in table
        if isinstance(tracing.LayerTable.get(e["hierarchy"], e["name"]), layertype)
    )


def num_conv_layers(model):
    return _count_layers(model, nn.Conv2d)


def num_linear_layers(model):
    return _count_layers(model, nn.Linear)


def count_parameters(model):
    return sum(torch.numel(p) for p in model.parameters())


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


class OneConvolution(nn.Module):
    def __init__(self, in_channels, out_features):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels=in_channels, out_channels=2, kernel_size=3, stride=1
        )
        self.relu1 = nn.ReLU()
        self.finalpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(2, out_features)

    def forward(self, x):
        assert x.dim() == 4
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.finalpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


class TwoConvolution(nn.Module):
    def __init__(self, in_channels, out_features):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels=in_channels, out_channels=32, kernel_size=3, stride=1
        )
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(
            in_channels=32, out_channels=64, kernel_size=3, stride=1)
        self.relu2 = nn.ReLU()
        self.finalpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, out_features)

    def forward(self, x):
        assert x.dim() == 4
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.finalpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


class BatchNormConvolution(nn.Module):
    def __init__(self, in_channels, out_features):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels=in_channels, out_channels=32, kernel_size=3, stride=1
        )
        self.relu1 = nn.ReLU()
        self.norm = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(
            in_channels=32, out_channels=64, kernel_size=3, stride=1)
        self.relu2 = nn.ReLU()
        self.finalpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, out_features)

    def forward(self, x):
        # print(x.size())

        assert x.dim() == 4
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.norm(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.finalpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


class LinearBlock(nn.Module):
    def __init__(self, in_features, hidden_features, out_features):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(hidden_features, out_features)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


class BlockedModel(nn.Module):
    def __init__(self, in_features, out_features) -> None:
        super().__init__()
        self.block1 = LinearBlock(in_features, 512, 1024)
        self.block2 = LinearBlock(1024, 1024, out_features)

    def forward(self, x):
        return self.block2(self.block1(x))


class SequentialModel(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Linear(in_features=in_features, out_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=out_features),
        )

    def forward(self, x):
        return self.seq(x)


class NonSequentialConvolution(nn.Module):
    def __init__(self, in_channels, out_features):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels=32, kernel_size=1)
        self.conv2 = nn.Conv2d(in_channels, out_channels=48, kernel_size=1)
        self.conv3 = nn.Conv2d(
            in_channels=48, out_channels=64, kernel_size=5, padding=2
        )
        self.finalpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(96, out_features)

    def forward(self, x):
        a = self.conv1(x)

        x = self.conv2(x)
        x = self.conv3(x)

        x = torch.cat([a, x], 1)

        x = self.finalpool(x)

        x = torch.flatten(x, 1)

        x = self.fc(x)
        return x


class RectangularKernel(nn.Module):
    def __init__(self, in_channels, out_features):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels,
            384,
            kernel_size=(1, 7),
            stride=(1, 1),
            padding=(0, 3),
            bias=False,
        )
        self.finalpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(384, out_features)

    def forward(self, x):
        x = self.conv1(x)
        x = self.finalpool(x)
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


def imagenet_inception(dropout=0.5):
    """27 million parameters"""

    # requires Nx3x75x75 (N>=1) input for evaluation, produces 1000-dim output!
    # https://pytorch.org/vision/0.12/generated/torchvision.models.inception_v3.html
    # requires Nx3x299x299 (N>=1) input for training, produces 2 1000-dim logit sets
    # https://pytorch.org/vision/main/models/generated/torchvision.models.inception_v3.html#torchvision.models.inception_v3
    return models.inception_v3(weights=None, init_weights=True, dropout=dropout)


def widen_inception(e):
    layer = tracing.LayerTable.get(e["hierarchy"], e["name"])
    if not isinstance(layer, nn.Conv2d):
        return 0
    if any("Inception" in type(c).__name__ for c in e["hierarchy"]):
        return 1 / (0.3**0.5)


def deepen_inception(e):
    curr_layer = tracing.LayerTable.get(e["hierarchy"], e["name"])
    if not isinstance(curr_layer, nn.Conv2d):
        return False
    if type(e["hierarchy"][-2]).__name__ not in {
        "InceptionC",
        "InceptionD",
        "InceptionE",
    }:
        return False
    if curr_layer.kernel_size[0] != curr_layer.kernel_size[1]:
        return True
    return False


def inception_ignoreset():
    return {models.inception.BasicConv2d.__name__}


class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)


class InceptionE(nn.Module):
    def __init__(self, in_channels, conv_block):
        super().__init__()
        if conv_block is None:
            conv_block = BasicConv2d
        self.branch1x1 = conv_block(in_channels, 320, kernel_size=1)

        self.branch3x3_1 = conv_block(in_channels, 384, kernel_size=1)
        self.branch3x3_2a = conv_block(
            384, 384, kernel_size=(1, 3), padding=(0, 1))
        self.branch3x3_2b = conv_block(
            384, 384, kernel_size=(3, 1), padding=(1, 0))

        self.branch3x3dbl_1 = conv_block(in_channels, 448, kernel_size=1)
        self.branch3x3dbl_2 = conv_block(448, 384, kernel_size=3, padding=1)
        self.branch3x3dbl_3a = conv_block(
            384, 384, kernel_size=(1, 3), padding=(0, 1))
        self.branch3x3dbl_3b = conv_block(
            384, 384, kernel_size=(3, 1), padding=(1, 0))

        self.branch_pool = conv_block(in_channels, 192, kernel_size=1)
        self.avgpool = nn.AdaptiveMaxPool2d((1, 1))
        self.fc = nn.Linear(2048, 10)

    def _forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch3x3 = self.branch3x3_1(x)
        branch3x3 = [
            self.branch3x3_2a(branch3x3),
            self.branch3x3_2b(branch3x3),
        ]
        branch3x3 = torch.cat(branch3x3, 1)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = [
            self.branch3x3dbl_3a(branch3x3dbl),
            self.branch3x3dbl_3b(branch3x3dbl),
        ]
        branch3x3dbl = torch.cat(branch3x3dbl, 1)

        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch3x3, branch3x3dbl, branch_pool]
        return outputs

    def forward(self, x):
        outputs = self._forward(x)
        outputs = torch.cat(outputs, 1)
        return outputs


