"""
NEEDSWORK document
"""

import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F
import torch
# import deepening


# def _count_layers(model, layertype):
#     table = tracing.LayerTable(model)
#     return sum(1 for e in table if isinstance(table.get(e["hierarchy"]), layertype))


# def num_conv_layers(model):
#     return _count_layers(model, nn.Conv2d)


# def num_linear_layers(model):
#     return _count_layers(model, nn.Linear)


def count_parameters(model):
    return sum(torch.numel(p) for p in model.parameters())


class FeedForwardNet2NetDeepenBlock(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(in_features, out_features), nn.ReLU())

    def forward(self, x):
        return self.layers(x)


class ConvolutionalNet2NetDeepenBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, **kwargs):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, **kwargs), nn.ReLU()
        )

    def forward(self, x):
        return self.layers(x)


class NormalizedConvolutionalNet2NetDeepenBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, **kwargs):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, **kwargs), nn.ReLU()
        )

    def forward(self, x):
        return self.layers(x)


class SmallFeedForward(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.feedforward = FeedForwardNet2NetDeepenBlock(in_features, 1024)
        self.linear = nn.Linear(1024, out_features)

    def forward(self, x):
        x = self.feedforward(x)
        x = self.linear(x)
        return x


class OneConvolution(nn.Module):
    def __init__(self, in_channels, out_features):
        super().__init__()
        self.conv_block = ConvolutionalNet2NetDeepenBlock(in_channels, 32, 3)
        self.finalpool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(32, out_features)

    def forward(self, x):
        x = self.conv_block(x)
        x = self.finalpool(x)
        x = torch.flatten(x, 1)
        x = self.linear(x)
        return x


class TwoConvolution(nn.Module):
    def __init__(self, in_channels, out_features):
        super().__init__()
        self.conv1 = ConvolutionalNet2NetDeepenBlock(in_channels, 32, 3)
        self.conv2 = ConvolutionalNet2NetDeepenBlock(32, 64, 3)
        self.finalpool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(64, out_features)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.finalpool(x)
        x = torch.flatten(x, 1)
        x = self.linear(x)
        return x


class BatchNormConvolution(nn.Module):
    def __init__(self, in_channels, out_features):
        super().__init__()
        self.conv1 = NormalizedConvolutionalNet2NetDeepenBlock(in_channels, 32, 3)
        self.conv2 = NormalizedConvolutionalNet2NetDeepenBlock(32, 64, 3)
        self.finalpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, out_features)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.finalpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


class NonSequentialConvolution(nn.Module):
    def __init__(self, in_channels, out_features):
        super().__init__()
        self.conv1 = ConvolutionalNet2NetDeepenBlock(in_channels, 32, 1)
        self.conv2 = ConvolutionalNet2NetDeepenBlock(in_channels, 48, 1)
        self.conv3 = ConvolutionalNet2NetDeepenBlock(48, 64, 5, padding=2)
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


def load_inception(weights, init_weights, dropout):
    if not hasattr(models.inception, "InceptionNet2NetDeepenBlock"):
        raise RuntimeError(
            "You have not loaded the instrumented version of InceptionNet"
        )
    return models.inception_v3(
        weights=weights, init_weights=init_weights, dropout=dropout
    )


def cifar10_inception(dropout=0.5):
    """~27 million parameters"""

    model = load_inception(None, True, dropout)
    model.fc = nn.Linear(2048, 10)
    return model


# def deepened_cifar10_inception(dropout=0.5):
#     model = cifar10_inception(dropout)
#     deepening.deepen_blocks(model, inception_deepen_filter_function, True)
#     return model


def tiny_imagenet_inception(dropout=0.5):
    """~27 million parameters"""

    model = load_inception(None, True, dropout)
    model.fc = nn.Linear(2048, 200)
    return model


def imagenet_inception(dropout=0.5):
    """27 million parameters"""

    # requires Nx3x75x75 (N>=1) input for evaluation, produces 1000-dim output!
    # https://pytorch.org/vision/0.12/generated/torchvision.models.inception_v3.html
    # requires Nx3x299x299 (N>=1) input for training, produces 2 1000-dim logit sets
    # https://pytorch.org/vision/main/models/generated/torchvision.models.inception_v3.html#torchvision.models.inception_v3
    return load_inception(None, True, dropout)


"""
def widen_inception(e):
    layer = tracing.LayerTable.get(e["hierarchy"], e["name"])
    if not isinstance(layer, nn.Conv2d):
        return 0
    if any("Inception" in type(c).__name__ for c in e["hierarchy"]):
        return 1 / (0.3**0.5)


def deepen_inception(e, curr_layer):
    if not isinstance(curr_layer, nn.Conv2d):
        return False
    if type(e["typehierarchy"][-2]).__name__ not in {
        "InceptionC",
        "InceptionD",
        "InceptionE",
    }:
        return False
    if curr_layer.kernel_size[0] != curr_layer.kernel_size[1]:
        return True
    return False
"""


def inception_deepen_filter_function(block, hierarchy):
    if type(hierarchy[-1]).__name__ not in {
        "InceptionC",
        "InceptionD",
        "InceptionE",
    }:
        return False
    conv_layer = block.layers[0]
    if conv_layer.kernel_size[0] != conv_layer.kernel_size[1]:
        return True
    return False


# def inception_ignoreset():
#     return {models.inception.BasicConv2d.__name__}


class InceptionNet2NetDeepenBlock(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, bias=False, **kwargs),
            nn.BatchNorm2d(out_channels, eps=0.001),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.layers(x)


class InceptionE(nn.Module):
    def __init__(self, in_channels, conv_block):
        super().__init__()
        if conv_block is None:
            conv_block = InceptionNet2NetDeepenBlock
        self.branch1x1 = conv_block(in_channels, 320, kernel_size=1)

        self.branch3x3_1 = conv_block(in_channels, 384, kernel_size=1)
        self.branch3x3_2a = conv_block(384, 384, kernel_size=(1, 3), padding=(0, 1))
        self.branch3x3_2b = conv_block(384, 384, kernel_size=(3, 1), padding=(1, 0))

        self.branch3x3dbl_1 = conv_block(in_channels, 448, kernel_size=1)
        self.branch3x3dbl_2 = conv_block(448, 384, kernel_size=3, padding=1)
        self.branch3x3dbl_3a = conv_block(384, 384, kernel_size=(1, 3), padding=(0, 1))
        self.branch3x3dbl_3b = conv_block(384, 384, kernel_size=(3, 1), padding=(1, 0))

        self.branch_pool = conv_block(in_channels, 192, kernel_size=1)
        self.avgpool = nn.AdaptiveMaxPool2d((1, 1))
        self.fc = nn.Linear(2048, 10)

    def _forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch3x3 = self.branch3x3_1(x)
        t1 = self.branch3x3_2a(branch3x3)
        t2 = self.branch3x3_2b(branch3x3)
        branch3x3 = torch.cat([t1, t2], 1)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        t3 = self.branch3x3dbl_3a(branch3x3dbl)
        t4 = self.branch3x3dbl_3b(branch3x3dbl)
        branch3x3dbl = torch.cat([t3, t4], 1)

        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch3x3, branch3x3dbl, branch_pool]
        return outputs

    def forward(self, x):
        outputs = self._forward(x)
        outputs = torch.cat(outputs, 1)
        return outputs


# model = InceptionE(3, None)
# blocks = tracing.get_all_deepen_blocks(model)
# print(blocks)

# blocks[0].conv2 = nn.Conv2d(3, 3, 3)
# print(model)
