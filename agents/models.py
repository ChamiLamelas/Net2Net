#!/usr/bin/env python3.8

import torch.nn as nn
import torch

DEEPEN_BLOCK_NAME = "Net2NetDeepenBlock"


def is_deepen_block(layer):
    return DEEPEN_BLOCK_NAME in type(layer).__name__


def count_parameters(model):
    return sum(torch.numel(p) for p in model.parameters())


def get_str_rep(layer):
    if is_deepen_block(layer):
        layer = layer.layers[0]
    if isinstance(layer, nn.Conv2d):
        return f"conv-{layer.in_channels}-{layer.out_channels}-{layer.kernel_size[0]}-{layer.kernel_size[1]}-{layer.padding[0]}-{layer.padding[1]}"
    elif isinstance(layer, nn.Linear):
        return f"linear-{layer.in_features}-{layer.out_features}"
    elif layer is None:
        return f"none-layer"


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
            nn.Conv2d(in_channels, out_channels, kernel_size, **kwargs),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.layers(x)


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


class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = NormalizedConvolutionalNet2NetDeepenBlock(3, 32, 3)
        self.conv2 = NormalizedConvolutionalNet2NetDeepenBlock(32, 32, 3)
        self.conv3 = NormalizedConvolutionalNet2NetDeepenBlock(32, 32, 3)
        self.finalpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = FeedForwardNet2NetDeepenBlock(32, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.finalpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


class LargeConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(
            NormalizedConvolutionalNet2NetDeepenBlock(3, 32, 3),
            NormalizedConvolutionalNet2NetDeepenBlock(32, 32, 3, padding=1),
        )
        self.conv2 = nn.Sequential(
            NormalizedConvolutionalNet2NetDeepenBlock(32, 32, 3),
            NormalizedConvolutionalNet2NetDeepenBlock(32, 32, 3, padding=1),
        )
        self.conv3 = nn.Sequential(
            NormalizedConvolutionalNet2NetDeepenBlock(32, 32, 3),
            NormalizedConvolutionalNet2NetDeepenBlock(32, 32, 3, padding=1),
        )
        self.finalpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            FeedForwardNet2NetDeepenBlock(32, 10), FeedForwardNet2NetDeepenBlock(10, 10)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.finalpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
