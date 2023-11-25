#!/usr/bin/env python3.8

import torch.nn as nn
import torch


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
