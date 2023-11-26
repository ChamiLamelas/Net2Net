#!/usr/bin/env python3.8

import torch
import torch.nn as nn


class SigmoidClassifier(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.linear = nn.Linear(in_features, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, encoding):
        return self.sigmoid(self.linear(encoding))


if __name__ == "__main__":
    input = torch.randn((1, 3))
    classifier = SigmoidClassifier(3)
    print(classifier(input))
