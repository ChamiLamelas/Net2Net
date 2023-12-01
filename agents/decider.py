#!/usr/bin/env python3.8

import torch
import torch.nn as nn
import numpy as np
import tracing
import models 


def make_decider_matrix(model):
    layers = tracing.get_all_layers(model)
    important_idxs = np.argwhere(list(map(tracing.is_important, layers))).flatten()
    decision_matrix = torch.zeros((len(important_idxs) + 1, len(layers) + 1))
    for i, idx in enumerate(important_idxs):
        decision_matrix[i, idx] = 1
    decision_matrix[-1, -1] = 1
    return decision_matrix


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

    dm = make_decider_matrix(models.ConvNet())
    print(dm)