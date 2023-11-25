#!/usr/bin/env python3.8

import torch 
import torch.nn as nn
import encoding 


class SigmoidClassifier(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.sigmoid = nn.Sigmoid()

    def forward(self, encoding):
        return self.sigmoid(self.linear(encoding))
    
    
class Controller(nn.Module):
    def __init__(self):
        self.classifier = SigmoidClassifier(10 + 50, 1)
        self.encoder = encoding.NetworkEncoder(50, 16)

    def forward(self, inputs):
        encoding = self.encoder(inputs["model"])
        # pass this concatenated with transformed features into classifier


if __name__ == "__main__":
    input = torch.randn((1, 3))
    classifier = SigmoidClassifier(3, 2)
    print(classifier(input))