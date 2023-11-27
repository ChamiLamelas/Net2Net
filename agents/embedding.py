#!/usr/bin/env python3.8

import torch.nn as nn
import torch
import sys
import os

sys.path.append(os.path.join("..", "src"))

import tracing
import models


class LayerEmbedder:
    def __init__(self, embedding_size):
        self.ids = {models.get_str_rep(None): 0}
        self.id = 1
        self.embedding_size = embedding_size

    def get_embeddings(self, model):
        ids = list()
        layers = tracing.get_all_important_layers(model)
        for layer in layers:
            str_rep = models.get_str_rep(layer)
            if str_rep not in self.ids:
                self.ids[str_rep] = self.id
                self.id += 1
            ids.append(self.ids[str_rep])
        ids.append(0)
        embedder = nn.Embedding(len(self.ids), self.embedding_size)
        embedder.eval()
        output = embedder(torch.LongTensor(ids))
        return output


if __name__ == "__main__":
    embedder = LayerEmbedder(16)
    print(embedder.get_embeddings(models.ConvNet()))
