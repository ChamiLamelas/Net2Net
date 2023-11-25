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
        self.id = 0
        self.ids = dict()
        self.embedding_size = embedding_size

    def get_embeddings(self, model):
        ids = list()
        hierarchies = tracing.get_all_important_layer_hierarchies(model)
        for hierarchy in hierarchies:
            key = "_".join(hierarchy)
            if key not in self.ids:
                self.ids[key] = self.id
                self.id += 1
            ids.append(self.ids[key])
        embedder = nn.Embedding(len(self.ids), self.embedding_size)
        return embedder(torch.LongTensor(ids))

if __name__ == "__main__":
    embedder = LayerEmbedder(16)
    print(embedder.get_embeddings(models.BatchNormConvolution(3, 5)))

