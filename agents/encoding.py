#!/usr/bin/env python3.8

import torch.nn as nn
import sys
import os

sys.path.append(os.path.join("..", "src"))

import embedding
import models 


class NetworkEncoder(nn.Module):
    def __init__(self, hidden_size, embedding_size):
        super().__init__()
        self.embedder = embedding.LayerEmbedder(embedding_size)
        self.encoder = nn.LSTM(
            embedding_size, hidden_size, bidirectional=True, batch_first=True
        )
        self.hidden_size = hidden_size

    def forward(self, model):
        embeddings = self.embedder.get_embeddings(model)
        output, _ = self.encoder(embeddings)
        return output

if __name__ == "__main__":
    encoder = NetworkEncoder(50, 16)
    print(encoder(models.BatchNormConvolution(3, 5)))

