#!/usr/bin/env python3.8

import torch.nn as nn
import embedding
import models
import torch
import gpu
import tracing


class NetworkEncoder(nn.Module):
    def __init__(self, hidden_size, embedding_size):
        super().__init__()
        self.encoder = nn.LSTM(
            embedding_size, hidden_size, bidirectional=True, batch_first=True
        )

    # def __init__(self, hidden_size, embedding_size, device):
    #     super().__init__()
    #     self.embedder = embedding.LayerEmbedder(embedding_size)
    #     self.encoder = nn.LSTM(
    #         embedding_size, hidden_size, bidirectional=True, batch_first=True
    #     )
    #     self.hidden_size = hidden_size
    #     self.device = device

    def forward(self, embeddings):
        return self.encoder(embeddings)[0]

    # def forward(self, model):
    #     embeddings, important_matrix = self.embedder.get_embeddings(model)
    #     print("forward", embeddings.requires_grad, important_matrix.requires_grad)
    #     # embeddings = torch.randn((len(tracing.get_all_layers(model)) + 1, 16))
    #     # important_matrix = torch.randn(
    #     #     (
    #     #         len(tracing.get_all_deepen_blocks(model)),
    #     #         len(tracing.get_all_layers(model)) + 1,
    #     #     )
    #     # )
    #     embeddings = embeddings.to(self.device)
    #     important_matrix = important_matrix.to(self.device)
    #     output, _ = self.encoder(embeddings)
    #     output = torch.matmul(important_matrix, output)
    #     print("forward", embeddings.requires_grad, important_matrix.requires_grad)
    #     return output.to(self.device)


if __name__ == "__main__":
    encoder = NetworkEncoder(50, 16, gpu.get_device()).to(gpu.get_device())
    print(encoder(models.ConvNet()).size())
