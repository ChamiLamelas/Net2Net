#!/usr/bin/env python3.8

import encoding
import decider
import torch
import torch.nn as nn
import gpu
import models
import tracing
import embedding


class Policy(nn.Module):
    def __init__(self, config, device):
        super().__init__()
        self.embedder = embedding.LayerEmbedder(config["embedding_size"])
        self.encoder = encoding.NetworkEncoder(
            config["hidden_size"], config["embedding_size"]  # , device
        )
        self.decider = decider.SigmoidClassifier((config["hidden_size"] * 2) + 2)
        self.softmax = nn.Softmax(dim=0)
        self.device = device

    def forward(self, state):
        embeddings, important_matrix = self.embedder.get_embeddings(state["model"])
        embeddings = embeddings.to(self.device)
        important_matrix = important_matrix.to(self.device)
        encodings = torch.matmul(important_matrix, self.encoder(embeddings))
        # encodings = torch.matmul(
        #     important_matrix,
        #     torch.randn((len(tracing.get_all_layers(state["model"]))) + 1, 100).to(
        #         self.device
        #     ),
        # )
        other_features = torch.tensor(
            [state["last_epoch_time"], state["timeleft"]],
            device=self.device,
        ).repeat((encodings.size()[0], 1))
        scores = self.decider(torch.cat([other_features, encodings], dim=1))
        return self.softmax(scores).reshape(scores.size()[0])


if __name__ == "__main__":
    p = Policy({"hidden_size": 50, "embedding_size": 16}, gpu.get_device()).to(
        gpu.get_device()
    )
    print(p({"model": models.ConvNet(), "last_epoch_time": 0.1, "timeleft": 0.5}))
