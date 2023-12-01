#!/usr/bin/env python3.8

import encoding
import decider
import torch
import torch.nn as nn
import gpu
import models
import embedding


class Policy(nn.Module):
    def __init__(self, config, device):
        super().__init__()
        self.vocab = embedding.Vocabulary(config["vocab"])
        self.encoder = encoding.NetworkEncoder(
            self.vocab, config["embedding_size"], config["hidden_size"]
        )
        self.decider = decider.SigmoidClassifier((config["hidden_size"] * 2) + 2)
        self.softmax = nn.Softmax(dim=0)
        self.device = device

    def forward(self, state):
        all_encodings = self.encoder(self.vocab.id(state["model"]).to(self.device))
        decision_matrix = decider.make_decider_matrix(state["model"]).to(self.device)
        decide_encodings = torch.matmul(decision_matrix, all_encodings)
        other_features = torch.tensor(
            [state["last_epoch_time"], state["timeleft"]],
            device=self.device,
        ).repeat((decide_encodings.size()[0], 1))
        scores = self.decider(torch.cat([other_features, decide_encodings], dim=1))
        return self.softmax(scores).reshape(scores.size()[0])

    def save_embeddings(self):
        self.vocab.save(self.encoder.embedder.state_dict())


if __name__ == "__main__":
    p = Policy(
        {"hidden_size": 50, "embedding_size": 16, "vocab": "models"}, gpu.get_device()
    ).to(gpu.get_device())
    print(p({"model": models.ConvNet(), "last_epoch_time": 0.1, "timeleft": 0.5}))
