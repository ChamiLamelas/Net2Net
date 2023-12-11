#!/usr/bin/env python3.8

import encoding
import decider
import torch
import torch.nn as nn
import gpu
import models
import embedding
import sys
import timeencoding


class Policy(nn.Module):
    def __init__(self, config, device):
        super().__init__()
        self.vocab = embedding.Vocabulary(config["vocab"])
        hidden_size = config.get("hidden_size", 50)
        self.encoder = encoding.NetworkEncoder(
            self.vocab, config.get("embedding_size", 16), hidden_size
        )
        self.time_encoding_size = config.get("time_encoding_size", 8)
        self.decider = decider.Decider2(
            (hidden_size * 2),
            config.get("decider_lstm_size", 32),
            config.get("decider_linear_size", 32),
            self.time_encoding_size,
            config["max_actions"],
        )
        self.device = device

    def forward(self, state):
        all_encodings = self.encoder(self.vocab.id(state["model"]).to(self.device))
        decision_matrix = decider.make_decider_matrix(state["model"]).to(self.device)
        decide_encodings = torch.matmul(decision_matrix, all_encodings)
        other_features = (
            torch.from_numpy(
                timeencoding.encode_time(
                    state["timeleft"], state["totaltime"], self.time_encoding_size
                )
            )
            .float()
            .to(self.device)
        )
        probabilities = self.decider(
            {"encodings": decide_encodings, "other": other_features}
        )
        return probabilities

    def save_embeddings(self):
        self.vocab.save(self.encoder.embedder.state_dict())


if __name__ == "__main__":
    # p = Policy(
    #     {"hidden_size": 50, "embedding_size": 16, "vocab": "models"}, gpu.get_device()
    # ).to(gpu.get_device())
    # print(p({"model": models.ConvNet(), "last_epoch_time": 0.1, "timeleft": 0.5}))
    pass
