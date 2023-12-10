#!/usr/bin/env python3.8

import torch
import torch.nn as nn
import gpu
import seed 


class Decider(nn.Module):
    def __init__(self):
        super().__init__()
        hsize = 128
        self.linear1 = nn.Linear(1, hsize)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hsize, 5)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.softmax(x)
        return x


def torchinputs(xs):
    return list(
        map(lambda x: torch.tensor(x, device=gpu.get_device()).reshape((1, 1)), xs)
    )


def torchlabels(cs):
    return list(
        map(lambda c: torch.tensor(c, device=gpu.get_device()).reshape((1,)), cs)
    )


def train():
    seed.set_seed()
    xs = torchinputs([26 / 400, 13 / 400])
    cs = torchlabels([0, 1])
    episodes = 100
    model = Decider()
    model = model.to(gpu.get_device())
    loss = nn.NLLLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    for ep in range(episodes):
        objective = 0
        for x, c in zip(xs, cs):
            ypred = model(x)
            objective += loss(ypred, c)
        optimizer.zero_grad()
        objective.backward()
        optimizer.step()
        print(f"Episode {ep + 1}: objective: {objective.item()}")
    print("Evaluation:")
    model.eval()
    for x in xs:
        print(
            f"{x.item()} -> {list(map(lambda e: f'{e.item():.4f}', model(x).flatten()))}"
        )


if __name__ == "__main__":
    train()
