#!/usr/bin/env python3.8

import torch
import torch.nn as nn
import gpu
from torch.distributions import Categorical
import seed 

class Decider(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(1, 128)
        self.relu = nn.Sigmoid()
        self.linear2 = nn.Linear(128, 5)
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
    gamma = 0.9
    xs = torchinputs([26 / 400, 13 / 400])
    cs = torchlabels([1, 2])
    episodes = 1000
    model = Decider()
    model = model.to(gpu.get_device())
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    for ep in range(episodes):
        log_probs = list()
        correct = list()
        for x, c in zip(xs, cs):
            ypred = model(x)
            selector = Categorical(ypred)
            action = selector.sample()
            log_prob = selector.log_prob(action)
            log_probs.append(log_prob)
            correct.append(action == c)
        goals = [0.6 * gamma, 0.6] if all(correct) else [0.4 * gamma, 0.4]
        objective = torch.mean(torch.cat([-p * g for p, g in zip(log_probs, goals)]))
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
    for n, p in model.named_parameters():
        print(f"{n}: {p.min().item():.4f} {p.max().item():.4f}")


if __name__ == "__main__":
    train()
