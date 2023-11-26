#!/usr/bin/env python3.8

import encoding
import decider
import torch
import reward
import torch.nn as nn
import torch.optim as optim
import models
import numpy as np
import gpu


class Policy(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.encoder = encoding.NetworkEncoder(
            config["hidden_size"], config["embedding_size"]
        )
        self.decider = decider.SigmoidClassifier((config["hidden_size"] * 2) + 2)

    def forward(self, state):
        encodings = self.encoder(state["model"])
        other_features = torch.tensor(
            [state["last_epoch_time"], state["timeleft"]], requires_grad=True
        ).repeat((encodings.size()[0], 1))
        return self.decider(torch.cat([other_features, encodings], dim=1))


class Agent:
    def __init__(self, config):
        config = config["agent"]
        self.policy = Policy(config)
        self.gamma = config["gamma"]
        self.optimizer = optim.Adam(self.policy.parameters(), config["alpha"])
        self.probabilities = None
        self.rewards = None
        self.final_weight = config["final_weight"]
        self.device = gpu.get_device(config["device"])

    def init(self):
        self.probabilities = list()
        self.rewards = [None]

    def action(self, state):
        assert self.probabilities is not None, "call init( )"
        if len(self.probabilities) > len(self.rewards):
            self.rewards.append(0)
        probabilities = self.policy(state)
        action = set(
            e[0] for e in filter(lambda x: x[1] >= 0.5, enumerate(probabilities))
        )
        self.probabilities.append(torch.sum(torch.log(probabilities)))
        return action

    def record_acc(self, acc):
        self.rewards.append(acc)

    def T(self):
        return len(self.probabilities)

    def compute_goals(self):
        Gs = list()
        G = 0
        for t in range(self.T() - 1, -1, -1):
            G = self.rewards[t] + self.gamma * G
            Gs.insert(0, G)
        return Gs

    def transform_rewards(self):
        self.rewards = [self.rewards[0]] + list(
            map(reward.acc_to_reward, self.rewards[1:])
        )
        self.rewards = [
            np.dot([1 - self.final_weight, self.final_weight], [self.rewards[-1], r])
            for r in self.rewards[1:-1]
        ]

    def update(
        self,
    ):
        self.transform_rewards()
        goals = self.compute_goals()
        raised_gamma = 1
        for t in range(self.T()):
            raised_gamma *= self.gamma
            loss = raised_gamma * goals[t] * self.probabilities[t]
            loss.backward()
            self.optimizer.step()


if __name__ == "__main__":
    p = Policy({"hidden_size": 50, "embedding_size": 16})
    print(
        p({"model": models.ConvNet(), "last_epoch_time": 0.1, "timeleft": 0.5}).size()
    )

    a = Agent(
        {
            "agent": {
                "hidden_size": 50,
                "embedding_size": 16,
                "gamma": 0.99,
                "alpha": 0.01,
                "final_weight": 0.5,
            }
        }
    )
    a.init()
    print(
        a.action({"model": models.ConvNet(), "last_epoch_time": 0.1, "timeleft": 0.5})
    )
    a.record_acc(0.5)
    a.record_acc(0.6)
    a.update()
