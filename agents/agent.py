#!/usr/bin/env python3.8

import policy
import reward
import torch.nn as nn
import torch.optim as optim
import models
import numpy as np
import gpu
from torch.distributions import Categorical
from abc import ABC, abstractmethod
from collections import deque
import torch


class BaseAgent(ABC):
    def __init__(self, config):
        self.performance = None
        self.config = config["agent"]
        self.device = gpu.get_device(self.config["device"])

    @abstractmethod
    def init(self):
        pass

    @abstractmethod
    def action(self, state):
        pass

    @abstractmethod
    def record_acc(self, acc, final=False):
        pass

    @abstractmethod
    def update(self):
        pass


class Agent(BaseAgent):
    def __init__(self, config):
        super().__init__(config)
        self.policy = policy.Policy(self.config, self.device)
        self.gamma = self.config["gamma"]
        self.optimizer = optim.Adam(self.policy.parameters(), self.config["alpha"])
        self.probabilities = None
        self.rewards = None
        self.final_weight = self.config["final_weight"]
        self.policy = self.policy.to(self.device)
        self.baseline_decay = self.config.get("baseline_decay", None)

    def init(self):
        self.probabilities = list()
        self.rewards = [None]

    def remove_unrewarded_probability(self):
        if len(self.probabilities) > (len(self.rewards) - 1):
            self.probabilities.pop(-1)

    def action(self, state):
        assert self.probabilities is not None, "call init( )"
        self.remove_unrewarded_probability()
        probabilities = self.policy(state)
        selector = Categorical(probabilities)
        action = selector.sample()
        self.probabilities.append(selector.log_prob(action))
        return action.item(), probabilities.shape[0]

    def record_acc(self, acc, final=False):
        if final:
            self.remove_unrewarded_probability()
        self.rewards.append(acc)

    def T(self):
        return len(self.probabilities)

    def compute_goals(self):
        Gs = deque()
        G = 0
        for t in range(self.T() - 1, -1, -1):
            G = self.rewards[t] + (self.gamma * G)
            Gs.appendleft(G)
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
        self.optimizer.zero_grad()
        objective = torch.sum(
            torch.stack([p * g for p, g in zip(self.probabilities, goals)])
        )
        loss = -objective
        loss.backward()
        self.optimizer.step()


if __name__ == "__main__":
    a = Agent(
        {
            "agent": {
                "vocab": "models",
                "hidden_size": 50,
                "embedding_size": 16,
                "gamma": 0.99,
                "alpha": 0.01,
                "final_weight": 0.5,
                "device": 0,
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
