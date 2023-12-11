#!/usr/bin/env python3.8

import policy
import reward
from collections import deque
import torch.optim as optim
import models
import numpy as np
import gpu
from torch.distributions import Categorical
from abc import ABC, abstractmethod
import torch
import sys


class BaseAgent(ABC):
    def __init__(self, config):
        self.performance = None
        self.folder = config["runner"]["folder"]
        self.config = config["agent"]
        self.device = gpu.get_device(self.config["device"])
        self.policy = None

    @abstractmethod
    def init(self):
        pass

    @abstractmethod
    def action(self, state):
        pass

    @abstractmethod
    def save_prob(self):
        pass

    @abstractmethod
    def record_acc(self, acc):
        pass

    @abstractmethod
    def update(self):
        pass

    @abstractmethod
    def save(self):
        pass


class Agent(BaseAgent):
    def __init__(self, config):
        super().__init__(config)
        self.policy = policy.Policy(self.config, self.device)
        self.gamma = self.config.get("gamma", 0.99)
        self.probabilities = None
        self.rewards = None
        self.policy = self.policy.to(self.device)
        self.optimizer = optim.Adam(
            self.policy.parameters(),
            lr=self.config.get("alpha", 1e-3),
        )
        self.learning_rate_decay = self.config.get("learning_rate_decay", 0.99)
        self.learning_rate = optim.lr_scheduler.ExponentialLR(
            self.optimizer, self.learning_rate_decay
        )
        self.decay_freq = self.config.get("decay_freq", 500)
        self.episodes = 0

    def init(self):
        self.probabilities = list()
        self.saved_probs = 0
        self.rewards = None
        self.episodes += 1

    def action(self, state):
        assert self.probabilities is not None, "call init( )"
        probabilities = self.policy(state)
        selector = Categorical(probabilities)
        action = selector.sample()
        self.probabilities.append(selector.log_prob(action))
        return action.item(), probabilities.flatten().tolist()

    def save_prob(self):
        self.saved_probs += 1

    def record_acc(self, acc):
        r = reward.acc_to_reward(acc)
        self.rewards = [r] * self.saved_probs

    def T(self):
        return self.saved_probs

    def compute_goals(self):
        return torch.tensor(self.rewards)

    def get_current_lr(self):
        return self.learning_rate.get_last_lr()[0]

    def update(self):
        goals = self.compute_goals()
        objective = torch.mean(
            torch.cat([-p * g for p, g in zip(self.probabilities, goals)])
        )
        output = objective.item()
        self.optimizer.zero_grad()
        objective.backward()
        self.optimizer.step()
        if self.episodes % self.decay_freq == 0 and self.get_current_lr() >= 1e-9:
            self.learning_rate.step()
        return output

    def save(self):
        raise RuntimeError("save using a logger")


if __name__ == "__main__":
    a = Agent(
        {
            "runner": {"folder": "test"},
            "agent": {
                "vocab": "models",
                "hidden_size": 50,
                "embedding_size": 16,
                "gamma": 0.99,
                "alpha": 0.01,
                "final_weight": 0.5,
                "device": 0,
            },
        }
    )
    a.init()
    print(
        a.action({"model": models.ConvNet(), "last_epoch_time": 0.1, "timeleft": 0.5})
    )
    a.record_acc(0.5)
    a.record_acc(0.6)
    print(a.update())
