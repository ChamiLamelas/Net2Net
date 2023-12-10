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
        self.gamma = self.config.get("gamma", 0.99)
        self.probabilities = None
        self.rewards = None
        # self.final_weight = self.config.get("final_weight", 0.9)
        self.policy = self.policy.to(self.device)
        self.optimizer = optim.Adam(
            self.policy.parameters(),
            lr=self.config.get("alpha", 1e-3),
        )
        self.learning_rate_decay = self.config.get("learning_rate_decay", 0.99)
        self.learning_rate = optim.lr_scheduler.ExponentialLR(
            self.optimizer, self.learning_rate_decay
        )

    def init(self):
        self.probabilities = list()
        self.saved_probs = 0
        self.rewards = None

    # def remove_unrewarded_probability(self):
    #     if len(self.probabilities) > (len(self.rewards) - 1):
    #         self.probabilities.pop(-1)

    def action(self, state):
        assert self.probabilities is not None, "call init( )"
        # self.remove_unrewarded_probability()
        probabilities = self.policy(state)
        # print("P=", [e.item() for e in probabilities], file=sys.stderr)
        selector = Categorical(probabilities)
        action = selector.sample()
        self.probabilities.append(selector.log_prob(action))
        return action.item(), probabilities.flatten().tolist()

    def save_prob(self):
        self.saved_probs += 1

    def record_acc(self, acc, final=False):
        # if final:
        #     self.remove_unrewarded_probability()
        r = reward.acc_to_reward(acc)
        self.rewards = [r] * self.saved_probs

    def T(self):
        return self.saved_probs

    def compute_goals(self):
        Gs = deque()
        G = 0
        for r in self.rewards[::-1]:
            G = r + (self.gamma * G)
            Gs.appendleft(G)
        Gs = torch.tensor(Gs)
        Gs = torch.tensor(self.rewards)
        # print("G=", [e.item() for e in Gs], file=sys.stderr)
        # if Gs.shape[0] > 1:
        #     Gs = (Gs - Gs.mean()) / (Gs.std() + 1e-9)
        # print("NG=", [e.item() for e in Gs], file=sys.stderr)
        return Gs

    def transform_rewards(self):
        pass
        # print("R=", self.rewards)
        # self.rewards = list(map(, self.rewards[1:]))
        # self.rewards = [
        #     np.dot([1 - self.final_weight, self.final_weight], [self.rewards[-1], r])
        #     for r in self.rewards[1:-1]
        # ]
        # print("TR=", self.rewards)

    def get_current_lr(self):
        return self.learning_rate.get_last_lr()[0]

    def update(self, decay):
        self.transform_rewards()
        goals = self.compute_goals()
        # print(goals, file=sys.stderr)
        # print(self.probabilities, file=sys.stderr)
        # print([-p * g for p, g in zip(self.probabilities, goals)])
        objective = torch.mean(
            torch.cat([-p * g for p, g in zip(self.probabilities, goals)])
        )
        # print("PG=", [(-p) * g for p, g in zip(self.probabilities, goals)])
        # print(list(self.policy.named_parameters()), file=sys.stderr)
        output = objective.item()
        self.optimizer.zero_grad()
        objective.backward()
        self.optimizer.step()
        if decay and self.get_current_lr() >= 1e-9:
            self.learning_rate.step()
        return output


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
