#!/usr/bin/env python3.8

import agent
import tracing
import random
import seed
import models
import torch


class RandomAgent(agent.BaseAgent):
    def __init__(self, config):
        super().__init__(config)

    def init(self):
        pass

    def action(self, state):
        nblocks = len(tracing.get_all_deepen_blocks(state["model"]))
        a = random.randint(0, nblocks)
        p = torch.zeros(nblocks + 1)
        p[a] = 1
        return a, p

    def record_acc(self, acc, final=True):
        pass

    def update(self):
        pass


class DeterministicAgent(agent.BaseAgent):
    def __init__(self, config):
        super().__init__(config)
        self.actions = self.config["action_sequence"]
        self.curr_action = None

    def init(self):
        self.curr_action = 0

    def action(self, state):
        assert self.curr_action is not None, "run init( ) first"
        nblocks = len(tracing.get_all_deepen_blocks(state["model"]))
        if self.curr_action >= len(self.actions):
            a = nblocks
        else:
            a = self.actions[self.curr_action]
            self.curr_action += 1
        p = torch.zeros(nblocks + 1)
        p[a] = 1
        return a, p

    def record_acc(self, acc, final=True):
        pass

    def update(self):
        pass


if __name__ == "__main__":
    seed.set_seed()
    states = [
        {"model": models.ConvNet(), "last_epoch_time": 0.1, "timeleft": 0.5},
        {"model": models.ConvNet(), "last_epoch_time": 0.1, "timeleft": 0.25},
    ]

    r = RandomAgent({"agent": {"device": 0}})
    r.init()
    print(r.action(states[0]))
    print(r.action(states[1]))

    d = DeterministicAgent({"agent": {"device": 0, "action_sequence": [3]}})
    d.init()
    print(d.action(states[0]))
    print(d.action(states[1]))
