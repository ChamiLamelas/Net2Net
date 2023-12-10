#!/usr/bin/env python3.8

import argparse
import numpy as np
from collections import deque
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import copy 

parser = argparse.ArgumentParser(description="PyTorch REINFORCE example")
parser.add_argument(
    "--gamma",
    type=float,
    default=0.99,
    metavar="G",
    help="discount factor (default: 0.99)",
)
parser.add_argument(
    "--seed", type=int, default=543, metavar="N", help="random seed (default: 543)"
)
parser.add_argument("--render", action="store_true", help="render the environment")
parser.add_argument(
    "--log-interval",
    type=int,
    default=1,
    metavar="N",
    help="interval between training status logs (default: 10)",
)
args = parser.parse_args()


# env = gym.make('CartPole-v1')
# env.reset(seed=args.seed)
torch.manual_seed(args.seed)

time_encoding_len = 8

policy_size = 128

middle = 1


class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()

        self.first = nn.Sequential(nn.Linear(time_encoding_len, policy_size), nn.ReLU())
        self.middle = nn.Sequential(
            *[
                nn.Sequential(nn.Linear(policy_size, policy_size), nn.ReLU())
                for _ in range(middle)
            ]
        )
        self.last = nn.Linear(policy_size, 5)

        self.saved_log_probs = []
        self.rewards = []

    def forward(self, x):
        x = self.first(x)
        x = self.middle(x)
        x = self.last(x)
        return F.softmax(x, dim=1)


policy = Policy()
optimizer = optim.Adam(policy.parameters(), lr=1e-3)
learning_rate_scheduler = optim.lr_scheduler.ExponentialLR(optimizer, 0.99)
eps = np.finfo(np.float32).eps.item()


def select_action(state):
    state = torch.from_numpy(state).float().unsqueeze(0)
    probs = policy(state)
    m = Categorical(probs)
    action = m.sample()
    policy.saved_log_probs.append(m.log_prob(action))
    return action.item(), probs


def finish_episode():
    R = 0
    policy_loss = []
    returns = deque()
    for r in policy.rewards[::-1]:
        R = r + args.gamma * R
        returns.appendleft(R)
    returns = torch.tensor(returns)
    returns = torch.tensor(policy.rewards)
    # returns = (returns - returns.mean()) / (returns.std() + eps)
    # print(returns)
    for log_prob, R in zip(policy.saved_log_probs, returns):
        policy_loss.append(-log_prob * R)
    # print(policy.saved_log_probs)
    # print(policy_loss)
    optimizer.zero_grad()
    policy_loss = torch.cat(policy_loss).mean()
    policy_loss.backward()
    optimizer.step()
    rewards = copy.deepcopy(policy.rewards)
    del policy.rewards[:]
    del policy.saved_log_probs[:]
    return policy_loss.item(), rewards 


def acc_to_reward(acc):
    return np.tan(acc * np.pi / 2) * 100


def get_reward(actions, correct_actions):
    acc = 0.4 + sum(a == ca for a, ca in zip(actions, correct_actions)) * 0.04
    return acc_to_reward(acc)


def main():
    nactions = 5
    original_states = [65, 52, 39, 26, 13][-nactions:]
    correct_actions = [0, 1, 2, 3, 4][:nactions]

    states = list(map(encode_time, original_states))
    print("=== States ===")
    for os, s in zip(original_states, states):
        print(f"{os} : {s}")
    print()

    for i_episode in range(5000):
        all_probs = list()
        actions = list()
        for state in states:
            action, probs = select_action(state)
            actions.append(action)
            all_probs.append(probs)

        reward = get_reward(actions, correct_actions)
        for _ in actions:
            policy.rewards.append(reward)

        objective, rewards = finish_episode()

        if (i_episode + 1) % 500 == 0 and learning_rate_scheduler.get_last_lr()[
            0
        ] >= 1e-9:
            learning_rate_scheduler.step()

        if i_episode == 0 or (i_episode + 1) % 100 == 0:
            print(f"=== Episode {i_episode + 1} ===")

            print("=== Probabilities ===")
            for probs in all_probs:
                print(" ".join(f"{p.item():.4f}" for p in probs.flatten()))

            print("=== Rewards ===")
            for r in rewards:
                print(f"{r:.4f}")

            print("=== Weights ===")
            for n, p in policy.named_parameters():
                print(f"{n}: mean={p.mean().item():.4f} std={p.std().item():.4f}")

            print("=== Learning Rate ===")
            print(f"{learning_rate_scheduler.get_last_lr()[0]:.4f}")

            print("=== Objective ===")
            print(f"{objective:.4f}")

            print()

    policy.eval()
    for original_state, state in zip(original_states, states):
        action, _ = select_action(state)
        print(f"{original_state} : {action}")


def encode_time(time_seconds, period=400, num_dimensions=time_encoding_len):
    # period is the length of the cycle (e.g., 24 hours in seconds)
    time_encoded = [
        np.sin(2 * np.pi * time_seconds / period * i)
        for i in range(1, num_dimensions + 1)
    ]
    return np.array(time_encoded)


if __name__ == "__main__":
    main()
