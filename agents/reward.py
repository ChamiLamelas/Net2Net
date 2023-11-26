#!/usr/bin/env python3.8

import math


def acc_to_reward(acc):
    return math.tan(acc * (math.pi / 2))


if __name__ == "__main__":
    print(acc_to_reward(0.9))