#!/usr/bin/env python3.8

import sys
import os

sys.path.append(os.path.join("..", "src"))

import torch.nn as nn
import distillation
import deepening
import models
import copy
import utils


def test_deeper_distillation():
    s = models.TwoConvolution(3, 5)

    # copy student to be the teacher and deepen it
    t = copy.deepcopy(s)
    deepening.deepen(t)

    # pretend we did some learning and update weights to be different from student
    t.conv1[0].weight = nn.Parameter(t.conv1[0].weight.data + 1)
    t.fc[0].weight = nn.Parameter(t.fc[0].weight.data + 1)

    distillation.deeper_weight_transfer(t, s)
    utils.checkequal(t.conv1[0].weight.data, s.conv1.weight.data)
    utils.checkequal(t.fc[0].weight.data, s.fc.weight.data)


def main():
    test_deeper_distillation()
    print("ALL DISTILLATION TESTS PASSED!")


if __name__ == "__main__":
    main()
