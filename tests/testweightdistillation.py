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
import torch


def test_deeper_distillation():
    s = models.OneConvolution(3, 5)

    # copy student to be the teacher and deepen it
    t = copy.deepcopy(s)
    deepening.deepen_blocks(t, add_batch_norm=False)

    # pretend we did some learning and update weights to be different from student
    t.conv_block.layers[0].weight = nn.Parameter(t.conv_block.layers[0].weight + 1)
    t.linear.weight = nn.Parameter(t.linear.weight.data + 1)

    # do the transfer 
    distillation.deeper_weight_transfer(t, s)

    # check the weights updated
    utils.checkequal(
        t.conv_block.layers[0].weight.data, s.conv_block.layers[0].weight.data
    )
    utils.checkequal(t.linear.weight.data, s.linear.weight.data)

    # make sure that the models work and give the same output
    x = torch.randn((1, 3, 5, 5))
    utils.checkequal(t(x), s(x))


def main():
    test_deeper_distillation()
    print("ALL DISTILLATION TESTS PASSED!")


if __name__ == "__main__":
    main()
