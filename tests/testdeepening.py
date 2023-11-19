#!/usr/bin/env python3.8

import sys
import os

sys.path.append(os.path.join("..", "src"))

from utils import check_adaptation
import models
import torch
import deepening


def test_deepen_feedforward():
    check_adaptation(
        models.SmallFeedForward,
        {"in_features": 4, "out_features": 5},
        torch.randn,
        ((1, 4)),
        deepening.deepen_blocks,
    )


def test_deepen_tiny_convolutional():
    check_adaptation(
        models.OneConvolution,
        {"in_channels": 1, "out_features": 5},
        torch.randn,
        ((1, 1, 28, 28)),
        deepening.deepen_blocks,
        add_batch_norm=False,
    )


def test_deepen_small_convolutional():
    check_adaptation(
        models.TwoConvolution,
        {"in_channels": 1, "out_features": 5},
        torch.randn,
        ((1, 1, 28, 28)),
        deepening.deepen_blocks,
        add_batch_norm=False,
    )


def test_deepen_norm_convolutional():
    check_adaptation(
        models.BatchNormConvolution,
        {"in_channels": 1, "out_features": 5},
        torch.normal,
        (torch.ones((1, 1, 28, 28)), torch.ones((1, 1, 28, 28))),
        deepening.deepen_blocks,
    )


def test_deepen_inception():
    def make_two():
        x = torch.randn((1, 3, 299, 299))
        return torch.cat([x, x], dim=0)

    check_adaptation(
        models.imagenet_inception,
        {"dropout": 0},
        make_two,
        [],
        deepening.deepen_blocks,
        filter_function=models.inception_deepen_filter_function,
        eval_=False,
    )


def test_deepen_subnet():
    check_adaptation(
        models.InceptionE,
        {"in_channels": 3, "conv_block": None},
        torch.randn,
        ((1, 3, 75, 75)),
        deepening.deepen_blocks,
        filter_function=models.inception_deepen_filter_function,
    )


def test_deepen_rectangular_kernel():
    check_adaptation(
        models.ConvolutionalNet2NetDeepenBlock,
        {"in_channels": 1, "out_channels": 3, "kernel_size": (7, 1)},
        torch.randn,
        ((1, 1, 28, 28)),
        deepening.deepen_blocks,
        add_batch_norm=False,
    )


def main():
    test_deepen_feedforward()
    test_deepen_tiny_convolutional()
    test_deepen_small_convolutional()
    test_deepen_norm_convolutional()
    test_deepen_rectangular_kernel()
    test_deepen_subnet()
    print("ALL NON-INCEPTION DEEPEN TESTS PASSED!")
    test_deepen_inception()


if __name__ == "__main__":
    main()
