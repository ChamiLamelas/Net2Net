#!/usr/bin/env python3.8

import training
import torch
import models
import sys
import os
sys.path.append(os.path.join("..", "src"))


def test_small_feedforward_reproducibility():
    training.set_seed(42)
    model = models.SmallFeedForward(5, 3)
    input_t = torch.ones((2, 5))
    output_t = model(input_t)
    expected = torch.tensor(
        [[0.3137, -0.1701, 0.2066], [0.3137, -0.1701, 0.2066]])
    assert torch.allclose(output_t, expected, atol=1e-4)


def test_small_convolution_reproducibility():
    training.set_seed(42)
    model = models.TwoConvolution(1, 3)
    input_t = torch.ones((2, 1, 16, 16))
    output_t = model(input_t)
    expected = torch.tensor(
        [[0.0454, -0.1450, 0.0823], [0.0454, -0.1450, 0.0823]])
    assert torch.allclose(output_t, expected, atol=1e-4)


def main():
    test_small_feedforward_reproducibility()
    test_small_convolution_reproducibility()
    print("ALL REPRODUCIBILITY TESTS PASSED!")


if __name__ == '__main__':
    main()
