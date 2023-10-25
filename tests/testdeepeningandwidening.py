import sys 
import os 
sys.path.append(os.path.join("..", "src"))

from utils import check_adaptation, widendeepen, deepenwiden
import models 
import torch 

def test_widen_and_deepen_feedforward():
    check_adaptation(
        models.SmallFeedForward,
        {"in_features": 4, "out_features": 5},
        torch.randn,
        ((1, 4)),
        widendeepen,
    )


def test_deepen_and_widen_feedforward():
    check_adaptation(
        models.SmallFeedForward,
        {"in_features": 4, "out_features": 5},
        torch.randn,
        ((1, 4)),
        deepenwiden,
    )


def test_deepen_and_widen_tiny_convolutional():
    check_adaptation(
        models.OneConvolution,
        {"in_channels": 1, "out_features": 5},
        torch.randn,
        ((1, 1, 28, 28)),
        deepenwiden,
    )





def test_widen_and_deepen_small_convolutional():
    check_adaptation(
        models.TwoConvolution,
        {"in_channels": 1, "out_features": 5},
        torch.randn,
        ((1, 1, 28, 28)),
        deepenwiden,
    )


def test_deepen_and_widen_small_convolutional():
    check_adaptation(
        models.TwoConvolution,
        {"in_channels": 1, "out_features": 5},
        torch.randn,
        ((1, 1, 28, 28)),
        widendeepen,
    )


def test_widen_and_deepen_norm_convolutional():
    check_adaptation(
        models.BatchNormConvolution,
        {"in_channels": 1, "out_features": 5},
        torch.randn,
        ((1, 1, 28, 28)),
        widendeepen,
    )


def test_deepen_and_widen_norm_convolutional():
    check_adaptation(
        models.BatchNormConvolution,
        {"in_channels": 1, "out_features": 5},
        torch.randn,
        ((1, 1, 28, 28)),
        deepenwiden,
    )


def main():
    test_deepen_and_widen_feedforward()
    test_deepen_and_widen_tiny_convolutional()
    test_deepen_and_widen_small_convolutional()
    test_deepen_and_widen_norm_convolutional()
    test_widen_and_deepen_feedforward()
    test_widen_and_deepen_small_convolutional()
    test_widen_and_deepen_norm_convolutional()
    print("ALL WIDEN AND DEEPEN TESTS PASSED!")

if __name__ == '__main__':
    main()