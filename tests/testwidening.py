import sys 
import os 
sys.path.append(os.path.join("..", "src"))

from utils import check_adaptation
import models 
import torch 
import widening

def test_widen_feedforward():
    check_adaptation(
        models.SmallFeedForward,
        {"in_features": 4, "out_features": 5},
        torch.randn,
        ((1, 4)),
        widening.widen,
    )


def test_widen_small_convolutional():
    check_adaptation(
        models.TwoConvolution,
        {"in_channels": 1, "out_features": 5},
        torch.randn,
        ((1, 1, 28, 28)),
        widening.widen,
    )

def test_widen_norm_convolutional():
    check_adaptation(
        models.BatchNormConvolution,
        {"in_channels": 1, "out_features": 5},
        torch.randn,
        ((1, 1, 28, 28)),
        widening.widen,
    )

def test_widen_nonsequential():
    check_adaptation(
        models.NonSequentialConvolution,
        {"in_channels": 3, "out_features": 5},
        torch.randn,
        ((1, 3, 75, 75)),
        widening.widen,
    )


def test_widen_inception():
    check_adaptation(
        models.imagenet_inception,
        {},
        torch.randn,
        ((1, 3, 75, 75)),
        widening.widen,
        models.inception_ignoreset(),
        models.widen_inception,
    )

def main():
    test_widen_feedforward()
    test_widen_small_convolutional()
    test_widen_norm_convolutional()
    test_widen_nonsequential()
    test_widen_inception()
    print("ALL WIDEN TESTS PASSED!")

if __name__ == '__main__':
    main()