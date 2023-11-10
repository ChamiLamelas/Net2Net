#!/usr/bin/env python3.8

import sys
import os

sys.path.append(os.path.join("..", "src"))

import data


def test_mnist():
    batch = next(iter(data.load_mnist(train=False, batch_size=2)))[0]
    assert list(batch.size()) == [2, 1, 28, 28]


def test_cifar10():
    batch = next(iter(data.load_cifar10(train=False, batch_size=2)))[0]
    assert list(batch.size()) == [2, 3, 299, 299]


def test_imagenet():
    batch = next(iter(data.load_imagenet(split="val", batch_size=2)))[0]
    assert list(batch.size()) == [2, 3, 224, 224]


def main():
    test_mnist()
    test_cifar10()
    test_imagenet()
    print("ALL DATA TESTS PASSED!")


if __name__ == "__main__":
    main()
