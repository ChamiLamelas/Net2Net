import models
import torch
import training
import data 

def test_mnist():
    batch = next(iter(data.load_mnist(train=False, batch_size=2)))[0]
    print(batch.size())


def test_cifar10():
    batch = next(iter(data.load_cifar10(train=False, batch_size=2)))[0]
    print(batch.size())


def test_small_feedforward_reproducibility():
    training.set_seed(42)
    model = models.SmallFeedForward(5, 3)
    input_t = torch.ones((2, 5))
    output_t = model(input_t)
    print(output_t)

def test_small_convolution_reproducibility():
    training.set_seed(42)
    model = models.SmallConvolution(1, 3)
    input_t = torch.ones((2, 1, 16, 16))
    output_t = model(input_t)
    print(output_t)

def main():
    test_mnist()
    test_cifar10()
    test_small_feedforward_reproducibility()
    test_small_convolution_reproducibility()


if __name__ == '__main__':
    main()