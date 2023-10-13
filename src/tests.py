import models
import torch
import training
import data
import dynamics
import prediction


def test_mnist():
    batch = next(iter(data.load_mnist(train=False, batch_size=2)))[0]
    return list(batch.size()) == [2, 1, 28, 28]


def test_cifar10():
    batch = next(iter(data.load_cifar10(train=False, batch_size=2)))[0]
    return list(batch.size()) == [2, 3, 32, 32]


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


def test_widen_feedforward():
    training.set_seed(42)
    model = models.SmallFeedForward(in_features=4, out_features=5)
    data = torch.randn((1, 4))
    pre_widen = prediction.forward(model, data)
    dynamics.widen(model)
    post_widen = prediction.forward(model, data)
    assert torch.allclose(pre_widen, post_widen)


def test_deepen_feedforward():
    training.set_seed(42)
    model = models.SmallFeedForward(in_features=4, out_features=5)
    data = torch.randn((1, 4))
    pre_deepen = prediction.forward(model, data)
    dynamics.deepen(model)
    post_deepen = prediction.forward(model, data)
    assert torch.allclose(pre_deepen, post_deepen)


def test_deepen_tiny_convolutional():
    training.set_seed(42)
    model = models.OneConvolution(in_channels=1, out_features=5)
    data = torch.randn((1, 1, 28, 28))
    pre_deepen = prediction.forward(model, data)
    dynamics.deepen(model)
    post_deepen = prediction.forward(model, data)
    assert torch.allclose(pre_deepen, post_deepen)


def test_deepen_and_widen_tiny_convolutional():
    training.set_seed(42)
    model = models.OneConvolution(in_channels=1, out_features=5)
    data = torch.randn((1, 1, 28, 28))
    pre_change = prediction.forward(model, data)
    # print(model)
    dynamics.deepen(model)
    # print(model)
    dynamics.widen(model)
    # print(model)
    post_change = prediction.forward(model, data)
    assert torch.allclose(pre_change, post_change)


def test_widen_small_convolutional():
    training.set_seed(42)
    model = models.TwoConvolution(in_channels=1, out_features=5)
    data = torch.randn((1, 1, 28, 28))
    pre_widen = prediction.forward(model, data)
    dynamics.widen(model)
    post_widen = prediction.forward(model, data)
    assert torch.allclose(pre_widen, post_widen)


def test_deepen_small_convolutional():
    training.set_seed(42)
    model = models.TwoConvolution(in_channels=1, out_features=5)
    data = torch.randn((1, 1, 28, 28))
    pre_deepen = prediction.forward(model, data)
    dynamics.deepen(model)
    post_deepen = prediction.forward(model, data)
    assert torch.allclose(pre_deepen, post_deepen)


def test_widen_and_deepen_small_convolutional():
    training.set_seed(42)
    model = models.TwoConvolution(in_channels=1, out_features=5)
    data = torch.randn((1, 1, 28, 28))
    pre_change = prediction.forward(model, data)
    dynamics.widen(model)
    dynamics.deepen(model)
    post_change = prediction.forward(model, data)
    assert torch.allclose(pre_change, post_change)


def test_deepen_and_widen_small_convolutional():
    training.set_seed(42)
    model = models.TwoConvolution(in_channels=1, out_features=5)
    data = torch.randn((1, 1, 28, 28))
    pre_change = prediction.forward(model, data)
    dynamics.deepen(model)
    dynamics.widen(model)
    post_change = prediction.forward(model, data)
    assert torch.allclose(pre_change, post_change)


def test_widen_norm_convolutional():
    training.set_seed(42)
    model = models.BatchNormConvolution(in_channels=1, out_features=5)
    data = torch.randn((1, 1, 28, 28))
    pre_widen = prediction.forward(model, data)
    # print(model.norm.running_mean)
    # print(model.norm.running_var)
    # print(model.norm.weight.data)
    # print(model.norm.bias.data)
    dynamics.widen(model)
    # print(model.norm.running_mean)
    # print(model.norm.running_var)
    # print(model.norm.weight.data)
    # print(model.norm.bias.data)
    post_widen = prediction.forward(model, data)
    assert torch.allclose(pre_widen, post_widen)


def test_deepen_norm_convolutional():
    training.set_seed(42)
    model = models.BatchNormConvolution(in_channels=1, out_features=5)
    data = torch.randn((1, 1, 28, 28))
    pre_widen = prediction.forward(model, data)
    dynamics.deepen(model)
    post_widen = prediction.forward(model, data)
    assert torch.allclose(pre_widen, post_widen)


def test_widen_and_deepen_norm_convolutional():
    training.set_seed(42)
    model = models.BatchNormConvolution(in_channels=1, out_features=5)
    data = torch.randn((1, 1, 28, 28))
    pre_change = prediction.forward(model, data)
    dynamics.widen(model)
    dynamics.deepen(model)
    post_change = prediction.forward(model, data)
    assert torch.allclose(pre_change, post_change)


def test_deepen_and_widen_norm_convolutional():
    training.set_seed(42)
    model = models.BatchNormConvolution(in_channels=1, out_features=5)
    data = torch.randn((1, 1, 28, 28))
    pre_change = prediction.forward(model, data)
    dynamics.deepen(model)
    dynamics.widen(model)
    post_change = prediction.forward(model, data)
    assert torch.allclose(pre_change, post_change)


def test_widen_inception():
    training.set_seed(42)
    model = models.cifar10_inception()
    data = torch.randn((1, 3, 75, 75))
    pre_widen = prediction.forward(model, data)
    dynamics.widen(model)
    post_widen = prediction.forward(model, data)
    assert torch.allclose(pre_widen, post_widen)


def test_deepen_inception():
    training.set_seed(42)
    model = models.cifar10_inception()
    data = torch.randn((1, 3, 75, 75))
    pre_widen = prediction.forward(model, data)
    dynamics.deepen(model)
    post_widen = prediction.forward(model, data)
    assert torch.allclose(pre_widen, post_widen)


def test_widen_and_deepen_inception():
    training.set_seed(42)
    model = models.cifar10_inception()
    data = torch.randn((1, 3, 75, 75))
    pre_change = prediction.forward(model, data)
    dynamics.widen(model)
    dynamics.deepen(model)
    post_change = prediction.forward(model, data)
    assert torch.allclose(pre_change, post_change)


def test_deepen_and_widen_inception():
    training.set_seed(42)
    model = models.cifar10_inception()
    data = torch.randn((1, 3, 75, 75))
    pre_change = prediction.forward(model, data)
    dynamics.deepen(model)
    dynamics.widen(model)
    post_change = prediction.forward(model, data)
    assert torch.allclose(pre_change, post_change)


def main():
    # test_mnist()
    # test_cifar10()
    # test_small_feedforward_reproducibility()
    # test_small_convolution_reproducibility()
    # test_widen_feedforward()
    # test_deepen_feedforward()
    # test_deepen_tiny_convolutional()
    # test_deepen_and_widen_tiny_convolutional()
    # test_widen_small_convolutional()
    # test_deepen_small_convolutional()
    # test_widen_and_deepen_small_convolutional()
    # test_deepen_and_widen_small_convolutional()
    # test_widen_norm_convolutional()
    # test_deepen_norm_convolutional()
    # test_widen_and_deepen_norm_convolutional()
    # test_deepen_and_widen_norm_convolutional()
    test_widen_inception()
    test_deepen_inception()
    test_widen_and_deepen_inception()
    test_deepen_and_widen_inception()
    print("ALL TESTS PASSED!")


if __name__ == "__main__":
    main()
