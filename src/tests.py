"""
NEEDSWORK document
"""

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
    expected = torch.tensor([[0.3137, -0.1701, 0.2066], [0.3137, -0.1701, 0.2066]])
    assert torch.allclose(output_t, expected, atol=1e-4)


def test_small_convolution_reproducibility():
    training.set_seed(42)
    model = models.TwoConvolution(1, 3)
    input_t = torch.ones((2, 1, 16, 16))
    output_t = model(input_t)
    expected = torch.tensor([[0.0454, -0.1450, 0.0823], [0.0454, -0.1450, 0.0823]])
    assert torch.allclose(output_t, expected, atol=1e-4)


def check_adaptation(
    model_func,
    model_kwargs,
    data_func,
    data_args,
    adaptation_func,
    adaptation_modifier=None,
):
    training.set_seed(42)
    data = data_func(*data_args)
    model = model_func(**model_kwargs)
    pre_widen = prediction.forward(model, data)
    if adaptation_func == dynamics.widen:
        pre_num = models.count_parameters(model)
    else:
        pre_num = models.num_conv_layers(model) + models.num_linear_layers(model)
    if adaptation_modifier is None:
        adaptation_func(model)
    else:
        adaptation_func(model, adaptation_modifier)
    post_widen = prediction.forward(model, data)
    if adaptation_func == dynamics.widen:
        post_num = models.count_parameters(model)
    else:
        post_num = models.num_conv_layers(model) + models.num_linear_layers(model)
    assert torch.allclose(pre_widen, post_widen, atol=1e-5)
    assert post_num > pre_num


def deepenwiden(
    model, deepenmodifier=lambda h, n: True, widenmodifier=lambda h, n: 1.5
):
    dynamics.deepen(model, deepenmodifier)
    dynamics.widen(model, widenmodifier)


def widendeepen(
    model, widenmodifier=lambda h, n: 1.5, deepenmodifier=lambda h, n: True
):
    dynamics.widen(model, widenmodifier)
    dynamics.deepen(model, deepenmodifier)


def test_widen_feedforward():
    check_adaptation(
        models.SmallFeedForward,
        {"in_features": 4, "out_features": 5},
        torch.randn,
        ((1, 4)),
        dynamics.widen,
    )


def test_deepen_feedforward():
    check_adaptation(
        models.SmallFeedForward,
        {"in_features": 4, "out_features": 5},
        torch.randn,
        ((1, 4)),
        dynamics.deepen,
    )


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


def test_deepen_tiny_convolutional():
    check_adaptation(
        models.OneConvolution,
        {"in_channels": 1, "out_features": 5},
        torch.randn,
        ((1, 1, 28, 28)),
        dynamics.deepen,
    )


def test_deepen_and_widen_tiny_convolutional():
    check_adaptation(
        models.OneConvolution,
        {"in_channels": 1, "out_features": 5},
        torch.randn,
        ((1, 1, 28, 28)),
        deepenwiden,
    )


def test_widen_small_convolutional():
    check_adaptation(
        models.TwoConvolution,
        {"in_channels": 1, "out_features": 5},
        torch.randn,
        ((1, 1, 28, 28)),
        dynamics.widen,
    )


def test_deepen_small_convolutional():
    check_adaptation(
        models.TwoConvolution,
        {"in_channels": 1, "out_features": 5},
        torch.randn,
        ((1, 1, 28, 28)),
        dynamics.deepen,
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


def test_widen_norm_convolutional():
    check_adaptation(
        models.BatchNormConvolution,
        {"in_channels": 1, "out_features": 5},
        torch.randn,
        ((1, 1, 28, 28)),
        dynamics.widen,
    )


def test_deepen_norm_convolutional():
    check_adaptation(
        models.BatchNormConvolution,
        {"in_channels": 1, "out_features": 5},
        torch.randn,
        ((1, 1, 28, 28)),
        dynamics.deepen,
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


def test_widen_inception():
    print(models.imagenet_inception())
    check_adaptation(
        models.imagenet_inception,
        {},
        torch.randn,
        ((1, 3, 75, 75)),
        dynamics.widen,
    )


def test_deepen_inception():
    check_adaptation(
        models.imagenet_inception,
        {},
        torch.randn,
        ((1, 3, 75, 75)),
        dynamics.deepen,
    )


def test_widen_and_deepen_inception():
    check_adaptation(
        models.imagenet_inception,
        {},
        torch.randn,
        ((1, 3, 75, 75)),
        widendeepen,
    )


def test_deepen_and_widen_inception():
    check_adaptation(
        models.imagenet_inception,
        {},
        torch.randn,
        ((1, 3, 75, 75)),
        deepenwiden,
    )


def test_widen_resnet18():
    check_adaptation(
        models.cifar10_resnet18,
        {},
        torch.randn,
        ((1, 3, 75, 75)),
        dynamics.widen,
    )


def test_deepen_resnet18():
    check_adaptation(
        models.cifar10_resnet18,
        {},
        torch.randn,
        ((1, 3, 75, 75)),
        dynamics.deepen,
    )


def test_widen_and_deepen_resnet18():
    check_adaptation(
        models.cifar10_resnet18,
        {},
        torch.randn,
        ((1, 3, 75, 75)),
        widendeepen,
    )


def test_deepen_and_widen_resnet18():
    check_adaptation(
        models.cifar10_resnet18,
        {},
        torch.randn,
        ((1, 3, 75, 75)),
        deepenwiden,
    )


# def test_widen_inception_net2net():
#     training.set_seed(42)
#     model = models.imagenet_inception()
#     data = torch.randn((1, 3, 75, 75))
#     pre_change = prediction.forward(model, data)
#     dynamics.widen(model, )
#     post_change = prediction.forward(model, data)
#     assert torch.allclose(pre_change, post_change)


def main():
    # test_mnist()
    # test_cifar10()
    # test_small_feedforward_reproducibility()
    # test_small_convolution_reproducibility()
    # test_widen_feedforward()
    # test_deepen_feedforward()
    # test_widen_and_deepen_feedforward()
    # test_deepen_and_widen_feedforward()
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
    test_widen_resnet18()
    test_deepen_resnet18()
    test_widen_and_deepen_resnet18()
    test_deepen_and_widen_resnet18()
    print("ALL TESTS PASSED!")


if __name__ == "__main__":
    main()
