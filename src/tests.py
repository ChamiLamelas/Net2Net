"""
NEEDSWORK document
"""

import models
import torch
import training
import data
import dynamics
import prediction


def asserts(actuals, expects):
    fails = list()
    for a, e in zip(actuals, expects):
        if a != e:
            fails.append((a, e))
    if len(fails) > 0:
        raise AssertionError(str(fails))


def test_layer_table_smallfeedforward():
    model = models.SmallFeedForward(4, 5)
    table = dynamics.LayerTable(model)
    itr = iter(table)
    e = next(itr)
    asserts([e["name"], e["prevname"], e["hierarchy"],
            e["prevhierarchy"]], ["fc1", None, [model], None])
    e = next(itr)
    asserts([e["name"], e["prevname"], e["hierarchy"],
            e["prevhierarchy"]], ["act", None, [model], None])
    e = next(itr)
    asserts([e["name"], e["prevname"], e["hierarchy"],
            e["prevhierarchy"]], ["fc2", "fc1", [model], [model]])


def test_layer_table_blockedmodel():
    model = models.BlockedModel(4, 5)
    table = dynamics.LayerTable(model)
    itr = iter(table)
    e = next(itr)
    asserts([e["name"], e["prevname"], e["hierarchy"],
            e["prevhierarchy"]], ["fc1", None, [model, model.block1], None])
    e = next(itr)
    asserts([e["name"], e["prevname"], e["hierarchy"],
            e["prevhierarchy"]], ["act", None, [model, model.block1], None])
    e = next(itr)
    asserts([e["name"], e["prevname"], e["hierarchy"],
            e["prevhierarchy"]], ["fc2", "fc1", [model, model.block1], [model, model.block1]])
    e = next(itr)
    asserts([e["name"], e["prevname"], e["hierarchy"],
            e["prevhierarchy"]], ["fc1", None, [model, model.block2], None])
    e = next(itr)
    asserts([e["name"], e["prevname"], e["hierarchy"],
            e["prevhierarchy"]], ["act", None, [model, model.block2], None])
    e = next(itr)
    asserts([e["name"], e["prevname"], e["hierarchy"],
            e["prevhierarchy"]], ["fc2", "fc1", [model, model.block2], [model, model.block2]])


def test_layer_table_sequentialmodel():
    model = models.SequentialModel(4, 5)
    table = dynamics.LayerTable(model)
    itr = iter(table)
    e = next(itr)
    asserts([e["name"], e["prevname"], e["hierarchy"],
            e["prevhierarchy"]], ["0", None, [model, model.seq], None])
    e = next(itr)
    asserts([e["name"], e["prevname"], e["hierarchy"],
            e["prevhierarchy"]], ["1", None, [model, model.seq], None])
    e = next(itr)
    asserts([e["name"], e["prevname"], e["hierarchy"],
            e["prevhierarchy"]], ["2", "0", [model, model.seq], [model, model.seq]])


def test_layer_table_nonsequentialconvolution():
    model = models.NonSequentialConvolution(3, 5)
    table = dynamics.LayerTable(model)
    itr = iter(table)
    e = next(itr)
    asserts([e["name"], e["prevname"], e["hierarchy"],
            e["prevhierarchy"]], ["conv1", None, [model], None])
    e = next(itr)
    asserts([e["name"], e["prevname"], e["hierarchy"],
            e["prevhierarchy"]], ["conv2", None, [model], None])
    e = next(itr)
    asserts([e["name"], e["prevname"], e["hierarchy"],
            e["prevhierarchy"]], ["conv3", "conv2", [model], [model]])
    e = next(itr)
    asserts([e["name"], e["prevname"], e["hierarchy"],
            e["prevhierarchy"]], ["finalpool", None, [model], None])
    e = next(itr)
    asserts([e["name"], e["prevname"], e["hierarchy"],
            e["prevhierarchy"]], ["fc", None, [model], None])


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


def check_adaptation(
    model_func,
    model_kwargs,
    data_func,
    data_args,
    adaptation_func,
    adaptation_ignore=None,
    adaptation_modifier=None,
):
    training.set_seed(42)
    data = data_func(*data_args)
    model = model_func(**model_kwargs)
    pre_widen = prediction.forward(model, data)
    if adaptation_func == dynamics.widen:
        pre_num = models.count_parameters(model)
    else:
        pre_num = models.num_conv_layers(
            model) + models.num_linear_layers(model)
    if adaptation_modifier is None:
        adaptation_func(model)
    else:
        adaptation_func(model, adaptation_ignore, adaptation_modifier)
    post_widen = prediction.forward(model, data)
    if adaptation_func == dynamics.widen:
        post_num = models.count_parameters(model)
    else:
        post_num = models.num_conv_layers(
            model) + models.num_linear_layers(model)
    assert torch.allclose(pre_widen, post_widen, atol=1e-5)
    assert post_num > pre_num


def deepenwiden(
    model, deepenmodifier=lambda e: True, widenmodifier=lambda e: 1.5
):
    dynamics.deepen(model, deepenmodifier)
    dynamics.widen(model, widenmodifier)


def widendeepen(
    model, widenmodifier=lambda e: 1.5, deepenmodifier=lambda e: True
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


def test_widen_nonsequential():
    check_adaptation(
        models.NonSequentialConvolution,
        {"in_channels": 3, "out_features": 5},
        torch.randn,
        ((1, 3, 75, 75)),
        dynamics.widen,
    )


def test_widen_inception():
    check_adaptation(
        models.imagenet_inception,
        {},
        torch.randn,
        ((1, 3, 75, 75)),
        dynamics.widen,
        models.inception_ignoreset(),
        models.widen_inception
    )


def test_deepen_inception():
    check_adaptation(
        models.imagenet_inception,
        {},
        torch.randn,
        ((1, 3, 75, 75)),
        dynamics.deepen,
        models.deepen_inception
    )


def main():
    # test_layer_table_smallfeedforward()
    # test_layer_table_blockedmodel()
    # test_layer_table_sequentialmodel()
    # test_layer_table_nonsequentialconvolution()
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
    # test_widen_nonsequential()
    test_widen_inception()
    test_deepen_inception()
    print("ALL TESTS PASSED!")


if __name__ == "__main__":
    main()
