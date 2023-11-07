import sys
import os
sys.path.append(os.path.join("..", "src"))

import device
import torch
import torchvision
import prediction
import config
import widening
import models
import deepening


def deepenwiden(model):
    deepening.deepen(model)
    widening.widen(model)


def widendeepen(model):
    widening.widen(model)
    deepening.deepen(model)


def checkequal(input, other, rtol=1e-5, atol=2e-2):
    if isinstance(input, torchvision.models.inception.InceptionOutputs):
        input = torch.cat([input.logits, input.aux_logits])
        other = torch.cat([other.logits, other.aux_logits])
    if not torch.allclose(input, other, rtol, atol):
        errmatrix = torch.abs(torch.sub(input, other))
        raise AssertionError(
            f"MAX AE: {torch.max(errmatrix)} MEAN AE: {torch.mean(errmatrix)}")


def asserts(actuals, expects):
    fails = list()
    for a, e in zip(actuals, expects):
        if a != e:
            fails.append((a, e))
    if len(fails) > 0:
        raise AssertionError(str(fails))


def check_adaptation(
    model_func,
    model_kwargs,
    data_func,
    data_args,
    adaptation_func,
    adaptation_ignore=None,
    adaptation_modifier=None,
):
    config.set_seed(42)
    data = data_func(*data_args)
    model = model_func(**model_kwargs)
    device.move(device.get_device(), data, model)
    pre_mod = prediction.forward(model, data)
    if adaptation_func == widening.widen:
        pre_num = models.count_parameters(model)
    else:
        pre_num = models.num_conv_layers(
            model) + models.num_linear_layers(model)
    if adaptation_modifier is None:
        adaptation_func(model)
    else:
        adaptation_func(model, adaptation_ignore, adaptation_modifier)
    post_mod = prediction.forward(model, data)
    if adaptation_func == widening.widen:
        post_num = models.count_parameters(model)
    else:
        post_num = models.num_conv_layers(
            model) + models.num_linear_layers(model)
    checkequal(pre_mod, post_mod)
    assert post_num > pre_num
