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
import copy


def deepenwiden(model):
    deepening.deepen(model)
    widening.widen(model)


def widendeepen(model):
    widening.widen(model)
    deepening.deepen(model)


def checkequal(input, other, rtol=1e-5, atol=2e-4):
    if isinstance(input, torchvision.models.inception.InceptionOutputs):
        input = torch.cat([input.logits, input.aux_logits])
        other = torch.cat([other.logits, other.aux_logits])
    if not torch.allclose(input, other, rtol, atol):
        errmatrix = torch.abs(torch.sub(input, other))
        raise AssertionError(
            f"MAX AE: {torch.max(errmatrix)} MEAN AE: {torch.mean(errmatrix)}"
        )


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
    **adaptation_kwargs,
):
    config.set_seed(42)
    data = data_func(*data_args)
    dataloader = [(data, None)]
    model = model_func(**model_kwargs)
    deepening.update_batchnorm_statistics(dataloader, model, device.get_device())
    adapted_model = copy.deepcopy(model)
    adaptation_func(
        adapted_model,
        dataloader,
        device.get_device(),
        **adaptation_kwargs,
    )
    eval_ = adaptation_kwargs.get("eval", True)
    device.move(device.get_device(), data, model, adapted_model)
    pre_num = models.count_parameters(model)
    pre_mod = prediction.forward(model, data, eval=eval_)
    post_mod = prediction.forward(adapted_model, data, eval=eval_)
    post_num = models.count_parameters(adapted_model)
    checkequal(pre_mod, post_mod)
    assert post_num > pre_num
