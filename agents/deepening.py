"""
NEEDSWORK document
"""

import gpu
import numpy as np
import torch.nn as nn
import tf_and_torch
import tracing
import copy


def _fc_only_deeper_tf_numpy(weight):
    deeper_w = np.eye(weight.shape[1])
    deeper_b = np.zeros(weight.shape[1])
    return deeper_w, deeper_b


def _fc_only_deeper(layer):
    weight = tf_and_torch.params_torch_to_tf_ndarr(layer, "weight")

    new_layer_w, new_layer_b = _fc_only_deeper_tf_numpy(weight)

    new_layer = nn.Linear(1, 1).to(gpu.get_device())
    tf_and_torch.params_tf_ndarr_to_torch(new_layer_w, new_layer, "weight")
    tf_and_torch.params_tf_ndarr_to_torch(new_layer_b, new_layer, "bias")

    return new_layer


def _conv_only_deeper_tf_numpy(weight):
    deeper_w = np.zeros(
        (weight.shape[0], weight.shape[1], weight.shape[3], weight.shape[3])
    )
    assert (
        weight.shape[0] % 2 == 1 and weight.shape[1] % 2 == 1
    ), "Kernel size should be odd"
    center_h = (weight.shape[0] - 1) // 2
    center_w = (weight.shape[1] - 1) // 2
    for i in range(weight.shape[3]):
        tmp = np.zeros((weight.shape[0], weight.shape[1], weight.shape[3]))
        tmp[center_h, center_w, i] = 1
        deeper_w[:, :, :, i] = tmp
    deeper_b = np.zeros(weight.shape[3])
    return deeper_w, deeper_b


def _conv_only_deeper(layer):
    weight = tf_and_torch.params_torch_to_tf_ndarr(layer, "weight")

    deeper_w, deeper_b = _conv_only_deeper_tf_numpy(weight)

    new_layer = nn.Conv2d(
        in_channels=layer.out_channels,
        out_channels=layer.out_channels,
        kernel_size=layer.kernel_size,
        stride=1,
        padding=(layer.kernel_size[0] // 2, layer.kernel_size[1] // 2),
    )

    tf_and_torch.params_tf_ndarr_to_torch(deeper_w, new_layer, "weight")
    tf_and_torch.params_tf_ndarr_to_torch(deeper_b, new_layer, "bias")

    return new_layer


def update_batchnorm_statistics(dataloader, model, device):
    model = model.to(device)
    model.train()
    for entry in dataloader:
        model(entry[0].to(device))


def deepen_block(parent, name, block, add_batch_norm):
    first_layer = block.layers[0]
    new_block = copy.deepcopy(block)
    new_block.layers = nn.Sequential()
    if isinstance(first_layer, nn.Conv2d):
        new_block.layers.append(_conv_only_deeper(first_layer))
        if add_batch_norm:
            new_block.layers.append(nn.BatchNorm2d(first_layer.out_channels))
        new_block.layers.append(nn.ReLU())
    elif isinstance(first_layer, nn.Linear):
        new_block.layers.append(_fc_only_deeper(first_layer))
        new_block.layers.append(nn.ReLU())
    else:
        raise tracing.UnsupportedLayer(str(type(first_layer)))
    setattr(parent, name, nn.Sequential(block, new_block))


def random_deepen_block(parent, name, block, add_batch_norm):
    first_layer = block.layers[0]
    new_block = copy.deepcopy(block)
    new_block.layers = nn.Sequential()
    if isinstance(first_layer, nn.Conv2d):
        new_block.layers.append(
            nn.Conv2d(
                in_channels=first_layer.out_channels,
                out_channels=first_layer.out_channels,
                kernel_size=first_layer.kernel_size,
                stride=1,
                padding=(
                    first_layer.kernel_size[0] // 2,
                    first_layer.kernel_size[1] // 2,
                ),
            )
        )
        if add_batch_norm:
            new_block.layers.append(nn.BatchNorm2d(first_layer.out_channels))
        new_block.layers.append(nn.ReLU())
    elif isinstance(first_layer, nn.Linear):
        new_block.layers.append(
            nn.Linear(first_layer.out_features, first_layer.out_features)
        )
        new_block.layers.append(nn.ReLU())
    else:
        raise tracing.UnsupportedLayer(str(type(first_layer)))
    setattr(parent, name, nn.Sequential(block, new_block))


def deepen_blocks(model, filter_function=None, add_batch_norm=None):
    if filter_function is None:
        filter_function = lambda b, h: True
    if add_batch_norm is None:
        add_batch_norm = True
    for hierarchy, name in tracing.get_all_deepen_blocks(model):
        block = getattr(hierarchy[-1], name)
        if filter_function(block, hierarchy):
            deepen_block(hierarchy[-1], name, block, add_batch_norm)


def random_deepen_blocks(model, filter_function=None, add_batch_norm=None):
    if filter_function is None:
        filter_function = lambda b, h: True
    if add_batch_norm is None:
        add_batch_norm = True
    for hierarchy, name in tracing.get_all_deepen_blocks(model):
        block = getattr(hierarchy[-1], name)
        if filter_function(block, hierarchy):
            deepen_block(hierarchy[-1], name, block, add_batch_norm)


# older deepening stuff ..

"""
def deeper(m):
    if isinstance(m, nn.Linear):
        new_layer = _fc_only_deeper(m)
    elif isinstance(m, nn.Conv2d):
        new_layer = _conv_only_deeper(m)
    else:
        raise tracing.UnsupportedLayer(str(type(m)))
    return nn.Sequential(m, nn.ReLU(), new_layer)


def random_deeper(m):
    if isinstance(m, nn.Linear):
        new_layer = nn.Linear(m.out_features, m.out_features)
    elif isinstance(m, nn.Conv2d):
        new_layer = nn.Conv2d(
            in_channels=m.out_channels,
            out_channels=m.out_channels,
            kernel_size=m.kernel_size,
            stride=1,
            padding=(m.kernel_size[0] // 2, m.kernel_size[1] // 2),
        )
    else:
        raise tracing.UnsupportedLayer(str(type(m)))
    return nn.Sequential(m, nn.ReLU(), new_layer)


def deepen(model, ignore=set(), modifier=lambda x, y: True):
    table = tracing.LayerTable(model, ignore)
    for e in table:
        curr = table.get(e["hierarchy"])
        if (isinstance(curr, nn.Conv2d) or isinstance(curr, nn.Linear)) and modifier(
            e, curr
        ):
            table.set(e["hierarchy"], deeper(curr))


def random_deepen(model, ignore=set(), modifier=lambda x, y: True):
    table = tracing.LayerTable(model, ignore)
    for e in table:
        curr = table.get(e["hierarchy"])
        if (isinstance(curr, nn.Conv2d) or isinstance(curr, nn.Linear)) and modifier(
            e, curr
        ):
            table.set(e["hierarchy"], random_deeper(curr))
"""
