"""
NEEDSWORK document
"""

import device
import numpy as np
import torch.nn as nn
import tf_and_torch
import tracing


def _fc_only_deeper_tf_numpy(weight):
    deeper_w = np.eye(weight.shape[1])
    deeper_b = np.zeros(weight.shape[1])
    return deeper_w, deeper_b


def _fc_only_deeper(layer):
    weight = tf_and_torch.params_torch_to_tf_ndarr(layer, "weight")

    new_layer_w, new_layer_b = _fc_only_deeper_tf_numpy(weight)

    new_layer = nn.Linear(1, 1).to(device.get_device())
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


def deepen_block(block, model, dataloader, device, **kwargs):
    add_batch_norm = kwargs.get("add_batch_norm", True)
    first_layer = block.layers[0]
    if isinstance(first_layer, nn.Conv2d):
        new_conv_layer = _conv_only_deeper(first_layer)
        if add_batch_norm:
            # new_batch_norm_layer = nn.BatchNorm2d(first_layer.out_channels)
            block.layers = nn.Sequential(
                *block.layers,
                new_conv_layer,
                nn.BatchNorm2d(first_layer.out_channels),
                nn.ReLU(),
            )
            # _update_batchnorm_weights(dataloader, model, device, new_batch_norm_layer)
            # new_batch_norm_layer.weight = nn.Parameter(
        #     torch.sqrt(new_batch_norm_layer.running_var + new_batch_norm_layer.eps)
        # )
        # new_batch_norm_layer.bias = nn.Parameter(new_batch_norm_layer.running_mean)

        else:
            block.layers = nn.Sequential(
                *block.layers,
                new_conv_layer,
                nn.ReLU(),
            )
    elif isinstance(first_layer, nn.Linear):
        block.layers = nn.Sequential(
            *block.layers,
            _fc_only_deeper(first_layer),
            nn.ReLU(),
        )
    else:
        raise tracing.UnsupportedLayer(str(type(first_layer)))


def random_deepen_block(block):
    first_layer = block.layers[0]
    if isinstance(first_layer, nn.Conv2d):
        new_conv_layer = nn.Conv2d(
            in_channels=first_layer.out_channels,
            out_channels=first_layer.out_channels,
            kernel_size=first_layer.kernel_size,
            stride=1,
            padding=(first_layer.kernel_size[0] // 2, first_layer.kernel_size[1] // 2),
        )
        new_batch_norm_layer = nn.BatchNorm2d(first_layer.out_channels)
        block.layers = nn.Sequential(
            *block.layers,
            new_conv_layer,
            new_batch_norm_layer,
            nn.ReLU(),
        )
    elif isinstance(first_layer, nn.Linear):
        block.layers = nn.Sequential(
            *block.layers,
            nn.Linear(first_layer.out_features, first_layer.out_features),
            nn.ReLU(),
        )
    else:
        raise tracing.UnsupportedLayer(str(type(first_layer)))


def deepen_blocks(model, dataloader, device, **kwargs):
    filter_function = kwargs.get("filter_function", lambda b, h: True)
    for block, hierarchy in tracing.get_all_deepen_blocks(model):
        if filter_function(block, hierarchy):
            deepen_block(block, model, dataloader, device, **kwargs)


def random_deepen_blocks(model, dataloader, device, **kwargs):
    filter_function = kwargs.get("filter_function", lambda b, h: True)
    for block, hierarchy in tracing.get_all_deepen_blocks(model):
        if filter_function(block, hierarchy):
            random_deepen_block(block, None, None, None, **kwargs)


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
