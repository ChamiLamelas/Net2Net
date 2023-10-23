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

    # import torch
    # x = torch.randn((1, layer.out_channels, 28, 28))
    # y = new_layer(x)
    # print(torch.max(torch.abs(x - y)).item())

    return new_layer


def deeper(m):
    if isinstance(m, nn.Linear):
        new_layer = _fc_only_deeper(m)
    elif isinstance(m, nn.Conv2d):
        new_layer = _conv_only_deeper(m)
    else:
        raise tracing.UnsupportedLayer(str(type(m)))
    return nn.Sequential(m, new_layer)


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
    return nn.Sequential(m, new_layer)


def deepen(model, ignore=set(), modifier=lambda _: True):
    table = tracing.LayerTable(model, ignore)
    for e in table:
        curr = table.get(e["hierarchy"], e["name"]) 
        if (isinstance(curr, nn.Conv2d) or isinstance(curr, nn.Linear)) and modifier(e):
            # print(deeper(curr), type(e["hierarchy"][-2]).__name__)
            table.set(e["hierarchy"], e["name"], deeper(curr))


def random_deepen(model, ignore=set(), modifier=lambda _: True):
    table = tracing.LayerTable(model, ignore)
    for e in table:
        curr = table.get(e["hierarchy"], e["name"])
        if (isinstance(curr, nn.Conv2d) or isinstance(curr, nn.Linear)) and modifier(e):
            table.set(e["hierarchy"], e["name"], random_deeper(curr))
