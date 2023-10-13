# NEEDSWORK
# - linear to conv / conv to linear -- we could just not support these two,
# alternative is convert linear to 1x1 convolution
# - only widen all layers with same layer before it (of linear or conv) and
# we duplicate all layers
# - cant get any deepening working for more than 1 output channel

# need support for no bias

import torch.nn as nn
import numpy as np
import tf_and_torch
import device
import torch


class UnsupportedLayer(Exception):
    pass


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

# https://discuss.pytorch.org/t/identity-convolution-weights-for-3-channel-image/155405/3


def _conv_only_deeper(layer):
    weight = tf_and_torch.params_torch_to_tf_ndarr(layer, "weight")

    deeper_w, deeper_b = _conv_only_deeper_tf_numpy(weight)

    new_layer = nn.Conv2d(in_channels=layer.out_channels, out_channels=layer.out_channels,
                          kernel_size=layer.kernel_size, stride=1, padding=(layer.kernel_size[0] // 2, layer.kernel_size[0] // 2))

    tf_and_torch.params_tf_ndarr_to_torch(deeper_w, new_layer, "weight")
    tf_and_torch.params_tf_ndarr_to_torch(deeper_b, new_layer, "bias")

    return new_layer


def _conv_only_wider_tf_numpy(teacher_w1, teacher_b1, teacher_w2, new_width):
    # print("RUNNING")
    rand = np.random.randint(
        teacher_w1.shape[3], size=(new_width - teacher_w1.shape[3])
    )
    replication_factor = np.bincount(rand)
    student_w1 = teacher_w1.copy()
    student_w2 = teacher_w2.copy()
    student_b1 = teacher_b1.copy()
    # target layer update (i)
    for i in range(len(rand)):
        teacher_index = rand[i]
        new_weight = teacher_w1[:, :, :, teacher_index]
        new_weight = new_weight[:, :, :, np.newaxis]
        student_w1 = np.concatenate((student_w1, new_weight), axis=3)
        student_b1 = np.append(student_b1, teacher_b1[teacher_index])
    # next layer update (i+1)
    for i in range(len(rand)):
        teacher_index = rand[i]
        factor = replication_factor[teacher_index] + 1
        assert factor > 1, "Error in Net2Wider"
        new_weight = teacher_w2[:, :, teacher_index, :] * (1.0 / factor)
        new_weight_re = new_weight[:, :, np.newaxis, :]
        student_w2 = np.concatenate((student_w2, new_weight_re), axis=2)
        student_w2[:, :, teacher_index, :] = new_weight
    return student_w1, student_b1, student_w2


# taken straight from abdullah's repo
def _fc_only_wider_tf_numpy(teacher_w1, teacher_b1, teacher_w2, new_width):
    rand = np.random.randint(
        teacher_w1.shape[1], size=(new_width - teacher_w1.shape[1])
    )
    replication_factor = np.bincount(rand)
    student_w1 = teacher_w1.copy()
    student_w2 = teacher_w2.copy()
    student_b1 = teacher_b1.copy()
    # target layer update (i)
    for i in range(len(rand)):
        teacher_index = rand[i]
        new_weight = teacher_w1[:, teacher_index]
        new_weight = new_weight[:, np.newaxis]
        student_w1 = np.concatenate((student_w1, new_weight), axis=1)
        student_b1 = np.append(student_b1, teacher_b1[teacher_index])
    # next layer update (i+1)
    for i in range(len(rand)):
        teacher_index = rand[i]
        factor = replication_factor[teacher_index] + 1
        assert factor > 1, "Error in Net2Wider"
        new_weight = teacher_w2[teacher_index, :] * (1.0 / factor)
        new_weight = new_weight[np.newaxis, :]
        student_w2 = np.concatenate((student_w2, new_weight), axis=0)
        student_w2[teacher_index, :] = new_weight
    return student_w1, student_b1, student_w2


def _fc_only_wider(layer1, layer2, new_width):
    teacher_w1 = tf_and_torch.params_torch_to_tf_ndarr(layer1, "weight")
    teacher_w2 = tf_and_torch.params_torch_to_tf_ndarr(layer2, "weight")
    teacher_b1 = tf_and_torch.params_torch_to_tf_ndarr(layer1, "bias")

    student_w1, student_b1, student_w2 = _fc_only_wider_tf_numpy(
        teacher_w1, teacher_b1, teacher_w2, new_width
    )

    tf_and_torch.params_tf_ndarr_to_torch(student_w1, layer1, "weight")
    tf_and_torch.params_tf_ndarr_to_torch(student_w2, layer2, "weight")
    tf_and_torch.params_tf_ndarr_to_torch(student_b1, layer1, "bias")

    return layer1, layer2, None


def _resize_with_zeros(t, newsize):
    newt = torch.zeros(newsize)
    newt[:t.shape[0]] = t
    return newt


def _resize_with_ones(t, newsize):
    newt = torch.ones(newsize)
    newt[:t.shape[0]] = t
    return newt


def _make_new_norm_layer(layer, new_width):
    if layer is not None:
        layer.running_mean = _resize_with_zeros(layer.running_mean, new_width)
        layer.running_var = _resize_with_ones(layer.running_var, new_width)
        if layer.affine:
            layer.weight = nn.Parameter(
                _resize_with_ones(layer.weight.data, new_width))
            layer.bias = nn.Parameter(
                _resize_with_zeros(layer.bias.data, new_width))
    return layer


def _conv_only_wider(layer1, layer2, norm_layer, new_width):
    teacher_w1 = tf_and_torch.params_torch_to_tf_ndarr(layer1, "weight")
    teacher_w2 = tf_and_torch.params_torch_to_tf_ndarr(layer2, "weight")
    teacher_b1 = tf_and_torch.params_torch_to_tf_ndarr(layer1, "bias")

    # print(teacher_w1.shape)
    # print(teacher_w2.shape)

    student_w1, student_b1, student_w2 = _conv_only_wider_tf_numpy(
        teacher_w1, teacher_b1, teacher_w2, new_width
    )

    # print(student_w1.shape)
    # print(student_w2.shape)

    tf_and_torch.params_tf_ndarr_to_torch(student_w1, layer1, "weight")
    tf_and_torch.params_tf_ndarr_to_torch(student_w2, layer2, "weight")
    tf_and_torch.params_tf_ndarr_to_torch(student_b1, layer1, "bias")

    return layer1, layer2, _make_new_norm_layer(norm_layer, new_width)


def _wider(m1, m2, new_width, batch_norm):
    if isinstance(m1, nn.Linear) and isinstance(m2, nn.Linear):
        return _fc_only_wider(m1, m2, new_width)
    elif isinstance(m1, nn.Conv2d) and isinstance(m2, nn.Conv2d):
        return _conv_only_wider(m1, m2, batch_norm, new_width)
    else:
        raise UnsupportedLayer(f"m1: {type(m1)} m2: {type(m2)}")


def _deeper(m):
    if isinstance(m, nn.Linear):
        new_layer = _fc_only_deeper(m)
    elif isinstance(m, nn.Conv2d):
        new_layer = _conv_only_deeper(m)
    else:
        raise UnsupportedLayer(str(type(m)))
    return nn.Sequential(m, new_layer)


class _LayerTable:
    def _helper(self, hierarchy, name, curr):
        if len(list(curr.children())) == 0:
            self.table.append((hierarchy, name))
        for n, child in curr.named_children():
            self._helper(hierarchy + [curr], n, child)

    def __init__(self, model):
        self.table = list()
        self._helper([], None, model)

    def __iter__(self):
        yield from self.table

    def get(self, hierarchy, name):
        parent = hierarchy[-1]
        return (
            parent[int(name)]
            if isinstance(parent, nn.Sequential)
            else getattr(parent, name)
        )

    def set(self, hierarchy, name, value):
        parent = hierarchy[-1]
        if isinstance(parent, nn.Sequential):
            parent[int(name)] = value
        else:
            setattr(parent, name, value)


def _is_conv(layer):
    return "Conv2d" in layer.__class__.__name__


def _is_linear(layer):
    return "Linear" in layer.__class__.__name__


def _is_batchnorm(layer):
    return "BatchNorm" in layer.__class__.__name__


def _get_out_size(layer):
    return layer.out_features if isinstance(layer, nn.Linear) else layer.out_channels


def widen(model, modifier=lambda h, l: 1.5):
    # print(model)
    table = _LayerTable(model)
    prev = None
    between_batchnorm = None
    for p, n in table:
        curr = table.get(p, n)
        if _is_batchnorm(curr):
            between_batchnorm = (p, n)
        elif _is_conv(curr) or _is_linear(curr):
            if prev is not None:
                old_layer1 = table.get(*prev)
                old_layer2 = curr
                if type(old_layer1) == type(old_layer2) and modifier(p, n) > 1:
                    old_batchnorm = (
                        table.get(*between_batchnorm)
                        if between_batchnorm is not None
                        else None
                    )
                    new_out_size = round(
                        modifier(p, n) * _get_out_size(old_layer1))
                    new_layer1, new_layer2, new_batchnorm = _wider(
                        old_layer1, old_layer2, new_out_size, old_batchnorm
                    )
                    table.set(*prev, new_layer1)
                    table.set(p, n, new_layer2)
                    if between_batchnorm is not None:
                        table.set(*between_batchnorm, new_batchnorm)
            prev = (p, n)
            between_batchnorm = None
    # print(model)


def deepen(model):
    table = _LayerTable(model)
    for p, n in table:
        curr = table.get(p, n)
        if _is_conv(curr) or _is_linear(curr):
            table.set(p, n, _deeper(curr))
