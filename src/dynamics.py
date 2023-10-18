"""
Notes: 

- This does not support linear to conv or conv to linear widening, one way 
to potentially solve this issue is to convert a linear layer to an
identical 1x1 convolutional layer (https://datascience.stackexchange.com/a/12833).

- Widening linear-linear, convolutional-convolutional and deepening linear layers
are taken from here: https://github.com/paengs/Net2Net

BUGS: 

- We are unable to handle nonsequential layer execution order, e.g. here's 
a forward function of a InceptionNet block. Note branch3x3 isn't passed
into following layers but is concatenated at the end. Hence, we can't 
arbitrarily widen all layers in a network, we need to have some knowledge
of which layers we can widen -- not sure how to determine this automatically.

def _forward(self, x: Tensor) -> List[Tensor]:
    branch3x3 = self.branch3x3(x)

    branch3x3dbl = self.branch3x3dbl_1(x)
    branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
    branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)

    branch_pool = F.max_pool2d(x, kernel_size=3, stride=2)

    outputs = [branch3x3, branch3x3dbl, branch_pool]
    return outputs

NEEDSWORK Document
"""

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


def _conv_only_deeper(layer):
    weight = tf_and_torch.params_torch_to_tf_ndarr(layer, "weight")

    deeper_w, deeper_b = _conv_only_deeper_tf_numpy(weight)

    new_layer = nn.Conv2d(
        in_channels=layer.out_channels,
        out_channels=layer.out_channels,
        kernel_size=layer.kernel_size,
        stride=1,
        padding=(layer.kernel_size[0] // 2, layer.kernel_size[0] // 2),
    )

    tf_and_torch.params_tf_ndarr_to_torch(deeper_w, new_layer, "weight")
    tf_and_torch.params_tf_ndarr_to_torch(deeper_b, new_layer, "bias")

    return new_layer


def _conv_only_wider_tf_numpy(teacher_w1, teacher_w2, new_width, teacher_b1):
    # print(teacher_w1.shape, teacher_w2.shape)
    rand = np.random.randint(
        teacher_w1.shape[3], size=(new_width - teacher_w1.shape[3])
    )
    replication_factor = np.bincount(rand)
    student_w1 = teacher_w1.copy()
    student_w2 = teacher_w2.copy()
    if teacher_b1 is not None:
        student_b1 = teacher_b1.copy()
    else:
        student_b1 = None 
    # target layer update (i)
    for i in range(len(rand)):
        teacher_index = rand[i]
        new_weight = teacher_w1[:, :, :, teacher_index]
        new_weight = new_weight[:, :, :, np.newaxis]
        student_w1 = np.concatenate((student_w1, new_weight), axis=3)
        if teacher_b1 is not None:
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


def _fc_only_wider_tf_numpy(teacher_w1, teacher_w2, new_width, teacher_b1):
    rand = np.random.randint(
        teacher_w1.shape[1], size=(new_width - teacher_w1.shape[1])
    )
    replication_factor = np.bincount(rand)
    student_w1 = teacher_w1.copy()
    student_w2 = teacher_w2.copy()
    if teacher_b1 is not None:
        student_b1 = teacher_b1.copy()
    else:
        student_b1 = None
    # target layer update (i)
    for i in range(len(rand)):
        teacher_index = rand[i]
        new_weight = teacher_w1[:, teacher_index]
        new_weight = new_weight[:, np.newaxis]
        student_w1 = np.concatenate((student_w1, new_weight), axis=1)
        if teacher_b1 is not None:
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
    if layer1.bias is not None:
        teacher_b1 = tf_and_torch.params_torch_to_tf_ndarr(layer1, "bias")
    else:
        teacher_b1 = None

    student_w1, student_b1, student_w2 = _fc_only_wider_tf_numpy(
        teacher_w1, teacher_w2, new_width, teacher_b1
    )

    tf_and_torch.params_tf_ndarr_to_torch(student_w1, layer1, "weight")
    tf_and_torch.params_tf_ndarr_to_torch(student_w2, layer2, "weight")
    if layer1.bias is not None:
        tf_and_torch.params_tf_ndarr_to_torch(student_b1, layer1, "bias")

    return layer1, layer2, None


def _resize_with_zeros(t, newsize):
    newt = torch.zeros(newsize)
    newt[: t.shape[0]] = t
    return newt


def _resize_with_ones(t, newsize):
    newt = torch.ones(newsize)
    newt[: t.shape[0]] = t
    return newt


def _make_new_norm_layer(layer, new_width):
    if layer is not None:
        layer.running_mean = _resize_with_zeros(layer.running_mean, new_width)
        layer.running_var = _resize_with_ones(layer.running_var, new_width)
        if layer.affine:
            layer.weight = nn.Parameter(_resize_with_ones(layer.weight.data, new_width))
            layer.bias = nn.Parameter(_resize_with_zeros(layer.bias.data, new_width))
    return layer


def _conv_only_wider(layer1, layer2, norm_layer, new_width):
    teacher_w1 = tf_and_torch.params_torch_to_tf_ndarr(layer1, "weight")
    teacher_w2 = tf_and_torch.params_torch_to_tf_ndarr(layer2, "weight")
    if layer1.bias is not None:
        teacher_b1 = tf_and_torch.params_torch_to_tf_ndarr(layer1, "bias")
    else:
        teacher_b1 = None

    student_w1, student_b1, student_w2 = _conv_only_wider_tf_numpy(
        teacher_w1, teacher_w2, new_width, teacher_b1
    )

    tf_and_torch.params_tf_ndarr_to_torch(student_w1, layer1, "weight")
    tf_and_torch.params_tf_ndarr_to_torch(student_w2, layer2, "weight")
    if layer1.bias is not None:
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


class LayerTable:
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


def widen(model, modifier=lambda h, l: 1.5):
    table = LayerTable(model)
    prev = None
    between_batchnorm = None
    for p, n in table:
        curr = table.get(p, n)
        if isinstance(curr, nn.BatchNorm2d):
            between_batchnorm = (p, n)
        elif isinstance(curr, nn.Conv2d) or isinstance(curr, nn.Linear):
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
                        modifier(p, n)
                        * (
                            old_layer1.out_features
                            if isinstance(old_layer1, nn.Linear)
                            else old_layer1.out_channels
                        )
                    )
                    new_layer1, new_layer2, new_batchnorm = _wider(
                        old_layer1, old_layer2, new_out_size, old_batchnorm
                    )
                    table.set(*prev, new_layer1)
                    table.set(p, n, new_layer2)
                    if between_batchnorm is not None:
                        table.set(*between_batchnorm, new_batchnorm)
            prev = (p, n)
            between_batchnorm = None


def deepen(model, modifier=lambda h, n: True):
    table = LayerTable(model)
    for p, n in table:
        curr = table.get(p, n)
        if (isinstance(curr, nn.Conv2d) or isinstance(curr, nn.Linear)) and modifier(
            p, n
        ):
            table.set(p, n, _deeper(curr))
