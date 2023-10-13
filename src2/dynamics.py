"""
Notes: 

- This does not support linear to conv or conv to linear widening, one way 
to potentially solve this issue is to convert a linear layer to an
identical 1x1 convolutional layer (https://datascience.stackexchange.com/a/12833).

- Widening linear-linear, convolutional-convolutional and deepening linear layers
are taken from here: https://github.com/paengs/Net2Net

- Widening batchnorm layers is taken from here: https://github.com/erogol/Net2Net

- Deepening convolutional layers (i.e. adding an identity kernel as noted in
original paper) is taken from here: 
https://discuss.pytorch.org/t/identity-convolution-weights-for-3-channel-image/155405/3

NEEDSWORK Document
"""

import torch.nn as nn
import numpy as np
import tf_and_torch
import device
import torch


#####################################################################################
############################## paengs code ##########################################
#####################################################################################

def _fc_only_deeper_tf_numpy(weight):
    deeper_w = np.eye(weight.shape[1])
    deeper_b = np.zeros(weight.shape[1])
    return deeper_w, deeper_b


def _conv_only_wider_tf_numpy(teacher_w1, teacher_b1, teacher_w2, new_width):
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

#####################################################################################
############################## erogol code ##########################################
#####################################################################################


def _make_new_norm_layer(layer, new_width):
    if layer is not None:
        layer.running_mean = layer.running_mean.clone().resize_(new_width)
        layer.running_var = layer.running_var.clone().resize_(new_width)
        if layer.affine:
            layer.weight = nn.Parameter(
                layer.weight.data.clone().resize_(new_width))
            layer.bias = nn.Parameter(
                layer.bias.data.clone().resize_(new_width))
    return layer

#####################################################################################
############################## identity convolution code ############################
#####################################################################################


def _conv_only_deeper(layer):
    wts = torch.zeros(1, 1, layer.kernel_size[0], layer.kernel_size[1])
    nn.init.dirac_(wts)
    wts = wts.repeat(layer.out_channels, 1, 1, 1)

    conv_layer = nn.Conv2d(
        layer.out_channels,
        layer.out_channels,
        kernel_size=layer.kernel_size,
        bias=False,
        padding="same",
        groups=layer.out_channels,
    )
    with torch.no_grad():
        conv_layer.weight.copy_(wts)

    return conv_layer

#####################################################################################
##########################  following code is my own ################################
#####################################################################################


def _fc_only_deeper(layer):
    weight = tf_and_torch.params_torch_to_tf_ndarr(layer, "weight")

    new_layer_w, new_layer_b = _fc_only_deeper_tf_numpy(weight)

    new_layer = nn.Linear(1, 1).to(device.get_device())
    tf_and_torch.params_tf_ndarr_to_torch(new_layer_w, new_layer, "weight")
    tf_and_torch.params_tf_ndarr_to_torch(new_layer_b, new_layer, "bias")

    return new_layer


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


def _conv_only_wider(layer1, layer2, norm_layer, new_width):
    teacher_w1 = tf_and_torch.params_torch_to_tf_ndarr(layer1, "weight")
    teacher_w2 = tf_and_torch.params_torch_to_tf_ndarr(layer2, "weight")
    teacher_b1 = tf_and_torch.params_torch_to_tf_ndarr(layer1, "bias")

    student_w1, student_b1, student_w2 = _conv_only_wider_tf_numpy(
        teacher_w1, teacher_b1, teacher_w2, new_width
    )

    tf_and_torch.params_tf_ndarr_to_torch(student_w1, layer1, "weight")
    tf_and_torch.params_tf_ndarr_to_torch(student_w2, layer2, "weight")
    tf_and_torch.params_tf_ndarr_to_torch(student_b1, layer1, "bias")

    return layer1, layer2, _make_new_norm_layer(norm_layer)


class UnsupportedLayer(Exception):
    pass


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
            self._table.append((hierarchy, name))
        for n, child in curr.named_children():
            self._helper(hierarchy + [curr], n, child)

    def __init__(self, model):
        self._table = list()
        self._helper([], None, model)

    def __iter__(self):
        yield from self._table

    def layer(self, hierarchy, name):
        parent = hierarchy[-1]
        return (
            parent[int(name)]
            if isinstance(parent, nn.Sequential)
            else getattr(parent, name)
        )

    def update_layer(self, hierarchy, name, value):
        parent = hierarchy[-1]
        if isinstance(parent, nn.Sequential):
            parent[int(name)] = value
        else:
            setattr(parent, name, value)


def widen(model, modifier=lambda h, l: 2):
    table = LayerTable(model)
    prev = None
    between_batchnorm = None
    for h, n in table:
        curr = table.layer(h, n)
        if isinstance(curr, nn.BatchNorm2d):
            between_batchnorm = (h, n)
        elif isinstance(curr, nn.Conv2d) or isinstance(curr, nn.Linear):
            if prev is not None:
                modification = modifier(h, curr)
                old_layer1 = table.layer(*prev)
                old_layer2 = curr
                if type(old_layer1) == type(curr) and modification > 1:
                    old_batchnorm = (
                        table.layer(*between_batchnorm)
                        if between_batchnorm is not None
                        else None
                    )
                    new_out_size = round(modification * (old_layer1.out_features if isinstance(
                        old_layer1, nn.Linear) else old_layer1.out_channels))
                    new_layer1, new_layer2, new_batchnorm = _wider(
                        old_layer1, old_layer2, new_out_size, old_batchnorm
                    )
                    table.update_layer(*prev, new_layer1)
                    table.update_layer(h, n, new_layer2)
                    if between_batchnorm is not None:
                        table.update_layer(*between_batchnorm, new_batchnorm)
            prev = (h, n)
            between_batchnorm = None


def deepen(model, modifier=lambda h, l: True):
    table = LayerTable(model)
    for h, n in table:
        curr = table.layer(h, n)
        if (isinstance(curr, nn.Conv2d) or isinstance(curr, nn.Linear)) and modifier(h, curr):
            table.update_layer(h, n, _deeper(curr))
