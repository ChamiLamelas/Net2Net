"""
Notes: 

- This does not support linear to conv or conv to linear widening, one way 
to potentially solve this issue is to convert a linear layer to an
identical 1x1 convolutional layer (https://datascience.stackexchange.com/a/12833).

- Widening linear-linear, convolutional-convolutional and deepening linear layers
are taken from here: https://github.com/paengs/Net2Net

https://web.stanford.edu/~nanbhas/blog/forward-hooks-pytorch/
https://github.com/pytorch/pytorch/issues/35600
https://pytorch.org/docs/stable/generated/torch.jit.trace.html#torch.jit.trace


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


def _filterout(iterable, filterset):
    return list(filter(lambda e: type(e).__name__ not in filterset, iterable))


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
            layer.weight = nn.Parameter(
                _resize_with_ones(layer.weight.data, new_width))
            layer.bias = nn.Parameter(
                _resize_with_zeros(layer.bias.data, new_width))
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


def wider(m1, m2, new_width, batch_norm):
    if isinstance(m1, nn.Linear) and isinstance(m2, nn.Linear):
        return _fc_only_wider(m1, m2, new_width)
    elif isinstance(m1, nn.Conv2d) and isinstance(m2, nn.Conv2d):
        return _conv_only_wider(m1, m2, batch_norm, new_width)
    else:
        raise UnsupportedLayer(f"m1: {type(m1)} m2: {type(m2)}")


def deeper(m):
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
            self.table.append({"hierarchy": hierarchy, "name": name})
        for n, child in curr.named_children():
            self._helper(hierarchy + [curr], n, child)

    def _find_prev(self, ignore):
        for i, e in enumerate(self.table):
            curr = LayerTable.get(e["hierarchy"], e["name"])
            j = i - 1
            found = False
            e["prevhierarchy"] = None
            e["prevname"] = None
            while j >= 0 and _filterout(self.table[j]["hierarchy"], ignore) == _filterout(e["hierarchy"], ignore) and not found:
                # print("===")
                # print([type(e).__name__ for e in _filterout(
                #     self.table[j]["hierarchy"], ignore)])
                # print(
                #     "\t", [type(e).__name__ for e in self.table[j]["hierarchy"]])
                # print([type(e).__name__ for e in _filterout(e["hierarchy"], ignore)])
                # print("\t", [type(e).__name__ for e in e["hierarchy"]])
                prevhierarchy = self.table[j]["hierarchy"]
                prevname = self.table[j]["name"]
                prev = LayerTable.get(prevhierarchy, prevname)
                if type(curr) == type(prev):
                    if isinstance(curr, nn.Linear):
                        found = prev.out_features == curr.in_features
                    elif isinstance(curr, nn.Conv2d):
                        found = prev.out_channels == curr.in_channels
                j -= 1
            if found:
                e["prevhierarchy"] = prevhierarchy
                e["prevname"] = prevname

    def __init__(self, model, ignore=set()):
        self.table = list()
        self._helper([], None, model)
        self._find_prev(ignore)

    def __iter__(self):
        yield from self.table

    @staticmethod
    def get(hierarchy, name):
        parent = hierarchy[-1]
        return (
            parent[int(name)]
            if isinstance(parent, nn.Sequential)
            else getattr(parent, name)
        )

    @staticmethod
    def set(hierarchy, name, value):
        parent = hierarchy[-1]
        if isinstance(parent, nn.Sequential):
            parent[int(name)] = value
        else:
            setattr(parent, name, value)


def widen(model, ignore=set(), modifier=lambda _: 1.5):
    table = LayerTable(model, ignore)
    between_batchnorm = None
    for e in table:
        curr = LayerTable.get(e["hierarchy"], e["name"])
        if isinstance(curr, nn.BatchNorm2d):
            between_batchnorm = e
        elif isinstance(curr, nn.Conv2d) or isinstance(curr, nn.Linear):
            print(e["prevname"])
            if e["prevname"] is not None:
                old_layer1 = LayerTable.get(e["prevhierarchy"], e["prevname"])
                old_layer2 = curr
                if (type(old_layer1) == type(old_layer2)) and modifier(e) > 1:
                    old_batchnorm = (
                        table.get(
                            between_batchnorm["hierarchy"], between_batchnorm["name"])
                        if between_batchnorm is not None
                        else None
                    )
                    new_out_size = round(
                        modifier(e)
                        * (
                            old_layer1.out_features
                            if isinstance(old_layer1, nn.Linear)
                            else old_layer1.out_channels
                        )
                    )
                    new_layer1, new_layer2, new_batchnorm = wider(
                        old_layer1, old_layer2, new_out_size, old_batchnorm
                    )
                    table.set(e["prevhierarchy"], e["prevname"], new_layer1)
                    table.set(e["hierarchy"], e["name"], new_layer2)
                    if between_batchnorm is not None:
                        table.set(
                            between_batchnorm["hierarchy"], between_batchnorm["name"], new_batchnorm)
            between_batchnorm = None


def deepen(model, ignore=set(), modifier=lambda _: True):
    table = LayerTable(model, ignore)
    for e in table:
        curr = table.get(e["hierarchy"], e["name"])
        if (isinstance(curr, nn.Conv2d) or isinstance(curr, nn.Linear)) and modifier(e):
            table.set(e["hierarchy"], e["name"], deeper(curr))


def shrink(model, factor):
    pass
