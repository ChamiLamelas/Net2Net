import torch.nn as nn
import net2net


class _LayerTable:
    def _helper(self, parent, name, curr):
        if len(list(curr.children())) == 0:
            self.table.append((parent, name))
        for n, child in curr.named_children():
            self._helper(curr, n, child)

    def __init__(self, model):
        self.table = list()
        self._helper(None, None, model)

    def __iter__(self):
        yield from self.table

    def get(self, parent, name):
        return (
            parent[int(name)]
            if isinstance(parent, nn.Sequential)
            else getattr(parent, name)
        )

    def set(self, parent, name, value):
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


def widen(model, scale):
    """
    NEEDSWORK

    Now it just widens every layer by the scaling factor, should have a way
    of customizing this
    """

    table = _LayerTable(model)
    prev = None
    between_batchnorm = None
    for p, n in table:
        curr = table.get(p, n)
        if _is_batchnorm(curr):
            between_batchnorm = (p, n)
        elif _is_conv(curr) or _is_linear(curr):
            if prev is not None:#  and type(prev) == type(curr):
                old_layer1 = table.get(*prev)
                old_layer2 = curr
                old_batchnorm = (
                    table.get(*between_batchnorm)
                    if between_batchnorm is not None
                    else None
                )
                new_out_size = round(scale * _get_out_size(old_layer1))
                new_layer1, new_layer2, new_batchnorm = net2net.wider(
                    old_layer1, old_layer2, new_out_size, old_batchnorm
                )
                table.set(*prev, new_layer1)
                table.set(p, n, new_layer2)
                if between_batchnorm is not None:
                    table.set(*between_batchnorm, new_batchnorm)
            prev = (p, n)
            between_batchnorm = None


def deepen(model):
    """
    NEEDSWORK

    Doesn't add additional batch norm layers or activation layers
    """

    table = _LayerTable(model)
    for p, n in table:
        curr = table.get(p, n)
        if _is_conv(curr) or _is_linear(curr):
            table.set(p, n, net2net.deeper(curr))
