import torch.nn as nn


DEEPEN_BLOCK_NAME = "Net2NetDeepenBlock"


class UnsupportedLayer(Exception):
    pass


class PrinterLayer(nn.Module):
    def __init__(self, label):
        super().__init__()
        self.label = label

    def forward(self, x):
        return x


def get_all_deepen_blocks(module):
    blocks = list()

    def _recursive_get_all_deepen_blocks(curr, hierarchy, name):
        if DEEPEN_BLOCK_NAME in type(curr).__name__:
            blocks.append((hierarchy, name))
        else:
            for name, child in curr.named_children():
                _recursive_get_all_deepen_blocks(child, hierarchy + (curr,), name)

    _recursive_get_all_deepen_blocks(module, tuple(), None)
    return blocks


def is_important(module):
    return any(isinstance(module, layer_type) for layer_type in [nn.Conv2d, nn.Linear])


def get_all_important_layer_hierarchies(module):
    layers = dict()

    def _recursive_get_all_important_layer_hierarchies(hierarchy, parent, curr):
        if is_important(curr):
            layers[hierarchy] = parent
        for name, child in curr.named_children():
            _recursive_get_all_important_layer_hierarchies(
                hierarchy + (name,), curr, child
            )

    _recursive_get_all_important_layer_hierarchies(tuple(), None, module)
    return layers


def get_all_important_layers(module):
    layers = list()

    def _recursive_get_all_important_layers(curr):
        if is_important(curr):
            layers.append(curr)
        for child in curr.children():
            _recursive_get_all_important_layers(child)

    _recursive_get_all_important_layers(module)
    return layers


# old tracing stuff ...

"""
def _filterout(iterable, filterset):
    return list(filter(lambda e: type(e).__name__ not in filterset, iterable))


class LayerTable:
    def _helper(self, hierarchy, typehierarchy, curr):
        if len(list(curr.children())) == 0:
            self.table.append({"hierarchy": hierarchy, "typehierarchy": typehierarchy})
        for n, child in curr.named_children():
            self._helper(
                hierarchy + [n],
                typehierarchy + [curr],
                child,
            )

    def _find_prev(self, ignore):
        for i, e in enumerate(self.table):
            curr = self.get(e["hierarchy"])
            j = i - 1
            found = False
            e["prevhierarchy"] = None
            while (
                j >= 0
                and _filterout(self.table[j]["hierarchy"], ignore)
                == _filterout(e["hierarchy"], ignore)
                and not found
            ):
                prevhierarchy = self.table[j]["hierarchy"]
                prev = self.get(prevhierarchy)
                if type(curr) == type(prev):
                    if isinstance(curr, nn.Linear):
                        found = prev.out_features == curr.in_features
                    elif isinstance(curr, nn.Conv2d):
                        found = prev.out_channels == curr.in_channels
                j -= 1
            if found:
                e["prevhierarchy"] = prevhierarchy

    def __init__(self, model, ignore=set()):
        self.table = list()
        self._helper([], [], model)
        self.model = model
        self._find_prev(ignore)

    def __iter__(self):
        yield from self.table

    @staticmethod
    def followhierarchy(obj, e):
        return obj[int(e)] if e.isdigit() else getattr(obj, e)

    def get(self, hierarchy):
        obj = self.model
        for e in hierarchy:
            obj = LayerTable.followhierarchy(obj, e)
        return obj

    def set(self, hierarchy, value):
        obj = self.model
        for e in hierarchy[:-1]:
            obj = LayerTable.followhierarchy(obj, e)
        setattr(obj, hierarchy[-1], value)
"""
