import torch.nn as nn


class UnsupportedLayer(Exception):
    pass


def _filterout(iterable, filterset):
    return list(filter(lambda e: type(e).__name__ not in filterset, iterable))


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
            while (
                j >= 0
                and _filterout(self.table[j]["hierarchy"], ignore)
                == _filterout(e["hierarchy"], ignore)
                and not found
            ):
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
        # print("init:", type(model))
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
