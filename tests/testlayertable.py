#!/usr/bin/env python3.8

import sys 
import os 
sys.path.append(os.path.join("..", "src"))

import models 
import tracing 
from utils import asserts

def test_layer_table_smallfeedforward():
    model = models.SmallFeedForward(4, 5)
    table = tracing.LayerTable(model)
    itr = iter(table)
    e = next(itr)
    asserts(
        [e["name"], e["prevname"], e["hierarchy"], e["prevhierarchy"]],
        ["fc1", None, [model], None],
    )
    e = next(itr)
    asserts(
        [e["name"], e["prevname"], e["hierarchy"], e["prevhierarchy"]],
        ["act", None, [model], None],
    )
    e = next(itr)
    asserts(
        [e["name"], e["prevname"], e["hierarchy"], e["prevhierarchy"]],
        ["fc2", "fc1", [model], [model]],
    )


def test_layer_table_blockedmodel():
    model = models.BlockedModel(4, 5)
    table = tracing.LayerTable(model)
    itr = iter(table)
    e = next(itr)
    asserts(
        [e["name"], e["prevname"], e["hierarchy"], e["prevhierarchy"]],
        ["fc1", None, [model, model.block1], None],
    )
    e = next(itr)
    asserts(
        [e["name"], e["prevname"], e["hierarchy"], e["prevhierarchy"]],
        ["act", None, [model, model.block1], None],
    )
    e = next(itr)
    asserts(
        [e["name"], e["prevname"], e["hierarchy"], e["prevhierarchy"]],
        ["fc2", "fc1", [model, model.block1], [model, model.block1]],
    )
    e = next(itr)
    asserts(
        [e["name"], e["prevname"], e["hierarchy"], e["prevhierarchy"]],
        ["fc1", None, [model, model.block2], None],
    )
    e = next(itr)
    asserts(
        [e["name"], e["prevname"], e["hierarchy"], e["prevhierarchy"]],
        ["act", None, [model, model.block2], None],
    )
    e = next(itr)
    asserts(
        [e["name"], e["prevname"], e["hierarchy"], e["prevhierarchy"]],
        ["fc2", "fc1", [model, model.block2], [model, model.block2]],
    )


def test_layer_table_sequentialmodel():
    model = models.SequentialModel(4, 5)
    table = tracing.LayerTable(model)
    itr = iter(table)
    e = next(itr)
    asserts(
        [e["name"], e["prevname"], e["hierarchy"], e["prevhierarchy"]],
        ["0", None, [model, model.seq], None],
    )
    e = next(itr)
    asserts(
        [e["name"], e["prevname"], e["hierarchy"], e["prevhierarchy"]],
        ["1", None, [model, model.seq], None],
    )
    e = next(itr)
    asserts(
        [e["name"], e["prevname"], e["hierarchy"], e["prevhierarchy"]],
        ["2", "0", [model, model.seq], [model, model.seq]],
    )


def test_layer_table_nonsequentialconvolution():
    model = models.NonSequentialConvolution(3, 5)
    table = tracing.LayerTable(model)
    itr = iter(table)
    e = next(itr)
    asserts(
        [e["name"], e["prevname"], e["hierarchy"], e["prevhierarchy"]],
        ["conv1", None, [model], None],
    )
    e = next(itr)
    asserts(
        [e["name"], e["prevname"], e["hierarchy"], e["prevhierarchy"]],
        ["conv2", None, [model], None],
    )
    e = next(itr)
    asserts(
        [e["name"], e["prevname"], e["hierarchy"], e["prevhierarchy"]],
        ["conv3", "conv2", [model], [model]],
    )
    e = next(itr)
    asserts(
        [e["name"], e["prevname"], e["hierarchy"], e["prevhierarchy"]],
        ["finalpool", None, [model], None],
    )
    e = next(itr)
    asserts(
        [e["name"], e["prevname"], e["hierarchy"], e["prevhierarchy"]],
        ["fc", None, [model], None],
    )

def main():
    test_layer_table_smallfeedforward()
    test_layer_table_blockedmodel()
    test_layer_table_sequentialmodel()
    test_layer_table_nonsequentialconvolution()
    print("ALL LAYER TABLE TESTS PASSED!")

if __name__ == '__main__':
    main()
