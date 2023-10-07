import models
import data
import toml
import os
import torch.optim as optim
from pathlib import Path
import shutil

CONFIG = os.path.join("..", "config")
RESULTS = os.path.join("..", "results")


def check_dict(d, *keys):
    for k in keys:
        assert k in d


def load_config(configfile):
    config = toml.load(configfile)
    check_dict(config, 'model', 'dataset', 'folder')

    config['model_args'] = config.get('model_args', dict())
    config['model'] = getattr(models, config['model'])(**config['model_args'])

    load_fn = getattr(data, "load_" + config['dataset'])
    batch_size = config.get('batchsize', 64)
    config['trainloader'] = load_fn(train=True, batch_size=batch_size)
    config['testloader'] = load_fn(train=False, batch_size=batch_size)

    config['optimizer'] = getattr(optim, config.get('optimizer', 'SGD'))

    config['folder'] = os.path.join(RESULTS, config['folder'])
    if os.path.isdir(config['folder']):
        shutil.rmtree(config['folder'])
    Path(config['folder']).mkdir(exist_ok=True, parents=True)

    config['scaleupepochs'] = config.get('scaleupepochs', list())
    config['scaledownepochs'] = config.get('scaledownepochs', list())

    del config['dataset']
    del config['model_args']
    if 'batchsize' in config:
        del config['batchsize']
    return config
