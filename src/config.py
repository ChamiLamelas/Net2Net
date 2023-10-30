"""
NEEDSWORK document
"""

import models
import data
import torch
import toml
import os
import torch.optim as optim
from pathlib import Path
import shutil
import deepening
import widening
import numpy as np
import random

CONFIG = os.path.join("..", "config")
RESULTS = os.path.join("..", "results")


def check_dict(d, *keys):
    for k in keys:
        assert k in d


class ConfigException(Exception):
    pass


def getdynamic(attr):
    if hasattr(deepening, attr):
        return getattr(deepening, attr)
    elif hasattr(widening, attr):
        return getattr(widening, attr)
    raise ConfigException(f"cannot find attribute {attr} in widening or deepening")


class ConfigLoader:
    def __init__(self, configfile):
        self.configfile = configfile
        self.config = toml.load(configfile)
        check_dict(self.config, "model", "dataset", "folder")
        self.loadseed()
        self.loadmodel()
        self.loaddata()
        self.loadoptimizer()
        self.prepfolder()
        self.loaddynamics("scaleupepochs")
        self.loaddynamics("scaledownepochs")
        self.cleanup()

    def loadmodel(self):
        self.config["model_args"] = self.config.get("model_args", dict())
        self.config["model"] = getattr(models, self.config["model"])(
            **self.config["model_args"]
        )
        if "model_weights" in self.config:
            self.config["model"].load_state_dict(
                torch.load(self.config["model_weights"])
            )

    def loaddata(self):
        load_fn = getattr(data, "load_" + self.config["dataset"])
        batch_size = self.config.get("batchsize", 64)
        self.config["trainloader"] = load_fn(train=True, batch_size=batch_size)
        self.config["testloader"] = load_fn(train=False, batch_size=batch_size)

    def loadoptimizer(self):
        self.config["optimizer"] = getattr(optim, self.config.get("optimizer", "Adam"))

    def prepfolder(self):
        self.config["folder"] = os.path.join(RESULTS, self.config["folder"])
        if os.path.isdir(self.config["folder"]):
            raise ConfigException(
                f"{self.config['folder']} already exists -- please delete it or specify a different folder"
            )
        Path(self.config["folder"]).mkdir(exist_ok=True, parents=True)
        if "desc" in self.config:
            Path(os.path.join(self.config["folder"], "description.txt")).write_text(
                self.config["desc"]
            )
        shutil.copyfile(
            self.configfile, os.path.join(self.config["folder"], "config.toml")
        )

    def loaddynamics(self, scaletype):
        if scaletype in self.config:
            for k in self.config[scaletype]:
                self.config[scaletype][k]["modifier"] = getdynamic(
                    self.config[scaletype][k]["modifier"]
                )
                if isinstance(self.config[scaletype][k]["ignore"], str):
                    self.config[scaletype][k]["ignore"] = getattr(
                        models, self.config[scaletype][k]["ignore"]
                    )()
                else:
                    self.config[scaletype][k]["ignore"] = set(
                        self.config[scaletype][k]["ignore"]
                    )
                self.config[scaletype][k]["modify"] = getattr(
                    models, self.config[scaletype][k]["modify"]
                )
        else:
            self.config[scaletype] = dict()

    def loadseed(self):
        self.config["seed"] = self.config.get("seed", 42)
        torch.manual_seed(self.config["seed"])
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.config["seed"])
        np.random.seed(self.config["seed"])
        random.seed(self.config["seed"])

    def cleanup(self):
        del self.config["dataset"]
        del self.config["model_args"]
        if "batchsize" in self.config:
            del self.config["batchsize"]


def load_config(configfile):
    return ConfigLoader(configfile).config
