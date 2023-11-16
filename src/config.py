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
import device
import torch.nn as nn

CONFIG = os.path.join("..", "config")
RESULTS = os.path.join("..", "results")


def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


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


class Config:
    def __init__(self, configfile):
        self.configfile = configfile
        self.config = toml.load(configfile)
        check_dict(self.config, "model", "dataset", "folder")
        self.loadseed()
        self.loadmodel()
        self.loaddata()
        self.loadoptimizer()
        self.prepfolder()
        self.loadscaleupepochs()
        self.loadscaledownepochs()
        self.loaddevice()
        self.load_kd_params()
        self.cleanup()

    def __getitem__(self, key):
        if key not in self.config:
            raise ConfigException(f"cannot find configuration for {key}")
        return self.config[key]

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
        self.config["optimizer_args"] = self.config.get("optimizer_args", dict())
        self.config["learning_rate_decay"] = self.config.get("learning_rate_decay", 0.9)

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

    def loadscaleupepochs(self):
        self.config["scaleupepochs"] = self.config.get("scaleupepochs", dict())
        self.config["scaleupepochs"] = {
            int(k): v for k, v in self.config["scaleupepochs"].items()
        }
        for k in self.config["scaleupepochs"]:
            self.config["scaleupepochs"][k]["modifier"] = getdynamic(
                self.config["scaleupepochs"][k]["modifier"]
            )
            self.config["scaleupepochs"][k]["filter_function"] = getattr(
                models, self.config["scaleupepochs"][k]["filter_function"]
            )

    def loadscaledownepochs(self):
        self.config["scaledownepochs"] = set(self.config.get("scaledownepochs", list()))

    def loadseed(self):
        self.config["seed"] = self.config.get("seed", 42)
        set_seed(self.config["seed"])

    def loaddevice(self):
        if "device" not in self.config:
            self.config["device"] = device.get_device()
        else:
            self.config["device"] = device.get_device(self.config["device"])
            if self.config["device"] is None:
                raise ConfigException(f"Invalid device {self.config['device']}")

    def cleanup(self):
        del self.config["dataset"]
        del self.config["model_args"]
        if "batchsize" in self.config:
            del self.config["batchsize"]

    def load_kd_params(self):
        self.config["T"] = self.config.get("T", 2)
        self.config["soft_target_loss_weight"] = self.config.get(
            "soft_target_loss_weight", 0.25
        )
        self.config["ce_loss_weight"] = self.config.get("ce_loss_weight", 0.75)
        self.config["weight_distillation"] = self.config.get(
            "weight_distillation", True
        )
        self.config["knowledge_distillation"] = self.config.get(
            "knowledge_distillation", True
        )

    def __repr__(self):
        rep = list()
        for k, v in self.config.items():
            rep.append(str(k))
            rep.append("\t" + (str(type(v)) if isinstance(v, nn.Module) else str(v)))
        return "\n".join(rep)

    def __str__(self):
        return repr(self)
