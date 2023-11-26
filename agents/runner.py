#!/usr/bin/env python3.8

import training
import environment
import agent
import toml
import torch
import numpy as np
import argparse
import random
import os
from logger import ML_Logger
import shutil
from pathlib import Path

RESULTS = "results"


def file(f):
    if not os.path.isfile(f):
        raise argparse.ArgumentTypeError(f"{f} is not a file")
    return f


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("configfile", type=file, help="configuration file")
    return parser.parse_args()


def set_seed(config):
    seed = config["seed"]
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def setup_folder(config, configfile):
    test_dir = config["folder"].startswith("test")
    config["folder"] = os.path.join(RESULTS, config["folder"])
    logger = ML_Logger(log_folder=config["folder"], persist=False)
    if os.path.isdir(config["folder"]):
        if test_dir:
            print("Specified a test folder -- deleting!")
            shutil.rmtree(config["folder"])
        else:
            raise RuntimeError(
                f"{config['folder']} already exists -- please delete it or specify a different folder"
            )
    Path(config["folder"]).mkdir(exist_ok=True, parents=True)
    if "desc" in config:
        Path(os.path.join(config["folder"], "description.txt")).write_text(
            config["desc"]
        )
    shutil.copyfile(configfile, os.path.join(config["folder"], "config.toml"))
    return logger


def loadconfig(args):
    return toml.load(args.configfile)


def run(config, total_eps, logger):
    max_digits = len(str(total_eps))
    env = environment.Environment(config)
    agt = agent.Agent(config)
    for ep in range(total_eps):
        env.scheduler.start()
        agt.init()
        tr = training.Trainer(config, env.scheduler, agt, logger)
        tr.train(f"training{str(ep).zfill(max_digits)}")
        agt.update()
        print(f"ep {ep}")


def main():
    args = get_args()
    config = loadconfig(args)
    runner_config = config["runner"]
    set_seed(runner_config)
    logger = setup_folder(runner_config, args.configfile)
    run(config, runner_config["total_episodes"], logger)


if __name__ == "__main__":
    main()
