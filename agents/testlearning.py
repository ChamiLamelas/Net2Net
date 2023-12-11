#!/usr/bin/env python3.8

import training
import environment
import agent
import toml
import argparse
import os
from logger import ML_Logger
import shutil
from pathlib import Path
import heuristics
import seed
import torch
import job
import copy
import tracing
import deepening
import sys

RESULTS = "results"


def file(f):
    if not os.path.isfile(f):
        raise argparse.ArgumentTypeError(f"{f} is not a file")
    return f


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("configfile", type=file, help="configuration file")
    return parser.parse_args()


def setup_folder(config, configfile):
    test_dir = config["folder"].startswith("test")
    config["folder"] = os.path.join(RESULTS, config["folder"])
    if os.path.isdir(config["folder"]):
        if test_dir:
            print("Specified a test folder -- deleting!", file=sys.stderr)
            shutil.rmtree(config["folder"])
        else:
            raise RuntimeError(
                f"{config['folder']} already exists -- please delete it or specify a different folder"
            )
    logger = ML_Logger(log_folder=config["folder"], persist=False)
    Path(config["folder"]).mkdir(exist_ok=True, parents=True)
    if "desc" in config:
        Path(os.path.join(config["folder"], "description.txt")).write_text(
            config["desc"]
        )
    shutil.copyfile(configfile, os.path.join(config["folder"], "config.toml"))
    return logger


def loadconfig(args):
    return toml.load(args.configfile)


def get_agent(config, run_config):
    for module in [agent, heuristics]:
        if hasattr(module, run_config["agent"]):
            return getattr(module, run_config["agent"])(config)
    raise RuntimeError("could not find agent")


def get_reward(actions, correct_actions):
    acc = 0.4 + sum(a == ca for a, ca in zip(actions, correct_actions)) * 0.04
    return acc


def should_log(ep, log_freq):
    return ep == 0 or (ep + 1) % log_freq == 0


def run(config, run_config, logger):
    log_freq = 100
    decay_freq = 500
    total_eps = run_config["total_episodes"]
    agt = get_agent(config, run_config)
    print(f"Policy:\n{agt.policy}", file=sys.stderr)
    agt_logger = copy.deepcopy(logger)
    agt_logger.start("agent", "agent", "learning")
    base_model = job.Job(config).model
    nactions = 5
    correct_actions = [0, 1, 2, 3, 4][:nactions]
    time_lefts = [65, 52, 39, 26, 13][-nactions:]
    for ep in range(total_eps):
        model = copy.deepcopy(base_model)
        if should_log(ep, log_freq):
            print(f"==== episode {ep + 1}/{total_eps} ====", file=sys.stderr)
        actions = list()
        agt.init()
        for tl, ca in zip(time_lefts, correct_actions):
            action, probs = agt.action(
                {"totaltime": 400, "timeleft": tl, "model": model}
            )
            actions.append(action)
            if should_log(ep, log_freq):
                print(
                    f"action = {action} ({'CORRECT' if action == ca else 'incorrect'})\nprobs = {' '.join(f'{e:.4f}' for e in probs)}\n--------",
                    file=sys.stderr,
                )
            deepening.deepen_model(model, index=action)
            agt.save_prob()
        agt.record_acc(get_reward(actions, correct_actions))
        if (ep + 1) % decay_freq == 0:
            obj = agt.update(True)
            print(
                f"Learning rate decayed to {agt.get_current_lr()}",
                file=sys.stderr,
            )
        else:
            obj = agt.update(False)
        if should_log(ep, log_freq):
            print("Weights:", file=sys.stderr)
            for n, p in agt.policy.named_parameters():
                print(f"{n}: {p.mean()} {p.std()}", file=sys.stderr)
            print(file=sys.stderr)

            print("Rewards:", file=sys.stderr)
            for r in agt.rewards:
                print(f"{r:.4f}", file=sys.stderr)
            print(f"objective = {obj}", file=sys.stderr)
        agt_logger.log_metrics({"objective": obj}, "episode")
    if run_config.get("save", False):
        agt.save()
    agt_logger.stop()


def main():
    # torch.autograd.set_detect_anomaly(True)
    args = get_args()
    config = loadconfig(args)
    runner_config = config["runner"]
    seed.set_seed(runner_config["seed"])
    logger = setup_folder(runner_config, args.configfile)
    run(config, runner_config, logger)


if __name__ == "__main__":
    main()
