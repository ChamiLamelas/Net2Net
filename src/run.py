#!/usr/bin/env python3.8

"""
NEEDSWORK document
"""

import config
import argparse
import training
import os


def check_file(fpath):
    fpath = os.path.join(config.CONFIG, fpath)
    if not os.path.isfile(fpath):
        raise argparse.ArgumentTypeError(f"{fpath} is not a file")
    return fpath


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("configfile", type=check_file)
    return parser.parse_args()


def main():
    args = get_args()
    cfg = config.load_config(args.configfile)
    trainer = training.Trainer(cfg)
    trainer.train()


if __name__ == "__main__":
    main()
