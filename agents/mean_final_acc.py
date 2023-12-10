#!/usr/bin/env python3.8

import numpy as np
import argparse
import logger
import os


def result_folder(folder):
    folder = os.path.join("results", folder)
    if os.path.isdir(folder):
        return folder
    raise argparse.ArgumentTypeError(f"{folder} does not exist")


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("folders", type=result_folder, nargs="+")
    return parser.parse_args()


def main():
    args = get_args()
    for folder in args.folders:
        accs = list()
        for entry in os.scandir(folder):
            if entry.name.endswith("test_acc"):
                acc = logger.get_final_metric(entry.path)
                accs.append(acc)
        print(f"{folder} mean {np.mean(accs)} std {np.std(accs)}")


if __name__ == "__main__":
    main()
