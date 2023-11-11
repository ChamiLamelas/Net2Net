"""
NEEDSWORK document
"""

from pathlib import Path
import matplotlib.pyplot as plt
import argparse
import config
from logger import read_json
import os
import numpy as np
import config

PLOTS = os.path.join("..", "plots")


def check_dir(dp):
    dp = os.path.join(config.RESULTS, dp)
    if not os.path.isdir(dp):
        raise argparse.ArgumentTypeError(f"{dp} is not a directory")
    return dp


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("start", type=int)
    parser.add_argument("ymin", type=float)
    parser.add_argument("ymax", type=float)
    parser.add_argument("file", type=str)
    parser.add_argument("folders", nargs="+", type=check_dir)
    return parser.parse_args()


def mkdir(dirpath):
    Path(dirpath).mkdir(exist_ok=True, parents=True)


def prep_paths(*files):
    for file in files:
        mkdir(os.path.dirname(file))


def save(file, format=None):
    file = os.path.join(PLOTS, file)
    prep_paths(file)
    plt.savefig(file, format=format, bbox_inches="tight")
    plt.close()


def make_plot_nice(
    ax,
    xlabel,
    ylabel,
    ymin,
    ymax,
    fontsize=16,
    legendcol=1,
    title=None,
    titlefontsize=None,
):
    if legendcol is not None:
        ax.legend(fontsize=fontsize, ncol=legendcol, frameon=False, loc="lower right")
    if title is not None:
        ax.suptitle(
            title, fontsize=titlefontsize if titlefontsize is not None else fontsize
        )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params("y", labelsize=fontsize)
    ax.tick_params("x", labelsize=fontsize)
    ax.set_xlabel(xlabel, fontsize=fontsize)
    ax.set_ylabel(ylabel, fontsize=fontsize)
    ax.set_ylim([ymin, ymax])
    ax.grid()


def breakdown_into_lists(results_folder):
    data = read_json(os.path.join(results_folder, "training_metrics.json"))
    train_epoch_times = list()
    train_epoch_accs = list()
    test_epoch_times = list()
    test_epoch_accs = list()
    train_batch_times = list()
    train_batch_accs = list()
    for entry in data:
        if "batch" in entry:
            train_batch_times.append(entry["time"])
            train_batch_accs.append(entry["train_acc"])
        elif "epoch" in entry:
            if "train_acc" in entry:
                train_epoch_times.append(entry["time"])
                train_epoch_accs.append(entry["train_acc"])
            elif "test_acc" in entry:
                test_epoch_times.append(entry["time"])
                test_epoch_accs.append(entry["test_acc"])
    windowsize = 100
    train_batch_times = train_batch_times[windowsize - 1 :]
    train_batch_accs = np.convolve(
        train_batch_accs, np.ones(windowsize) * (1 / windowsize), "valid"
    )
    return {
        "train_epoch_times": train_epoch_times,
        "train_epoch_accs": train_epoch_accs,
        "test_epoch_times": test_epoch_times,
        "test_epoch_accs": test_epoch_accs,
        "train_batch_times": train_batch_times,
        "train_batch_accs": train_batch_accs,
    }
