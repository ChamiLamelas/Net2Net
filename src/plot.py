"""
NEEDSWORK document
"""

from pathlib import Path
import matplotlib.pyplot as plt
import argparse
import config
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
        ax.legend(fontsize=fontsize, ncol=legendcol, frameon=False)
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


def to_min(secs):
    return [sec // 60 for sec in secs]
