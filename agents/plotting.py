#!/usr/bin/env python3.8

from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import toml
import csv
import os

RESULTS = "results"
PLOTS = "plots"

BASELINE_FOLDERS = ["baseline1", "baseline2", "baseline3", "baseline4", "baseline5"]
AGENT_FOLDERS = ["middle", "early"]

EXTENSIONS = {"train": "train_acc", "test": "test_acc"}


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


def _get_final_acc(path):
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            pass
        return float(row[1])


def _get_final_accs(folder):
    final_accs = defaultdict(list)
    for entry in os.scandir(folder):
        for k, v in EXTENSIONS.items():
            if entry.name.endswith(v):
                final_accs[k].append(_get_final_acc(entry.path))
    return final_accs


def _get_adaptation_point(folder):
    return toml.load(os.path.join(folder, "config.toml"))["scheduler"]["gpu_changes"][
        0
    ]["time"]


def random_plot():
    times = list()
    all_final_accs = defaultdict(list)
    for folder in BASELINE_FOLDERS:
        folder = os.path.join(RESULTS, folder)
        times.append(_get_adaptation_point(folder))
        final_accs = _get_final_accs(folder)
        for k, v in final_accs.items():
            all_final_accs[k].append(v)
    xs = np.arange(len(times))
    for k, v in all_final_accs.items():
        _, ax = plt.subplots()
        means = [np.mean(e) for e in v]
        stds = [np.std(e) for e in v]
        ax.errorbar(xs, means, stds, fmt="ok")
        ax.set_xticks(xs)
        ax.set_xticklabels(times)
        make_plot_nice(
            ax, "Adaptation Time (s)", f"{k.title()} Accuracy", 0, 1, legendcol=None
        )
        save(f"{k.lower()}_baseline.png")


def _dict_to_list(d):
    l = [None] * len(d)
    for k, v in d.items():
        l[k] = v
    return l


def agent_plot():
    for folder in AGENT_FOLDERS:
        final_accs = defaultdict(dict)
        folder = os.path.join(RESULTS, folder)
        for entry in os.scandir(folder):
            for k, v in EXTENSIONS.items():
                if entry.name.endswith(v):
                    episode = int(entry.name[len("training") : entry.name.index(".")])
                    final_accs[k][episode] = _get_final_acc(entry.path)
        final_accs = {k: _dict_to_list(v) for k, v in final_accs.items()}
        for k, v in final_accs.items():
            _, ax = plt.subplots()
            x = np.arange(len(v))
            ax.plot(x, v)
            make_plot_nice(ax, "Episode", f"{k.title()} Accuracy", 0, 1, legendcol=None)
            save(f"{k.lower()}_{os.path.basename(folder)}.png")


def main():
    random_plot()
    agent_plot()


if __name__ == "__main__":
    main()
