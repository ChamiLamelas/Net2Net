#!/usr/bin/env python3.8

from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import csv
import logger
import os

RESULTS = "results"
PLOTS = "plots"

SMALL_FOLDER = "no_adaptation2"

BASELINE_FOLDERS = [
    "baseline_early2",
    "baseline_early_middle2",
    "baseline_middle2",
    "baseline_middle_late2",
    "baseline_late2",
]

AGENT_FOLDERS = [
    "simulation_early_learn",
    "simulation_middle_learn",
    "simulation_late_learn",
    "simulation_early_to_middle_learn",
    "simulation_early_to_late_learn",
    "simulation_middle_to_early_learn",
    "simulation_middle_to_late_learn",
    "simulation_late_to_early_learn",
    "simulation_late_to_middle_learn",
]

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


def _get_final_accs(folder):
    final_accs = defaultdict(list)
    for entry in os.scandir(folder):
        for k, v in EXTENSIONS.items():
            if entry.name.endswith(v):
                final_accs[k].append(logger.get_final_metric(entry.path))
    return final_accs


def _transform_labels(labels):
    new_labels = list()
    for label in labels:
        split_label = label.split("_")
        new_labels.append("-".join([e[0].upper() for e in split_label]))
    return new_labels


def random_plot():
    small_final_accs = dict()
    for k, v in EXTENSIONS.items():
        small_final_accs[k] = logger.get_final_metric(
            os.path.join(RESULTS, SMALL_FOLDER, f"training0.epoch.{v}")
        )
    all_final_accs = defaultdict(list)
    labels = list()
    for folder in BASELINE_FOLDERS:
        labels.append(folder[folder.index("_") + 1 :])
        folder = os.path.join(RESULTS, folder)
        final_accs = _get_final_accs(folder)
        for k, v in final_accs.items():
            all_final_accs[k].append(v)
    for k, v in all_final_accs.items():
        _, ax = plt.subplots()
        means = [np.mean(e) for e in v]
        stds = [np.std(e) for e in v]
        xs = np.arange(len(labels))
        ax.errorbar(xs, means, stds, fmt="ok")
        ax.set_xticks(xs)
        ax.set_xticklabels(_transform_labels(labels))
        ax.axhline(small_final_accs[k], 0, 1, linewidth=0.5, label="no adaptation")
        make_plot_nice(
            ax, "Adaptation Time (s)", f"{k.title()} Accuracy", 0, 1, legendcol=None
        )
        save(f"{k.lower()}_baseline.png")
        print(f"{k}: {small_final_accs[k]}")


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
                    final_accs[k][episode] = logger.get_final_metric(entry.path)
        final_accs = {k: _dict_to_list(v) for k, v in final_accs.items()}
        for k, v in final_accs.items():
            _, ax = plt.subplots()
            x = np.arange(len(v))
            ax.plot(x, v)
            print(f"{folder}: {k}: {np.max(v)}")
            make_plot_nice(ax, "Episode", f"{k.title()} Accuracy", 0, 1, legendcol=None)
            save(f"{k.lower()}_{os.path.basename(folder)}.png")


def load_objectives(path):
    objectives = list()
    with open(path, "r") as f:
        reader = csv.reader(f)
        for row in reader:
            objectives.append(float(row[-1]))
    return objectives


def objective_plot():
    window_size = 50
    for folder in AGENT_FOLDERS:
        folder = os.path.join(RESULTS, folder)
        _, ax = plt.subplots()
        objectives = load_objectives(os.path.join(folder, "agent.episode.objective"))
        mean_objectives = np.convolve(
            objectives, np.ones(window_size) / window_size, mode="valid"
        )
        x = np.arange(len(objectives))
        ax.plot(x, objectives, label="objective")
        ax.plot(x[: -window_size + 1], mean_objectives, label="running mean")
        make_plot_nice(
            ax,
            "Episode",
            "Objective",
            np.min(objectives),
            np.max(objectives),
            legendcol=None,
        )
        save(f"objective_{os.path.basename(folder)}.png")


def main():
    random_plot()
    agent_plot()
    objective_plot()


if __name__ == "__main__":
    main()
