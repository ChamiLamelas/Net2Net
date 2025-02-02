#!/usr/bin/env python3.8

from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import csv
import logger
import os
import toml
import agent
import simulation
import torch
import tracing
import job
import deepening
import random

RESULTS = os.path.join("..", "results")
PLOTS = os.path.join("..", "plots")

SMALL_FOLDER = "no_adaptation2"

BASELINE_FOLDERS = [
    "baseline_early2",
    "baseline_early_middle2",
    "baseline_middle2",
    "baseline_middle_late2",
    "baseline_late2",
]

AGENT_FOLDERS = [
    "simulation_early_learn2",
    "simulation_middle_learn2",
    "simulation_late_learn2",
    # "simulation_middle_to_late_learn1",
]

ACC_AGENT_FOLDERS = {
    "time_stages": {
        "agentfolder": "simulation_early_to_middle_to_late_learn1",
        "folders": [
            "simulation_early",
            "simulation_early_middle",
            "simulation_middle",
            "simulation_middle_late",
            "simulation_late",
        ],
        "names": ["E", "E-M", "M", "M-L", "L"],
    }
}

GROUPED_AGENT_FOLDERS = {
    "early_to_middle_explorations": {
        "folders": [
            "simulation_early_to_middle_learn1",
            "simulation_early_to_middle_learn10",
            "simulation_early_to_middle_learn100",
        ],
        "comparison": "compare_exploration",
    },
    "early_to_late_explorations": {
        "folders": [
            "simulation_early_to_late_learn1",
            "simulation_early_to_late_learn10",
            "simulation_early_to_late_learn100",
        ],
        "comparison": "compare_exploration",
    },
    "middle_to_early_explorations": {
        "folders": [
            "simulation_middle_to_early_learn1",
            "simulation_middle_to_early_learn10",
            "simulation_middle_to_early_learn100",
        ],
        "comparison": "compare_exploration",
    },
    "middle_to_late_explorations": {
        "folders": [
            "simulation_middle_to_late_learn1",
            "simulation_middle_to_late_learn10",
            "simulation_middle_to_late_learn100",
        ],
        "comparison": "compare_exploration",
    },
    "late_to_early_explorations": {
        "folders": [
            "simulation_late_to_early_learn1",
            "simulation_late_to_early_learn10",
            "simulation_late_to_early_learn100",
        ],
        "comparison": "compare_exploration",
    },
    "late_to_middle_explorations": {
        "folders": [
            "simulation_late_to_middle_learn1",
            "simulation_late_to_middle_learn10",
            "simulation_late_to_middle_learn100",
        ],
        "comparison": "compare_exploration",
    },
    "middle_transfer": {
        "folders": [
            "simulation_middle_learn",
            "simulation_early_to_middle_learn10",
        ],
        "comparison": "transfer",
        "threshold": 5,
        "jumpstart_pos": (5000, 200),
        "time_to_threshold_pos": (3000, 20),
    },
}

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
    # ax.grid()


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


def get_name(folder, comparison):
    ending_digits = folder[folder.index("learn") + len("learn") :]
    if len(ending_digits) == 0:
        if comparison == "compare_exploration" or comparison == "transfer":
            return "scratch"
        elif comparison == "compare_scenarios":
            folder = folder[len("simulation_") :]
            return folder[: folder.index("_")]
        else:
            raise RuntimeError(f"invalid comparison option {comparison}")
    if comparison == "transfer":
        return "transfer"
    return f"T = {ending_digits}"


def draw_arrow(ax, label, arrow_start, arrow_end, label_pos):
    arrow_properties = dict(facecolor="black", edgecolor="black", arrowstyle="<->")
    ax.annotate("", xy=arrow_start, xytext=arrow_end, arrowprops=arrow_properties)
    ax.annotate(label, xy=label_pos, fontsize=16)


def get_threshold_pt(objs, threshold):
    for i, o in enumerate(objs):
        if o <= threshold:
            return i
    raise RuntimeError("never passed threshold!")


def grouped_objective_plot():
    window_size = 50
    for groupname, groupinfo in GROUPED_AGENT_FOLDERS.items():
        _, ax = plt.subplots()
        ymax = 0
        starts = list()
        thresholds = list()
        for folder in groupinfo["folders"]:
            name = get_name(folder, groupinfo["comparison"])
            folder = os.path.join(RESULTS, folder)
            objectives = load_objectives(
                os.path.join(folder, "agent.episode.objective")
            )
            x = np.arange(len(objectives))
            mean_objectives = np.convolve(
                objectives, np.ones(window_size) / window_size, mode="valid"
            )
            ax.plot(x[: -window_size + 1], mean_objectives, label=name)
            ymax = max(ymax, np.max(mean_objectives))
            if "threshold" in groupinfo:
                starts.append(mean_objectives[0])
                thresholds.append(
                    get_threshold_pt(mean_objectives, groupinfo["threshold"])
                )
        if len(groupinfo["folders"]) == 2:
            ax.axhline(
                groupinfo["threshold"],
                label="threshold",
                color="black",
                linestyle="dashed",
            )
            draw_arrow(
                ax,
                "Jumpstart",
                (1, starts[0]),
                (1, starts[1]),
                groupinfo["jumpstart_pos"],
            )
            draw_arrow(
                ax,
                "Time to Threshold",
                (thresholds[0], groupinfo["threshold"] + 5),
                (thresholds[1], groupinfo["threshold"] + 5),
                groupinfo["time_to_threshold_pos"],
            )
        make_plot_nice(
            ax,
            "Episode",
            "Objective",
            0,
            ymax,
            legendcol=1,
        )
        save(f"objective_{groupname}.png")


def acc_plot():
    for groupname, groupinfo in ACC_AGENT_FOLDERS.items():
        folder = groupinfo["agentfolder"]

        print(f"=== Folder ===\n{folder}")

        folder = os.path.join(RESULTS, folder)
        config = toml.load(os.path.join(folder, "config.toml"))

        agt = agent.Agent(config)
        agt.policy.load_state_dict(
            torch.load(os.path.join(folder, "agent.finalmodel.pth"))
        )
        agt.eval()

        model = job.Job(config).model
        action_set_size = len(tracing.get_all_deepen_blocks(model)) + 1

        _, ax = plt.subplots()
        no_adaptations = list()
        baselines = list()
        agents = list()
        bests = list()

        for name, folder in zip(groupinfo["names"], groupinfo["folders"]):
            folder = os.path.join(RESULTS, folder)
            config = toml.load(os.path.join(folder, "config.toml"))
            sim = simulation.Simulation(config)

            actions = list()
            for _ in range(sim.get_num_actions()):
                action, _ = agt.action(
                    {
                        "totaltime": sim.get_total_time(),
                        "timeleft": sim.get_time_left(actions),
                        "model": model,
                    }
                )
                deepening.deepen_model(model, index=action)
                actions.append(action)

            all_rankings = sim.get_acc_ranking()

            actions = tuple(actions)

            no_adaptations.append(
                sim.get_acc(tuple([action_set_size - 1] * sim.get_num_actions()))
            )

            bests.append(all_rankings[0][1])

            baselines.append(np.mean(random.sample([e[1] for e in all_rankings], k=10)))

            agents.append(sim.get_acc(actions))

            print(
                f"{name}\t: agent : {' '.join(map(str, actions))} ({agents[-1]:.4f}) best : {' '.join(map(str, all_rankings[0][0]))} ({bests[-1]:.4f})"
            )

        no_adaptation = np.mean(no_adaptations)

        ax.axhline(
            no_adaptation, color="black", linestyle="dashed", label="no adaptation"
        )

        xs = np.arange(len(baselines))
        ax.set_xticks(xs)
        ax.set_xticklabels(groupinfo["names"])
        ax.scatter(xs, baselines, label="baseline")
        ax.scatter(xs, agents, label="agent")
        ax.scatter(xs, bests, label="best")

        make_plot_nice(
            ax,
            "Stage",
            "Accuracy",
            0,
            1,
            legendcol=2,
        )
        save(f"acc_{groupname}.png")


def main():
    # random_plot()
    # agent_plot()
    objective_plot()
    # grouped_objective_plot()
    # acc_plot()


if __name__ == "__main__":
    main()
