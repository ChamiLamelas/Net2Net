#!/usr/bin/env python3.8

import matplotlib.pyplot as plt
import numpy as np
import sys
import os

sys.path.append(os.path.join("..", "src"))

import plot
import config


def plot6():
    teacher_metrics = plot.breakdown_into_lists(
        os.path.join(config.RESULTS, "TeacherInceptionCIFAR105")
    )
    big_metrics = plot.breakdown_into_lists(
        os.path.join(config.RESULTS, "BigInceptionCIFAR105")
    )
    net2net_only = plot.breakdown_into_lists(
        os.path.join(config.RESULTS, "AdaptedInceptionCIFAR108")
    )
    random_deepen = plot.breakdown_into_lists(
        os.path.join(config.RESULTS, "AdaptedInceptionCIFAR109")
    )
    _, ax = plt.subplots()
    ax.plot(
        teacher_metrics["train_batch_times"],
        teacher_metrics["train_batch_accs"],
        color="blue",
        linestyle="-",
        label="teacher",
    )
    ax.plot(
        big_metrics["train_batch_times"],
        big_metrics["train_batch_accs"],
        color="red",
        linestyle="-",
        label="big",
    )
    ax.plot(
        np.add(
            net2net_only["train_batch_times"], teacher_metrics["train_epoch_times"][3]
        ),
        net2net_only["train_batch_accs"],
        color="green",
        linestyle="-",
        label="net2net"
    )
    ax.plot(
        np.add(
            random_deepen["train_batch_times"], teacher_metrics["train_epoch_times"][3]
        ),
        random_deepen["train_batch_accs"],
        color="orange",
        linestyle="-",
        label="random"
    )
    plot.make_plot_nice(ax, "time (s)", "accuracy", 0, 1)
    plot.save("plot6.png")


def plot7():
    teacher_metrics = plot.breakdown_into_lists(
        os.path.join(config.RESULTS, "TeacherInceptionCIFAR105")
    )
    big_metrics = plot.breakdown_into_lists(
        os.path.join(config.RESULTS, "BigInceptionCIFAR105")
    )
    net2net_only = plot.breakdown_into_lists(
        os.path.join(config.RESULTS, "AdaptedInceptionCIFAR108")
    )
    random_deepen = plot.breakdown_into_lists(
        os.path.join(config.RESULTS, "AdaptedInceptionCIFAR109")
    )
    _, ax = plt.subplots()
    ax.scatter(
        teacher_metrics["train_epoch_times"],
        teacher_metrics["train_epoch_accs"],
        color="blue",
        marker="o",
        label="teacher",
    )
    ax.scatter(
        big_metrics["train_epoch_times"],
        big_metrics["train_epoch_accs"],
        color="red",
        marker="*",
        label="big",
    )
    ax.scatter(
        np.add(
            net2net_only["train_epoch_times"], teacher_metrics["train_epoch_times"][3]
        ),
        net2net_only["train_epoch_accs"],
        color="green",
        marker="x",
        label="net2net"
    )
    ax.scatter(
        np.add(
            random_deepen["train_epoch_times"], teacher_metrics["train_epoch_times"][3]
        ),
        random_deepen["train_epoch_accs"],
        color="orange",
        marker="+",
        label="random"
    )
    plot.make_plot_nice(ax, "time (s)", "accuracy", 0, 1)
    plot.save("plot7.png")

def plot8():
    net2net_kd_only = plot.breakdown_into_lists(
        os.path.join(config.RESULTS, "AdaptedInceptionCIFAR106")
    )
    net2net_wd_kd = plot.breakdown_into_lists(
        os.path.join(config.RESULTS, "AdaptedInceptionCIFAR107")
    )
    _, ax = plt.subplots()
    ax.scatter(
        net2net_kd_only["train_epoch_times"],
        net2net_kd_only["train_epoch_accs"],
        color="blue",
        marker="o",
        label="kd",
    )
    ax.scatter(
        net2net_wd_kd["train_epoch_times"],
        net2net_wd_kd["train_epoch_accs"],
        color="red",
        marker="*",
        label="kd + wd",
    )
    plot.make_plot_nice(ax, "time (s)", "accuracy", 0, 1)
    plot.save("plot8.png")

if __name__ == "__main__":
    plot6()
    plot7()
    plot8()
