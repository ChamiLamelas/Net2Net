#!/usr/bin/env python3.8

import matplotlib.pyplot as plt
import numpy as np
import sys
import os

sys.path.append(os.path.join("..", "src"))

import plot
import config
from logger import ML_Logger as LOG


def plot6():
    metric = "train_acc"

    teacher_metrics = LOG.load_metrics(
        os.path.join(config.RESULTS, "TeacherInceptionTinyImageNet_11_19_1"),
        "training",
        metric,
        "epoch",
    )

    big_metrics = LOG.load_metrics(
        os.path.join(config.RESULTS, "BigInceptionTinyImageNet_11_19_2"),
        "training",
        metric,
        "epoch",
    )
    _, ax = plt.subplots()
    ax.plot(
        plot.to_min(teacher_metrics["times"]),
        teacher_metrics["metrics"],
        color="blue",
        linestyle="-",
        label="teacher",
    )
    ax.plot(
        plot.to_min(big_metrics["times"]),
        big_metrics["metrics"],
        color="red",
        linestyle="-",
        label="big",
    )
    plot.make_plot_nice(ax, "time (min)", metric, 0, 1)
    plot.save("plot6.png")


def plot7():
    metric = "test_acc"

    teacher_metrics = LOG.load_metrics(
        os.path.join(config.RESULTS, "TeacherInceptionTinyImageNet_11_19_1"),
        "training",
        metric,
        "epoch",
    )

    big_metrics = LOG.load_metrics(
        os.path.join(config.RESULTS, "BigInceptionTinyImageNet_11_19_2"),
        "training",
        metric,
        "epoch",
    )
    _, ax = plt.subplots()
    ax.plot(
        plot.to_min(teacher_metrics["times"]),
        teacher_metrics["metrics"],
        color="blue",
        linestyle="-",
        label="teacher",
    )
    ax.plot(
        plot.to_min(big_metrics["times"]),
        big_metrics["metrics"],
        color="red",
        linestyle="-",
        label="big",
    )
    plot.make_plot_nice(ax, "time (min)", metric, 0, 1)
    plot.save("plot7.png")


def plot8():
    kd_metrics = plot.breakdown_into_lists(
        os.path.join(config.RESULTS, "AdaptedInceptionCIFAR10_11_11_6")
    )
    kd_and_wd_metrics = plot.breakdown_into_lists(
        os.path.join(config.RESULTS, "AdaptedInceptionCIFAR10_11_11_5")
    )
    no_kd_metrics = plot.breakdown_into_lists(
        os.path.join(config.RESULTS, "AdaptedInceptionCIFAR10_11_11_8")
    )
    wd_metrics = plot.breakdown_into_lists(
        os.path.join(config.RESULTS, "AdaptedInceptionCIFAR10_11_11_7")
    )
    _, ax = plt.subplots()
    ax.scatter(
        kd_metrics["test_epoch_times"],
        kd_metrics["test_epoch_accs"],
        color="blue",
        marker="o",
        label="kd",
    )
    ax.scatter(
        wd_metrics["test_epoch_times"],
        wd_metrics["test_epoch_accs"],
        color="green",
        marker="x",
        label="wd",
    )
    ax.scatter(
        no_kd_metrics["test_epoch_times"],
        no_kd_metrics["test_epoch_accs"],
        color="yellow",
        marker="^",
        label="baseline",
    )
    ax.scatter(
        kd_and_wd_metrics["test_epoch_times"],
        kd_and_wd_metrics["test_epoch_accs"],
        color="red",
        marker="*",
        label="kd + wd",
    )
    plot.make_plot_nice(ax, "time (s)", "accuracy", 0, 1)
    plot.save("plot8.png")


if __name__ == "__main__":
    plot6()
    plot7()
    # plot8()
