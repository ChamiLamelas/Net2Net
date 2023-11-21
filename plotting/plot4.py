#!/usr/bin/env python3.8

import matplotlib.pyplot as plt
import numpy as np
import sys
import os

sys.path.append(os.path.join("..", "src"))

import plot
import config
from logger import ML_Logger as LOG


def add_time_starter(metrics, starter):
    if starter is not None:
        return {
            "times": starter[0]["times"][: starter[1] + 1]
            + list(np.add(metrics["times"], starter[0]["times"][starter[1]])),
            "metrics": starter[0]["metrics"][: starter[1] + 1] + metrics["metrics"],
        }
    return metrics


def plot6(
    metric,
    granularity,
    folder1,
    folder2,
    output,
    label1,
    label2,
    starter1=None,
    starter2=None,
):
    model1_metrics = add_time_starter(
        LOG.load_metrics(
            os.path.join(config.RESULTS, folder1),
            "training",
            metric,
            granularity,
        ), starter1
    )

    model2_metrics = add_time_starter(
        LOG.load_metrics(
            os.path.join(config.RESULTS, folder2),
            "training",
            metric,
            granularity,
        ), starter2
    )

    _, ax = plt.subplots()
    ax.plot(
        plot.to_min(model1_metrics["times"]),
        model1_metrics["metrics"],
        color="blue",
        linestyle="solid",
        label=label1,
    )
    ax.plot(
        plot.to_min(model2_metrics["times"]),
        model2_metrics["metrics"],
        color="red",
        linestyle="dotted",
        label=label2,
    )
    plot.make_plot_nice(ax, "time (min)", metric, 0, 1)
    plot.save(output)


def plot7(
    metric,
    granularity,
    folder1,
    folder2,
    folder3,
    folder4,
    output,
    label1,
    label2,
    label3,
    label4,
):
    model1_metrics = LOG.load_metrics(
        os.path.join(config.RESULTS, folder1),
        "training",
        metric,
        granularity,
    )
    model2_metrics = LOG.load_metrics(
        os.path.join(config.RESULTS, folder2),
        "training",
        metric,
        granularity,
    )
    model3_metrics = LOG.load_metrics(
        os.path.join(config.RESULTS, folder3),
        "training",
        metric,
        granularity,
    )
    model4_metrics = LOG.load_metrics(
        os.path.join(config.RESULTS, folder4),
        "training",
        metric,
        granularity,
    )
    _, ax = plt.subplots()
    ax.plot(
        plot.to_min(model1_metrics["times"]),
        model1_metrics["metrics"],
        color="blue",
        linestyle="solid",
        label=label1,
    )
    ax.plot(
        plot.to_min(model2_metrics["times"]),
        model2_metrics["metrics"],
        color="red",
        linestyle="dotted",
        label=label2,
    )
    ax.plot(
        plot.to_min(model3_metrics["times"]),
        model3_metrics["metrics"],
        color="green",
        linestyle="dashed",
        label=label3,
    )
    ax.plot(
        plot.to_min(model4_metrics["times"]),
        model4_metrics["metrics"],
        color="orange",
        linestyle="dashdot",
        label=label4,
    )
    plot.make_plot_nice(ax, "time (min)", metric, 0, 1)
    plot.save(output)


if __name__ == "__main__":
    plot6(
        "train_acc",
        "epoch",
        "TeacherInceptionCIFAR10_11_20_1a",
        "BigInceptionCIFAR10_11_20_1b",
        "plot10.png",
        "teacher",
        "big",
    )
    plot6(
        "train_acc",
        "batch",
        "TeacherInceptionCIFAR10_11_20_1a",
        "BigInceptionCIFAR10_11_20_1b",
        "plot11.png",
        "teacher",
        "big",
    )
    plot6(
        "test_acc",
        "epoch",
        "TeacherInceptionCIFAR10_11_20_1a",
        "BigInceptionCIFAR10_11_20_1b",
        "plot12.png",
        "teacher",
        "big",
    )
    plot6(
        "train_acc",
        "epoch",
        "TeacherInceptionCIFAR10_11_20_1a",
        "AdaptedInceptionCIFAR10_11_20_2a",
        "plot13.png",
        "teacher",
        "net2net",
        starter2=(
            LOG.load_metrics(
                os.path.join(config.RESULTS, "TeacherInceptionCIFAR10_11_20_1a"),
                "training",
                "train_acc",
                "epoch",
            ),
            3,
        ),
    )
    plot6(
        "train_acc",
        "batch",
        "TeacherInceptionCIFAR10_11_20_1a",
        "AdaptedInceptionCIFAR10_11_20_2a",
        "plot14.png",
        "teacher",
        "net2net",
        starter2=(
            LOG.load_metrics(
                os.path.join(config.RESULTS, "TeacherInceptionCIFAR10_11_20_1a"),
                "training",
                "train_acc",
                "epoch",
            ),
            3,
        ),
    )
    plot6(
        "test_acc",
        "epoch",
        "TeacherInceptionCIFAR10_11_20_1a",
        "AdaptedInceptionCIFAR10_11_20_2a",
        "plot15.png",
        "teacher",
        "net2net",
        starter2=(
            LOG.load_metrics(
                os.path.join(config.RESULTS, "TeacherInceptionCIFAR10_11_20_1a"),
                "training",
                "test_acc",
                "epoch",
            ),
            3,
        ),
    )
    plot6(
        "train_acc",
        "epoch",
        "AdaptedInceptionCIFAR10_11_20_2a",
        "AdaptedInceptionCIFAR10_11_20_2b",
        "plot16.png",
        "net2net",
        "random",
    )
    plot6(
        "train_acc",
        "batch",
        "AdaptedInceptionCIFAR10_11_20_2a",
        "AdaptedInceptionCIFAR10_11_20_2b",
        "plot17.png",
        "net2net",
        "random",
    )
    plot6(
        "test_acc",
        "epoch",
        "AdaptedInceptionCIFAR10_11_20_2a",
        "AdaptedInceptionCIFAR10_11_20_2b",
        "plot18.png",
        "net2net",
        "random",
    )
    plot7(
        "train_acc",
        "epoch",
        "AdaptedInceptionCIFAR10_11_20_3a",
        "AdaptedInceptionCIFAR10_11_20_3b",
        "AdaptedInceptionCIFAR10_11_20_3c",
        "AdaptedInceptionCIFAR10_11_20_3d",
        "plot19.png",
        "kd + wd",
        "kd",
        "wd",
        "baseline",
    )
    plot7(
        "train_acc",
        "batch",
        "AdaptedInceptionCIFAR10_11_20_3a",
        "AdaptedInceptionCIFAR10_11_20_3b",
        "AdaptedInceptionCIFAR10_11_20_3c",
        "AdaptedInceptionCIFAR10_11_20_3d",
        "plot20.png",
        "kd + wd",
        "kd",
        "wd",
        "baseline",
    )
    plot7(
        "test_acc",
        "epoch",
        "AdaptedInceptionCIFAR10_11_20_3a",
        "AdaptedInceptionCIFAR10_11_20_3b",
        "AdaptedInceptionCIFAR10_11_20_3c",
        "AdaptedInceptionCIFAR10_11_20_3d",
        "plot21.png",
        "kd + wd",
        "kd",
        "wd",
        "baseline",
    )
