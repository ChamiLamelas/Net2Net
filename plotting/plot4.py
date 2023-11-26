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


def plot5(metric, granularity, folder1, output, label1):
    model1_metrics = LOG.load_metrics(
        os.path.join(config.RESULTS, folder1),
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

    plot.make_plot_nice(ax, "time (min)", metric, 0, 1)
    plot.save(output)


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
    unit="min",
):
    model1_metrics = add_time_starter(
        LOG.load_metrics(
            os.path.join(config.RESULTS, folder1),
            "training",
            metric,
            granularity,
        ),
        starter1,
    )

    model2_metrics = add_time_starter(
        LOG.load_metrics(
            os.path.join(config.RESULTS, folder2),
            "training",
            metric,
            granularity,
        ),
        starter2,
    )

    if unit == "min":
        model1_metrics["times"] = plot.to_min(model1_metrics["times"])
        model2_metrics["times"] = plot.to_min(model2_metrics["times"])

    _, ax = plt.subplots()
    ax.plot(
        model1_metrics["times"],
        model1_metrics["metrics"],
        color="blue",
        linestyle="solid",
        label=label1,
    )
    ax.plot(
        model2_metrics["times"],
        model2_metrics["metrics"],
        color="red",
        linestyle="dotted",
        label=label2,
    )
    plot.make_plot_nice(ax, f"time ({unit})", metric, 0, 1)
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
        "AdaptedInceptionCIFAR10_11_21_2a",
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
            1,
        ),
    )
    plot6(
        "train_acc",
        "batch",
        "TeacherInceptionCIFAR10_11_20_1a",
        "AdaptedInceptionCIFAR10_11_21_2a",
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
            1,
        ),
    )
    plot6(
        "test_acc",
        "epoch",
        "TeacherInceptionCIFAR10_11_20_1a",
        "AdaptedInceptionCIFAR10_11_21_2a",
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
            1,
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
    plot6(
        "train_acc",
        "epoch",
        "TinyModelCIFAR10_11_21_1",
        "BigModelCIFAR10_11_21_1",
        "plot22.png",
        "teacher",
        "big",
        unit="sec",
    )
    plot6(
        "test_acc",
        "epoch",
        "TinyModelCIFAR10_11_21_1",
        "BigModelCIFAR10_11_21_1",
        "plot23.png",
        "teacher",
        "big",
        unit="sec",
    )
    plot6(
        "train_acc",
        "epoch",
        "TinyModelCIFAR10_11_21_1",
        "AdaptedTinyModelCIFAR10_11_21_1",
        "plot24.png",
        "teacher",
        "net2net",
        unit="sec",
        starter2=(
            LOG.load_metrics(
                os.path.join(config.RESULTS, "TinyModelCIFAR10_11_21_1"),
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
        "TinyModelCIFAR10_11_21_1",
        "AdaptedTinyModelCIFAR10_11_21_1",
        "plot25.png",
        "teacher",
        "net2net",
        unit="sec",
        starter2=(
            LOG.load_metrics(
                os.path.join(config.RESULTS, "TinyModelCIFAR10_11_21_1"),
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
        "TeacherInceptionTinyImageNet_11_22_1",
        "BigInceptionTinyImageNet_11_22_2",
        "plot26.png",
        "teacher",
        "big",
        unit="min",
    )
    plot6(
        "test_acc",
        "epoch",
        "TeacherInceptionTinyImageNet_11_22_1",
        "BigInceptionTinyImageNet_11_22_2",
        "plot27.png",
        "teacher",
        "big",
        unit="min",
    )
    plot6(
        "train_acc",
        "epoch",
        "TeacherInceptionCIFAR10_11_20_1a",
        "BigInceptionCIFAR10_11_22_1b",
        "plot28.png",
        "teacher",
        "big",
    )
    plot6(
        "test_acc",
        "epoch",
        "TeacherInceptionCIFAR10_11_20_1a",
        "BigInceptionCIFAR10_11_22_1b",
        "plot29.png",
        "teacher",
        "big",
    )
    plot6(
        "train_acc",
        "epoch",
        "TeacherInceptionTinyImageNet_11_19_1",
        "BigInceptionTinyImageNet_11_19_2",
        "plot30.png",
        "teacher",
        "big",
        unit="min",
    )
    plot6(
        "test_acc",
        "epoch",
        "TeacherInceptionTinyImageNet_11_19_1",
        "BigInceptionTinyImageNet_11_19_2",
        "plot31.png",
        "teacher",
        "big",
        unit="min",
    )
    plot6(
        "train_acc",
        "epoch",
        "TeacherInceptionCIFAR10_11_20_1a",
        "AdaptedInceptionCIFAR10_11_20_2a",
        "plot32.png",
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
        "plot33.png",
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
        "TeacherInceptionTinyImageNet_11_22_2",
        "BigInceptionTinyImageNet_11_22_3",
        "plot34.png",
        "teacher",
        "big",
        unit="min",
    )
    plot6(
        "test_acc",
        "epoch",
        "TeacherInceptionTinyImageNet_11_22_2",
        "BigInceptionTinyImageNet_11_22_3",
        "plot35.png",
        "teacher",
        "big",
        unit="min",
    )
    plot6(
        "train_acc",
        "epoch",
        "TeacherInceptionTinyImageNet_11_22_3",
        "BigInceptionTinyImageNet_11_22_3",
        "plot36.png",
        "teacher",
        "big",
        unit="min",
    )
    plot6(
        "test_acc",
        "epoch",
        "TeacherInceptionTinyImageNet_11_22_3",
        "BigInceptionTinyImageNet_11_22_3",
        "plot37.png",
        "teacher",
        "big",
        unit="min",
    )
    plot6(
        "train_acc",
        "epoch",
        "TeacherInceptionTinyImageNet_11_23_1",
        "BigInceptionTinyImageNet_11_23_1",
        "plot38.png",
        "teacher",
        "big",
        unit="min",
    )
    plot6(
        "test_acc",
        "epoch",
        "TeacherInceptionTinyImageNet_11_23_1",
        "BigInceptionTinyImageNet_11_23_1",
        "plot39.png",
        "teacher",
        "big",
        unit="min",
    )
    plot6(
        "train_acc",
        "epoch",
        "TeacherInceptionTinyImageNet_11_23_2",
        "BigInceptionTinyImageNet_11_23_2",
        "plot40.png",
        "teacher",
        "big",
        unit="min",
    )
    plot6(
        "test_acc",
        "epoch",
        "TeacherInceptionTinyImageNet_11_23_2",
        "BigInceptionTinyImageNet_11_23_2",
        "plot41.png",
        "teacher",
        "big",
        unit="min",
    )
    plot6(
        "train_acc",
        "epoch",
        "TeacherInceptionTinyImageNet_11_24_1",
        "BigInceptionTinyImageNet_11_24_1",
        "plot42.png",
        "teacher",
        "big",
        unit="min",
    )
    plot6(
        "test_acc",
        "epoch",
        "TeacherInceptionTinyImageNet_11_24_1",
        "BigInceptionTinyImageNet_11_24_1",
        "plot43.png",
        "teacher",
        "big",
        unit="min",
    )
    plot6(
        "train_acc",
        "epoch",
        "TeacherInceptionTinyImageNet_11_24_1",
        "AdaptedInceptionTinyImageNet_11_24_1",
        "plot44.png",
        "teacher",
        "net2net",
        starter2=(
            LOG.load_metrics(
                os.path.join(config.RESULTS, "TeacherInceptionTinyImageNet_11_24_1"),
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
        "TeacherInceptionTinyImageNet_11_24_1",
        "AdaptedInceptionTinyImageNet_11_24_1",
        "plot45.png",
        "teacher",
        "net2net",
        starter2=(
            LOG.load_metrics(
                os.path.join(config.RESULTS, "TeacherInceptionTinyImageNet_11_24_1"),
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
        "TeacherInceptionTinyImageNet_11_24_1",
        "AdaptedInceptionTinyImageNet_11_24_2",
        "plot46.png",
        "teacher",
        "net2net",
        starter2=(
            LOG.load_metrics(
                os.path.join(config.RESULTS, "TeacherInceptionTinyImageNet_11_24_1"),
                "training",
                "train_acc",
                "epoch",
            ),
            1,
        ),
    )
    plot6(
        "test_acc",
        "epoch",
        "TeacherInceptionTinyImageNet_11_24_1",
        "AdaptedInceptionTinyImageNet_11_24_2",
        "plot47.png",
        "teacher",
        "net2net",
        starter2=(
            LOG.load_metrics(
                os.path.join(config.RESULTS, "TeacherInceptionTinyImageNet_11_24_1"),
                "training",
                "test_acc",
                "epoch",
            ),
            1,
        ),
    )
    plot6(
        "train_acc",
        "epoch",
        "TeacherInceptionTinyImageNet_11_24_1",
        "AdaptedInceptionTinyImageNet_11_25_2",
        "plot48.png",
        "teacher",
        "net2net",
        starter2=(
            LOG.load_metrics(
                os.path.join(config.RESULTS, "TeacherInceptionTinyImageNet_11_24_1"),
                "training",
                "train_acc",
                "epoch",
            ),
            1,
        ),
    )
    plot6(
        "test_acc",
        "epoch",
        "TeacherInceptionTinyImageNet_11_24_1",
        "AdaptedInceptionTinyImageNet_11_25_2",
        "plot49.png",
        "teacher",
        "net2net",
        starter2=(
            LOG.load_metrics(
                os.path.join(config.RESULTS, "TeacherInceptionTinyImageNet_11_24_1"),
                "training",
                "test_acc",
                "epoch",
            ),
            1,
        ),
    )
    plot6(
        "train_acc",
        "epoch",
        "TeacherInceptionTinyImageNet_11_24_1",
        "AdaptedInceptionTinyImageNet_11_25_3",
        "plot50.png",
        "teacher",
        "net2net",
        starter2=(
            LOG.load_metrics(
                os.path.join(config.RESULTS, "TeacherInceptionTinyImageNet_11_24_1"),
                "training",
                "train_acc",
                "epoch",
            ),
            1,
        ),
    )
    plot6(
        "test_acc",
        "epoch",
        "TeacherInceptionTinyImageNet_11_24_1",
        "AdaptedInceptionTinyImageNet_11_25_3",
        "plot51.png",
        "teacher",
        "net2net",
        starter2=(
            LOG.load_metrics(
                os.path.join(config.RESULTS, "TeacherInceptionTinyImageNet_11_24_1"),
                "training",
                "test_acc",
                "epoch",
            ),
            1,
        ),
    )
    plot6(
        "train_acc",
        "epoch",
        "TeacherInceptionTinyImageNet_11_24_1",
        "AdaptedInceptionTinyImageNet_11_25_4",
        "plot52.png",
        "teacher",
        "net2net",
        starter2=(
            LOG.load_metrics(
                os.path.join(config.RESULTS, "TeacherInceptionTinyImageNet_11_24_1"),
                "training",
                "train_acc",
                "epoch",
            ),
            1,
        ),
    )
    plot6(
        "test_acc",
        "epoch",
        "TeacherInceptionTinyImageNet_11_24_1",
        "AdaptedInceptionTinyImageNet_11_25_4",
        "plot53.png",
        "teacher",
        "net2net",
        starter2=(
            LOG.load_metrics(
                os.path.join(config.RESULTS, "TeacherInceptionTinyImageNet_11_24_1"),
                "training",
                "test_acc",
                "epoch",
            ),
            1,
        ),
    )

