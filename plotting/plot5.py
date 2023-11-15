#!/usr/bin/env python3.8

import matplotlib.pyplot as plt
import numpy as np
import sys
import os

sys.path.append(os.path.join("..", "src"))

import plot
import config


def plot9():
    kd_and_wd_metrics = plot.breakdown_into_lists(
        os.path.join(config.RESULTS, "AdaptedInceptionCIFAR10_11_14_1")
    )
    _, ax = plt.subplots()
    ax.scatter(
        kd_and_wd_metrics["test_epoch_times"],
        kd_and_wd_metrics["test_epoch_accs"],
        color="red",
        marker="*",
        label="kd + wd",
    )
    plot.make_plot_nice(ax, "time (s)", "accuracy", 0, 1)
    plot.save("plot9.png")

if __name__ == "__main__":
    plot9()
