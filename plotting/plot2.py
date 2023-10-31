import matplotlib.pyplot as plt
import numpy as np
import sys
import os

sys.path.append(os.path.join("..", "src"))

import plot

MARKERS = ["+", "o", "x"]


def train_acc_plot(args):
    _, ax = plt.subplots()
    for m, f in zip(MARKERS, args.folders):
        epoch_lists = plot.make_epoch_lists(f)
        base = os.path.basename(f)
        start = args.start if base.startswith('Adapted') else 0
        train_acc = epoch_lists["train_acc"][start:]
        xs = np.arange(start, len(train_acc)) + 1
        ax.scatter(xs, train_acc, label=base[: base.index("Inception")], marker=m)
    plot.make_plot_nice(ax, "Epochs", "Train Accuracy", args.ymin, args.ymax)
    plot.save(args.file)


def main():
    args = plot.get_args()
    train_acc_plot(args)


if __name__ == "__main__":
    main()
