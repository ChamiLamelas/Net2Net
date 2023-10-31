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
        # start = args.start if base.startswith("Adapted") else 0
        test_acc = epoch_lists["test_acc"]
        xs = np.arange(len(test_acc)) + 1
        ax.scatter(xs, test_acc, label=base[: base.index("Inception")], marker=m)
    plot.make_plot_nice(ax, "Epochs", "Test Accuracy", args.ymin, args.ymax)
    # ax.legend(loc='lower right')
    plot.save(args.file)


def main():
    args = plot.get_args()
    train_acc_plot(args)


if __name__ == "__main__":
    main()
