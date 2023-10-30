import matplotlib.pyplot as plt
import numpy as np
import sys
import os

sys.path.append(os.path.join("..", "src"))

import plot

MARKERS = ["+", "o", "x"]


def train_loss_plot(args):
    _, ax = plt.subplots()
    for m, f in zip(MARKERS, args.folders):
        epoch_lists = plot.make_epoch_lists(f)
        base = os.path.basename(f)
        start = args.start if base.startswith('Adapted') else 0
        train_loss = epoch_lists["train_loss"][start:]
        xs = np.arange(start, len(train_loss)) + 1
        ax.scatter(xs, train_loss, label=base[: base.index("Inception")], marker=m)
    plot.make_plot_nice(ax, "Epochs", "Train Loss", args.ymin, args.ymax)
    plot.save("plot1.png")


def main():
    args = plot.get_args()
    train_loss_plot(args)


if __name__ == "__main__":
    main()
