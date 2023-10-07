from pathlib import Path 
import os 
import matplotlib.pyplot as plt
import argparse 
import config 
from logger import read_json

PLOTS = os.path.join("..", "plots")

def check_dir(s):
    s = os.path.join(config.RESULTS, s)
    if not os.path.isdir(s):
        raise argparse.ArgumentTypeError(f'{s} is not a directory')
    return s

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('folder', type=check_dir)
    return parser.parse_args()

def mkdir(dirpath):
    Path(dirpath).mkdir(exist_ok=True, parents=True)


def prep_paths(*files):
    for file in files:
        mkdir(os.path.dirname(file))

def save(file, format=None):
    prep_paths(file)
    plt.savefig(file, format=format, bbox_inches='tight')
    plt.close()


def make_plot_nice(ax, xlabel, ylabel, ymin, ymax, fontsize=16, legendcol=1, title=None, titlefontsize=None):
    if legendcol is not None:
        ax.legend(fontsize=fontsize, ncol=legendcol, frameon=False)
    if title is not None:
        ax.suptitle(
            title, fontsize=titlefontsize if titlefontsize is not None else fontsize)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params('y', labelsize=fontsize)
    ax.tick_params('x', labelsize=fontsize)
    ax.set_xlabel(xlabel, fontsize=fontsize)
    ax.set_ylabel(ylabel, fontsize=fontsize)
    ax.set_ylim([ymin, ymax])
    ax.grid()

def make_epoch_lists(results_folder):
    data = read_json(os.path.join(
        results_folder, "training_metrics.json"))
    epochs = list()
    train_loss = list()
    test_acc = list()
    for e in data:
        if 'train_loss' in e:
            epochs.append(e['epoch'])
            train_loss.append(e['train_loss'])
        if 'test_acc' in e:
            test_acc.append(e['test_acc'])
    return epochs, train_loss, test_acc

def main():
    args = get_args()
    plot_folder = os.path.join(PLOTS, os.path.basename(args.folder))
    print(plot_folder)

if __name__ == '__main__':
    main()