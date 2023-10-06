import argparse
import matplotlib.pyplot as plt
from pathlib import Path
import os
import sys
sys.path.append("../")
from src.logger import read_json


MODELS = {'teacher_training': 'small', 'wider_deeper_teacher':
          'large (rand)', 'wider_teacher': 'large (rand)', 'dynamic_wider_training': 'transfer', 'dynamic_wider_deeper_training': 'transfer'}


def get_one_argument(argument, description, **kwargs):
    """
    Sets up ArgumentParser to parse a single command line argument

    Parameters:
        argument: str
            The name of the argument 

        description: str
            The help text for the program

        kwargs: **
            Any other keyword arguments for the argument (e.g. help, type, choices, etc.)

    Returns:
        The parsed argument
    """

    parser = argparse.ArgumentParser(
        description=description, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument(
        argument, **kwargs)
    return next(iter(vars(parser.parse_args()).values()))


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


def make_epoch_lists(model_name, dataset):
    data = read_json(os.path.join(
        "logs", dataset, model_name + "_metrics.json"))
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


def plot(metric_lists, models, models_name):
    nepochs = len(metric_lists[0][0])
    mid = nepochs // 2
    _, ax = plt.subplots()
    for model, (epochs, metric) in zip(models, metric_lists):
        ax.plot(epochs, metric, label=MODELS[model])
    ax.axvline(x=mid, linestyle='--', color='red')
    return ax


def plot_train_loss(train_loss_lists, dataset, models, models_name):
    ymin, ymax = (0, 900) if dataset == 'mnist' else (0, 2000)
    make_plot_nice(plot(train_loss_lists, models, models_name),
                   'epoch', 'train loss', ymin, ymax)
    save(os.path.join('plots', dataset, f'{models_name}_train_loss.png'))


def plot_test_acc(test_acc_lists, dataset, models, models_name):
    ymin, ymax = (94, 100) if dataset == 'mnist' else (50, 80)
    make_plot_nice(plot(test_acc_lists, models, models_name),
                   'epoch', 'test accuracy', ymin, ymax)
    save(os.path.join('plots', dataset, f'{models_name}_test_acc.png'))


def plots(models, models_name, dataset):
    train_loss_lists = list()
    test_acc_lists = list()
    for model in models:
        epochs, train_loss, test_acc = make_epoch_lists(model, dataset)
        train_loss_lists.append((epochs, train_loss))
        test_acc_lists.append((epochs, test_acc))
    plot_train_loss(train_loss_lists, dataset, models, models_name)
    plot_test_acc(test_acc_lists, dataset, models, models_name)


def main():
    dataset = get_one_argument('dataset', 'plotter', help='name of dataset', choices=[
                               'cifar10', 'mnist'], type=str)
    wider_deeper = [m for m in MODELS if 'deeper' in m] + ['teacher_training']
    wider = [m for m in MODELS if 'deeper' not in m]
    plots(wider_deeper, 'wider_deeper', dataset)
    plots(wider, 'wider', dataset)


if __name__ == '__main__':
    main()
