"""
NEEDSWORK document
"""

import config
import argparse
import training
import os


def check_file(fpath):
    fpath = os.path.join(config.CONFIG, fpath)
    if not os.path.isfile(fpath):
        raise argparse.ArgumentTypeError(f'{fpath} is not a file')
    return fpath


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('configfile', type=check_file)
    return parser.parse_args()


def main():
    args = get_args()
    cfg = config.load_config(args.configfile)
    training.train(cfg['device'], cfg['model'], cfg['trainloader'], cfg['testloader'], cfg['epochs'],
                   cfg['scaleupepochs'], cfg['scaledownepochs'], cfg['folder'], cfg['optimizer'], **cfg['optimizer_args'])


if __name__ == '__main__':
    main()
