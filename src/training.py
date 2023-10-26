"""
NEEDSWORK document
"""

import torch.nn.functional as F
from logger import ML_Logger
import numpy as np
import random
import torch
import prediction
import device
from torchvision.models.inception import InceptionOutputs


def set_seed(seed):
    # Seeds all RNGs used by torch and its dependencies

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def train(seed, model, train_loader, test_loader, total_epochs, scale_up_epochs, scale_down_epochs, folder, init_optimizer, **optim_args):
    set_seed(seed)
    logger = ML_Logger(log_folder=folder, persist=False)
    logger.start(task='training', log_file='training', metrics_file='training')
    model = model.to(device.get_device())
    optimizer = init_optimizer(model.parameters(), **optim_args)
    scale_up_epochs = sorted(set(scale_up_epochs))
    scale_down_epochs = sorted(set(scale_down_epochs))
    for epoch in range(total_epochs):
        if epoch in scale_up_epochs:
            config = scale_up_epochs[epoch]
            model = config["modifier"](
                model, config["modify_args"]["ignore"], config["modify_args"]["modifier"])
        elif epoch in scale_down_epochs:
            config = scale_down_epochs[epoch]
            model = config["modifier"](
                model, config["modify_args"]["ignore"], config["modify_args"]["modifier"])
        train_epoch(model, train_loader, epoch, optimizer, logger)
        prediction.predict(model, train_loader, epoch, logger, 'train')
        prediction.predict(model, test_loader, epoch, logger, 'test')
    logger.stop()


def train_epoch(model, train_loader, epoch, optimizer, logger):
    model.train()
    total_loss = 0
    for (data, target) in train_loader:
        data, target = device.move(device.get_device(), data, target)
        optimizer.zero_grad()
        output = model(data)
        if isinstance(output, InceptionOutputs):
            output = output.logits
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    logger.log_metrics({'epoch': epoch, 'train_loss': total_loss})
