import torch.nn.functional as F
from logger import ML_Logger
import os
import numpy as np 
import random 
import torch
import prediction

def set_seed(seed=None):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def train(model, train_loader, test_loader, total_epochs, scale_up_epochs, scale_down_epochs, folder, init_optimizer, **optim_args):
    logger = ML_Logger(log_folder=folder, persist=False)
    logger.start(task='training', log_file='training', metrics_file='training')
    model = model.cuda()
    optimizer = init_optimizer(model.parameters(), **optim_args)
    scale_up_epochs = sorted(set(scale_up_epochs))
    scale_down_epochs = sorted(set(scale_down_epochs))
    i = 0
    j = 0
    for epoch in range(total_epochs):
        if i < len(scale_up_epochs) and epoch == scale_up_epochs[i]:
            # scale up
            i += 1
        elif j < len(scale_down_epochs) and epoch == scale_down_epochs[j]:
            # scale down 
            j += 1
        train_epoch(model, train_loader, epoch, optimizer, logger)    
        prediction.predict(model, train_loader, epoch, logger, 'train')
        prediction.predict(model, test_loader, epoch, logger, 'test')
    logger.stop()

def train_epoch(model, train_loader, epoch, optimizer, logger):
    model.train()
    total_loss = 0
    for (data, target) in train_loader:
        data = data.cuda()
        target = target.cuda()
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    logger.log_metrics({'epoch': epoch, 'train_loss': total_loss})