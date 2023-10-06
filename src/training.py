import torch.nn.functional as F
from logger import ML_Logger
import os
import numpy as np 
import random 
import torch

LOGS_FOLDER = os.path.join("..", "logs")

def set_seed(seed=None):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def train(model, train_loader, total_epochs, scale_up_epochs, scale_down_epochs, folder):
    logger = ML_Logger(log_folder=os.path.join(LOGS_FOLDER, folder), persist=False)
    logger.start(f'training')
    model = model.cuda()
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
        train_epoch(model, train_loader, epoch)    
    logger.stop()

def train_epoch(model, train_loader, epoch, optimizer, logger):
    model.train()
    total_loss = 0
    for (data, target) in train_loader:
        data = data.cuda()
        target = target.cuda()
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    logger.log_metrics({'epoch': epoch, 'train_loss': total_loss})