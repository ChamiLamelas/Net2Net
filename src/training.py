"""
NEEDSWORK document + reimplement as OOP (train has so many parameters..)
"""

import torch.nn.functional as F
from logger import ML_Logger
import torch
import prediction
from torchvision.models.inception import InceptionOutputs
from tqdm import tqdm
import models 
import device 
import copy


def train(
    dev,
    model,
    train_loader,
    test_loader,
    total_epochs,
    scale_up_epochs,
    scale_down_epochs,
    folder,
    init_optimizer,
    **optim_args,
):
    smaller = list()
    teacher = None 
    logger = ML_Logger(log_folder=folder, persist=False)
    logger.start(task="training", log_file="training", metrics_file="training")
    optimizer = init_optimizer(model.parameters(), **optim_args)
    for epoch in range(total_epochs):
        if epoch in scale_up_epochs:
            backup = copy.deepcopy(model)
            if len(smaller) == 0 or models.count_parameters(model) > models.count_parameters(smaller[-1]):
                smaller.append(backup)
            elif models.count_parameters(model) == models.count_parameters(smaller[-1]):
                smaller[-1] = backup
            config = scale_up_epochs[epoch]
            config["modifier"](
                model,
                config["ignore"],
                config["modify"],
            )
            
        elif epoch in scale_down_epochs:
            teacher = copy.deepcopy(model)
            model = smaller[-1] 
        train_epoch(dev, model, train_loader, epoch, optimizer, logger, teacher)
        prediction.predict(model, train_loader, epoch, logger, "train")
        test_acc = prediction.predict(
            model, test_loader, epoch, logger, "test")
        logger.save_model(model, test_acc, epoch)
    logger.stop()


def train_epoch(dev, model, train_loader, epoch, optimizer, logger, teacher):
    model = model.to(dev)
    model.train()
    total_loss = 0
    for data, target in tqdm(
        train_loader, total=len(train_loader), desc=f"training epoch {epoch}"
    ):
        data, target = device.move(dev, data, target)
        optimizer.zero_grad()
        loss = compute_loss(model, data, target, teacher)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    logger.log_metrics({"epoch": epoch, "train_loss": total_loss})

def get_logits(model, data):
    output = model(data)
    if isinstance(output, InceptionOutputs):
        output = output.logits
    return output 


def compute_loss(model, data, target, teacher):
    output = get_logits(model, data)
    if teacher is None:
        loss = F.cross_entropy(output, target)
    else:
        T = 2
        soft_target_loss_weight = 0.25
        ce_loss_weight = 0.75
        with torch.no_grad():
            teacher_logits = teacher(data)
        student_logits = get_logits(model, data)
        soft_targets = F.softmax(teacher_logits / T, dim=-1)
        soft_prob = F.log_softmax(student_logits / T, dim=-1)
        soft_targets_loss = - \
            torch.sum(soft_targets * soft_prob) / soft_prob.size()[0] * (T**2)
        label_loss = F.cross_entropy(student_logits, target)
        loss = soft_target_loss_weight * soft_targets_loss + ce_loss_weight * label_loss
    return loss
