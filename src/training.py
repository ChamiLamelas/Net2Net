"""
NEEDSWORK document
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


class Trainer:
    def __init__(self, config):
        self.device = config["device"]
        self.model = config["model"]
        self.train_loader = config["trainloader"]
        self.test_loader = config["testloader"]
        self.scale_up_epochs = config["scaleupepochs"]
        self.scale_down_epochs = config["scaledownepochs"]
        self.folder = config["folder"]
        self.total_epochs = config["epochs"]
        optimizer_fn = config["optimizer"]
        optimizer_args = config["optimizer_args"]
        self.optimizer = optimizer_fn(self.model.parameters(), **optimizer_args)
        self.T = config["T"]
        self.soft_target_loss_weight = config["soft_target_loss_weight"]
        self.ce_loss_weight = config["ce_loss_weight"]

    def train(self):
        smaller = list()
        teacher = None
        logger = ML_Logger(log_folder=self.folder, persist=False)
        logger.start(task="training", log_file="training", metrics_file="training")
        for epoch in range(self.total_epochs):
            if epoch in self.scale_up_epochs:
                backup = copy.deepcopy(self.model)
                if len(smaller) == 0 or models.count_parameters(
                    self.model
                ) > models.count_parameters(smaller[-1]):
                    smaller.append(backup)
                elif models.count_parameters(self.model) == models.count_parameters(
                    smaller[-1]
                ):
                    smaller[-1] = backup
                config = self.scale_up_epochs[epoch]
                config["modifier"](
                    self.model,
                    config["ignore"],
                    config["modify"],
                )
            elif epoch in self.scale_down_epochs:
                teacher = copy.deepcopy(self.model)
                self.model = smaller[-1]
            self.train_epoch(epoch, self.optimizer, logger, teacher)
            prediction.predict(self.model, self.train_loader, epoch, logger, "train")
            prediction.predict(self.model, self.test_loader, epoch, logger, "test")
        logger.stop()

    def train_epoch(self, epoch, optimizer, logger, teacher):
        self.model = self.model.to(self.device)
        self.model.train()
        total_loss = 0
        for data, target in tqdm(
            self.train_loader,
            total=len(self.train_loader),
            desc=f"training epoch {epoch}",
        ):
            data, target = device.move(self.device, data, target)
            optimizer.zero_grad()
            loss = self.compute_loss(data, target, teacher)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        logger.log_metrics({"train_loss": total_loss}, "epcoh")

    def get_logits(self, data):
        output = self.model(data)
        if isinstance(output, InceptionOutputs):
            output = output.logits
        return output

    def compute_loss(self, data, target, teacher):
        output = self.get_logits(data)
        if teacher is None:
            loss = F.cross_entropy(output, target)
        else:
            with torch.no_grad():
                teacher_logits = teacher(data)
            student_logits = self.get_logits(data)
            soft_targets = F.softmax(teacher_logits / self.T, dim=-1)
            soft_prob = F.log_softmax(student_logits / self.T, dim=-1)
            soft_targets_loss = (
                -torch.sum(soft_targets * soft_prob)
                / soft_prob.size()[0]
                * (self.T**2)
            )
            label_loss = F.cross_entropy(student_logits, target)
            loss = (
                self.soft_target_loss_weight * soft_targets_loss
                + self.ce_loss_weight * label_loss
            )
        return loss
