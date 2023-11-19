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
import distillation
import torch.optim as optim


class Trainer:
    def __init__(self, config):
        self.device = config["device"]
        self.optimizer_fn = config["optimizer"]
        self.optimizer_args = config["optimizer_args"]
        self.learning_rate_decay = config["learning_rate_decay"]
        self.model = config["model"]
        self.update_optimizer()
        self.train_loader = config["trainloader"]
        self.test_loader = config["testloader"]
        self.scale_up_epochs = config["scaleupepochs"]
        self.scale_down_epochs = config["scaledownepochs"]
        self.folder = config["folder"]
        self.total_epochs = config["epochs"]
        self.T = config["T"]
        self.soft_target_loss_weight = config["soft_target_loss_weight"]
        self.ce_loss_weight = config["ce_loss_weight"]
        self.weight_distillation = config["weight_distillation"]
        self.knowledge_distillation = config["knowledge_distillation"]
        self.logger = ML_Logger(log_folder=self.folder, persist=False)

    def train(self):
        smaller = list()
        teacher = None
        self.logger.start(task="training", log_file="training", metrics_file="training")
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
                    config["filter_function"],
                    config["add_batch_norm"]
                )
                self.update_optimizer()
            elif epoch in self.scale_down_epochs:
                teacher = copy.deepcopy(self.model)
                self.model = smaller[-1]
                self.update_optimizer()
                if self.weight_distillation:
                    distillation.deeper_weight_transfer(teacher, self.model)
                if not self.knowledge_distillation:
                    teacher = None
            self.train_epoch(epoch, self.optimizer, teacher)
            test_acc = prediction.predict(self.model, self.test_loader)
            self.logger.log_metrics({"test_acc": test_acc}, "epoch", self.model)
        self.logger.stop()

    def update_optimizer(self):
        self.optimizer = self.optimizer_fn(
            self.model.parameters(), **self.optimizer_args
        )
        self.learning_rate = optim.lr_scheduler.ExponentialLR(
            self.optimizer, self.learning_rate_decay
        )

    def train_epoch(self, epoch, optimizer, teacher):
        self.model = self.model.to(self.device)
        self.model.train()
        total_correct = 0
        total_size = 0
        for data, target in tqdm(
            self.train_loader,
            total=len(self.train_loader),
            desc=f"training epoch {epoch}",
        ):
            data, target = device.move(self.device, data, target)
            optimizer.zero_grad()
            loss, correct = self.compute_loss(data, target, teacher)
            loss.backward()
            optimizer.step()
            # self.logger.log_metrics({"train_acc": correct / data.size()[0]}, "batch")
            total_correct += correct
            total_size += data.size()[0]
        self.learning_rate.step()
        self.logger.log_metrics({"train_acc": total_correct / total_size}, "epoch")

    @staticmethod
    def get_logits(model, data):
        output = model(data)
        if isinstance(output, InceptionOutputs):
            output = output.logits
        return output

    def compute_loss(self, data, target, teacher):
        student_logits = Trainer.get_logits(self.model, data)
        correct = prediction.num_correct(student_logits, target)
        loss = F.cross_entropy(student_logits, target)
        self.logger.log_metrics({"cross_entropy_loss": loss.item()}, "batch")
        if teacher is not None:
            with torch.no_grad():
                teacher_logits = Trainer.get_logits(teacher, data)
            soft_targets = F.softmax(teacher_logits / self.T, dim=-1)
            soft_prob = F.log_softmax(student_logits / self.T, dim=-1)
            soft_targets_loss = (
                -torch.sum(soft_targets * soft_prob)
                / soft_prob.size()[0]
                * (self.T**2)
            )
            self.logger.log_metrics(
                {"soft_targets_loss": soft_targets_loss.item()}, "batch"
            )
            loss = (
                self.soft_target_loss_weight * soft_targets_loss
                + self.ce_loss_weight * loss
            )
        return loss, correct
