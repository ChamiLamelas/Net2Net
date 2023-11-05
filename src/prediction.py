"""
NEEDSWORK document
"""

import torch
import device
from tqdm import tqdm


def forward(model, data, eval=False):
    model, data = device.move(device.get_device(), model, data)
    if eval:
        model.eval()
    else:
        model.train()
    with torch.no_grad():
        return model(data)


def predict_batch(model, epoch, split, batch, logger):
    model.eval()
    with torch.no_grad():
        data, target = device.move(device.get_device(), *batch)
        correct = num_correct(model(data), target)
        total += data.size()[0]
        if logger is None:
            logger.log_metrics({f"{split}_acc": correct / total})
    return correct, total


def predict(model, data_loader, epoch, epoch_logger, batch_logger, split):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in tqdm(
            data_loader, total=len(data_loader), desc=f"prediction epoch {epoch}"
        ):
            batch_correct, batch_total = predict_batch(
                model, epoch, split, batch, batch_logger
            )
            correct += batch_correct
            total += batch_total
    acc = correct / total
    epoch_logger.log_metrics({"epoch": epoch, f"{split}_acc": acc})
    return acc


def num_correct(output, target):
    pred = output.data.max(1, keepdim=True)[1]
    return pred.eq(target.data.view_as(pred)).cpu().sum().item()
