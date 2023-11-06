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


def predict(model, data_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in tqdm(
            data_loader, total=len(data_loader), desc=f"prediction epoch"
        ):
            data, target = device.move(device.get_device(), *batch)
            correct += num_correct(model(data), target)
            total += data.size()[0]
    return correct / total


def num_correct(output, target):
    pred = output.data.max(1, keepdim=True)[1]
    return pred.eq(target.data.view_as(pred)).cpu().sum().item()
