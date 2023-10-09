import torch
import torch.nn.functional as F

def forward(model, data):
    model = model.cuda()
    data = data.cuda()
    model.eval()
    with torch.no_grad():
        return model(data)

def predict(model, data_loader, epoch, logger, split):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for (data, target) in data_loader:
            data = data.cuda()
            target = target.cuda()
            output = model(data)
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).cpu().sum().item()
            total += data.size()[0]
    logger.log_metrics({'epoch': epoch, f'{split}_acc': correct / total})
