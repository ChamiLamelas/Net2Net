import torch
import torch.nn.functional as F

def forward(model, data):
    model = model.cuda()
    data = data.cuda()
    model.eval()
    with torch.no_grad():
        return model(data)

def predict(model, data_loader, epoch, logger):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for (data, target) in data_loader:
            data = data.cuda()
            target = target.cuda()
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).cpu().sum().item()
            total += data.size()[0]
    logger.log_metrics({'epoch': epoch, 'test_acc': correct / total})
    logger.log_metrics({'epoch': epoch, 'test_loss': test_loss})
