import os

from utils import PlotLearning
import numpy as np
import copy

import argparse
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import sys
sys.path.append('../')

from logger import ML_Logger, MyTimer
from net2net import wider, deeper


# Training settings
parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=80, metavar='N',
                    help='number of epochs to train (default: 40)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum (default: 0.9)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=1000, metavar='N',
                    help='how many batches to wait before logging status')
parser.add_argument('--noise', type=int, default=1,
                    help='noise or no noise 0-1')
parser.add_argument('--weight_norm', type=int, default=1,
                    help='norm or no weight norm 0-1')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

train_transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

test_transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


train_loader = torch.utils.data.DataLoader(
    datasets.CIFAR10('./data', train=True, download=True,
                     transform=train_transform),
    batch_size=args.batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(
    datasets.CIFAR10('./data', train=False, transform=test_transform),
    batch_size=args.test_batch_size, shuffle=True)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 8, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(8)
        self.pool1 = nn.MaxPool2d(3, 2)
        self.conv2 = nn.Conv2d(8, 16, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(16)
        self.pool2 = nn.MaxPool2d(3, 2)
        self.conv3 = nn.Conv2d(16, 32, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(32)
        self.pool3 = nn.AvgPool2d(5, 1)
        self.fc1 = nn.Linear(32 * 3 * 3, 10)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, np.sqrt(2. / n))
                m.bias.data.fill_(0.0)
            if isinstance(m, nn.Linear):
                m.bias.data.fill_(0.0)

    def forward(self, x):
        try:
            x = self.conv1(x)
            x = self.bn1(x)
            x = F.relu(x)
            x = self.pool1(x)
            x = self.conv2(x)
            x = self.bn2(x)
            x = F.relu(x)
            x = self.pool2(x)
            x = self.conv3(x)
            x = self.bn3(x)
            x = F.relu(x)
            x = self.pool3(x)
            x = x.view(-1, x.size(1) * x.size(2) * x.size(3))
            x = self.fc1(x)
            return F.log_softmax(x, dim=1)
        except RuntimeError:
            print(x.size())

    def net2net_wider(self):
        self.conv1, self.conv2, _ = wider(self.conv1, self.conv2, 12,
                                          self.bn1, noise=args.noise)
        self.conv2, self.conv3, _ = wider(self.conv2, self.conv3, 24,
                                          self.bn2, noise=args.noise)
        self.conv3, self.fc1, _ = wider(self.conv3, self.fc1, 48,
                                        self.bn3, noise=args.noise)

    def net2net_deeper(self):
        s = deeper(self.conv1, nn.ReLU, bnorm_flag=True,
                   weight_norm=args.weight_norm, noise=args.noise)
        self.conv1 = s
        s = deeper(self.conv2, nn.ReLU, bnorm_flag=True,
                   weight_norm=args.weight_norm, noise=args.noise)
        self.conv2 = s
        s = deeper(self.conv3, nn.ReLU, bnorm_flag=True,
                   weight_norm=args.weight_norm, noise=args.noise)
        self.conv3 = s

    def net2net_deeper_nononline(self):
        s = deeper(self.conv1, None, bnorm_flag=True,
                   weight_norm=args.weight_norm, noise=args.noise)
        self.conv1 = s
        s = deeper(self.conv2, None, bnorm_flag=True,
                   weight_norm=args.weight_norm, noise=args.noise)
        self.conv2 = s
        s = deeper(self.conv3, None, bnorm_flag=True,
                   weight_norm=args.weight_norm, noise=args.noise)
        self.conv3 = s

    def define_wider(self):
        self.conv1 = nn.Conv2d(3, 12, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(12)
        self.conv2 = nn.Conv2d(12, 24, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(24)
        self.conv3 = nn.Conv2d(24, 48, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(48)
        self.fc1 = nn.Linear(48*3*3, 10)

    def define_wider_deeper(self):
        self.conv1 = nn.Sequential(nn.Conv2d(3, 12, kernel_size=3, padding=1),
                                   nn.BatchNorm2d(12),
                                   nn.ReLU(),
                                   nn.Conv2d(12, 12, kernel_size=3, padding=1))
        self.bn1 = nn.BatchNorm2d(12)
        self.conv2 = nn.Sequential(nn.Conv2d(12, 24, kernel_size=3, padding=1),
                                   nn.BatchNorm2d(24),
                                   nn.ReLU(),
                                   nn.Conv2d(24, 24, kernel_size=3, padding=1))
        self.bn2 = nn.BatchNorm2d(24)
        self.conv3 = nn.Sequential(nn.Conv2d(24, 48, kernel_size=3, padding=1),
                                   nn.BatchNorm2d(48),
                                   nn.ReLU(),
                                   nn.Conv2d(48, 48, kernel_size=3, padding=1))
        self.bn3 = nn.BatchNorm2d(48)
        self.fc1 = nn.Linear(48*3*3, 10)


def net2net_deeper_recursive(model):
    """
    Apply deeper operator recursively any conv layer.
    """
    for name, module in model._modules.items():
        if isinstance(module, nn.Conv2d):
            s = deeper(module, nn.ReLU, bnorm_flag=False)
            model._modules[name] = s
        elif isinstance(module, nn.Sequential):
            module = net2net_deeper_recursive(module)
            model._modules[name] = module
    return model


def train(epoch):
    model.train()
    avg_loss = 0
    avg_accu = 0
    total_loss = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        # get the index of the max log-probability
        pred = output.data.max(1, keepdim=True)[1]
        avg_accu += pred.eq(target.data.view_as(pred)).cpu().sum()
        avg_loss += loss.item()
        if batch_idx % args.log_interval == 0:
            LOGGER.debug('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
        total_loss += loss.item()
    avg_loss /= batch_idx + 1
    avg_accu = avg_accu / len(train_loader.dataset)
    LOGGER.log_metrics({'epoch': epoch, 'train_loss': total_loss})
    return avg_accu, avg_loss


def test(epoch):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            if args.cuda:
                data, target = data.cuda(), target.cuda()
            output = model(data)
            # sum up batch loss
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            # get the index of the max log-probability
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    test_loss /= len(test_loader.dataset)
    accuracy = (100. * correct / len(test_loader.dataset)).item()
    LOGGER.debug('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        accuracy))
    LOGGER.log_metrics({'epoch': epoch, 'test_acc': accuracy})
    return correct / len(test_loader.dataset), test_loss


def run_training(model, run_name,  epoch_end, epoch_start=1, plot=None):
    global optimizer
    model.cuda()
    optimizer = optim.SGD(model.parameters(), lr=args.lr,
                          momentum=args.momentum)
    if plot is None:
        plot = PlotLearning('./plots/cifar/', 10, prefix=run_name)
    for epoch in range(epoch_start, epoch_end):
        accu_train, loss_train = train(epoch)
        accu_test, loss_test = test(epoch)
        logs = {}
        logs['acc'] = accu_train
        logs['val_acc'] = accu_test
        logs['loss'] = loss_train
        logs['val_loss'] = loss_test
        plot.plot(logs)
    return plot

def do_single_forward(model):
    model.eval()
    with torch.no_grad():
        data, _ = next(iter(test_loader))
        return model(data.cuda())


if __name__ == "__main__":
    TOTAL_TIMER = MyTimer()
    TOTAL_TIMER.start('train_cifar10')

    LOGGER = ML_Logger(log_folder=os.path.join('logs', 'cifar10'), persist=False)

    # start_t = time.time()
    # LOGGER.start(log_file='teacher_training', task='Teacher Training')
    # model = Net()
    # model.cuda()
    criterion = nn.NLLLoss()
    # plot = run_training(model, 'Teacher_', args.epochs + 1)
    # LOGGER.stop()

    # # wider student training
    # LOGGER.start(log_file='wider_student', task="Wider Student training ... ")
    # model_ = Net()
    # model_ = copy.deepcopy(model)

    # del model
    # model = model_
    # model.net2net_wider()
    # plot = run_training(model, 'Wider_student_', args.epochs + 1, plot)
    # LOGGER.stop()

    # # wider + deeper student training
    # LOGGER.start(log_file='wider_deeper_student',
    #              task="Wider+Deeper Student training ... ")
    # model_ = Net()
    # model_.net2net_wider()
    # model_ = copy.deepcopy(model)

    # del model
    # model = model_
    # model.net2net_deeper_nononline()
    # run_training(model, 'WiderDeeper_student_', args.epochs + 1, plot)
    # LOGGER.stop()

    # # wider teacher training
    # start_t = time.time()
    # LOGGER.start(log_file='wider_teacher', task="Wider teacher training ... ")
    # model_ = Net()

    # del model
    # model = model_
    # model.define_wider()
    # model.cuda()
    # run_training(model, 'Wider_teacher_', args.epochs + 1)
    # LOGGER.stop()

    # # wider deeper teacher training
    # LOGGER.start(log_file='wider_deeper_teacher',
    #              task="Wider+Deeper teacher training ... ")

    # start_t = time.time()
    # model_ = Net()

    # del model
    # model = model_
    # model.define_wider_deeper()
    # run_training(model, 'Wider_Deeper_teacher_', args.epochs + 1)
    # LOGGER.stop()

    LOGGER.start(log_file='dynamic_wider_training', task='Dynamic Wider Training')

    model = Net()
    model.cuda()


    total = args.epochs + 1
    mid = total // 2

    run_training(model, 'dynamic_wider_training', mid)

    before_transfer = do_single_forward(model)

    model_ = Net()
    model_ = copy.deepcopy(model)

    del model
    model = model_
    model.net2net_wider()
    model.cuda()

    after_transfer = do_single_forward(model)

    print(f"MSE: {F.mse_loss(before_transfer, after_transfer).item()}")

    run_training(model, 'dynamic_wider_training', total, mid)

    LOGGER.stop()

    LOGGER.start(log_file='dynamic_wider_deeper_training', task='Dynamic Wider Deeper Training')

    model = Net()
    model.cuda()


    total = args.epochs + 1
    mid = total // 2

    run_training(model, 'dynamic_wider_deeper_training', mid)

    before_transfer = do_single_forward(model)

    model_ = Net()
    model_.net2net_wider()
    model_ = copy.deepcopy(model)

    del model
    model = model_
    model.net2net_deeper_nononline()
    model.cuda()

    after_transfer = do_single_forward(model)

    print(f"MSE: {F.mse_loss(before_transfer, after_transfer).item()}")

    run_training(model, 'dynamic_wider_deeper_training', total, mid)
    LOGGER.stop()

    TOTAL_TIMER.stop()
