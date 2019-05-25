
import sys
import os

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

from IPython import embed


class Net(nn.Module):
    def __init__(self, input_size, output_size):
        super(Net, self).__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.fc1 = nn.Linear(input_size, 500)
        self.fc2 = nn.Linear(500, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)
    
def train_model(args, loss_fn, model, device, train_loader, optimizer, epoch, dataset_iter):
    assert loss_fn in ['kl', 'nll']

    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)

        if loss_fn == "kl":
            loss = F.kl_div(output, target, reduction='batchmean')
        elif loss_fn == "nll":
            loss = F.nll_loss(output, target.long())

        loss.backward()
        optimizer.step()

        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {0} ({1}) [{2}/{3} ({4:.0f}%)]\tLoss: {5:.6f}'.format(
                epoch,
                dataset_iter,
                batch_idx * len(data),
                len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.item())
            )
            sys.stdout.flush()

def test_model(args, loss_fn, model, device, test_loader):
    assert loss_fn in ['kl', 'nll']

    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)

            if loss_fn == "kl":
                test_loss += F.kl_div(output, target, reduction='sum').item()
                target_max = target.argmax(dim=1, keepdim=True)
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target_max.view_as(pred)).sum().item()

            elif loss_fn == "nll":
                target = target.long()
                test_loss += F.nll_loss(output, target, reduction='sum').item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss,
        correct,
        len(test_loader.dataset),
        100. * correct / len(test_loader.dataset))
    )
    sys.stdout.flush()
