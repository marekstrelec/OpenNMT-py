
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

        self.dropout = nn.Dropout(p=0.5)

        self.fc1 = nn.Linear(input_size, 1000)
        self.fc2 = nn.Linear(1000, 500)
        self.fc3 = nn.Linear(500, 500)
        self.fc4_dist = nn.Linear(500, output_size)
        self.fc3_conf = nn.Linear(500, 1)

        self.sig = nn.Sigmoid()

    def forward(self, x):
        body = self.dropout(F.relu(self.fc1(x)))
        body = self.dropout(F.relu(self.fc2(body)))

        dist_head = self.dropout(F.relu(self.fc3(body)))
        dist_head = self.fc4_dist(dist_head)
        dist_head = F.log_softmax(dist_head, dim=1)

        conf_head = self.sig(self.fc3_conf(body))

        return dist_head, conf_head

    
def train_model(args, loss_fn, model, device, train_loader, optimizer, epoch, autoencoder, dataset_iter):
    assert loss_fn in ['kl', 'nll']

    train_loss_dist = 0
    train_loss_conf = 0
    correct_dist = 0
    correct_conf = 0
    model.train()
    for batch_idx, (data, (target_dist, target_conf)) in enumerate(train_loader):
        data, target_dist, target_conf = data.to(device), target_dist.to(device), target_conf.to(device)
        batch_size = data.shape[0]

        if autoencoder:
            data, _ = autoencoder(data)

        optimizer.zero_grad()
        out_dist, out_conf = model(data)

        criterion_conf = nn.BCELoss(reduction='mean')
        loss_conf = criterion_conf(out_conf.view(-1), target_conf)
        out_conf_binary = out_conf.view(-1) > 0.5
        correct_conf += out_conf_binary.eq(target_conf == 1).sum().item()
        train_loss_conf += loss_conf.item() * batch_size

        if loss_fn == "kl":
            loss_dist = F.kl_div(out_dist, target_dist, reduction='batchmean')

            target_max = target_dist.argmax(dim=1, keepdim=True)
            pred = out_dist.argmax(dim=1, keepdim=True)
            correct_dist += pred.eq(target_max.view_as(pred)).sum().item()
        elif loss_fn == "nll":
            target_dist = target_dist.long()
            loss_dist = F.nll_loss(out_dist, target_dist.long(), reduction='mean')

            pred = out_dist.argmax(dim=1, keepdim=True)
            correct_dist += pred.eq(target_dist.view_as(pred)).sum().item()

        train_loss_dist += loss_dist.item() * batch_size

        loss = loss_dist + loss_conf

        loss.backward()
        optimizer.step()

        if batch_idx > 0 and batch_idx % args.log_interval == 0:
            print('Train Epoch: {0} ({1}) [{2}/{3} ({4:.0f}%)]'.format(
                epoch,
                dataset_iter,
                batch_idx * len(data),
                len(train_loader.dataset),
                100. * batch_idx / len(train_loader))
            )
            sys.stdout.flush()

    return train_loss_dist, train_loss_conf, correct_dist, correct_conf

def test_model(args, loss_fn, model, device, test_loader, autoencoder, log=True):
    assert loss_fn in ['kl', 'nll']

    model.eval()
    test_loss_dist = 0
    test_loss_conf = 0
    correct_dist = 0
    correct_conf = 0
    with torch.no_grad():
        for data, (target_dist, target_conf) in test_loader:
            data, target_dist, target_conf = data.to(device), target_dist.to(device), target_conf.to(device)

            if autoencoder:
                data, _ = autoencoder(data)

            out_dist, out_conf = model(data)

            criterion_conf = nn.BCELoss(reduction='mean')
            test_loss_conf += criterion_conf(out_conf.view(-1), target_conf)
            out_conf_binary = out_conf.view(-1) > 0.5
            correct_conf += out_conf_binary.eq(target_conf == 1).sum().item()

            if loss_fn == "kl":
                test_loss_dist += F.kl_div(out_dist, target_dist, reduction='sum').item()
                target_max = target_dist.argmax(dim=1, keepdim=True)
                pred = out_dist.argmax(dim=1, keepdim=True)
                correct_dist += pred.eq(target_max.view_as(pred)).sum().item()

            elif loss_fn == "nll":
                target_dist = target_dist.long()
                test_loss_dist += F.nll_loss(out_dist, target_dist, reduction='sum').item()
                pred = out_dist.argmax(dim=1, keepdim=True)
                correct_dist += pred.eq(target_dist.view_as(pred)).sum().item()

    if log:
        print('\nTest set: Average dist loss: {0:.4f}, Average conf loss: {1:.4f}, Accuracy dist: {2}/{6} ({3:.0f}%), Accuracy conf: {4}/{6} ({5:.0f}%)\n'.format(
            test_loss_dist / len(test_loader.dataset),
            test_loss_conf / len(test_loader.dataset),
            correct_dist,
            100. * correct_dist / len(test_loader.dataset),
            correct_conf,
            100. * correct_conf / len(test_loader.dataset),
            len(test_loader.dataset))
        )
        sys.stdout.flush()

    return test_loss_dist, test_loss_conf, correct_dist, correct_conf
