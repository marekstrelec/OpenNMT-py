import sys
import os

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

from IPython import embed


class Autoencoder(nn.Module):
    def __init__(self, input_size):
        super(Autoencoder, self).__init__()
        self.input_size = input_size

        assert input_size == 500 * 3

        hid_layer = 350
        latent_dim = 200

        self.bn1 = nn.BatchNorm1d(num_features=1500)
        self.bn2 = nn.BatchNorm1d(num_features=1000)

        self.fc1 = nn.Linear(input_size, hid_layer)
        self.fc2 = nn.Linear(hid_layer, latent_dim)
        self.fc3 = nn.Linear(latent_dim, hid_layer)
        self.fc4 = nn.Linear(hid_layer, input_size)

        self.losses = []

    def should_stop(self, min_epochs=10, tolerance=0.0001):
        if len(self.losses) <= max(3, min_epochs):
            return False

        return abs(self.losses[-1]-self.losses[-2]) + abs(self.losses[-2]-self.losses[-3]) < tolerance

    def encoder(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc2(h1)

    def decoder(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.tanh(self.fc4(h3))

    def forward(self, x):
        enc = self.encoder(x)
        rec = self.decoder(enc)
        return enc, rec


def train_auto(args, loss_fn, model, device, train_loader, optimizer, epoch, dataset_iter):
    assert loss_fn in ['kl', 'nll']

    criterion = nn.MSELoss()

    model.train()

    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device)

        # ===================forward=====================
        _, recon_batch = model(data)
        loss = criterion(recon_batch, data)
        train_loss += loss.item()

        # ===================backward====================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch_idx > 0 and batch_idx % args.log_interval == 0:
            print('Train Epoch: {0} ({1}) [{2}/{3} ({4:.0f}%)]\tLoss: {5:.6f}'.format(
                epoch,
                dataset_iter,
                batch_idx * len(data),
                len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.item())
            )
            sys.stdout.flush()
    print('====> Epoch: {0} Average loss: {1:.4f}'.format(
          epoch,
          train_loss / len(train_loader)
          )
    )

    # save in the model for early stopping
    model.losses.append(train_loss / len(train_loader))



class VAE(nn.Module):
    def __init__(self, input_size):
        super(VAE, self).__init__()
        self.input_size = input_size

        assert input_size == 500

        hid_layer = 350
        latent_dim = 200

        self.fc1 = nn.Linear(input_size, hid_layer)
        self.fc21 = nn.Linear(hid_layer, latent_dim)
        self.fc22 = nn.Linear(hid_layer, latent_dim)
        self.fc3 = nn.Linear(latent_dim, hid_layer)
        self.fc4 = nn.Linear(hid_layer, input_size)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.tanh(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


def loss_function(recon_x, x, mu, logvar, input_size=500):
    # BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')

    reconstruction_function = nn.MSELoss(reduction='mean')
    BCE = reconstruction_function(recon_x, x)

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE, KLD


def train_vae(args, loss_fn, model, device, train_loader, optimizer, epoch, dataset_iter):
    assert loss_fn in ['kl', 'nll']

    model.train()

    train_loss = 0
    bce_loss = 0
    kld_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()

        recon_batch, mu, logvar = model(data)
        BCE, KLD = loss_function(recon_batch, data, mu, logvar)
        loss = BCE + KLD
        loss.backward()

        if epoch == 100:
            embed()
            sys.exit()

        train_loss += loss.item()
        bce_loss += BCE.item()
        kld_loss += KLD.item()
        optimizer.step()

        if batch_idx > 0 and batch_idx % args.log_interval == 0:
            print('Train Epoch: {0} ({1}) [{2}/{3} ({4:.0f}%)]\tLoss: {5:.6f}'.format(
                epoch,
                dataset_iter,
                batch_idx * len(data),
                len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.item())
            )
            sys.stdout.flush()
    print('====> Epoch: {0} Average loss: {1:.4f} .... {2:.4f} {3:.4f}'.format(
          epoch,
          train_loss / len(train_loader),
          bce_loss / len(train_loader),
          kld_loss / len(train_loader)
          )
    )