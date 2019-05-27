
import sys
import os
import time
import random
import argparse

import torch
from torch.utils.data import DataLoader
import pickle

import numpy as np

from collections import Counter
from pathlib import Path
from IPython import embed

from model import Net, train_model, test_model
from auto import Autoencoder, VAE, train_auto, train_vae
from dataset import ExploreDataset


def main():
    # Parser
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=100, metavar='N',
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=500, metavar='N',
                        help='how many batches to wait before logging training status')

    parser.add_argument('--working_dir', type=str, default=None, metavar='N', required=True)
    parser.add_argument('--model_name', type=str, default=None, metavar='N', required=True)
    
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()


    output_dir = Path(args.working_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    # VARS
    INPUT_SIZE = 500
    OUTPUT_SIZE = 24725

    # DATASET_MODE = 'dist'
    # TEST_LOSS_FN = 'kl'

    DATASET_MODE = 'max'
    TEST_LOSS_FN = 'nll'

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if use_cuda else "cpu")

    # AUTOENCODER
    model_auto = Autoencoder(input_size=INPUT_SIZE).to(device)
    optimiser_auto = torch.optim.Adam(model_auto.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=True)  # https://medium.com/@Biboswan98/optim-adam-vs-optim-sgd-lets-dive-in-8dbf1890fbdc

    # TRAIN AUTOENCODER
    for epoch in range(1, args.epochs + 1):
        print("\n<< EPOCH {0} >>".format(epoch))
        se_time = time.time()

        pickle_path = Path("/local/scratch/ms2518/collected/e0_small20.pickle")
        shard_train_dataset = ExploreDataset(pickle_path, output_size=OUTPUT_SIZE, mode=DATASET_MODE)
        train_loader = DataLoader(shard_train_dataset, batch_size=128, shuffle=True, num_workers=4)

        train_auto(args, TEST_LOSS_FN, model_auto, device, train_loader, optimiser_auto, epoch, dataset_iter="{0}/{1}".format(1, 1))

        if model_auto.should_stop():
            print("Early stopping!")
            break

    # SAVE THE MODEL
    model_save_path = output_dir.joinpath("{0}".format(args.model_name))
    torch.save(model_auto.state_dict(), str(model_save_path))


if __name__ == "__main__":
    main()
