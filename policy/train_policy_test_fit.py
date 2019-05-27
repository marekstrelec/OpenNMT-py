
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

    parser.add_argument('--working_dir', type=str, default=None, metavar='N', required=False)
    parser.add_argument('--auto', type=str, default=None, metavar='N', required=False)
    
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()


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
    model_auto = None
    if args.auto:
        model_auto = Autoencoder(input_size=INPUT_SIZE).to(device)
        print("* Loading autoencoder model: {0}".format(args.auto))
        model_auto.load_state_dict(torch.load(args.auto + ".model"))

        INPUT_SIZE = 200

    # PROCESS
    model = Net(input_size=INPUT_SIZE, output_size=OUTPUT_SIZE).to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=True)  # https://medium.com/@Biboswan98/optim-adam-vs-optim-sgd-lets-dive-in-8dbf1890fbdc

    last_opt_save_path = None
    for epoch in range(1, args.epochs + 1):
        print("\n<< EPOCH {0} >>".format(epoch))
        se_time = time.time()

        # TRAIN
        pickle_path = Path("/local/scratch/ms2518/collected/e0_small20.pickle")
        shard_train_dataset = ExploreDataset(pickle_path, output_size=OUTPUT_SIZE, mode=DATASET_MODE)
        train_loader = DataLoader(shard_train_dataset, batch_size=128, shuffle=True, num_workers=4)

        train_model(args, TEST_LOSS_FN, model, device, train_loader, optimiser, epoch, autoencoder=model_auto, dataset_iter="{0}/{1}".format(1, 1))

        # TEST
        pickle_path = Path("/local/scratch/ms2518/collected/e0_small20.pickle")
        shard_dataset = ExploreDataset(pickle_path, output_size=OUTPUT_SIZE, mode=DATASET_MODE)
        test_loader = DataLoader(shard_dataset, batch_size=128, shuffle=True, num_workers=4)

        test_model(args, TEST_LOSS_FN, model, device, test_loader, autoencoder=model_auto)

        print("<< Epoch finished: {0:.2f}s >>\n".format(time.time() - se_time))



if __name__ == "__main__":
    main()
