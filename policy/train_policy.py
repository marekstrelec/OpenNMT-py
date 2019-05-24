
import sys
import os
import time
import random
import argparse

import torch
from torch.utils.data import Dataset, DataLoader
import pickle

import numpy as np

from collections import Counter
from pathlib import Path
from IPython import embed

from model import Net, train_model, test_model
from dataset import Dataset


def main():
    # Parser
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
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
    
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()


    # VARS
    INPUT_SIZE = 500
    OUTPUT_SIZE = 24725

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if use_cuda else "cpu")

    data_dir = Path("/local/scratch/ms2518/collected/explore")
    data_paths = list(data_dir.glob("*.pickle"))

    output_dir = Path("policy_models")
    output_dir.mkdir(exist_ok=True, parents=True)

    # PROCESS
    model = Net(input_size=INPUT_SIZE, output_size=OUTPUT_SIZE).to(device)
    # https://medium.com/@Biboswan98/optim-adam-vs-optim-sgd-lets-dive-in-8dbf1890fbdc
    optimiser = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=True)

    for epoch in range(1, args.epochs + 1):
        se_time = time.time()

        # shuffle datasets
        random.shuffle(data_paths)

        # load data
        for pickle_idx, pickle_path in enumerate(data_paths):
            sp_time = time.time()
            shard_dataset = Dataset(pickle_path, output_size=OUTPUT_SIZE, mode='dist')
            train_loader = DataLoader(shard_dataset, batch_size=64, shuffle=True, num_workers=4)

            train_model(args, model, device, train_loader, optimiser, epoch, dataset_iter="{0}/{1}".format(pickle_idx+1, len(data_paths)))
            # test_model(args, model, device, train_loader)

            print("Dataset finished: {0:.2f}s".format(time.time() - sp_time))

        print("<< Epoch finished: {0:.2f}s >> ".format(time.time() - se_time))

        torch.save(model.state_dict(), output_dir.joinpath("{0}.{1}.model".format(epoch, int(time.time()))))


if __name__ == "__main__":
    main()
