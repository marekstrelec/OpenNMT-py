
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

    parser.add_argument('--load', type=str, default=None, metavar='N')
    
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()


    # VARS
    INPUT_SIZE = 500
    OUTPUT_SIZE = 24725

    DATASET_MODE = 'dist'
    TEST_LOSS_FN = 'kl'

    # DATASET_MODE = 'max'
    # TEST_LOSS_FN = 'nll'

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if use_cuda else "cpu")

    # data_dir = Path("/local/scratch/ms2518/collected/explore")
    # data_paths = list(data_dir.glob("*.pickle"))
    data_paths = [Path("/local/scratch/ms2518/collected/explore/e0.pickle")]

    # PROCESS
    model = Net(input_size=INPUT_SIZE, output_size=OUTPUT_SIZE).to(device)
    
    models_dir = Path(args.load)
    model_paths = list(models_dir.glob("*.model"))
    model_paths = sorted(model_paths, key=lambda x: int(str(x).split('/')[-1].split('.')[0]))

    last_opt_save_path = None
    for model_path in model_paths:
        print("*** Loading model: {0}".format(str(model_path)))
        model.load_state_dict(torch.load(str(model_path)))

        sp_time = time.time()

        for pickle_idx, pickle_path in enumerate(data_paths):
            sp_time = time.time()

            # TEST
            shard_dataset = ExploreDataset(pickle_path, output_size=OUTPUT_SIZE, mode=DATASET_MODE)
            test_loader = DataLoader(shard_dataset, batch_size=64, shuffle=True, num_workers=4)

            test_model(args, TEST_LOSS_FN, model, device, test_loader)

            print("Dataset finished: {0:.2f}s".format(time.time() - sp_time))

        print("Dataset finished: {0:.2f}s".format(time.time() - sp_time))


if __name__ == "__main__":
    main()
