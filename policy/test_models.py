
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

from tqdm import tqdm


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
    parser.add_argument('--auto', type=str, default=None, metavar='N', required=False)
    
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

    data_dir = Path("/local/scratch/ms2518/collected/")
    data_paths = list(data_dir.glob("*/*.pickle"))
    # data_paths = [Path("/local/scratch/ms2518/collected/explore/e0.pickle")]

    # AUTOENCODER
    model_auto = None
    if args.auto:
        model_auto = Autoencoder(input_size=INPUT_SIZE).to(device)
        print("* Loading autoencoder model: {0}".format(args.auto))
        model_auto.load_state_dict(torch.load(args.auto + ".model"))

        INPUT_SIZE = 200

    # PROCESS
    model = Net(input_size=INPUT_SIZE, output_size=OUTPUT_SIZE).to(device)
    
    # models_dir = Path(args.load)
    # model_paths = list(models_dir.glob("*.model"))
    model_paths = [Path("policy_models/run0/9.1558998717.model")]
    model_paths = sorted(model_paths, key=lambda x: int(str(x).split('/')[-1].split('.')[0]))

    last_opt_save_path = None
    for model_path in model_paths:
        print("*** Loading model: {0}".format(str(model_path)), end="\t")
        s_time = time.time()
        model.load_state_dict(torch.load(str(model_path)))
        print("\t(Loaded in {0:.2f}s)".format(time.time() - s_time))

        sp_time = time.time()

        acc_loss_dist = 0
        acc_loss_conf = 0
        acc_correct_dist = 0
        acc_correct_conf = 0
        acc_length = 0
        with tqdm(total=len(data_paths)) as pbar:
            for pickle_idx, pickle_path in enumerate(data_paths):
                sp_time = time.time()

                # TEST
                shard_dataset = ExploreDataset(pickle_path, output_size=OUTPUT_SIZE, mode=DATASET_MODE)
                test_loader = DataLoader(shard_dataset, batch_size=128, shuffle=True, num_workers=4)

                test_loss_dist, test_loss_conf, correct_dist, correct_conf = test_model(args, TEST_LOSS_FN, model, device, test_loader, autoencoder=model_auto, log=False)
                acc_loss_dist += test_loss_dist
                acc_loss_conf += test_loss_conf
                acc_correct_dist += correct_dist
                acc_correct_conf += correct_conf
                acc_length += len(test_loader.dataset)

                pbar.update(1)

        print("Test finished: {0:.2f}s".format(time.time() - sp_time))

        print('\nTest set: Average dist loss: {0:.4f}, Average conf loss: {1:.4f}, Accuracy dist: {2}/{6} ({3:.0f}%), Accuracy conf: {4}/{6} ({5:.0f}%)\n'.format(
            acc_loss_dist / acc_length,
            acc_loss_conf / acc_length,
            acc_correct_dist,
            100. * acc_correct_dist / acc_length,
            acc_correct_conf,
            100. * acc_correct_conf / acc_length,
            acc_length)
        )
        sys.stdout.flush()


if __name__ == "__main__":
    main()
