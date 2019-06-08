
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
from auto import Autoencoder


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
    parser.add_argument('--load', type=str, default=None, metavar='N')
    parser.add_argument('--start_epoch', type=int, default=None, metavar='N')
    parser.add_argument('--nooptload', action='store_true', default=False)
    parser.add_argument('--auto', type=str, default=None, metavar='N', required=False)

    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()


    # VARS
    INPUT_SIZE = 500 * 3  # + 100
    OUTPUT_SIZE = 24725

    # DATASET_MODE = 'dist'
    # TEST_LOSS_FN = 'kl'

    DATASET_MODE = 'max'
    TEST_LOSS_FN = 'nll'

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if use_cuda else "cpu")

    data_dir = Path("/local/scratch/ms2518/collected/")
    data_paths = list(data_dir.glob("*/*.pickle"))

    output_dir = Path(args.working_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

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

    # Load model
    if args.load:
        assert args.start_epoch
        print("* Loading model: {0}".format(args.load))
        model.load_state_dict(torch.load(args.load + ".model"))
        if not args.nooptload:
            print("* Loading optimiser: {0}".format(args.load))
            optimiser.load_state_dict(torch.load(args.load + ".opt"))

    # embed()
    # sys.exit(0)

    if args.start_epoch is None:
        args.start_epoch = 1

    last_opt_save_path = None
    for epoch in range(args.start_epoch, args.epochs + 1):
        print("\n<< EPOCH {0} >>".format(epoch))
        se_time = time.time()

        # shuffle datasets
        random.shuffle(data_paths)

        # load data
        acc_train_loss_dist = 0
        acc_train_loss_conf = 0
        acc_correct_dist = 0
        acc_correct_conf = 0
        acc_length = 0
        for pickle_idx, pickle_path in enumerate(data_paths):
            sp_time = time.time()

            # TRAIN
            shard_train_dataset = ExploreDataset(pickle_path, output_size=OUTPUT_SIZE, mode=DATASET_MODE, oversample=20)
            train_loader = DataLoader(shard_train_dataset, batch_size=128, shuffle=True, num_workers=4)

            res = train_model(args, TEST_LOSS_FN, model, device, train_loader, optimiser, epoch, autoencoder=model_auto, dataset_iter="{0}/{1}".format(pickle_idx+1, len(data_paths)))
            train_loss_dist, train_loss_conf, correct_dist, correct_conf = res
            acc_train_loss_dist += train_loss_dist
            acc_train_loss_conf += train_loss_conf
            acc_correct_dist += correct_dist
            acc_correct_conf += correct_conf
            acc_length += len(train_loader.dataset)

            print("Dataset finished: {0:.2f}s".format(time.time() - sp_time))

        print('\nTraining stats (e={0}): Average dist loss: {1:.4f}, Average conf loss: {2:.4f}, Accuracy dist: {3}/{7} ({4:.0f}%), Accuracy conf: {5}/{7} ({6:.0f}%)\n'.format(
            epoch,
            acc_train_loss_dist / acc_length,
            acc_train_loss_conf / acc_length,
            acc_correct_dist,
            100. * acc_correct_dist / acc_length,
            acc_correct_conf,
            100. * acc_correct_conf / acc_length,
            acc_length)
        )
        sys.stdout.flush()

        print("<< Epoch finished: {0:.2f}s >>\n".format(time.time() - se_time))

        save_time = int(time.time())
        model_save_path = output_dir.joinpath("{0}.{1}.model".format(epoch, save_time))
        opt_save_path = output_dir.joinpath("{0}.{1}.opt".format(epoch, save_time))
        torch.save(model.state_dict(), str(model_save_path))
        torch.save(optimiser.state_dict(), str(opt_save_path))

        if last_opt_save_path:
            last_opt_save_path.unlink()

        last_opt_save_path = opt_save_path


if __name__ == "__main__":
    main()
