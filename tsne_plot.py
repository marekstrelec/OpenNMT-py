
import sys
import os
import time
import random
import argparse

import torch
from torch.utils.data import DataLoader
import pickle

import numpy as np
import matplotlib.pyplot as plt

from collections import Counter
from pathlib import Path
from IPython import embed

from sklearn.manifold import TSNE


def main():

	with open("tsne.pickle", "rb") as f:
		tsne_data, data_conf = pickle.load(f)

		negative_data = []
		negative_c = []
		positive_data = []
		positive_c = []
		for v, c in zip(tsne_data, data_conf):
			if c == 0:
				negative_data.append(v)
				negative_c.append(0)
			else:
				positive_data.append(v)
				positive_c.append(3)

		negative_data = np.asarray(negative_data)
		positive_data = np.asarray(positive_data)

		# embed()

		
		plt.scatter(negative_data[:, 0], negative_data[:, 1], c="r", alpha=0.5, zorder=0)
		plt.scatter(positive_data[:, 0], positive_data[:, 1], c="b", alpha=0.8, zorder=100)
		
		plt.savefig("tsne.png")


if __name__ == "__main__":
    main()