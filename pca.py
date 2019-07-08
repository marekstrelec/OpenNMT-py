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
from sklearn.decomposition import PCA, IncrementalPCA


def load_data(pickle_path):
        print("* Loading {0}".format(str(pickle_path)), end="\t")
        s_time = time.time()

        X = []
        with open(str(pickle_path), "rb") as f:
            data = pickle.load(f)


            for d_idx, d in enumerate(data):
                for b in d:
                    for t_idx, t in enumerate(b):
                        for origin in t:
                            t_vec = np.zeros(100, dtype=np.float32)
                            t_vec[t_idx] = 1.0

                            X.append(np.hstack([
                                origin['h_out'],
                                origin['d_out'],
                                origin['attn'],
                                t_vec
                            ]))

        print("\t(Loaded in {0:.2f}s)".format(time.time() - s_time))

        return X

def main():

    data_dir = Path("/local/scratch/ms2518/collected/")
    data_paths = list(data_dir.glob("run0/*.pickle"))
    data = []
    for pickle_idx, pickle_path in enumerate(data_paths):
        print("{0}/{1}".format(pickle_idx+1, len(data_paths)))
        data.extend(load_data(pickle_path))

    print("Total size: {0}".format(len(data)))

    print("Computing PCA")
    # model = PCA(n_components=518)
    model = IncrementalPCA(n_components=518, batch_size=None)
    # model.fit(data)
    chunk_size = 20000
    for i in range(0, len(data)//chunk_size):
        print("{0}/{1}".format(i, len(data)//chunk_size))
        model.partial_fit(data[i*chunk_size:(i+1)*chunk_size])
    
    variance = model.explained_variance_ratio_
    print("Done!")

    with open("pca.pickle", "wb") as f:
        pickle.dump(variance, f)

    with open("pca.model", "wb") as f:
        pickle.dump(model, f)



if __name__ == "__main__":
    main()