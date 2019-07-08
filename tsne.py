
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


def load_data(pickle_path):
        print("* Loading {0}".format(str(pickle_path)), end="\t")
        s_time = time.time()

        X_pos = []
        X_neg = []
        with open(str(pickle_path), "rb") as f:
            data = pickle.load(f)


            for d_idx, d in enumerate(data):
                for b in d:
                    for t_idx, t in enumerate(b):
                        for origin in t:
                            t_vec = np.zeros(100, dtype=np.float32)
                            t_vec[t_idx] = 1.0

                            if origin['conf'] == 1.0:
                                X_pos.append(np.hstack([
                                    origin['h_out'],
                                    origin['d_out'],
                                    origin['attn'],
                                    t_vec
                                ]))
                            else:
                                X_neg.append(np.hstack([
                                    origin['h_out'],
                                    origin['d_out'],
                                    origin['attn'],
                                    t_vec
                                ]))

                if len(X_pos) >= 2000:
                    break

        print("\t(Loaded in {0:.2f}s)".format(time.time() - s_time))

        random.shuffle(X_neg)
        X_neg = X_neg[:int(len(X_pos)*2)]

        return {
            'pos': X_pos,
            'neg': X_neg,
        }

def main():
    pickle_path = Path("/local/scratch/ms2518/collected/run0/e0.pickle")
    data = load_data(pickle_path)

    print(len(data['pos']))
    print(len(data['neg']))

    embed()
    sys.exit(0)

    # data_conf = [n['conf'] for n in data['targets']]
    # first_n=4000
    # second_n=2000
    # negative_data = []
    # negative_c = []
    # positive_data = []
    # positive_c = []
    # for v, c in zip(data['inputs'], data_conf):
    #     if c == 0:
    #         negative_data.append(v)
    #         negative_c.append(0)
    #     else:
    #         positive_data.append(v)
    #         positive_c.append(3)
    # input_data = negative_data[:first_n] + positive_data[:second_n]
    # input_data = np.asarray(input_data)

    # s_time = time.time()
    # data_conf = negative_c[:first_n] + positive_c[:second_n]
    # tsne_data = TSNE(n_components=2, init='pca', random_state=0, n_iter=1500, verbose=1, perplexity=150).fit_transform(input_data)
    # print("\t(t-SNE finished in {0:.2f}s)".format(time.time() - s_time))

    s_time = time.time()
    input_data = data['neg'] + data['pos']
    input_data = np.asarray(input_data)
    tsne_data = TSNE(n_components=2, init='pca', random_state=0, n_iter=2000, verbose=1, perplexity=150).fit_transform(input_data)
    print("\t(t-SNE finished in {0:.2f}s)".format(time.time() - s_time))


    with open("tsne.pickle", "wb") as f:
        pickle.dump([tsne_data, len(data['neg']), len(data['pos'])], f)






if __name__ == "__main__":
    main()