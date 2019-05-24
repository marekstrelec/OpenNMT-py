
import time
import pickle

import numpy as np
import torch
from torch.utils import data

from collections import Counter


class Dataset(data.Dataset):

    def __init__(self, pickle_path, output_size, mode):
        assert pickle_path.exists()
        assert output_size > 0
        assert mode in ['onehot', 'dist']

        self.pickle_path = pickle_path
        self.output_size = output_size

        self.data = self.load_data(self.pickle_path)
        self.mode = mode

    def load_data(self, pickle_path):
        print("Loading {0}".format(str(pickle_path)))
        s_time = time.time()

        X = []
        Y = []
        with open(str(pickle_path), "rb") as f:
            data = pickle.load(f)

            for d_idx, d in enumerate(data):
                for b in d:
                    for t in b:
                        for origin in t:
                            X.append(origin['dec'])
                            Y.append(origin['vals'])

        print("Loaded in {0:.2f}s".format(time.time() - s_time))

        assert len(X) == len(Y)
        return {
            'inputs': X,
            'targets': Y
        }

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.data['inputs'])

    def __getitem__(self, index):
        'Generates one sample of data'

        X = self.data['inputs'][index]
        targets = self.data['targets'][index]
        
        vec = np.zeros(self.output_size, dtype=np.float32)
        cnt = Counter(targets)
        if self.mode == "onehot":
            word_id = cnt.most_common()[0][0][0]
            vec[word_id] = 1.
        elif self.mode == "dist":
            denom = sum(cnt.values())
            for k, v in cnt.items():
                vec[k[0]] = v / denom

        return X, vec