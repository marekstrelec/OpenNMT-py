
import time
import pickle

import numpy as np
import torch
from torch.utils import data

from collections import Counter
from IPython import embed


class ExploreDataset(data.Dataset):

    def __init__(self, pickle_path, output_size, mode, oversample):
        assert pickle_path.exists()
        assert output_size > 0
        assert mode in ['onehot', 'max', 'dist']
        assert oversample > 0

        self.pickle_path = pickle_path
        self.output_size = output_size
        self.mode = mode
        self.oversample = oversample

        self.data = self.load_data(self.pickle_path)

    def load_data(self, pickle_path):
        print("* Loading {0}".format(str(pickle_path)), end="\t")
        s_time = time.time()

        X = []
        Y = []
        with open(str(pickle_path), "rb") as f:
            data = pickle.load(f)

            for d_idx, d in enumerate(data):
                for b in d:
                    # conf_used = False
                    for t_idx, t in enumerate(b):
                        for origin in t:

                            repeat = 1
                            if origin['conf'] == 1.0:
                                repeat = self.oversample

                            for _ in range(repeat):
                                t_vec = np.zeros(100, dtype=np.float32)
                                t_vec[t_idx] = 1.0
                                X.append(np.hstack([
                                    origin['h_out'],
                                    origin['d_out'],
                                    origin['attn'],
                                    # t_vec
                                ]))

                                Y.append({
                                    'dist': origin['vals'],
                                    'conf': origin['conf']
                                })

        print("\t(Loaded in {0:.2f}s)".format(time.time() - s_time))

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
        targets = self.data['targets'][index]['dist']
        conf_score = np.asarray(self.data['targets'][index]['conf'], dtype=np.float32)
        
        vec = np.zeros(self.output_size, dtype=np.float32)
        cnt = Counter(targets)
        if self.mode == "onehot":
            word_id = cnt.most_common()[0][0][0]
            vec[word_id] = 1.
        elif self.mode == "max":
            word_id = cnt.most_common()[0][0][0]
            vec = np.array(word_id, dtype=np.float32)
        elif self.mode == "dist":
            denom = sum(cnt.values())
            for k, v in cnt.items():
                vec[k[0]] = v / denom

        return X, (vec, conf_score)