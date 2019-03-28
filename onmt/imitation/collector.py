
import sys
import pickle

from pathlib import Path
from IPython import embed
from typing import List

from collections import Counter


class CollectorNode(object):

    def __init__(self, word_id, vocab_size=24725):
        self.word_id = word_id
        self.children = {}
        self.count = 1

    def prob(self):
        word_ids = []
        for k in self.children:
            for _ in range(self.children[k].count):
                word_ids.append(k)

        return word_ids

    def next_words(self, fields):
        word_ids = self.prob()

        return Counter(map(lambda x: fields['tgt'].base_field.vocab.itos[x], word_ids))

    def add_child(self, word_id):
        if word_id not in self.children:
            self.children[word_id] = CollectorNode(word_id)
        else:
            self.children[word_id].count += 1

    def has_child(self, word_id):
        return word_id in self.children

    def get_child(self, word_id):
        return self.children.get(word_id, None)


class Collector(object):

    def __init__(self, fields):
        self.fields = fields

    def build_graph(self, beam_data) -> "CollectorNode":

        graph_root = CollectorNode(None)

        for b in beam_data:
            node = graph_root
            for word_id in b[1]['hyp']:
                node.add_child(word_id)
                node = node.get_child(word_id)
                assert node is not None

        return graph_root

    def iterate_until_branch(self, graph) -> "CollectorNode":
        
        while True:
            probs = graph.prob()
            if not len(probs):
                break

            if probs.count(probs[0]) == len(probs):
                word = self.fields['tgt'].base_field.vocab.itos[probs[0]]
                print(word, end=" ")

                graph = graph.get_child(probs[0])

            else:
                break

        return graph

    def build_batch_graphs(self, explorer_data_path) -> List["CollectorNode"]:
        if not explorer_data_path.exists():
            raise Exception("Explored data not found!")

        graphs = []
        with open(str(explorer_data_path), "rb") as f:
            batch_beam_data, dec_data = pickle.load(f)

            for n in batch_beam_data:
                graph = self.build_graph(n)
                graphs.append(graph)

        return graphs

    def build_new_data(self, explorer_data_path, graphs):
        with open(str(explorer_data_path), "rb") as f:
            _, dec_data = pickle.load(f)

        embed()
        sys.exit(0)


    def process_collection(self, explorer_data_path, output_data_path):
        if output_data_path.exists():
            raise Exception("Output file already exists!")

        graphs = self.build_batch_graphs(explorer_data_path)
        data = self.build_new_data(explorer_data_path, graphs)

        with open(str(output_data_path), "wb") as f:
            pickle.dump(data, f)










if __name__ == "__main__":
    cltr = Collector()
    cltr.build_batch_graphs(Path("collected/large/e0.pickle"))




