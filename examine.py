
import pickle
import sys

from collections import Counter
from pathlib import Path
from IPython import embed


def main():

	pickle_path = Path("/local/scratch/ms2518/collected/run0/e0.pickle")
	with open(str(pickle_path), "rb") as f:
		loaded_data = pickle.load(f)

	data = []

	for s_idx, s in enumerate(loaded_data):
		print("{0}/{1}".format(s_idx+1, len(loaded_data)))
		for ch in s:
			stop = False
			for t in ch:
				for b in t:
					if b['conf'] == 1.:
						word = Counter(b['vals']).most_common()[0][0][1]
						data.append(word)

						stop = True
						break
						
					if stop:
						break
				if stop:
					break


	embed()





if __name__ == "__main__":
	main()