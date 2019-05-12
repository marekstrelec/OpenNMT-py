
import sys
import pickle

from IPython import embed

if __name__ == "__main__":
	with open(sys.argv[1], "rb") as f:
		data = pickle.load(f)

	embed()