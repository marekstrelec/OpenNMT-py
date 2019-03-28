
import sys
import pickle

from pathlib import Path
from IPython import embed


def collect_data(large_beam_data, small_beam_data, small_dec_data):
	embed()


def process_pickles(path_large, path_small):
	if not path_large.exists() or not path_small.exists():
		raise Exception("Missing input files!")

	with open(str(path_large), "rb") as f:
		large_beam_data = pickle.load(f)

	with open(str(path_small), "rb") as f:
		small_beam_data, small_dec_data = pickle.load(f)

	assert len(large_beam_data) == len(small_beam_data) == len(small_dec_data)
	
	data = []
	for lbd, sbd, sdd in zip(large_beam_data, small_beam_data, small_dec_data):
		res = collect_data(lbd, sbd, sdd)
		data.extend(res)

		break

	return data


if __name__ == "__main__":
	large_files = [n.name for n in Path("collected/large/").glob("*.pickle")]
	small_files = [n.name for n in Path("collected/small/").glob("*.pickle")]
	
	for lf in large_files:
		if lf not in small_files:
			raise Exception("Small file not found! {0}".format(str(lf)))

		process_pickles(Path("collected/large/").joinpath(lf), Path("collected/small/").joinpath(lf))