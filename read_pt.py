
import sys
import torch, errno


if __name__ == "__main__":
	

	try:
		data = torch.load(sys.argv[1])
		for d in list(data):
			src = " ".join(d.src[0])
			tgt = " ".join(d.tgt[0])

			print("{0}\t{1}".format(src, tgt))

	except BrokenPipeError as e:
		pass