import os
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset

class FileLatentDataset(Dataset):
	def __init__(self, src_file, dst_file, device="cpu", dtype=torch.float16):
		assert os.path.isfile(src_file), f"src bin missing! ({src_file})"
		assert os.path.isfile(dst_file), f"dst bin missing! ({dst_file})"
		self.src_data = torch.load(src_file).to(dtype).to(device)
		self.dst_data = torch.load(dst_file).to(dtype).to(device)
		assert self.src_data.shape[0] == self.dst_data.shape[0], "Data size mismatch!"

	def __len__(self):
		return self.src_data.shape[0]

	def __getitem__(self, index):
		return {
			"src": self.src_data[index].float(),
			"dst": self.dst_data[index].float(),
		}

class Shard:
	def __init__(self, paths):
		self.paths = paths
		self.data = None

	def exists(self):
		return all([os.path.isfile(x) for x in self.paths.values()])

	def get_data(self):
		if self.data is not None: return self.data
		return {k:self.load_latent(v) for k,v in self.paths.items()}

	def load_latent(self, path):
		lat = torch.from_numpy(np.load(path))
		if lat.shape[0] == 1:
			lat = torch.squeeze(lat, 0)
		assert not torch.isnan(torch.sum(lat.float()))
		return lat

	def preload(self):
		self.data = self.get_data()

class LatentDataset(Dataset):
	def __init__(self, src_root, dst_root, preload=True):
		assert os.path.isdir(src_root), f"Source folder missing! ({src_root})"
		assert os.path.isdir(dst_root), f"Destination folder missing! ({dst_root})"

		print("Dataset: Parsing data from disk")
		fnames = list(
			set(os.listdir(src_root)).intersection(
			set(os.listdir(dst_root)))
		)
		assert len(fnames) > 0, "Source/destination have no overlapping files"

		self.shards = []
		for fname in tqdm(fnames):
			src_path = os.path.join(src_root, fname)
			dst_path = os.path.join(dst_root, fname)
			name, ext = os.path.splitext(fname)
			if ext not in [".npy"]:
				continue
			shard = Shard({
				"src": src_path,
				"dst": dst_path,
			})
			if shard.exists():
				self.shards.append(shard)
		assert len(self.shards) > 0, "No valid files found."

		if preload: # cache to RAM
			print("Dataset: Preloading data to system RAM")
			[x.preload() for x in tqdm(self.shards)]

		print(f"Dataset: OK, {len(self)} items")

	def __len__(self):
		return len(self.shards)

	def __getitem__(self, index):
		return self.shards[index].get_data()

def load_evals(evals):
	data = {}
	for name, paths in evals.items():
		shard = Shard(paths)
		assert shard.exists(), f"Eval data missing ({name})"
		data[name] = {}
		for k, v in shard.get_data().items():
			if len(v.shape) == 3:
				v = v.unsqueeze(0)
			data[name][k] = v.float()
	return data
