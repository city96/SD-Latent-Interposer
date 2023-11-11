# Custom dataset to load encoded latents from disk.
#  Files should contain latents as (1, C, H, W) or (C, H, W)
#  Latents should be in their original format without scaling

	######### Folder Layout #########
	#  latents                      #
	#   |- test_v1_768px.npy <=eval #
	#   |- test_xl_768px.npy <=^    #
	#   |- v1_768px <= ver/res      #
	#   |   |- 000001.npy           #
	#   |   |- 000002.npy           #
	#   |   |   ...                 #
	#   |   |- 000999.npy           #
	#   |   \- 001000.npy           #
	#   |- xl_768px                 #
	#     ...                       #
	#################################

import os
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset

DEFAULT_ROOT = "latents"
ALLOWED_EXTS = [".npy"]

class Shard:
	"""
	Shard to store groups of latents in
	paths: List containing paths to latent encoded images
	"""
	def __init__(self, paths):
		self.paths = paths
		self.data = None

	def exists(self):
		return all([os.path.isfile(x) for x in self.paths])

	def get_data(self):
		if self.data is not None: return self.data
		return tuple([self.load_latent(x) for x in self.paths])

	def load_latent(self, path):
		lat = torch.from_numpy(np.load(path))
		if lat.shape[0] == 1:
			lat = torch.squeeze(lat, 0)
		assert not torch.isnan(torch.sum(lat.float()))
		return lat

	def preload(self):
		self.data = self.get_data()

class LatentDataset(Dataset):
	def __init__(self, specs, res=768, root=DEFAULT_ROOT, preload=False):
		"""
		Main dataset that returns list of requested images as (C, H, W) latents
		specs: List of latent versions in the other to return them in
		res: Native resolution of images (before latent encoding)
		root: Path to folder with sorted files
		preload: Load all files into memory on initialization
		"""
		print("Dataset: Parsing data from disk")
		self.specs  = specs
		self.res    = res
		self.root   = root
		self.shards = []
		for fname in tqdm(os.listdir(f"{root}/{specs[0]}_{res}px")):
			name, ext = os.path.splitext(fname)
			if ext not in ALLOWED_EXTS: continue
			shard = Shard([f"{root}/{x}_{res}px/{name}{ext}" for x in specs])
			if shard.exists():
				self.shards.append(shard)

		if preload: # cache to RAM
			print("Dataset: Preloading data to system RAM")
			[x.preload() for x in tqdm(self.shards)]
		print(f"Dataset: OK, {len(self)} items")

	def __len__(self):
		return len(self.shards)

	def __getitem__(self, index):
		return self.shards[index].get_data()

	def get_eval(self):
		shard = Shard([f"{self.root}/test_{x}_{self.res}px.npy" for x in self.specs])
		data = shard.get_data() if shard.exists() else self[0]
		return tuple([x.unsqueeze(0).to(torch.float32) for x in data])
