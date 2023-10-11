import os
import torch
import torch.nn as nn
from safetensors.torch import load_file
from huggingface_hub import hf_hub_download


class Interposer(nn.Module):
	"""
		Basic NN layout, ported from:
		https://github.com/city96/SD-Latent-Interposer/blob/main/interposer.py
	"""
	version = 3.1 # network revision
	def __init__(self):
		super().__init__()
		self.chan = 4
		self.hid = 128

		self.head_join  = nn.ReLU()
		self.head_short = nn.Conv2d(self.chan, self.hid, kernel_size=3, stride=1, padding=1)
		self.head_long  = nn.Sequential(
			nn.Conv2d(self.chan, self.hid, kernel_size=3, stride=1, padding=1),
			nn.LeakyReLU(0.1),
			nn.Conv2d(self.hid,  self.hid, kernel_size=3, stride=1, padding=1),
			nn.LeakyReLU(0.1),
			nn.Conv2d(self.hid,  self.hid, kernel_size=3, stride=1, padding=1),
		)
		self.core = nn.Sequential(
			Block(self.hid),
			Block(self.hid),
			Block(self.hid),
		)
		self.tail = nn.Sequential(
			nn.ReLU(),
			nn.Conv2d(self.hid, self.chan, kernel_size=3, stride=1, padding=1)
		)

	def forward(self, x):
		y = self.head_join(
			self.head_long(x)+
			self.head_short(x)
		)
		z = self.core(y)
		return self.tail(z)

class Block(nn.Module):
	def __init__(self, size):
		super().__init__()
		self.join = nn.ReLU()
		self.long = nn.Sequential(
			nn.Conv2d(size, size, kernel_size=3, stride=1, padding=1),
			nn.LeakyReLU(0.1),
			nn.Conv2d(size, size, kernel_size=3, stride=1, padding=1),
			nn.LeakyReLU(0.1),
			nn.Conv2d(size, size, kernel_size=3, stride=1, padding=1),
		)
	def forward(self, x):
		y = self.long(x)
		z = self.join(y + x)
		return z


class LatentInterposer:
	def __init__(self):
		pass

	@classmethod
	def INPUT_TYPES(s):
		return {
			"required": {
				"samples": ("LATENT", ),
				"latent_src": (["v1", "xl"],),
				"latent_dst": (["v1", "xl"],),
			}
		}

	RETURN_TYPES = ("LATENT",)
	FUNCTION = "convert"
	CATEGORY = "latent"

	def convert(self, samples, latent_src, latent_dst):
		if latent_src == latent_dst:
			return (samples,)
		model = Interposer()
		model.eval()
		filename = f"{latent_src}-to-{latent_dst}_interposer-v{model.version}.safetensors"
		local = os.path.join(
			os.path.join(os.path.dirname(os.path.realpath(__file__)),"models"),
			filename
		)

		if os.path.isfile(local):
			print("LatentInterposer: Using local model")
			weights = local
		else:
			print("LatentInterposer: Using HF Hub model")
			weights = str(hf_hub_download(
				repo_id="city96/SD-Latent-Interposer",
				filename=filename)
			)

		model.load_state_dict(load_file(weights))
		lt = samples["samples"]
		lt = model(lt)
		del model
		return ({"samples": lt},)

NODE_CLASS_MAPPINGS = {
	"LatentInterposer": LatentInterposer,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LatentInterposer": "Latent Interposer"
}
