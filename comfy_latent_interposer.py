import os
import torch
import torch.nn as nn
from safetensors.torch import load_file
from huggingface_hub import hf_hub_download

# v1 = Stable Diffusion 1.x
# xl = Stable Diffusion Extra Large (SDXL)
# cc = Stable Cascade (Stage C) [not used]
# ca = Stable Cascade (Stage A/B)
config = {
	"v1-to-xl": {"ch_in": 4, "ch_out": 4, "ch_mid": 64, "scale": 1.0, "blocks": 12},
	"xl-to-v1": {"ch_in": 4, "ch_out": 4, "ch_mid": 64, "scale": 1.0, "blocks": 12},
	"ca-to-v1": {"ch_in": 4, "ch_out": 4, "ch_mid": 64, "scale": 0.5, "blocks": 12},
	"ca-to-xl": {"ch_in": 4, "ch_out": 4, "ch_mid": 64, "scale": 0.5, "blocks": 12},
	"v3-to-v1": {"ch_in":16, "ch_out": 4, "ch_mid": 64, "scale": 1.0, "blocks": 12},
	"v3-to-xl": {"ch_in":16, "ch_out": 4, "ch_mid": 64, "scale": 1.0, "blocks": 12},
}

class ResBlock(nn.Module):
	"""Block with residuals"""
	def __init__(self, ch):
		super().__init__()
		self.join = nn.ReLU()
		self.norm = nn.BatchNorm2d(ch)
		self.long = nn.Sequential(
			nn.Conv2d(ch, ch, kernel_size=3, stride=1, padding=1),
			nn.SiLU(),
			nn.Conv2d(ch, ch, kernel_size=3, stride=1, padding=1),
			nn.SiLU(),
			nn.Conv2d(ch, ch, kernel_size=3, stride=1, padding=1),
			nn.Dropout(0.1)
		)
	def forward(self, x):
		x = self.norm(x)
		return self.join(self.long(x) + x)

class ExtractBlock(nn.Module):
	"""Increase no. of channels by [out/in]"""
	def __init__(self, ch_in, ch_out):
		super().__init__()
		self.join  = nn.ReLU()
		self.short = nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1)
		self.long  = nn.Sequential(
			nn.Conv2d( ch_in, ch_out, kernel_size=3, stride=1, padding=1),
			nn.SiLU(),
			nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1),
			nn.SiLU(),
			nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1),
			nn.Dropout(0.1)
		)
	def forward(self, x):
		return self.join(self.long(x) + self.short(x))

class InterposerModel(nn.Module):
	"""
	NN layout, ported from:
	https://github.com/city96/SD-Latent-Interposer/blob/main/interposer.py
	"""
	def __init__(self, ch_in=4, ch_out=4, ch_mid=64, scale=1.0, blocks=12):
		super().__init__()
		self.ch_in  = ch_in
		self.ch_out = ch_out
		self.ch_mid = ch_mid
		self.blocks = blocks
		self.scale  = scale

		self.head = ExtractBlock(self.ch_in, self.ch_mid)
		self.core = nn.Sequential(
			nn.Upsample(scale_factor=self.scale, mode="nearest"),
			*[ResBlock(self.ch_mid) for _ in range(blocks)],
			nn.BatchNorm2d(self.ch_mid),
			nn.SiLU(),
		)
		self.tail = nn.Conv2d(self.ch_mid, self.ch_out, kernel_size=3, stride=1, padding=1)

	def forward(self, x):
		y = self.head(x)
		z = self.core(y)
		return self.tail(z)

class ComfyLatentInterposer:
	"""Custom node"""
	def __init__(self):
		self.version = 4.0 # network revision
		self.loaded = None # current model name
		self.model = None  # current model

	@classmethod
	def INPUT_TYPES(s):
		return {
			"required": {
				"samples": ("LATENT", ),
				"latent_src": (["v1", "xl", "v3", "ca"],),
				"latent_dst": (["v1", "xl"],),
			}
		}

	RETURN_TYPES = ("LATENT",)
	FUNCTION     = "convert"
	CATEGORY     = "latent"
	TITLE        = "Latent Interposer"

	def get_model_path(self, model_name):
		fname = f"{model_name}_interposer-v{self.version}.safetensors"
		path  = os.path.join(os.path.dirname(os.path.realpath(__file__)),"models")

		# local path: [models/xl-to-v1_interposer-v4.2.safetensors]
		if os.path.isfile(os.path.join(path, fname)):
			print("LatentInterposer: Using local model")
			return os.path.join(path, fname)

		# local path: [models/v4.2/xl-to-v1_interposer-v4.2.safetensors]
		if os.path.isfile(os.path.join(path, os.path.join(f"v{self.version}", fname))):
			print("LatentInterposer: Using local model")
			return os.path.join(path, os.path.join(f"v{self.version}", fname))

		# huggingface hub fallback
		print("LatentInterposer: Using HF Hub model")
		return str(hf_hub_download(
			repo_id   = "city96/SD-Latent-Interposer",
			subfolder = f"v{self.version}",
			filename  = fname,
		))

	def convert(self, samples, latent_src, latent_dst):
		samples = samples.copy()
		if latent_src == latent_dst:
			return (samples,)

		model_name = f"{latent_src}-to-{latent_dst}"
		if model_name not in config:
			raise ValueError(f"No model exists for this conversion! ({model_name})")

		# only reload if changed
		if self.loaded != model_name or self.model is None:
			# load/init model
			path = self.get_model_path(model_name)
			model = InterposerModel(**config[model_name])
			model.eval()
			model.load_state_dict(load_file(path))
			# keep for later runs
			self.model = model
			self.loaded = model_name

		lt = samples["samples"]
		with torch.no_grad():
			# force FP32, always run on CPU
			lt = self.model(
				lt.cpu().float()
			).to(lt.device).to(lt.dtype)
		samples["samples"] = lt
		return (samples,)

NODE_CLASS_MAPPINGS = {
	"LatentInterposer": ComfyLatentInterposer,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LatentInterposer": ComfyLatentInterposer.TITLE,
}
