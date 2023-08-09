import torch
import torch.nn as nn
from safetensors.torch import load_file
from huggingface_hub import hf_hub_download


class Interposer(nn.Module):
	"""
		Basic NN layout, ported from:
		https://github.com/city96/SD-Latent-Interposer/blob/main/interposer.py
	"""
	version = 1.1 # network revision
	def __init__(self):
		super().__init__()
		module_list = [
			nn.Conv2d(4, 32, kernel_size=5, padding=2),
			nn.ReLU(),
			nn.Conv2d(32, 128, kernel_size=7, padding=3),
			nn.ReLU(),
			nn.Conv2d(128, 32, kernel_size=7, padding=3),
			nn.ReLU(),
			nn.Conv2d(32, 4, kernel_size=5, padding=2),
		]
		self.sequential = nn.Sequential(*module_list)
	def forward(self, x: torch.Tensor) -> torch.Tensor:
		return self.sequential(x)


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
		weights = str(hf_hub_download(
			repo_id="city96/SD-Latent-Interposer",
			filename=f"{latent_src}-to-{latent_dst}_interposer-v{model.version}.safetensors")
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
