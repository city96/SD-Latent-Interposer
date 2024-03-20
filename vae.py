import torch
from diffusers import AutoencoderKL

DTYPE = torch.float16
DEVICE = "cuda:0"

class SDv1_VAE:
	scale = 1/8
	channels = 4
	def __init__(self, device=DEVICE, dtype=DTYPE, dec_only=False):
		self.device = device
		self.dtype = dtype
		self.model = AutoencoderKL.from_pretrained(
			"stabilityai/sd-vae-ft-mse"
		)
		self.model.eval().to(self.dtype).to(self.device)
		if dec_only:
			del self.model.encoder

	def encode(self, image):
		image = image.to(self.dtype).to(self.device)
		image = (image * 2.0) - 1.0 # assuming input is [0;1]
		with torch.no_grad():
			latent = self.model.encode(image).latent_dist.sample()
		return latent.to(image.dtype).to(image.device)

	def decode(self, latent, grad=False):
		latent = latent.to(self.dtype).to(self.device)
		if grad:
			out = self.model.decode(latent)[0]
		else:
			with torch.no_grad():
				out = self.model.decode(latent).sample
			out = torch.clamp(out, min=-1.0, max=1.0)
			out = (out + 1.0) / 2.0
		return out.to(latent.dtype).to(latent.device)

class SDXL_VAE(SDv1_VAE):
	scale = 1/8
	channels = 4
	def __init__(self, device=DEVICE, dtype=DTYPE, dec_only=False):
		self.device = device
		self.dtype = dtype
		self.model = AutoencoderKL.from_pretrained(
			"madebyollin/sdxl-vae-fp16-fix"
		)
		self.model.eval().to(self.dtype).to(self.device)
		if dec_only:
			del self.model.encoder

class CascadeC_VAE(SDv1_VAE):
	scale = 1/32
	channels = 16
	def __init__(self, device=DEVICE, dtype=DTYPE, **kwargs):
		self.device = device
		self.dtype = dtype

		#For now this is just piggybacking off of koyha-ss/sd-scripts
		from library import stable_cascade as sc
		from safetensors.torch import load_file
		from huggingface_hub import hf_hub_download

		self.model = sc.EfficientNetEncoder()
		self.model.load_state_dict(load_file(
			str(hf_hub_download(
			repo_id   = "stabilityai/stable-cascade",
			filename  = "effnet_encoder.safetensors",
			))
		))
		self.model.eval().to(self.dtype).to(self.device)

class CascadeA_VAE():
	scale = 1/4
	channels = 4
	def __init__(self, device=DEVICE, dtype=DTYPE, dec_only=False):
		self.device = device
		self.dtype = dtype
		
		# not sure if this will change in the future?
		from diffusers.pipelines.wuerstchen.modeling_paella_vq_model import PaellaVQModel
		self.model = PaellaVQModel.from_pretrained(
			"stabilityai/stable-cascade",
			subfolder="vqgan"
		)
		self.model.eval().to(self.dtype).to(self.device)
		if dec_only:
			del self.model.encoder

	def encode(self, image):
		image = image.to(self.dtype).to(self.device)
		with torch.no_grad():
			latent = self.model.encode(image).latents
		return latent.to(image.dtype).to(image.device)

	def decode(self, latent, grad=False):
		latent = latent.to(self.dtype).to(self.device)
		if grad:
			out = self.model.decode(latent)[0]
		else:
			with torch.no_grad():
				out = self.model.decode(latent).sample
			out = torch.clamp(out, min=0.0, max=1.0)
		return out.to(latent.dtype).to(latent.device)

def load_vae(ver, *args, **kwargs):
	if ver == "v1":
		VAE = SDv1_VAE
	elif ver == "xl":
		VAE = SDXL_VAE
	elif ver == "cc":
		VAE = CascadeC_VAE
	elif ver == "ca":
		VAE = CascadeA_VAE
	return VAE(*args, **kwargs)
