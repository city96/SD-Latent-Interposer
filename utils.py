#
# This file just has all the random saving/logging/eval related code
#
import os
import torch
from tqdm import tqdm
from diffusers import AutoencoderKL
from safetensors.torch import save_file
from torchvision.utils import save_image

LOSS_MEMORY = 500
LOG_EVERY_N = 500
SAVE_FOLDER = "models"

class ModelWrapper:
	def __init__(self, name, specs, model, optimizer, criterion, scheduler, device="cpu", evals=[None,None], stdout=True):
		self.name   = name
		self.specs  = specs
		self.losses = []

		self.model = model
		self.optimizer = optimizer
		self.criterion = criterion
		self.scheduler = scheduler

		self.device = device
		self.vae    = self.get_vae(self.specs[1], fp16=True)
		self.eval_src = evals[0]
		self.eval_dst = evals[1]

		os.makedirs(SAVE_FOLDER, exist_ok=True)
		self.csvlog = open(f"{SAVE_FOLDER}/{self.name}.csv", "w")
		self.stdout = stdout

	def log_step(self, loss, step=None):
		self.losses.append(loss)
		step = step if step else len(self.losses)
		if step % LOG_EVERY_N == 0:
			self.log_main(step)

	def log_main(self, step=None):
		lr = float(self.scheduler.get_last_lr()[0])
		avg = sum(self.losses[-LOSS_MEMORY:])/LOSS_MEMORY
		evl = self.eval_model()[0]
		if self.stdout:
			tqdm.write(f"{str(step):<10} {avg:.4e}|{evl:.4e} @ {lr:.4e}")
		if self.csvlog:
			self.csvlog.write(f"{step},{avg},{evl},{lr}\n")
			self.csvlog.flush()

	def eval_model(self):
		with torch.no_grad():
			pred = self.model(self.eval_src.to(self.device))
			loss = self.criterion(pred, self.eval_dst.to(self.device))
		return loss, pred

	def save_model(self, step=None, epoch=None):
		step = step if step else len(self.losses)
		if epoch is None and step >= 10**6:
			epoch = f"_e{round(step/10**6,2)}M"
		elif epoch is None:
			epoch = f"_e{round(step/10**3)}K"
		output_name = f"./{SAVE_FOLDER}/{self.name}{epoch}"
		if self.vae:
			out = self.eval_model()[1]
			img = self.vae_decode(out).detach()
			save_image(img, f"{output_name}.png")
			torch.cuda.empty_cache()
		save_file(self.model.state_dict(), f"{output_name}.safetensors")
		torch.save(self.optimizer.state_dict(), f"{output_name}.optim.pth")

	def close(self):
		del self.vae
		self.csvlog.close()

	def vae_decode(self, latent):
		latent = latent.to(torch.float16).to("cuda")
		out = self.vae.decode(latent).sample
		out = out.float().to(latent.device)
		out = torch.clamp(out, min=-1.0, max=1.0)
		return ((out + 1.0) / 2.0)

	def get_vae(self, version, file_path=None, fp16=False):
		"""Load VAE from file or default hf repo. fp16 only works from hf"""
		vae = None
		dtype = torch.float16 if fp16 else torch.float32
		if version == "v1" and file_path:
			vae = AutoencoderKL.from_single_file(
				file_path,
				image_size=512,
			)
		elif version == "v1":
			vae = AutoencoderKL.from_pretrained(
				"runwayml/stable-diffusion-v1-5",
				subfolder="vae",
				torch_dtype=dtype,
			)
		elif version == "xl" and file_path:
			vae = AutoencoderKL.from_single_file(
				file_path,
				image_size=1024
			)
		elif version == "xl" and fp16:
			vae = AutoencoderKL.from_pretrained(
				"madebyollin/sdxl-vae-fp16-fix",
				torch_dtype=torch.float16,
			)
		elif version == "xl":
			vae = AutoencoderKL.from_pretrained(
				"stabilityai/stable-diffusion-xl-base-1.0",
				subfolder="vae"
			)
		else:
			raise NotImplementedError(f"Unknown VAE version '{version}'")

		# save VRAM
		vae.to(dtype).to("cuda")
		vae.decoder.eval()
		vae.set_use_memory_efficient_attention_xformers(True)
		vae.enable_xformers_memory_efficient_attention()
		vae.enable_gradient_checkpointing()
		del vae.encoder
		return vae
