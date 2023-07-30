import os
import torch
import numpy as np
from torchvision import transforms
from diffusers import AutoencoderKL
from tqdm import tqdm
from PIL import Image

from vae import get_vae

def encode(vae, img):
	"""image [PIL Image] -> latent [np array]"""
	inp = transforms.ToTensor()(img).unsqueeze(0)
	inp = inp.to("cuda") # move to GPU
	latent = vae.encode(inp*2.0-1.0)
	latent = latent.latent_dist.sample()
	return latent.cpu().detach()

def process_folder(vae, v):
	if not os.path.isdir(f"latent_{v}"):
		os.mkdir(f"latent_{v}")

	vae.to("cuda")
	for i in tqdm(os.listdir("images")):
		src = os.path.join("images", i)
		img = Image.open(src)
		dst = os.path.join(f"latent_{v}", f"{os.path.splitext(i)[0]}.npy")
		latent = encode(vae, img)
		np.save(dst, latent)
	vae.to("cpu")

def run_v1(file_path=None):
	vae = get_vae("v1", file_path)
	process_folder(vae, "v1")
	del vae

def run_v2(file_path=None):
	vae = get_vae("v2", file_path)
	process_folder(vae, "v2")
	del vae

def run_xl(file_path=None):
	vae = get_vae("xl", file_path)
	process_folder(vae, "xl")
	del vae

if __name__ == "__main__":
	# run_v1("./vae/ft-mse-840000.ckpt") # probably doesn't reflect internal SD latent
	run_v1()
	# run_v2() # v2 and v1 share a latent space
	run_xl("./vae/sdxl_v0.9.safetensors") # 1.0 has artifacts 
