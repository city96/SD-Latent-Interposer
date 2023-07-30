import torch
from diffusers import AutoencoderKL

def get_vae(version, file_path=None, fp16=False):
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
	elif version == "v2" and file_path:
		vae = AutoencoderKL.from_single_file(
			file_path,
			image_size=768,
		)
	elif version == "v2":
		vae = AutoencoderKL.from_pretrained(
			"stabilityai/stable-diffusion-2-1",
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
		input("Invalid VAE version. Press any key to exit")
		exit(1)
	return vae
