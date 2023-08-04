import os
import torch
import torch.nn as nn
import numpy as np
import argparse
import random
from PIL import Image
from tqdm import tqdm
from safetensors.torch import save_file, load_file

from interposer import Interposer
from vae import get_vae

def parse_args():
	parser = argparse.ArgumentParser(description="Train latent interposer model")
	parser.add_argument("--steps", type=int, default=500000, help="No. of training steps")
	parser.add_argument('--bs', type=int, default=1, help="Batch size")
	parser.add_argument('--lr', default="1e-8", help="Learning rate")
	parser.add_argument("-n", "--save_every_n", type=int, dest="save", default=50000, help="Save model/sample periodically")
	parser.add_argument('--src', choices=["v1","xl"], required=True, help="Source latent format")
	parser.add_argument('--dst', choices=["v1","xl"], required=True, help="Destination latent format")
	parser.add_argument('--resume', help="Checkpoint to resume from")
	args = parser.parse_args()
	if args.src == args.dst:
		parser.error("--src and --dst can't be the same")
	try:
		float(args.lr)
	except:
		parser.error("--lr must be a valid float eg. 0.001 or 1e-3")
	return args

class Latent:
	def __init__(self, md5, lat_src, lat_dst, dev):
		if lat_src == "v1": src = os.path.join("latent_v1", f"{md5}.npy")
		if lat_src == "xl": src = os.path.join("latent_xl", f"{md5}.npy")

		if lat_dst == "v1": dst = os.path.join("latent_v1", f"{md5}.npy")
		if lat_dst == "xl": dst = os.path.join("latent_xl", f"{md5}.npy")

		self.src = torch.from_numpy(np.load(src)).to(dev)
		self.dst = torch.from_numpy(np.load(dst)).to(dev)

def load_latents(src, dst, dev):
	print("Loading latents from disk")
	latents = []
	for i in tqdm(os.listdir("images")):
		md5 = os.path.splitext(i)[0]
		latents.append(Latent(md5, src, dst, dev))
	return latents

vae = None
def sample_decode(latent, filename, version):
	global vae
	if not vae:
		vae = get_vae(version, fp16=True)
		vae.to("cuda")

	latent = latent.half().to("cuda")
	out = vae.decode(latent).sample
	out = out.cpu().detach().numpy()
	out = np.squeeze(out, 0)
	out = out.transpose((1, 2, 0))
	out = np.clip(out, -1.0, 1.0)
	out = (out+1)/2 * 255
	out = out.astype(np.uint8)
	out = Image.fromarray(out)
	out.save(filename)

if __name__ == "__main__":
	args = parse_args()
	target_dev = "cuda"
	latent_src = args.src
	latent_dst = args.dst

	latents = load_latents(latent_src, latent_dst, target_dev)

	if not os.path.isdir("models"): os.mkdir("models")
	log = open(f"models/{latent_src}-to-{latent_dst}_interposer.csv", "w")

	if os.path.isfile(f"test_{latent_src}.npy") and os.path.isfile(f"test_{latent_dst}.npy"):
		ss_latent = torch.from_numpy(np.load(f"test_{latent_src}.npy")).to(target_dev)
		st_latent = torch.from_numpy(np.load(f"test_{latent_dst}.npy")).to(target_dev)
	else:
		sample_latent = random.choice(latents)
		ss_latent = sample_latent.src.to(target_dev)
		st_latent = sample_latent.dst.to(target_dev)

	model = Interposer()
	if args.resume:
		model.load_state_dict(load_file(args.resume))
	model.to(target_dev)

	criterion = torch.nn.MSELoss(size_average=False)
	optimizer = torch.optim.SGD(model.parameters(), lr=float(args.lr)/args.bs)

	for t in tqdm(range(int(args.steps/args.bs)), unit_scale=args.bs):
		step = t*args.bs
		# input batch
		lts = [random.choice(latents) for _ in range(args.bs)]
		src = torch.cat([x.src for x in lts],0)
		dst = torch.cat([x.dst for x in lts],0)

		y_pred = model(src) # forward
		loss = criterion(y_pred, dst) # loss

		# backward
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

		# print loss
		if step%1000 == 0:
			# test loss
			with torch.no_grad():
				t_pred = model(ss_latent)
				t_loss = criterion(t_pred, st_latent)
			tqdm.write(f"{step} - {loss.data.item()/args.bs:.2f}|{t_loss.data.item()/args.bs:.2f}")
			log.write(f"{step},{loss.data.item()/args.bs:.2f},{t_loss.data.item()/args.bs:.2f}\n")
			log.flush()

		# sample/save
		if step%args.save == 0:
			out = model(ss_latent)
			output_name = f"./models/{latent_src}-to-{latent_dst}_interposer_e{step/1000}k"
			sample_decode(out, f"{output_name}.png", latent_dst)
			save_file(model.state_dict(), f"{output_name}.safetensors")
	# save final output
	output_name = f"./models/{latent_src}-to-{latent_dst}_interposer_e{args.steps/1000}k"
	sample_decode(out, f"{output_name}.png", "v1")
	save_file(model.state_dict(), f"{output_name}.safetensors")
	log.close()
