import os
import torch
import torch.nn as nn
import numpy as np
import argparse
from PIL import Image
from tqdm import tqdm
from safetensors.torch import save_file, load_file
from torch.utils.data import DataLoader, Dataset

from interposer import Interposer
from vae import get_vae

torch.backends.cudnn.benchmark = True
torch.manual_seed(0)

def parse_args():
	parser = argparse.ArgumentParser(description="Train latent interposer model")
	parser.add_argument("--steps", type=int, default=500000, help="No. of training steps")
	parser.add_argument('--bs', type=int, default=4, help="Batch size")
	parser.add_argument('--lr', default="1e-4", help="Learning rate")
	parser.add_argument("-n", "--save_every_n", type=int, dest="save", default=50000, help="Save model/sample periodically")
	parser.add_argument('--src', choices=["v1","xl"], required=True, help="Source latent format")
	parser.add_argument('--dst', choices=["v1","xl"], required=True, help="Destination latent format")
	parser.add_argument('--resume', help="Checkpoint to resume from")
	parser.add_argument('--cosine', action=argparse.BooleanOptionalAction, help="Use cosine scheduler to taper off LR")
	args = parser.parse_args()
	if args.src == args.dst:
		parser.error("--src and --dst can't be the same")
	try:
		float(args.lr)
	except:
		parser.error("--lr must be a valid float eg. 0.001 or 1e-3")
	return args

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

def get_eval_data(dataset, src_path, dst_path, target_dev):
	if os.path.isfile(src_path) and os.path.isfile(dst_path):
		src = LatentDataset.load_latent(None, src_path)
		dst = LatentDataset.load_latent(None, dst_path)
	else:
		src = dataset[0][0]
		dst = dataset[0][1]
	src = src.float().to(target_dev).unsqueeze(0)
	dst = dst.float().to(target_dev).unsqueeze(0)
	return(src, dst)

def eval_model(step, model, criterion, scheduler, src, dst):
	with torch.no_grad():
		t_pred = model(src)
		t_loss = criterion(t_pred, dst)
	tqdm.write(f"{str(step):<10} {loss.data.item():.4e}|{t_loss.data.item():.4e} @ {float(scheduler.get_last_lr()[0]):.4e}")
	log.write(f"{step},{loss.data.item()},{t_loss.data.item()},{float(scheduler.get_last_lr()[0])}\n")
	log.flush()

def save_model(step, model, optim, lat, src, dst):
	with torch.no_grad():
		out = model(lat)
	output_name = f"./models/{src}-to-{dst}_interposer_e{round(step/1000)}k"
	sample_decode(out, f"{output_name}.png", dst)
	save_file(model.state_dict(), f"{output_name}.safetensors")
	torch.save(optim.state_dict(), f"{output_name}.optim.pth")

class LatentDataset(Dataset):
	class Shard:
		def __init__(self, root, fname, res, src, dst):
			self.fname = fname
			self.src_path = f"{root}/{src}_{res}px/{fname}.npy"
			self.dst_path = f"{root}/{dst}_{res}px/{fname}.npy"

	def __init__(self, res, src, dst, root="latents"):
		print("Loading latents from disk")
		self.latents = []
		for i in tqdm(os.listdir(f"{root}/{src}_{res}px")):
			fname, ext = os.path.splitext(i)
			assert ext == ".npy"
			s = self.Shard(root, fname, res, src, dst)
			if os.path.isfile(s.src_path) and os.path.isfile(s.dst_path):
				self.latents.append(s)

	def __len__(self):
		return len(self.latents)

	def __getitem__(self, index):
		s = self.latents[index]
		src = self.load_latent(s.src_path)
		dst = self.load_latent(s.dst_path)
		return (src, dst)

	def load_latent(self, path):
		lat = torch.from_numpy(np.load(path))
		if lat.shape[0] == 1:
			lat = torch.squeeze(lat, 0)
		assert not torch.isnan(torch.sum(lat.float()))
		return lat

if __name__ == "__main__":
	args = parse_args()
	target_dev = "cuda"
	resolution = 768

	dataset = LatentDataset(resolution, args.src, args.dst)
	loader = DataLoader(
		dataset,
		batch_size=args.bs,
		shuffle=True,
		num_workers=0,
		# num_workers=4,
		# persistent_workers=True,
	)
	eval_src, eval_dst = get_eval_data(
		dataset,
		f"latents/test_{args.src}_{resolution}px.npy",
		f"latents/test_{args.dst}_{resolution}px.npy",
		target_dev,
	)

	os.makedirs("models", exist_ok=True)
	log = open(f"models/{args.src}-to-{args.dst}_interposer.csv", "w")

	model = Interposer()

	criterion = torch.nn.L1Loss()
	optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.lr))
	# import bitsandbytes as bnb
	# optimizer = bnb.optim.AdamW8bit(model.parameters(), lr=float(args.lr))

	scheduler = None
	if args.cosine:
		print("Using CosineAnnealingLR")
		scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
			optimizer, T_max = int(args.steps/args.bs),
		)
	else:
		print("Using LinearLR")
		scheduler = torch.optim.lr_scheduler.LinearLR(
			optimizer,
			start_factor = 0.1,
			end_factor   = 1.0,
			total_iters  = int(5000/args.bs),
		)

	if args.resume:
		model.load_state_dict(load_file(args.resume))
		model.to(target_dev)
		optimizer.load_state_dict(torch.load(
			f"{os.path.splitext(args.resume)[0]}.optim.pth"
		))
	else:
		model.to(target_dev)

	progress = tqdm(total=args.steps)
	while progress.n < args.steps:
		for src, dst in loader:
			src = src.to(target_dev)
			dst = dst.to(target_dev)
			with torch.cuda.amp.autocast():
				y_pred = model(src) # forward
				loss = criterion(y_pred, dst) # loss

			# backward
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
			scheduler.step()

			# eval/save
			progress.update(args.bs)
			if progress.n % (1000 + 1000%args.bs) == 0:
				eval_model(progress.n, model, criterion, scheduler, eval_src, eval_dst)
			if progress.n % (args.save + args.save%args.bs) == 0:
				save_model(progress.n, model, optimizer, eval_src, args.src, args.dst)
			if progress.n >= args.steps:
				break
	progress.close()

	# save final output
	eval_model(progress.n, model, criterion, scheduler, eval_src, eval_dst)
	save_model(progress.n, model, optimizer, eval_src, args.src, args.dst)
	log.close()
