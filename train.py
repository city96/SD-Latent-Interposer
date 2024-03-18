import os
import yaml
import torch
import argparse
from tqdm import tqdm
from torch.utils.data import DataLoader
from safetensors.torch import save_file, load_file

from interposer import InterposerModel
from dataset import LatentDataset, FileLatentDataset, load_evals
from vae import load_vae

torch.backends.cudnn.benchmark = True
torch.manual_seed(0)

def parse_args():
	parser = argparse.ArgumentParser(description="Train latent interposer model")
	parser.add_argument("--config", help="Config for training")
	args = parser.parse_args()
	with open(args.config) as f:
		conf = yaml.safe_load(f)
	args.dataset = argparse.Namespace(**conf.pop("dataset"))
	args.model = argparse.Namespace(**conf.pop("model"))
	return argparse.Namespace(**vars(args), **conf)

def eval_images(model, vae, evals):
	preds = eval_model(model, evals, loss=False)
	out = {}
	for name, pred in preds.items():
		images = vae.decode(pred).cpu().float()
		# for image in images: # eval isn't batched
		out[f"eval/{name}"] = images[0]
	return out

def eval_model(model, evals, loss=True):
	model.eval()
	preds = {}
	losses = []
	for name, data in evals.items():
		src = data["src"].to(args.device)
		dst = data["dst"].to(args.device)
		with torch.no_grad():
			pred = model(src)
		if loss:
			loss = torch.nn.functional.l1_loss(dst, pred)
			losses.append(loss)
		else:
			preds[name] = pred
	model.train()
	if loss:
		return (sum(losses) / len(losses)).data.item()
	else:
		return preds

# from pytorch GAN tutorial
def weights_init(m):
	classname = m.__class__.__name__
	if classname.find('Conv') != -1:
		torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
	elif classname.find('BatchNorm') != -1:
		torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
		torch.nn.init.constant_(m.bias.data, 0)

if __name__ == "__main__":
	args = parse_args()
	base_name = f"models/{args.model.src}-to-{args.model.dst}_interposer-{args.model.rev}"

	# dataset
	if os.path.isfile(args.dataset.src):
		dataset = FileLatentDataset(
			args.dataset.src,
			args.dataset.dst,
		)
	elif os.path.isdir(args.dataset.src):
		dataset = LatentDataset(
			args.dataset.src,
			args.dataset.dst,
			args.dataset.preload
		)
	else:
		raise OSError(f"Missing dataset source {args.dataset.src}")
	loader = DataLoader(
		dataset,
		batch_size  = args.batch,
		shuffle     = True,
		drop_last   = True,
		pin_memory  = False,
		num_workers = 0,
		# num_workers = 6,
		# persistent_workers=True,
	)

	# evals
	try:
		evals = load_evals(args.dataset.evals)
	except:
		print(f"No evals, fallback to dataset.")
		evals = dataset[0]

	# defaults
	crit = torch.nn.L1Loss()
	optim_args = {
		"lr": args.optim["lr"],
		"betas": (args.optim["beta1"], args.optim["beta2"])
	}

	# model
	model = InterposerModel(**args.model.args)
	model.apply(weights_init)
	model.to(args.device)
	optim = torch.optim.AdamW(model.parameters(), **optim_args)

	# aux model for reverse pass
	model_back = InterposerModel(
			ch_in  = args.model.args["ch_out"],
			ch_mid = args.model.args["ch_mid"],
			ch_out = args.model.args["ch_in"],
			scale  = 1.0 / args.model.args["scale"],
			blocks = args.model.args["blocks"],
	)
	model_back.apply(weights_init)
	model_back.to(args.device)
	optim_back = torch.optim.AdamW(model_back.parameters(), **optim_args)

	# scheduler
	scheduler = None
	if args.cosine:
		scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
			optim,
			T_max = (args.steps - args.fconst),
			eta_min = 1e-8,
		)

	# vae
	vae = None
	if args.save_image:
		vae = load_vae(args.model.dst, device=args.device, dtype=torch.float16, dec_only=True)

	# main loop
	import time
	from torch.utils.tensorboard import SummaryWriter
	writer = SummaryWriter(log_dir=f"{base_name}_{int(time.time())}")

	pbar = tqdm(total=args.steps)
	while pbar.n < args.steps:
		for batch in loader:
			# get training data
			src = batch.get("src").to(args.device)
			dst = batch.get("dst").to(args.device)

			### Train main model ###
			optim.zero_grad()
			logs = {}
			loss = []
			with torch.cuda.amp.autocast():
				# pass first model
				pred = model(src)

				p_loss = crit(pred, dst) * args.p_loss_weight
				loss.append(p_loss)
				logs["p_loss"] = p_loss.data.item()

				# pass second model
				if args.r_loss_weight:
					pred_back = model_back(pred)

					r_loss = crit(pred_back, src) * args.r_loss_weight
					loss.append(r_loss)
					logs["r_loss"] = r_loss.data.item()

			# loss logic
			loss = sum(loss)
			logs["main"] = loss.data.item()
			loss.backward()
			optim.step()

			# logging
			for name, value in logs.items():
				writer.add_scalar(f"loss/{name}", value, pbar.n)

			### Train backwards model ###
			if args.r_loss_weight:
				optim_back.zero_grad()
				logs = {}
				loss = []
				with torch.cuda.amp.autocast():
					# pass second model
					pred = model_back(dst)

					p_loss = crit(pred, src) * args.b_loss_weight
					loss.append(p_loss)
					logs["p_loss"] = p_loss.data.item()

					# pass first model
					if args.h_loss_weight: # better w/o this?
						pred_back = model(pred)

						r_loss = crit(pred_back, dst) * args.h_loss_weight
						loss.append(r_loss)
						logs["r_loss"] = r_loss.data.item()

				# loss logic
				loss = sum(loss)
				logs["main"] = loss.data.item()
				loss.backward()
				optim_back.step()

				# logging
				for name, value in logs.items():
					writer.add_scalar(f"loss_aux/{name}", value, pbar.n)

			# run eval/save eval image
			if args.eval_model and pbar.n % args.eval_model == 0:
				writer.add_scalar("loss/eval_loss", eval_model(model, evals), pbar.n)
			if args.save_image and pbar.n % args.save_image == 0:
				for name, image in eval_images(model, vae, evals).items():
					writer.add_image(name, image, pbar.n)

			# scheduler logic main
			if scheduler is not None and pbar.n >= args.fconst:
				lr = scheduler.get_last_lr()[0]
				scheduler.step()
			else:
				lr = args.optim["lr"]
			writer.add_scalar("lr/model", lr, pbar.n)

			# aux model doesn't have a scheduler
			writer.add_scalar("lr/model_aux", args.optim["lr"], pbar.n)

			# step
			pbar.update()
			if pbar.n > args.steps:
				break

			# hacky workaround when the colors are off.
			# Save the last n versions and just pick the best one later.
			# if pbar.n > (args.steps-2500) and pbar.n%500==0:
				# from torchvision.utils import save_image
				# save_file(model.state_dict(), f"{base_name}_{pbar.n:07}.safetensors")
				# for name, image in eval_images(model, vae, evals).items():
					# name = f"models/{name.replace('/', '_')}_{pbar.n:07}.png"
					# save_image(image, name)

	# final save/cleanup
	pbar.close()
	writer.close()

	save_file(model.state_dict(), f"{base_name}.safetensors")
	torch.save(optim.state_dict(), f"{base_name}.optim.pth")
