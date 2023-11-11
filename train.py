import os
import torch
import argparse
from tqdm import tqdm
from torch.utils.data import DataLoader
from safetensors.torch import load_file

from interposer import InterposerModel as Model
from dataset import LatentDataset
from utils import ModelWrapper

torch.backends.cudnn.benchmark = True
torch.manual_seed(0)

TARGET_DEV = "cuda"

def parse_args():
	parser = argparse.ArgumentParser(description="Train latent interposer model")
	parser.add_argument("-s", "--steps", type=int, default=500000, help="No. of training steps")
	parser.add_argument("-b", "--batch", type=int, default=     1, help="Batch size")
	parser.add_argument("-n", "--nsave", type=int, default= 50000, help="Save model/sample periodically")
	parser.add_argument('--rev', default="v4.0-rc1", help="Revision/log ID")
	parser.add_argument('--src', choices=["v1","xl"], required=True, help="Source latent format")
	parser.add_argument('--dst', choices=["v1","xl"], required=True, help="Destination latent format")
	parser.add_argument('--lr', default="1e-4", help="Learning rate")
	parser.add_argument('--lrskip', type=int, default=0, help="Constant lr for first N steps")
	parser.add_argument('--cosine', action=argparse.BooleanOptionalAction, help="Use cosine scheduler")
	parser.add_argument('--resume', help="Checkpoint to resume from")
	args = parser.parse_args()
	if args.src == args.dst:
		parser.error("--src and --dst can't be the same")
	try:
		float(args.lr)
	except:
		parser.error("--lr must be a valid float eg. 0.001 or 1e-3")
	return args

if __name__ == "__main__":
	args = parse_args()

	dataset = LatentDataset([args.src, args.dst])
	loader = DataLoader(
		dataset,
		batch_size  = args.batch,
		shuffle     = True,
		drop_last   = True,
		pin_memory  = False,
		# num_workers = 0,
		num_workers = 4,
		persistent_workers=True,
	)
	model = Model() # TODO: handle scale factor/channels for non-sd VAEs
	criterion = torch.nn.L1Loss()
	optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.lr))
	scheduler = None
	if args.cosine:
		print("Using CosineAnnealingLR")
		scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
			optimizer, T_max = int(args.steps/args.batch),
		)
	else:
		print("Using LinearLR")
		scheduler = torch.optim.lr_scheduler.LinearLR(
			optimizer,
			start_factor = 0.1,
			end_factor   = 1.0,
			total_iters  = int(5000/args.batch),
		)

	if args.resume:
		model.load_state_dict(load_file(args.resume))
		model.to(TARGET_DEV)
		optimizer.load_state_dict(torch.load(
			f"{os.path.splitext(args.resume)[0]}.optim.pth"
		))
		optimizer.param_groups[0]['lr'] = scheduler.base_lrs[0]
	else:
		model.to(TARGET_DEV)

	wrapper = ModelWrapper( # model wrapper for saving/eval/etc
		name      = f"{args.src}-to-{args.dst}_interposer-{args.rev}",
		specs     = [args.src, args.dst],
		model     = model,
		evals     = dataset.get_eval(),
		device    = TARGET_DEV,
		criterion = criterion,
		optimizer = optimizer,
		scheduler = scheduler,
	)

	progress = tqdm(total=args.steps)
	while progress.n < args.steps:
		for src, dst in loader:
			src = src.to(TARGET_DEV)
			dst = dst.to(TARGET_DEV)
			with torch.cuda.amp.autocast():
				y_pred = model(src) # forward
				loss = criterion(y_pred, dst) # loss

			# backward
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
			if progress.n >= args.lrskip: scheduler.step()

			# eval/save
			progress.update(args.batch)
			wrapper.log_step(loss.data.item(), progress.n)
			if args.nsave > 0 and progress.n % (args.nsave + args.nsave%args.batch) == 0:
				wrapper.save_model(step=progress.n)
			if progress.n >= args.steps:
				break
	progress.close()
	wrapper.save_model(epoch="") # final save
	wrapper.close()
