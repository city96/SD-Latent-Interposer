import os
import hashlib
import argparse
from tqdm import tqdm
from PIL import Image
from queue import Queue
from threading import Thread

if not os.path.isdir("images"):
	os.mkdir("images")

def parse_args():
	parser = argparse.ArgumentParser(description="Preprocess images")
	parser.add_argument("-r", "--res", type=int, default=768, help="Target resolution")
	parser.add_argument("-t", "--threads", type=int, default=4, help="No. of CPU threads to use")
	parser.add_argument('--src', default="raw", help="Source folder with images")
	return parser.parse_args()

def process(fname, folder, resolution):
	src = os.path.join(folder, fname)
	md5 = hashlib.md5(open(src,'rb').read()).hexdigest()
	out = os.path.join("images", f"{md5}.png")
	if os.path.isfile(out):
		return
	img = Image.open(src)
	img = img.convert('RGB')
	target = (resolution, resolution)
	if min(img.height, img.width) < 256:
		return

	if img.width > img.height:
		target = (int(img.width/img.height*resolution), resolution)
	elif img.height > img.width:
		target = (resolution, int(img.height/img.width*resolution))
	img = img.resize(target, Image.LANCZOS)
	img = img.crop([0,0,resolution,resolution])
	img.save(out)

def thread(queue, pbar, folder, resolution):
	while not queue.empty():
		fname = queue.get()
		try: process(fname, folder, resolution)
		except: pass
		queue.task_done()
		pbar.update()

args = parse_args()
files = os.listdir(args.src)
pbar = tqdm(total=len(files),unit="img")
queue = Queue()
[queue.put(x) for x in files]

for _ in range(args.threads):
	Thread(
		target=thread,
		args=(
			queue,
			pbar,
			args.src,
			args.res,
		),
		daemon=True,
	).start()

queue.join()
