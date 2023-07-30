import os
import hashlib
from tqdm import tqdm
from PIL import Image
from queue import Queue
from threading import Thread

# target resolution [latent res * 8]
resolution = 768
# threads used for resizing
threads = 4
# source folder with images
folder = "raw"

if not os.path.isdir("images"):
	os.mkdir("images")

def process(fname):
	global folder
	global resolution
	src = os.path.join(folder, fname)
	md5 = hashlib.md5(open(src,'rb').read()).hexdigest()
	out = os.path.join("images", f"{md5}.png")
	if os.path.isfile(out):
		return
	img = Image.open(src)
	img = img.convert('RGB')
	img = img.resize((resolution,resolution), Image.LANCZOS)
	img.save(out)

def thread(queue, pbar):
	while not queue.empty():
		fname = queue.get()
		process(fname)
		queue.task_done()
		pbar.update()

files = os.listdir(folder)
pbar = tqdm(total=len(files),unit="img")
queue = Queue()
[queue.put(x) for x in files]

for _ in range(threads):
	Thread(
		target=thread,
		args=(queue,pbar),
		daemon=True,
	).start()

queue.join()
