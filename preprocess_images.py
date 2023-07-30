import os
import hashlib
from tqdm import tqdm
from PIL import Image

# target resolution [latent res * 8]
resolution = 768

if not os.path.isdir("images"):
	os.mkdir("images")

for i in tqdm(os.listdir("raw")):
	src = os.path.join("raw", i)
	md5 = hashlib.md5(open(src,'rb').read()).hexdigest()
	out = os.path.join("images", f"{md5}.png")
	if os.path.isfile(out):
		continue
	img = Image.open(src)
	img = img.convert('RGB')
	img = img.resize((resolution,resolution), Image.LANCZOS)
	img.save(out)
