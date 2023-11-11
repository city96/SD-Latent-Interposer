import os
import math
import matplotlib.pyplot as plt

files = [f"models/{x}" for x in os.listdir("models") if x.endswith(".csv")]
train_loss = {}
eval_loss = {}
lr_vals = {}
fskip = 0

offsets = { # offset to display resumed training runs
}
sep = ".csv"
rep = "_interposer"
model = "Latent Interposer"

def process_lines(lines):
	global train_loss
	global eval_loss
	name = fp.split("/")[1]
	print(name)
	if sep: name = name.split(sep)[0]
	if rep: name = name.replace(rep,"")
	vals = [x.split(",") for x in lines]
	train_loss[name] = (
		[int(x[0]) for x in vals],
		[math.log(float(x[1])+1e-10) for x in vals],
	)
	if len(vals[0]) >= 3:
		eval_loss[name] = (
			[int(x[0]) for x in vals],
			[math.log(float(x[2])+1e-10) for x in vals],
		)
	if len(vals[0]) >= 4:
		lr_vals[name] = (
			[int(x[0]) for x in vals],
			[float(x[3]) for x in vals],
		)

# https://stackoverflow.com/a/49357445
def smooth(scalars, weight):
	last = scalars[0]
	smoothed = list()
	for point in scalars:
		smoothed_val = last * weight + (1 - weight) * point
		smoothed.append(smoothed_val)
		last = smoothed_val
	return smoothed

def plot(data, fname, title=None, smw=0.9):
	fig, ax = plt.subplots()
	plt.tight_layout()
	ax.grid()
	dmax = 0
	for name, val in data.items():
		data = [x + offsets[name] for x in val[0]] if name in offsets.keys() else val[0]
		dmax = max(dmax, round(data[-1],10000))
		sval = val[1][:fskip] + smooth(val[1][fskip:], smw) # skip first N
		ax.plot(data, sval, label=name)
	ax.set_xticks([dmax//10*x for x in range(10)])
	plt.legend(loc="lower left", bbox_to_anchor=(0.00, -0.20), ncol=5)
	if title: plt.title(title) 
	plt.savefig(fname, bbox_inches='tight')

if __name__ == "__main__":
	for fp in files:
		with open(fp) as f:
			lines = f.readlines()
		process_lines(lines)
	plot(train_loss, "loss.png", f"{model} Training loss", 0.2)
	plot(eval_loss, "loss-eval.png", f"{model} Eval. loss", 0.7)
	plot(lr_vals, "loss-lr.png", f"{model} Learning rate", 0.0)
