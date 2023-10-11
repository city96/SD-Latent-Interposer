import os
import math
import matplotlib.pyplot as plt

files = [f"models/{x}" for x in os.listdir("models") if x.endswith(".csv")]
train_loss = {}
eval_loss = {}
	
def process_lines(lines):
	global train_loss
	global eval_loss
	name = fp.split("/")[1].split("_")[0]
	vals = [x.split(",") for x in lines]
	train_loss[name] = (
		[int(x[0]) for x in vals],
		[math.log(float(x[1])) for x in vals],
	)
	if len(vals[0]) >= 3:
		eval_loss[name] = (
			[int(x[0]) for x in vals],
			[math.log(float(x[2])) for x in vals],
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

def plot(data, fname):
	fig, ax = plt.subplots()
	ax.grid()
	for name, val in data.items():
		ax.plot(val[0], smooth(val[1], 0.9), label=name)
	plt.legend(loc="upper right")
	plt.savefig(fname, dpi=300, bbox_inches='tight')

for fp in files:
	with open(fp) as f:
		lines = f.readlines()
	process_lines(lines)

plot(train_loss, "loss.png")
plot(eval_loss, "loss-eval.png")
