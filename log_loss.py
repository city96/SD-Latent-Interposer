import os
import math
import matplotlib.pyplot as plt

files = [f"models/{x}" for x in os.listdir("models") if x.endswith(".csv")]
step = []
data = {}
for fp in files:
	with open(fp) as f:
		lines = f.readlines()
	if not step:
		step = [int(x.split(",")[0]) for x in lines]
	name = fp.split("/")[1].split("_")[0]
	data[name] = (
		[int(x.split(",")[0]) for x in lines],
		[math.log(float(x.split(",")[1])) for x in lines],
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

fig, ax = plt.subplots()
ax.grid()
for name, val in data.items():
	ax.plot(val[0], smooth(val[1], 0.7), label=name)
plt.legend(loc="upper right")
plt.savefig('loss.png', dpi=300, bbox_inches='tight')
