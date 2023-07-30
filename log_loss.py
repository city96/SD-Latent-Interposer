import os
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
	data[name] = [float(x.split(",")[1]) for x in lines]

fig, ax = plt.subplots()
ax.grid()
for name, val in data.items():
	ax.plot(step, val, label=name)
plt.legend(loc="upper right")
plt.savefig('loss.png', dpi=300, bbox_inches='tight')
