import torch
import torch.nn as nn

class ResBlock(nn.Module):
	"""Block with residuals"""
	def __init__(self, ch):
		super().__init__()
		self.join = nn.ReLU()
		self.norm = nn.BatchNorm2d(ch)
		self.long = nn.Sequential(
			nn.Conv2d(ch, ch, kernel_size=3, stride=1, padding=1),
			nn.SiLU(),
			nn.Conv2d(ch, ch, kernel_size=3, stride=1, padding=1),
			nn.SiLU(),
			nn.Conv2d(ch, ch, kernel_size=3, stride=1, padding=1),
			nn.Dropout(0.1)
		)
	def forward(self, x):
		x = self.norm(x)
		return self.join(self.long(x) + x)

class ExtractBlock(nn.Module):
	"""Increase no. of channels by [out/in]"""
	def __init__(self, ch_in, ch_out):
		super().__init__()
		self.join  = nn.ReLU()
		self.short = nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1)
		self.long  = nn.Sequential(
			nn.Conv2d( ch_in, ch_out, kernel_size=3, stride=1, padding=1),
			nn.SiLU(),
			nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1),
			nn.SiLU(),
			nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1),
			nn.Dropout(0.1)
		)
	def forward(self, x):
		return self.join(self.long(x) + self.short(x))

class InterposerModel(nn.Module):
	"""Main neural network"""
	def __init__(self, ch_in=4, ch_out=4, ch_mid=64, scale=1.0, blocks=12):
		super().__init__()
		self.ch_in  = ch_in
		self.ch_out = ch_out
		self.ch_mid = ch_mid
		self.blocks = blocks
		self.scale  = scale

		self.head = ExtractBlock(self.ch_in, self.ch_mid)
		self.core = nn.Sequential(
			nn.Upsample(scale_factor=self.scale, mode="nearest"),
			*[ResBlock(self.ch_mid) for _ in range(blocks)],
			nn.BatchNorm2d(self.ch_mid),
			nn.SiLU(),
		)
		self.tail = nn.Conv2d(self.ch_mid, self.ch_out, kernel_size=3, stride=1, padding=1)

	def forward(self, x):
		y = self.head(x)
		z = self.core(y)
		return self.tail(z)
