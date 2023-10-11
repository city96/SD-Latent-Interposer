import torch
import torch.nn as nn
import numpy as np

class Block(nn.Module):
	def __init__(self, size):
		super().__init__()
		self.join = nn.ReLU()
		self.long = nn.Sequential(
			nn.Conv2d(size, size, kernel_size=3, stride=1, padding=1),
			nn.LeakyReLU(0.1),
			nn.Conv2d(size, size, kernel_size=3, stride=1, padding=1),
			nn.LeakyReLU(0.1),
			nn.Conv2d(size, size, kernel_size=3, stride=1, padding=1),
			nn.Dropout(0.2)
		)
	def forward(self, x):
		y = self.long(x)
		z = self.join(y + x)
		return z

class Interposer(nn.Module):
	def __init__(self):
		super().__init__()
		self.chan = 4       # in/out channels
		self.hid = 128

		# expand channels
		self.head_join  = nn.ReLU()
		self.head_short = nn.Conv2d(self.chan, self.hid, kernel_size=3, stride=1, padding=1)
		self.head_long  = nn.Sequential(
			nn.Conv2d(self.chan, self.hid, kernel_size=3, stride=1, padding=1),
			nn.LeakyReLU(0.1),
			nn.Conv2d(self.hid,  self.hid, kernel_size=3, stride=1, padding=1),
			nn.LeakyReLU(0.1),
			nn.Conv2d(self.hid,  self.hid, kernel_size=3, stride=1, padding=1),
		)
		# not sure if this is how residuals work
		self.core = nn.Sequential(
			Block(self.hid),
			Block(self.hid),
			Block(self.hid),
		)
		# reduce channels
		self.tail = nn.Sequential(
			nn.ReLU(),
			nn.Conv2d(self.hid, self.chan, kernel_size=3, stride=1, padding=1)
		)

	def forward(self, x):
		y = self.head_join(
			self.head_long(x)+
			self.head_short(x)
		)
		z = self.core(y)
		return self.tail(z)
