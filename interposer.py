import torch
import torch.nn as nn
import numpy as np

class Interposer(nn.Module):
	def __init__(self):
		super().__init__()
		
		# it looks like a spaceship if you squint :P
		module_list = [
		#############)
		#############)
			    #||#
			    #||#
			nn.Conv2d(4, 32, kernel_size=5, padding=2),
			nn.ReLU(),
			nn.Conv2d(32, 128, kernel_size=7, padding=3),
			nn.ReLU(),
			nn.Conv2d(128, 32, kernel_size=7, padding=3),
			nn.ReLU(),
			nn.Conv2d(32, 4, kernel_size=5, padding=2),
			    #||#
			    #||#
		#############)
		#############)
		]

		self.sequential = nn.Sequential(*module_list)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		return self.sequential(x)
