import torch
import torch.nn as nn
from torch.autograd import Variable, Function
import random
#
class LinearFunction(Function):
	def forward():

def randomarraygenerator(sparsity):
	seed(1)
	max = numel() - 1
	num_sparsed_els = (sparsity/100)*max
	return random.sample(range(0, max), num_sparsed_els)

def numel(self):
		return int(sum(mask.view(-1).size(0) for mask in self.masks))

class SparseLinear(nn.Module):
	def __init__(self, num_inputs, num_outputs, sparsity):
		self.num_inputs = num_inputs
		self.num_outputs = num_outputs
		self.sparsity = sparsity
		
		device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		
		self.weights = nn.Parameter()
		
		masks = torch.zeros_like(weights.view(-1))
		zero_indicies = randomarraygenerator(sparsity)
		for zi in zero_indicies:
			m[zi] = 1
		self.masks = torch.reshape(masks, weights.shape)


		

