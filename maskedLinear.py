import torch
import torch.nn as nn
from torch.autograd import Variable, Function
import random
#
def randomarraygenerator(sparsity, max):
	num_sparsed_els = (sparsity/100)*max
	return random.sample(range(0, (max-1)), num_sparsed_els)
'''
class LinearFunction(Function):
	def forward():
'''

class SparseLinear(nn.Module):
	def __init__(self, num_inputs, num_outputs, sparsity):
		self.num_inputs = num_inputs
		self.num_outputs = num_outputs
		self.sparsity = sparsity
		
		device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		
		self.weights = nn.Parameter()

		masks = torch.zeros_like(weights.view(-1))
		max = masks.size()[0]
		zero_indicies = randomarraygenerator(sparsity, max)
		for zi in zero_indicies:
			masks[zi] = 1
		self.masks = torch.reshape(masks, weights.shape)

	def numel(self):
		return int(sum(mask.view(-1).size(0) for mask in self.masks))



