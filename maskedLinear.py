import torch
import torch.nn as nn
from torch.autograd import Variable, Function
import random
#
def randomarraygenerator(sparsity, mask_els):
	num_sparsed_els = int((sparsity/100)*mask_els)
	return random.sample(range(0, (mask_els-1)), num_sparsed_els)
'''
class LinearFunction(Function):
	def forward():
'''

class SparseLinear(nn.Module):
	def __init__(self, num_inputs, num_outputs, sparsity):
		super(SparseLinear, self).__init__() 
		self.num_inputs = num_inputs
		self.num_outputs = num_outputs
		self.sparsity = sparsity
		
		device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		self.weights = nn.Parameter(torch.Tensor(num_outputs, num_inputs))

		masks = torch.zeros_like(self.weights.view(-1))
		mask_els = masks.size()[0]
		zero_indicies = randomarraygenerator(sparsity, mask_els)
		for zi in zero_indicies:
			masks[zi] = 1
		self.masks = torch.reshape(masks, self.weights.shape)

	def numel(self):
		return int(sum(mask.view(-1).size(0) for mask in self.masks))
