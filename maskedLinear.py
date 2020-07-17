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
	def __init__ (self, self.masks):
		super(LinearFunction, self).__init__()
		self.masks = self.masks
	def forward(ctx, self, input, weight):
		ctx.save_for_backward(input, weight)
		output = input.mm(weight.mul_(self.mask.data).T)
'''

class SparseLinear(nn.Module):
	def __init__(self, num_inputs, num_outputs, sparsity):
		super(SparseLinear, self).__init__() 
		self.num_inputs = num_inputs
		self.num_outputs = num_outputs
		self.sparsity = sparsity
		
		device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

		weights = torch.Tensor(num_outputs, num_inputs)
		wetghts =  nn.init.xavier_normal(weights)
		self.weights = nn.Parameter(weights)

		masks = torch.zeros_like(self.weights.view(-1))
		mask_els = masks.size()[0]
		zero_indicies = randomarraygenerator(sparsity, mask_els)
		for zi in zero_indicies:
			masks[zi] = 1
		masks = torch.reshape(masks, self.weights.shape)
		self.masks = nn.Parameter(masks)
		self.masks.requires_grad = False

	def numel(self):
		return int(sum(mask.view(-1).size(0) for mask in self.masks))
