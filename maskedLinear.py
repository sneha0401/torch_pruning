import torch
import torch.nn as nn
from torch.autograd import Variable, Function
import random

def randomarraygenerator(sparsity, mask_els):
	num_sparsed_els = int((1 - sparsity/100)*mask_els)
	return random.sample(range(0, (mask_els-1)), num_sparsed_els)

class LinearFunction(Function):
	def __init__ (self, masks):
		self.masks = masks

	@staticmethod

	def forward(ctx, self, input, weight):
		ctx.save_for_backward(input, weight)
		
		output = input.mm(weight.mul_(self.masks.data).T)
		return output

	@staticmethod

	def backward(ctx, grad_output):
		input, weight = ctx.saved_tensors

		grad_input = grad_weight = None
		cl_weight = weight.clone()
		wt_copy = cl_weight.mul_(mask.data)
		if ctx.needs_input_grad[0]:
			grad_input = grad_output.mm(wt_copy)
		if ctx.needs_input_grad[1]: 
			grad_weight = grad_output.clone().t().mm(input)
			grad_weight = grad_weight.mul_(mask)

		return grad_input, grad_weight

class SparseLinear(nn.Module):
	def __init__(self, num_inputs, num_outputs, sparsity):
		super(SparseLinear, self).__init__() 
		self.num_inputs = num_inputs
		self.num_outputs = num_outputs
		self.sparsity = sparsity
		
		device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

		weights = torch.Tensor(num_outputs, num_inputs)
		weights =  nn.init.xavier_normal(weights)
		self.weights = nn.Parameter(weights)

		masks = torch.zeros_like(self.weights.view(-1))
		mask_els = masks.size()[0]
		zero_indicies = randomarraygenerator(sparsity, mask_els)
		for zi in zero_indicies:
			masks[zi] = 1
		masks = torch.reshape(masks, self.weights.shape)
		self.masks = nn.Parameter(masks)
		self.masks.requires_grad = False

	def forward(self, input):
		return LinearFunction.apply(input, self.weights)
		
