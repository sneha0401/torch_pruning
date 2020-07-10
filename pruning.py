import torch
import torch.nn as nn
import torch.nn.functional as F

def cut(pruning_rate, flat_params):

	assert flat_params.dim() == 1
	with torch.no_grad():
		cutoff_index = round(pruning_rate * flat_params.size()[0])
        values, __indices = torch.sort(torch.abs(flat_params))
        cutoff = values[cutoff_index]
    return cutoff

class MagnitudePruning():

	def __init__(self, params, pruning_rate = 0.25, local = True, 
				 exclude_biases = False):

		self.local = bool(local)
		self.pruning_rate = float(pruning_rate)

		if exclude_biases: 
			self.params = [p for p in params if p.dim > 1]

		else:
			self.params = [p for p in params]

		masks = []
		for p in self.params:
			masks.append(torch.ones_like(p))
		self.masks = masks

	def count_nonzero(self):
		return int(sum(mask.sum() for mask in self.masks).item())

	def numel(self):
		return int(sum(mask.view(-1).size(0) for mask in self.masks))

	def clone_params(self):
		return [p.clone() for p in self.params]

	def rewind(self, cloned_params):
		for p_old, p_new in zip(self.params, cloned_params):
			p_old.data = p_new.data

	def step(self):
		if self.local:
			for i, (m, p) in enumerate(zip(self.masks, self.params)):
				flat_params = p[m==1].view(-1)
				cutoff = cut(self.pruning_rate, flat_params)
				new_mask = torch.where(torch.abs(p) < cutoff,
										torch.zeros_like(p), m)
				self.masks[i] = new_mask

		else: 
			flat_params = torch.cat([p[m==1].view(-1) 
									 for p, m in zip(self.params, self.masks)]) 
			cutoff = cut(self.pruning_rate, flat_params)
			for i, (m, p) in enumerate(zip(self.masks, self.params)):
				new_mask = torch.where(torch.abs(p) < cutoff,
										torch.zeros_like(p), m)
				self.masks[i] = new_mask

	def zero_params(self, masks = None):
		masks = masks if masks is not None else self.masks
		for m, p in zip(self.masks, self.params):
			p.data = m * p.data





