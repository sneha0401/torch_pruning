import torch
import torch.utils.data as data
import torch.nn as nn
import numpy as np
import math

from pruning import MagnitudePruning

def print_ticket(masks, params):
	for i, (m, p) in enumerate(zip(masks, params)):
		print("\tLayer", i)
		for row, col in m.nonzero():
			print("\t[{},{}]: {}".format(row, col, p[row, col]))

def make_sum_data(n_feats, n_train, n_test, train_range, test_range):
	x_train = torch.zeros(n_train, n_feats).uniform_(*train_range)
	y_train = x_train.sum(1)
	train_set = list(zip(x_train, y_train))
	x_test = torch.zeros(n_test, n_feats).uniform_(*test_range)
	y_test = x_test.sum(1)
	test_set = list(zip(x_test, y_test))
	return train_set, test_set

def main():
	n_hidden = 1000
	n_feats = 4
	n_train = 5000
	n_test = 1000
	train_range = [-1, 1]
	test_range = [10, 20]

	pruning_rounds = 1000
	pruning_rate = 0.25
	rewind_to = 1
	local = True

	n_epochs = 20
	batch_size = 16
	lr, wd = 1e-2, 1e-5

	def stop_criterion(nz, n):
		return nz/2 < n_feats*2

	train_set, test_set = make_sum_data(n_feats, n_train, n_test,
                                        train_range, test_range)

	net = nn.Sequential(nn.Linear(n_feats, n_hidden, bias=False),
						nn.Linear(n_hidden, 1, bias=False))
	print(net)

	criterion = nn.MSELoss()
	optimizer = torch.optim.SGD(net.parameters(), lr,
								weight_decay=wd)
	pruning = MagnitudePruning(net.parameters(), pruning_rate, local=local)

	def train(net, masks = None, snapshot_at = rewind_to, verbose = False):
		print("Training for", n_epochs, "epochs")
		w_k = pruning.clone_params() if snapshot_at == 0 else None
		net.train()
		for epoch in range(1, n_epochs+1):
			train_loader = data.DataLoader(train_set,
											batch_size = batch_size,
											shuffle = True)

			epoch_loss = 0
			for x, y in train_loader:
				pruning.zero_params(masks)
				y_hat = net(x)
				loss = criterion(y_hat, y.unsqueeze(1))
				optimizer.zero_grad()
				loss.backward()
				optimizer.step()
				epoch_loss += loss.item()
			if verbose:
				print("Epoch {}: {:.4f}".format(epoch, epoch_loss))
			
			if epoch == snapshot_at:
				w_k = pruning.clone_params()
		return w_k

	w_k = None
	last_nz = 0
	for i in range(pruning_rounds):
		if i > 0:
			print("Rewinding params to epoch", rewind_to)
			pruning.rewind(w_k) 
		w_k = train(net, verbose=True)
		pruning.step()
		nz, n = pruning.count_nonzero(), pruning.numel()
		print("Pruning {} weights => {} weights still active ~= {:3.2f}%"
				.format(pruning_rate, nz, nz / n * 100))
		if stop_criterion(nz, n):
			print("Stopping criterion met.")
			break
		last_nz = nz

	pruning.rewind(w_k)
	train(net)

	def validate(net, masks=None):
		test_loader = data.DataLoader(test_set,
 									batch_size=batch_size,
									shuffle=False)
		net.eval()
		val_loss = 0.
		for x, y in test_loader:
			pruning.zero_params(masks=masks)
			y_hat = net(x)
			loss = criterion(y_hat, y.unsqueeze(1))
			val_loss += loss.item()
		return val_loss

	rmse = np.sqrt(validate(net))
	print("Winning ticket RMSE:", rmse)

	for layer in net:
		layer.reset_parameters()
	train(net, snapshot_at=-1) 
	rmse = np.sqrt(validate(net))
	print("Reinit Random ticket RSME:", rmse)
	perm_masks = [torch.tensor(np.random.permutation(m.numpy()))
				  for m in pruning.masks]
	for layer in net:
		layer.reset_parameters()
	train(net, masks=perm_masks, snapshot_at=-1)  
	rmse = np.sqrt(validate(net, masks=perm_masks))
	print("Permute+Reinit Random ticket RSME:", rmse)

	ones_masks = [torch.ones_like(m) for m in pruning.masks]
	for layer in net:
		layer.reset_parameters()
	train(net, masks=ones_masks, snapshot_at=-1)
	rmse = np.sqrt(validate(net, masks=ones_masks))
	print("Full-model RSME:", rmse)

if __name__ == "__main__":
	main()
