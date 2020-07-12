import torch
import torch.nn.functional as F
import torch.utils.data as data
import torch.nn as nn
from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np
import math

from pruning import MagnitudePruning

def print_ticket(masks, params):
	for i, (m, p) in enumerate(zip(masks, params)):
		print("\tLayer", i)
		for row, col in m.nonzero():
			print("\t[{},{}]: {}".format(row, col, p[row, col]))

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.fc0 = nn.Linear(28 * 28, 512)
        self.fc1 = nn.Linear(512, 512)
        self.fc2 = nn.Linear(512, 10)

    def forward(self, h1):
    	h1 = h1.view(-1, 28*28)
    	h1 = F.relu(self.fc0(h1))
    	h1 = F.dropout(h1, p=0.5, training=self.training)
    	h2 = F.relu(self.fc1(h1))
    	h2 = F.dropout(h2, p=0.5, training=self.training)
    	h3 = self.fc2(h2)
    	return h3


def main():
	n_hidden = 1000
#	n_feats = 4
#	n_train = 5000
#	n_test = 1000
	train_range = [-1, 1]
	test_range = [10, 20]
	num_workers = 0
	pruning_rounds = 1000
	pruning_rate = 0.25
	rewind_to = 1
	local = True

	n_epochs = 20
	batch_size = 30
	lr, wd = 1e-2, 1e-5
	transform = transforms.ToTensor()
	train_data = datasets.MNIST(root = 'data', train = True, download = True, transform = transform)
	test_data = datasets.MNIST(root = 'data', train = False, download = True, transform = transform)
	num_train = len(train_data)
	indices = list(range(num_train))
	np.random.shuffle(indices)
	
	def stop_criterion(nz, n):
		return nz < n*0.75

	net = Net()
	print(net)

	criterion = nn.CrossEntropyLoss()
	optimizer = torch.optim.SGD(net.parameters(), lr,
								weight_decay=wd)
	pruning = MagnitudePruning(net.parameters(), pruning_rate, local=local)

	def train(net, masks = None, snapshot_at = rewind_to, verbose = False):
		print("Training for", n_epochs, "epochs")
		w_k = pruning.clone_params() if snapshot_at == 0 else None
		net.train()
		for epoch in range(1, n_epochs+1):
			'''train_loader = data.DataLoader(train_set,
											batch_size = batch_size,
											shuffle = True)'''
			train_loader = data.DataLoader(train_data, batch_size = batch_size,
											shuffle = True)


			epoch_loss = 0
			for x, y in train_loader:
				pruning.zero_params(masks)
				y_hat = net(x)
				loss = criterion(y_hat, y.view(-1))
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
		'''test_loader = data.DataLoader(test_set,
 									batch_size=batch_size,
		
									shuffle=False)'''
		test_loader = data.DataLoader(test_data, batch_size = batch_size,
                                      shuffle = True)
		net.eval()
		val_loss = 0.
		correct = 0
		for x, y in test_loader:
			y_hat = net(data)
			val_loss += F.nll_loss(y_hat, y, size_average=False).item()
			pred = y_hat.x.max(1, keepdim=True)[1]
			correct += pred.eq(y.x.view_as(pred)).sum()
		val_loss /= len(test_loader.dataset)
		val_losses.append(val_loss)
		print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
		    val_loss, correct, len(test_loader.dataset),
		    100. * correct / len(test_loader.dataset)))

#	validate()	
	for epoch in range(1, n_epochs + 1):
		validate(net)

if __name__ == "__main__":
	main()
