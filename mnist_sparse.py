import torch
import torch.nn.functional as F
import torch.utils.data as data
import torch.nn as nn
from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np
import math

from maskedLinear import SparseLinear

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.fc0 = nn.Linear(28 * 28, 512)
        self.fc1 = nn.Linear(512, 512)
        self.fc2 = nn.Linear(512, 10)

    def forward(self, h1):
    	h1 = h1.view(-1, 28*28)
    	h1 = F.relu(self.fc0(h1))
    	h2 = F.relu(self.fc1(h1))
    	h3 = self.fc2(h2)
    	return h3


def main():
	n_epochs = 20
	batch_size = 30
	lr, wd = 1e-2, 1e-5
	transform = transforms.ToTensor()
	train_data = datasets.MNIST(root = 'data', train = True, download = True, transform = transform)
	test_data = datasets.MNIST(root = 'data', train = False, download = True, transform = transform)

	net = Net()
	print(net)
	criterion = nn.CrossEntropyLoss()
	optimizer = torch.optim.SGD(net.parameters(), lr,
								weight_decay=wd)

	def train(net, verbose = False):
		print("Training for", n_epochs, "epochs")
		net.train()
		for epoch in range(1, n_epochs+1):
			
			train_loader = data.DataLoader(train_data, batch_size = batch_size,
											shuffle = True)


			epoch_loss = 0
			for x, y in train_loader:
				y_hat = net(x)
				loss = criterion(y_hat, y.view(-1))
				optimizer.zero_grad()
				loss.backward()
				optimizer.step()
				epoch_loss += loss.item()
			if verbose:
				print("Epoch {}: {:.4f}".format(epoch, epoch_loss))
	
	val_losses = []

	train(net)

	def validate(net, masks=None):
		
		test_loader = data.DataLoader(test_data, batch_size = batch_size,
                                      shuffle = True)
		net.eval()
		val_loss = 0.
		correct = 0
		
		for x, y in test_loader:
			y_hat = net(x)
			val_loss += F.nll_loss(y_hat, y, size_average=False).item()
			pred = y_hat.data.max(1, keepdim=True)[1]
			correct += pred.eq(y.data.view_as(pred)).sum()
		val_loss /= len(test_loader.dataset)
		val_losses.append(val_loss)
		print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
		    val_loss, correct, len(test_loader.dataset),
		    100. * correct / len(test_loader.dataset)))
	validate(net)

if __name__ == "__main__":
	main()