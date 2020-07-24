import torch
from maskedLinear import SparseLinear
from maskedLinear import LinearFunction
from torch.autograd import Variable, Function
from torch.autograd import gradcheck

mask = torch.zeros(50,20)
for i in range(20):
    x = torch.randperm(50)
    for j in range(5):
        mask[x[j]][i] = 1
mask =  mask.double()
mask = Variable(mask, requires_grad=False)
weight = torch.randn(50,20)
linear = LinearFunction.apply
print(linear)

input = (Variable(torch.randn(20,20).double(), requires_grad=True), Variable(torch.randn(50,20).double(), requires_grad=True), mask)
test = gradcheck(linear, input, eps=1e-6, atol=1e-4)
print(test)
print(mask.data)
