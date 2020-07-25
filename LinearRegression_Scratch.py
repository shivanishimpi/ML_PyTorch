import numpy as np
import torch 

inputs = np.array([[73, 67, 43], 
                   [91, 88, 64], 
                   [87, 134, 58], 
                   [102, 43, 37], 
                   [69, 96, 70]], dtype='float32')

labels = np.array([[56, 70], 
                    [81, 101], 
                    [119, 133], 
                    [22, 37], 
                    [103, 119]], dtype='float32')
inputs = torch.from_numpy(inputs)
labels = torch.from_numpy(labels)

weights = torch.randn(2,3,requires_grad=True) #2x3 weights matrix
bias = torch.randn(2,requires_grad=True) #two bias units

def model(x):
  return x@weights.t() + bias #@ is matrix multiplication and .t() is transpose

def MSE(val1,val2):
  delta = (val1-val2)**2
  return torch.sum(delta)/delta.numel()

numEpoch = int(input('numEpoch: '))
learning_rate = 1e-4

for i in range(numEpoch):
  predictions = model(inputs)
  loss = MSE(predictions,labels)
  loss.backward()
  with torch.no_grad():
    weights -= weights.grad * learning_rate
    bias -= bias.grad * learning_rate
    weights.grad.zero_()
    bias.grad.zero_()

predictions = model(inputs)
loss = MSE(predictions, labels)
print(f'Loss = {loss}')
#print(f'Predictions:\n{predictions}')
#print(f'Labels:\n{labels}')
