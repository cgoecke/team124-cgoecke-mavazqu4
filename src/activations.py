#Activations code file
import numpy as np

#ReLU activation
class ReLU:
  def forward(self,input):
    self.input = input
    return np.maximum(0, input)

  def backward(self, grad_output):
    grad = grad_output.copy()
    grad[self.input<=0] = 0
    return grad

#Softmax activation
class Softmax:
  def forward(self,input):
    exp_values = np.exp(input-np.max(input, axis=1, keepdims=True))
    self.output=exp_values/np.sum(exp_values, axis=1, keepdims=True)
    return self.output

  def backward(self, grad_output):
    return grad_output #handled with cross entropy in loss.py
