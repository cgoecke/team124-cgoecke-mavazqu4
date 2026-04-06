#Dense layers

import numpy as np

class Dense:
  def __init__(self, input_size, output_size):
    self.weights = 0.01 * np.random.randn(input_size, output_size)
    self.biases = np.zeros((1, output_size))

  def forward(self, input):
    self.input = input
    return np.dot(input, self.weights) + self.biases

  def backward(self, grad_output):
    self.grad_weights = np.dot(self.input.T, grad_output)
    self.grad_biases = np.sum(grad_output, axis=0, keepdims=True)
    return np.dot(grad_output, self.weights.T)

  def update_params(self, optimizer):
    optimizer.update(self)
    
