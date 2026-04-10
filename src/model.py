#Imports Dense from layers
#Uses ReLU and Softmax activation from activations
#Updates with the optimizer

from layers import Dense
from activations import ReLu, Softmax

class Model:
  def __init__(self):
    self.layers = [
      Dense(784, 128),
      ReLU(),
      Dense(128, 64),
      ReLU(),
      Dense(64, 10),
      Softmax()
    ]

  def forward(self, X):
    output = X
    for layer in self.layers:
      output = layer.forward(output)
    return output

  def backward(self, grad):
    for layer in reversed(self.layers):
      grad = layer.backward(grad)

  def update(self, optimizer):
    for layer in self.layers:
      if hasattr(layer, "update_params"):
        layer.update_params(optimizer)
      
