#Tests dense forward/backward
#Can be used with pytest in a local terminal

import numpy as np
from layers import Dense 

def test_dense_forward_shape():
  layer = Dense(4, 3)
  x = np.random.randn(2,4)
  out = layer.forward(x)
  assert out.shape == (2, 3)

def test_dense_backward_shapes():
  layer = Dense(4, 3)
  x = np.random.randn(2, 4)
  layer.forward(x)
  grad_output = np.random.randn(2, 3)
  grad_input = layer.backward(grad_output)
  assert grad_input.shape == (2, 4)
  assert layer.grad_weights.shape == (4, 3)
