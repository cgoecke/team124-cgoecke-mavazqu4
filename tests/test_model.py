#Tests forward pass, softmax output, and backward pass

import numpy as np
from model import Model

def test_model_forward_shape():
  model = Model()
  x = np.random.randn(5, 784)
  out = model.forward(x)
  assert out.shape == (5,10)

def test_model_softmax_output():
  model = Model()
  x = np.random.randn(5, 784)
  out = model.forward(x)
  assert np.allclose(np.sum(out, axis=1), 1.0)

def test_model_backward_runs():
  model = Model()
  x = np.random.randn(5, 784)
  out = model.forward(x)
  grad = np.random.randn(5, 10)
  model.backward(grad)
