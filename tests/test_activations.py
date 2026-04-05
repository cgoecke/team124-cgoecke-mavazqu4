#Tests ReLU and Softmax
#Can run with pytest in a local terminal

import numpy and np
from activations import ReLU, Softmax

def test_relu_forward():
  relu = ReLU()
  x = np.array([[-1, 0, 2]])
  out = relu.forward(x)
  assert np.array_equal(out, np.array([[0, 0, 2]])

def test_relu_backward():
  relu = ReLU()
  x = np.array([[-1, 0, 2]])
  relu.forward(x)
  grad_output = np.array([[1, 1, 1]])
  grad_input = relu.backward(grad_output)
  assert np.array_equal(grad_input, np.array([[0, 0, 1]]))

def test_softmax_output_sums_to_one():
  softmax = Softmax()
  x = np.array([[1, 2, 3]])
  out = softmax.forward(x)
  assert np.allclose(np.sum(out, axis=1), 1.0)
