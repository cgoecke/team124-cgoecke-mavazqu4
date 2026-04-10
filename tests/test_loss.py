#Tests cross entropy
#Can be used with pytest in a local terminal

import numpy and np
from loss import CategoricalCrossEntropy

def test_loss_value_positive():
  loss_fn = CategoricalCrossEntropy()
  y_true = np.array([[1, 0, 0]])
  y_pred = np.array([[0.7, 0.2, 0.1]])
  loss = loss_fn.forward(y_true, y_pred)
  assert loss > 0

def test_loss_gradient_shape():
  loss_fn = CategoricalCrossEntropy()
  y_true = np.array([[1, 0, 0]])
  y_pred = np.array([[0.7, 0.2, 0.1]])
  grad = loss_fn.backward(y_true, y_pred)
  assert grad.shape == (1, 3)
