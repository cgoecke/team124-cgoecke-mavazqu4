#Tests Adam optimizer
#Can be used with pytest in a local terminal

import numpy as np
from layers import Dense
from optimizer import Adam

def test_adam_updates_weights():
  layer = Dense(2, 2)
  optimizer = Adam(learning_rate=0.01)
  layer.grad_weights = np.ones_like(layer.weights)
  layer.grad_biases = np.ones_like(layer.biases)
  old_weights = layer.weights.copy()
  optimizer.update(layer)
  assert not np.array_equal(layer.weights, old_weights)
