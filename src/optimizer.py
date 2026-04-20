#Adam Optimizer

import numpy as np

class Adam:
  def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-7):
    self.lr = learning_rate
    self.beta1 = beta1
    self.beta2 = beta2
    self.epsilon = epsilon
    self.t = 0

  def update(self,layer):
    if not hasattr(layer, 'm_w'):
      layer.m_w = np.zeros_like(layer.weights)
      layer.v_w = np.zeros_like(layer.weights)
      layer.m_b = np.zeros_like(layer.biases)
      layer.v_b = np.zeros_like(layer.biases)

    self.t += 1

    #Update moments
    layer.m_w = self.beta1 * layer.m_w + (1 - self.beta1) * layer.grad_weights
    layer.v_w = self.beta2 * layer.v_w + (1 - self.beta2) * (layer.grad_weights ** 2)
    layer.m_b = self.beta1 * layer.m_b + (1 - self.beta1) * layer.grad_biases
    layer.v_b = self.beta2 * layer.v_b + (1 - self.beta2) * (layer.grad_biases ** 2)

    #Bias Correction
    m_w_hat = layer.m_w / (1 - self.beta1 ** self.t)
    v_w_hat = layer.v_w / (1 - self.beta2 ** self.t)
    m_b_hat = layer.m_b / (1 - self.beta1 ** self.t)
    v_b_hat = layer.v_b / (1 - self.beta2 ** self.t)

    #Update Parameters
    layer.weights -= self.lr * m_w_hat / (np.sqrt(v_w_hat) + self.epsilon)
    layer.biases -= self.lr * m_b_hat / (np.sqrt(v_b_hat) + self.epsilon)
