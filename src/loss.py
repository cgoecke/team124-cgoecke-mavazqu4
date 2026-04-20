import numpy as np

class CategoricalCrossEntropy:
  def forward(self, y_true, y_pred):
    samples = y_pred.shape[0]
    y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)
    correct_confidences = np.sum(y_true * y_pred_clipped, axis=1)
    loss = -np.log(correct_confidences)
    return np.mean(loss)

  def backwards(self, y_true, y_pred):
    samples = y_pred.shape[0]
    return (y_pred - y_true) / samples
