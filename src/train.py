#Trains and evaluates the MLP model using the MNIST dataset

import numpy as np
from tensorflow.keras.datasets import mnist

from model import Model
from loss import CategoricalCrossEntropy
from optimizer import Adam

def one_hot(y,num_classes=10):
  return np.eye(num_classes)[y]

def accuracy(y_true, y_pred):
  predictions = np.argmax(y_pred, axis=1)
  labels = np.argmax(y_true, axis=1)
  return np.mean(predicitons == labels)

#Load Data
(X_train, y_train), (X_test, y_test) = mnist.load_data()

#Preprocess
X_train = X_train.reshape(-1, 784) / 255.0
X_test = X_test.reshape(-1, 784) / 255.0

y_train = one_hot(y_train)
y_test = one_hot(y_test) 

#Initialize
model = Model()
loss_fn = CategoricalCrossEntropy()
optimizer = Adam(learning_rate=0.001)
epochs = 20
batch_size = 128

#Training Loop
for epoch in range(epochs):
  indices = np.arange(X_train.shape[0])
  np.random.shuffle(indices)
  X_train = X_train[indices]
  for i in range(0,X_train.shape[0], batch_size):
    X_batch = X_train[i:i+batch_size]
    y_batch = y_train[i:i+batch_size]
    output = model.forward(X_batch) #Forward
    loss = loss_fn.forward(y_batch, output) #Loss
    grad = loss_fn.backward(y_batch, output) #Backward
    model.backward(grad)
    model.update(optimizer) #Update

#Evaluate
train_acc = accuracy(y_train, model.forward(X_train))
test_acc = accuracy(y_test, model.forward(X_test))

print(f"Epoch {epoch+1}/{epochs} - Loss: {loss:.4f} -Train Acc: {train_acc:.4f} -Test Acc: {test_acc:.4f}")
