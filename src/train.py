#Trains and evaluates the MLP model using the MNIST dataset

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist

from model import Model
from loss import CategoricalCrossEntropy
from optimizer import Adam

# Set up - improved for evaluation

def one_hot(y, num_classes=10):
    return np.eye(num_classes)[y]

def accuracy(y_true, y_pred):
    predictions = np.argmax(y_pred, axis=1)
    labels = np.argmax(y_true, axis=1)
    return np.mean(predictions == labels)

def confusion_matrix(y_true, y_pred, num_classes=10):
    cm = np.zeros((num_classes, num_classes), dtype=int)
    true_labels = np.argmax(y_true, axis=1)
    pred_labels = np.argmax(y_pred, axis=1)

    for t, p in zip(true_labels, pred_labels):
        cm[t, p] += 1
    return cm

def precision_recall_f1(cm):
    num_classes = cm.shape[0]
    precision = np.zeros(num_classes)
    recall = np.zeros(num_classes)
    f1 = np.zeros(num_classes)

    for i in range(num_classes):
        tp = cm[i, i]
        fp = np.sum(cm[:, i]) - tp
        fn = np.sum(cm[i, :]) - tp

        precision[i] = tp / (tp + fp + 1e-8)
        recall[i] = tp / (tp + fn + 1e-8)
        f1[i] = 2 * precision[i] * recall[i] / (precision[i] + recall[i] + 1e-8)

    macro_precision = np.mean(precision)
    macro_recall = np.mean(recall)
    macro_f1 = np.mean(f1)

    return precision, recall, f1, macro_precision, macro_recall, macro_f1

def plot_curves(train_losses, test_losses, train_accs, test_accs):
    epochs = len(train_losses)

    plt.figure(figsize=(12,5))

    # Loss
    plt.subplot(1,2,1)
    plt.plot(range(epochs), train_losses, label="Train Loss")
    plt.plot(range(epochs), test_losses, label="Test Loss")
    plt.title("Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    # Accuracy
    plt.subplot(1,2,2)
    plt.plot(range(epochs), train_accs, label="Train Acc")
    plt.plot(range(epochs), test_accs, label="Test Acc")
    plt.title("Accuracy Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.tight_layout()
    plt.show()

def show_errors(X, y_true, y_pred, num_examples=10):
    true_labels = np.argmax(y_true, axis=1)
    pred_labels = np.argmax(y_pred, axis=1)

    errors = np.where(true_labels != pred_labels)[0]

    print(f"\nTotal misclassified examples: {len(errors)}")

    for i in errors[:num_examples]:
        print(f"True: {true_labels[i]} | Pred: {pred_labels[i]}")

        plt.imshow(X[i].reshape(28,28), cmap='gray')
        plt.title(f"True: {true_labels[i]}, Pred: {pred_labels[i]}")
        plt.axis('off')
        plt.show()


# Load Data

(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train.reshape(-1, 784) / 255.0
X_test = X_test.reshape(-1, 784) / 255.0

y_train = one_hot(y_train)
y_test = one_hot(y_test)


# Initialize

model = Model()
loss_fn = CategoricalCrossEntropy()
optimizer = Adam(learning_rate=0.001)

epochs = 20
batch_size = 128

train_losses = []
test_losses = []
train_accs = []
test_accs = []


# Training Loop

for epoch in range(epochs):
    indices = np.arange(X_train.shape[0])
    np.random.shuffle(indices)

    X_train = X_train[indices]
    y_train = y_train[indices]

    batch_losses = []

    for i in range(0, X_train.shape[0], batch_size):
        X_batch = X_train[i:i+batch_size]
        y_batch = y_train[i:i+batch_size]

        # Forward
        output = model.forward(X_batch)

        # Loss
        loss = loss_fn.forward(y_batch, output)
        batch_losses.append(loss)

        # Backward
        grad = loss_fn.backward(y_batch, output)
        model.backward(grad)

        # Update
        model.update(optimizer)

    # Epoch metrics
    train_output = model.forward(X_train)
    test_output = model.forward(X_test)

    train_loss = np.mean(batch_losses)
    test_loss = loss_fn.forward(y_test, test_output)

    train_acc = accuracy(y_train, train_output)
    test_acc = accuracy(y_test, test_output)

    train_losses.append(train_loss)
    test_losses.append(test_loss)
    train_accs.append(train_acc)
    test_accs.append(test_acc)

    print(f"Epoch {epoch+1}/{epochs} | "
          f"Loss: {train_loss:.4f} | Test Loss: {test_loss:.4f} | "
          f"Train Acc: {train_acc:.4f} | Test Acc: {test_acc:.4f}")


# Improved Evaluation

print("\n--- Final Evaluation ---")

train_preds = model.forward(X_train)
test_preds = model.forward(X_test)

train_acc = accuracy(y_train, train_preds)
test_acc = accuracy(y_test, test_preds)

print(f"Final Train Accuracy: {train_acc:.4f}")
print(f"Final Test Accuracy: {test_acc:.4f}")

# Generalization gap
gap = train_acc - test_acc
print(f"Generalization Gap: {gap:.4f}")

# Confusion matrix
cm = confusion_matrix(y_test, test_preds)
print("\nConfusion Matrix:")
print(cm)

# Precision / Recall / F1
precision, recall, f1, macro_p, macro_r, macro_f1 = precision_recall_f1(cm)

print("\nPer-class Metrics:")
for i in range(10):
    print(f"Digit {i}: Precision={precision[i]:.4f}, "
          f"Recall={recall[i]:.4f}, F1={f1[i]:.4f}")

print("\nMacro Averages:")
print(f"Precision: {macro_p:.4f}, Recall: {macro_r:.4f}, F1: {macro_f1:.4f}")

# Curves
plot_curves(train_losses, test_losses, train_accs, test_accs)

# Error analysis
show_errors(X_test, y_test, test_preds, num_examples=10)
