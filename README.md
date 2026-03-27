# Handwritten Digit Recognition MLP

This project builds a **Multi-Layer Perceptron (MLP)** neural network from scratch in Python to recognize handwritten digits (0–9) using the MNIST dataset. The goal is to help automate reading ZIP codes and addresses for postal and logistics tasks.

---

## Project Structure

- `src/` – all the Python code for the neural network:
  - `layers.py` – defines the linear layers
  - `activations.py` – defines ReLU and Softmax functions
  - `loss.py` – cross-entropy loss calculation
  - `optimizer.py` – Adam optimizer updates
  - `model.py` – MLP model that combines layers
  - `train.py` – code to run training
- `data/` – placeholder folder for datasets (`.gitkeep` included)
- `utils.py` – helper functions, e.g., for checking accuracy and building a confusion matrix

---

## Getting Started

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/team124-cgoecke-mavazqu4.git

Install required packages:

pip install numpy keras
How to Use

Train the model:

python src/train.py
The model will print the loss and accuracy during training.
You can also use utils.py functions to check test accuracy or build a confusion matrix.

Tips for Working Together:

Always pull changes before starting work:

git pull origin main

Commit your changes often with clear messages:

git add .
git commit -m "Describe your changes here"
git push origin main
Use branches if experimenting with new features to avoid conflicts.
