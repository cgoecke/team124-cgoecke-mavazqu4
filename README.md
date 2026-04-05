Handwritten Digit Recognition MLP

This project builds a **Multi-Layer Perceptron (MLP)** neural network from scratch in Python to recognize handwritten digits (0–9) using the MNIST dataset. The goal is to help automate reading ZIP codes and addresses for postal and logistics tasks.

---

Project Structure

- `src/` - all the Python code for the neural network:
  - `layers.py` - defines the linear layers
  - `activations.py` - defines ReLU and Softmax functions
  - `loss.py` - cross-entropy loss calculation
  - `optimizer.py` - Adam optimizer updates
  - `model.py` - MLP model that combines layers
  - `train.py` - code to run training
- `data/` - placeholder folder for datasets (`.gitkeep` included)
- `test/` - all the tests to debug the code files. Use pytest to run in your local terminal.
  - `test_activations.py` - tests ReLU and Softmax activations
  -  `test_layers.py` - tests dense forward/backward
  -  `test_loss.py` - tests cross entropy
  -   `test_optimizer.py` - tests Adam optimizer
  -   `test_model.py` - tests forward and backward pass

---

Getting Started

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/team124-cgoecke-mavazqu4.git

2. Install required packages:

pip install numpy keras

3. Test/debug each code file before moving on to the next. 

Tips for Working Together:

Always pull changes before starting work
Commit your changes often with clear messages
Use branches if experimenting with new features to avoid conflicts.
