# Micrograd: Autograd Engine from Scratch

A minimal scalar-valued automatic differentiation engine and a neural network library built from scratch using Python.

This project explores the fundamentals of **backpropagation, computational graphs, automatic differentiation, and neural networks** by implementing them from the ground up.

## =>Features

- **Automatic Differentiation:** A custom `Value` class that tracks operations and computes gradients through backpropagation.
- **Computational Graphs:** Builds a directed graph of operations to represent mathematical expressions.
- **Backpropagation:** Computes gradients using the chain rule and topological ordering.
- **Neural Network Components:** Implements `Neuron`, `Layer`, and `MLP` classes.
- **Activation Function:** Uses `tanh` as the neuron's activation function.
- **Trainable Parameters:** Supports gradient-based parameter updates.
- **From-Scratch Implementation:** Implements the core autograd engine without relying on deep learning frameworks like PyTorch or TensorFlow.

## => Project Structure

```text
micrograd-autograd-engine/
│
├── engine.py                    # Value class and autograd engine
├── nn.py                        # Neuron, Layer, and MLP classes
├── Micrograd_scratchpad.ipynb   # Original implementation and experiments
└── README.md
```

## =>How It Works

### 1. Autograd Engine (`engine.py`)

The `Value` class represents scalar values and tracks the operations that produce them.

Each value stores:

- Its numerical data.
- Its gradient.
- References to previous values in the computational graph.
- The operation used to create it.
- The backward operation used to propagate gradients.

Supported operations include arithmetic operations, exponentiation, and hyperbolic tangent activation.

Calling `.backward()` propagates gradients through the computational graph using the chain rule.

### 2. Neural Networks (`nn.py`)

The neural network module builds on the autograd engine.

It contains three classes:

- **`Neuron`**: Computes a weighted sum of inputs, adds a bias, and applies `tanh`.
- **`Layer`**: Groups multiple neurons into a layer.
- **`MLP`**: Combines multiple layers to construct a multilayer perceptron.

The network parameters are represented using `Value` objects, allowing gradients to flow through the entire network.

## =>Usage

Keep `engine.py` and `nn.py` in the same directory.

### Import the classes

```python
from engine import Value
from nn import Neuron, Layer, MLP
```

### Test automatic differentiation

```python
a = Value(2.0)
b = Value(3.0)

c = a * b
c.backward()

print(c.data)  # 6.0
print(a.grad)  # 3.0
print(b.grad)  # 2.0
```

### Build a neural network

```python
model = MLP(3, [4, 4, 1])

x = [2.0, 3.0, -1.0]
output = model(x)

print(output)
```

This creates a network with three input features, two hidden layers containing four neurons each, and one output neuron.

## => Notebook

The `Micrograd_scratchpad.ipynb` notebook documents the development process, including experiments with derivatives, computational graphs, backpropagation, and neural network construction.

[Open the notebook in Google Colab](https://colab.research.google.com/github/srishanthgouda/micrograd-autograd-engine/blob/main/Micrograd_scratchpad.ipynb)

## => Technologies

- Python
- NumPy and Matplotlib for numerical experiments and visualization in the notebook
- Graphviz for computational graph visualization in the notebook

The core implementation uses Python rather than an existing deep learning framework.

## => Objective

The goal of this project is to understand how automatic differentiation and neural networks work internally, rather than treating them as black boxes.

Building the engine from scratch provides a foundation for understanding gradient computation, optimization, and the mechanics behind modern deep learning frameworks.

---

**Built from scratch as a hands-on exploration of automatic differentiation and neural networks.**
