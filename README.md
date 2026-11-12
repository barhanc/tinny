# tinny 🪶🧠

[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-3776AB.svg?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![NumPy](https://img.shields.io/badge/NumPy-2.0+-013243.svg?style=flat&logo=numpy&logoColor=white)](https://numpy.org/)
[![Tests](https://img.shields.io/badge/tests-10%20passed-brightgreen.svg?style=flat&logo=pytest&logoColor=white)](https://docs.pytest.org/)
[![Theory Notes](https://img.shields.io/badge/Theory_Notes-barhanc.github.io-8A2BE2.svg?style=flat)](https://barhanc.github.io/notes)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

A minimal, educational neural network and energy-based model library implemented from scratch in pure **NumPy**.

`tinny` was created to demystify deep learning internals: no opaque autograd magic, no heavy framework bloat — just explicit mathematical operations, vectorized tensor calculus, and property-tested analytical gradient backpropagation.

---

## ⚡ Core Highlights

- **Pure NumPy Core**: Transparent matrix implementations of forward and backward passes with zero black-box dependencies.
- **Energy-Based Models (EBMs)**: Complete implementations of Restricted Boltzmann Machines (**RBMs**) and Deep Belief Networks (**DBNs**), trained via Contrastive Divergence ($CD_k$) and Persistent Contrastive Divergence (**PCD**).
- **Extensible Tensor Layer API**: Layers are defined as parametrized tensor functions $y = f(x; \theta)$ supporting sequential composition and residual skip connections (`Residual`).
- **Comprehensive Layer Suite**: Dense (Linear), 2D Convolution (`Conv2D` with vectorized im2col), `MaxPool2D`, `Flatten`, `Dropout`, and `BatchNorm`.
- **Rigorously Tested Gradients**: Analytical gradients are validated against numerical finite differences via property-based testing with **Hypothesis** and **pytest**.
- **Accompanying Theory Notes**: Mathematical derivations and conceptual notes available at [barhanc.github.io/notes](https://barhanc.github.io/notes).

---

## 🚀 Quickstart

Ensure you have [uv](https://docs.astral.sh/uv/) installed.

### 1. Installation

```bash
git clone https://github.com/barhanc/tinny.git
cd tinny

# Synchronize the environment and install dependencies in seconds
uv sync
```

### 2. Building & Training a Model

```python
import numpy as np
from tinny import nn

# Define a Multi-Layer Perceptron
model = nn.Sequential(
    nn.Linear(784, 128),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(128, 10),
    nn.Softmax(),
)

# Forward pass
X_batch = np.random.randn(32, 784)
predictions = model.forward(X_batch)

# Backward pass with cross-entropy gradient
grad_output = predictions - y_onehot
model.backward(grad_output)

# Update parameters via optimizer
optimizer = nn.Adam(model.parameters(), lr=1e-3)
optimizer.step()
```

---

## 🧪 Model Zoo & Notebook Experiments

Explore complete, runnable implementations in the [`notebooks/`](./notebooks/) directory:

| Experiment | Description | Architecture / Method |
| :--- | :--- | :--- |
| [**`mnist_mlp.ipynb`**](./notebooks/mnist_mlp.ipynb) | Multi-Layer Perceptron on MNIST | Dense + ReLU + Softmax, mini-batch SGD / Adam |
| [**`mnist_cnn.ipynb`**](./notebooks/mnist_cnn.ipynb) | Convolutional Neural Network on MNIST | `Conv2D` (im2col/col2im) + `MaxPool2D` + Cross-Entropy |
| [**`mnist_ae.ipynb`**](./notebooks/mnist_ae.ipynb) | Autoencoder Reconstruction | Symmetric encoder-decoder with Mean Squared Error (MSE) |
| [**`mnist_dbn.ipynb`**](./notebooks/mnist_dbn.ipynb) | Deep Belief Network Pretraining & Sampling | Stacked RBMs trained via Contrastive Divergence ($CD_k$) |
| [**`mnist_gan.ipynb`**](./notebooks/mnist_gan.ipynb) | Generative Adversarial Network | Minimax adversarial game with Generator & Discriminator |

---

## 🎓 Academic Coursework Solutions

The [`notebooks/course/`](./notebooks/course/) folder contains comprehensive solutions to the **Neural Networks and Deep Learning** course at AGH University (Labs 01 through 13), accompanied by personal commentary, mathematical notes, and critiques:

- **Lab 01–03**: Perceptron foundations, backpropagation from scratch, and optimization dynamics.
- **Lab 04–06**: Modern architectures, regularizations (Dropout, BatchNorm), and convolutional operators.
- **Lab 07–09**: Recurrent architectures, sequence modeling, and representation learning.
- **Lab 10–13**: Generative models, Autoencoders, Energy-Based Models, and modern advancements.

---

## 🛠️ Running Tests

Validate analytical layer gradients against numerical finite-difference approximations:

```bash
uv run pytest
```
