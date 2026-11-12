# tinny

[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-3776AB.svg?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![NumPy](https://img.shields.io/badge/NumPy-2.0+-013243.svg?style=flat&logo=numpy&logoColor=white)](https://numpy.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

A minimal, educational neural network and energy-based model library implemented from scratch in pure **NumPy**.

`tinny` was created to demystify deep learning internals: no opaque autograd magic, no heavy framework bloat — just explicit mathematical operations, vectorized tensor calculus, and property-tested analytical gradient backpropagation. See the [`notebooks/`](./notebooks/) directory for experiment implementations (including MLPs, CNNs, Autoencoders, GANs, and DBNs) along with solutions and commentary for course assignments.

## Core Highlights

- **Pure NumPy Core**: Transparent matrix implementations of forward and backward passes with zero black-box dependencies.
- **Energy-Based Models (EBMs)**: Complete implementations of Restricted Boltzmann Machines (**RBMs**) and Deep Belief Networks (**DBNs**), trained via Contrastive Divergence ($CD_k$) and Persistent Contrastive Divergence (**PCD**).
- **Extensible Tensor Layer API**: Layers are defined as parametrized tensor functions $y = f(x; \theta)$ supporting sequential composition and residual skip connections (`Residual`).
- **Comprehensive Layer Suite**: Dense (Linear), 2D Convolution (`Conv2D` with vectorized im2col), `MaxPool2D`, `Flatten`, `Dropout`, and `BatchNorm`.
- **Rigorously Tested Gradients**: Analytical gradients are validated against numerical finite differences via property-based testing with **Hypothesis** and **pytest**.
- **Accompanying Theory Notes**: Mathematical derivations and conceptual notes available at [barhanc.github.io/notes](https://barhanc.github.io/notes).

## Quickstart

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
from tinny import nn
from tinny.utils import softmax, chunks

# Define a Multi-Layer Perceptron
model = nn.Sequential(
    nn.Linear(vsize=784, hsize=128),
    nn.ReLU(),
    nn.Dropout(p=0.2),
    nn.Linear(vsize=128, hsize=10),
)

optimizer = nn.Adam(model.parameters(), lr=1e-3)

# Training step across mini-batches
for X_batch, y_batch in zip(chunks(X_train, 128), chunks(y_train, 128)):
    # Forward pass
    logits = model.forward(X_batch, training=True)

    # Compute cross-entropy loss gradient
    grad_y = (softmax(logits) - y_batch) / len(X_batch)

    # Backward pass & parameter update
    model.backward(grad_y)
    optimizer.apply(model.gradients())
```
