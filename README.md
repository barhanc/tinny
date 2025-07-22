# nn-core

This is a small educational neural network library implemented using NumPy only, which originated
from my experiments during the Neural Networks and Deep Learning course at AGH University (see the
[course](/notebooks/course/) directory for Jupyter notebooks containing my solutions to course
assignments, along with personal commentary and critique). 

Currently it contains implementations of:

* Energy-based models (EBMs): Restricted Boltzmann Machines (RBMs) and Deep Belief Networks (DBNs),
  trained using Contrastive Divergence (CD-K) and Persistent Contrastive Divergence (PCD).

* Some conventional feed-forward networks trained via backpropagation.

Unlike more general-purpose libraries this project does not implement a full autograd engine.
Instead, it adopts a simplified approach where layers are defined as parametrized tensor functions
of the form `y = Layer(x; θ)`. These functions accept a single tensor input and return a single
tensor output, resulting in a path-like computation graph. While this limits flexibility (e.g. for
branching architectures), it remains extensible. Features like residual connections are still
achievable via helper layers (e.g. a custom `Residual` layer that ensures correct gradient flow).

The current recommended way to install this package is from source
```bash
git clone https://github.com/barhanc/nn-core.git
cd nn-core
python3 -m venv venv
source venv/bin/activate
pip install -e .
```

---

This repository also serves as a personal sandbox for implementing and experimenting with different
models, whether in pure NumPy, PyTorch, or alternative frameworks like tinygrad. As such, you can
expect new and varied commits over time. See the [notebooks](/notebooks/) directory for examples.

You can also find accompanying theory notes and derivations
[here](https://barhanc.github.io/notes/machine-learning/deep/dl.pdf).

