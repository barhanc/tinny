import numpy as np
import matplotlib.pyplot as plt

from numpy.typing import NDArray


def tiles(imgs: NDArray):
    """Display 2D matrices as grayscale tiles from a 4D tensor (rows, cols, height, width)."""
    space = 2
    rows, cols, h, w = imgs.shape

    img_matrix = np.empty(shape=(rows * (h + space) - space, cols * (w + space) - space))
    img_matrix.fill(np.nan)

    for r in range(rows):
        for c in range(cols):
            x = r * (h + space)
            y = c * (w + space)
            _min = np.min(imgs[r, c])
            _max = np.max(imgs[r, c])
            img_matrix[x : x + h, y : y + w] = (imgs[r, c] - _min) / (_max - _min)

    plt.matshow(img_matrix, cmap="gray")
    plt.axis("off")
    plt.show()
