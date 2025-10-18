import os
import numpy as np
from numpy.typing import NDArray, DTypeLike


type FloatType = np.float32 | np.float64
type BoolType = np.bool_
type IntpType = np.intp


# NOTE: This is a hack to allow GPU execution of NumPy code
# fmt:off
if os.environ.get("GPU") == "1":
    try:
        import cupy as cp
        xp = cp
    except ImportError:
        print("Cupy is not available!")
else:
    xp = np
# fmt:on
