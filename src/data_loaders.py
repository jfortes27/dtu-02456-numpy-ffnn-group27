import numpy as np
from typing import Tuple
from sklearn.datasets import fetch_openml
from .utils import train_val_split

def load_fashion_mnist(val_ratio: float = 0.2, seed: int = 42) -> Tuple[np.ndarray, ...]:
    """
    Loads Fashion-MNIST from OpenML (no TF/PyTorch), normalizes to [0,1],
    flattens to vectors, returns train/val splits.
    """
    # OpenML name is exactly "Fashion-MNIST"
    ds = fetch_openml("Fashion-MNIST", version=1, as_frame=False)
    X = ds["data"].astype(np.float32) / 255.0   # (70000, 784)
    y = ds["target"].astype(np.int64)           # labels 0..9 as strings sometimes -> cast

    X_tr, y_tr, X_va, y_va = train_val_split(X, y, val_ratio=val_ratio, seed=seed)
    return X_tr, y_tr, X_va, y_va
