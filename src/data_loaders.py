import numpy as np
from typing import Tuple
from sklearn.datasets import fetch_openml
from .utils import train_val_split, train_test_split

def load_fashion_mnist(val_ratio: float = 0.2, seed: int = 42) -> Tuple[np.ndarray, ...]:
    """
    Loads Fashion-MNIST from OpenML (no TF/PyTorch), normalizes to [0,1],
    flattens to vectors, returns train/val/test splits.
    """
    ds = fetch_openml("Fashion-MNIST", version=1, as_frame=False)
    X = ds["data"].astype(np.float32) / 255.0   # (70000, 784)
    y = ds["target"].astype(np.int64)  


    X_tr, y_tr, X_te, y_te = train_test_split(X, y, test_ratio=0.2, seed=seed) 
    X_tr, y_tr, X_va, y_va = train_val_split(X_tr, y_tr, val_ratio=val_ratio, seed=seed)
    return X_tr, y_tr, X_va, y_va, X_te, y_te

def load_cifar10(val_ratio: float = 0.2, seed: int = 42):
    """
    Loads CIFAR-10 from OpenML (no TF/PyTorch), normalizes to [0,1],
    flattens to 3072-dim vectors, returns train/val/test splits.
    """
    ds = fetch_openml("CIFAR_10", version=1, as_frame=False)

    X = ds["data"].astype(np.float32) / 255.0
    y = ds["target"].astype(np.int64)

    X_img = X.reshape(-1, 32, 32, 3)


    X_gray = (
        0.299 * X_img[:, :, :, 0] +
        0.587 * X_img[:, :, :, 1] +
        0.114 * X_img[:, :, :, 2]
    )


    X_flat = X_gray.reshape(len(X_gray), -1)

    X_tr, y_tr, X_te, y_te = train_test_split(X_flat, y, test_ratio=0.2, seed=seed) 
    X_tr, y_tr, X_va, y_va = train_val_split(X_tr, y_tr, val_ratio=val_ratio, seed=seed)
    return X_tr, y_tr, X_va, y_va, X_te, y_te

    