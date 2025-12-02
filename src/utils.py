import numpy as np
from sklearn.metrics import confusion_matrix
from typing import Iterator, Tuple

def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)

def one_hot(y: np.ndarray, n_classes: int) -> np.ndarray:
    y = y.astype(int).ravel()
    oh = np.zeros((y.size, n_classes), dtype=float)
    oh[np.arange(y.size), y] = 1.0
    return oh

def batch_iter(
    X: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool = True, seed: int | None = None
) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    n = len(X)
    idx = np.arange(n)
    if shuffle:
        rng = np.random.default_rng(seed)
        rng.shuffle(idx)
    for i in range(0, n, batch_size):
        sel = idx[i : i + batch_size]
        yield X[sel], y[sel]

def train_val_split(X, y, val_ratio: float = 0.2, seed: int = 42):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X))
    n_val = int(len(X) * val_ratio)
    val_idx, train_idx = idx[:n_val], idx[n_val:]
    return (X[train_idx], y[train_idx], X[val_idx], y[val_idx])

def train_test_split(X, y, test_ratio: float = 0.2, seed: int = 42):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X))
    n_test = int(len(X) * test_ratio)
    test_idx, train_idx = idx[:n_test], idx[n_test:]
    return (X[train_idx], y[train_idx], X[test_idx], y[test_idx])

def confusion_matrix_from_probs(probs: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    y_pred = probs.argmax(axis=1)
    return confusion_matrix(y_true, y_pred)