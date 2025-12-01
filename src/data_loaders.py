import numpy as np
from typing import Tuple
from sklearn.datasets import fetch_openml
from keras.datasets import cifar10
from .utils import train_val_split


def load_fashion_mnist(val_ratio: float = 0.1, test_ratio: float = 0.1, seed: int = 42
                       ) -> Tuple[np.ndarray, ...]:
    """
    Loads Fashion-MNIST from OpenML, normalizes to [0,1],
    splits into train/val/test sets.
    """
    # Load entire dataset from OpenML
    ds = fetch_openml("Fashion-MNIST", version=1, as_frame=False)
    X = ds["data"].astype(np.float32) / 255.0     # (70000, 784)
    y = ds["target"].astype(np.int64)

    # Shuffle first for consistent splitting
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X))
    X, y = X[idx], y[idx]

    # Compute split sizes
    n_total = len(X)
    n_test = int(n_total * test_ratio)
    n_val = int(n_total * val_ratio)

    # Test set
    X_test = X[:n_test]
    y_test = y[:n_test]

    # Train + validation pool
    X_pool = X[n_test:]
    y_pool = y[n_test:]

    # Now split pool into train/val
    X_train, y_train, X_val, y_val = train_val_split(X_pool, y_pool,
                                                     val_ratio=val_ratio,
                                                     seed=seed)

    return X_train, y_train, X_val, y_val, X_test, y_test



def load_cifar10(val_ratio: float = 0.2, seed: int = 42
                 ) -> Tuple[np.ndarray, ...]:
    """
    Loads CIFAR-10 from keras.datasets.
    Normalizes to [0,1], flattens images,
    returns train/val/test splits.
    """
    # Load raw CIFAR-10
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()

    # Normalize to [0,1]
    X_train = X_train.astype(np.float32) / 255.0
    X_test  = X_test.astype(np.float32) / 255.0

    # Flatten: (N, 32,32,3) -> (N, 3072)
    X_train = X_train.reshape(len(X_train), -1)
    X_test  = X_test.reshape(len(X_test), -1)

    # Flatten labels
    y_train = y_train.reshape(-1).astype(np.int64)
    y_test  = y_test.reshape(-1).astype(np.int64)

    # Train → Train/Validation split
    X_train, y_train, X_val, y_val = train_val_split(
        X_train, y_train, val_ratio=val_ratio, seed=seed
    )

    return X_train, y_train, X_val, y_val, X_test, y_test
