import os
import tarfile
import pickle
from pathlib import Path
from typing import Union
import gzip
import struct
from urllib.request import urlretrieve
import numpy as np
from .utils import train_val_split


_CIFAR_URL = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"

def _download_and_extract_cifar10(data_root: Union[str, Path] = "data") -> Path:
    """
    Download and extract CIFAR-10 (python version) if not already present.
    Returns the path to the cifar-10-batches-py directory.
    """
    project_root = Path(__file__).resolve().parents[1]
    data_root = project_root / data_root
    cifar_dir = data_root / "cifar-10-batches-py"

    if cifar_dir.exists():
        return cifar_dir

    data_root.mkdir(parents=True, exist_ok=True)
    tar_path = data_root / "cifar-10-python.tar.gz"

    if not tar_path.exists():
        print(f"Downloading CIFAR-10 from {_CIFAR_URL} ...")
        urlretrieve(_CIFAR_URL, tar_path)
        print("Download complete.")

    print("Extracting CIFAR-10 archive...")
    with tarfile.open(tar_path, "r:gz") as tar:
        tar.extractall(path=data_root)
    print("Extraction complete.")

    return cifar_dir


def _load_cifar10_numpy() -> tuple[np.ndarray, np.ndarray]:
    """
    Load CIFAR-10 training data (50k images) as NumPy arrays.
    Returns X (N, 3072) in [0,1] and y (N,) int labels 0..9.
    """
    cifar_dir = _download_and_extract_cifar10()
    data_list = []
    labels_list = []

    # training batches data_batch_1..5
    for i in range(1, 6):
        batch_path = cifar_dir / f"data_batch_{i}"
        with open(batch_path, "rb") as f:
            batch = pickle.load(f, encoding="bytes")
        data_list.append(batch[b"data"])        # shape (10000, 3072)
        labels_list.extend(batch[b"labels"])    # list of length 10000

    X = np.vstack(data_list).astype(np.float32) / 255.0  # (50000, 3072)
    y = np.array(labels_list, dtype=np.int64)            # (50000,)
    return X, y


def load_cifar10(val_ratio: float = 0.2, seed: int = 42):
    """
    Public loader for CIFAR-10, returning train/val split:
    X_train, y_train, X_val, y_val
    """
    X, y = _load_cifar10_numpy()
    return train_val_split(X, y, val_ratio=val_ratio, seed=seed)




_FASHION_BASE_URL = "https://github.com/zalandoresearch/fashion-mnist/raw/master/data/fashion"


def _download_fashion_file(filename: str, data_root: Union[str, Path] = "data") -> Path:
    project_root = Path(__file__).resolve().parents[1]
    data_root = project_root / data_root / "fashion-mnist"
    data_root.mkdir(parents=True, exist_ok=True)

    out_path = data_root / filename
    if not out_path.exists():
        url = f"{_FASHION_BASE_URL}/{filename}"
        print(f"Downloading {filename} from {url} ...")
        urlretrieve(url, out_path)
        print("Download complete.")
    return out_path


def _read_idx_images(path: Path) -> np.ndarray:
    with gzip.open(path, "rb") as f:
        magic, num, rows, cols = struct.unpack(">IIII", f.read(16))
        data = np.frombuffer(f.read(), dtype=np.uint8)
    return data.reshape(num, rows * cols)


def _read_idx_labels(path: Path) -> np.ndarray:
    with gzip.open(path, "rb") as f:
        magic, num = struct.unpack(">II", f.read(8))
        labels = np.frombuffer(f.read(), dtype=np.uint8)
    return labels


def load_fashion_mnist(val_ratio: float = 0.2, seed: int = 42):
    """
    Load Fashion-MNIST (no OpenML, no TF) as NumPy arrays.
    Returns X_train, y_train, X_val, y_val with X in [0,1].
    """
    # Download raw gzip files (if needed)
    train_images_path = _download_fashion_file("train-images-idx3-ubyte.gz")
    train_labels_path = _download_fashion_file("train-labels-idx1-ubyte.gz")
    test_images_path  = _download_fashion_file("t10k-images-idx3-ubyte.gz")
    test_labels_path  = _download_fashion_file("t10k-labels-idx1-ubyte.gz")

    # Load arrays
    X_train = _read_idx_images(train_images_path).astype(np.float32) / 255.0
    y_train = _read_idx_labels(train_labels_path).astype(np.int64)

    X_test = _read_idx_images(test_images_path).astype(np.float32) / 255.0
    y_test = _read_idx_labels(test_labels_path).astype(np.int64)

    # Option 1: use only original train split and carve out validation
    X = X_train
    y = y_train

    # If you prefer train+test all together, use:
    # X = np.vstack([X_train, X_test])
    # y = np.concatenate([y_train, y_test])

    return train_val_split(X, y, val_ratio=val_ratio, seed=seed)
