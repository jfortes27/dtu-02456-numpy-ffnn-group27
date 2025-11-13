import numpy as np

def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0, x)

def relu_grad(x: np.ndarray) -> np.ndarray:
    return (x > 0).astype(x.dtype)

def tanh(x: np.ndarray) -> np.ndarray:
    return np.tanh(x)

def tanh_grad(x: np.ndarray) -> np.ndarray:
    t = np.tanh(x)
    return 1.0 - t * t

def softmax(x: np.ndarray) -> np.ndarray:
    """Row-wise stable softmax."""
    x_shift = x - np.max(x, axis=1, keepdims=True)
    ex = np.exp(x_shift)
    return ex / np.sum(ex, axis=1, keepdims=True)

