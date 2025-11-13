import numpy as np

def cross_entropy(probs: np.ndarray, y_onehot: np.ndarray, eps: float = 1e-12) -> float:
    """Mean cross-entropy for one-hot targets."""
    p = np.clip(probs, eps, 1.0 - eps)
    return float(-np.mean(np.sum(y_onehot * np.log(p), axis=1)))

def mse(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    return float(np.mean((y_pred - y_true) ** 2))

