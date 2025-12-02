import numpy as np
from typing import List
from .activations import relu, relu_grad, tanh, tanh_grad, identity, identity_grad

_ACT = {
    "relu": (relu, relu_grad),
    "tanh": (tanh, tanh_grad),
    "identity": (identity, identity_grad)
}


def dropout(x, drop_prob, seed=None):
    if drop_prob <= 0.0:
        return x, None
    if seed is not None:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()
    mask = rng.random(x.shape) > drop_prob
    out = x * mask / (1.0 - drop_prob)
    return out, mask


class Dense:
    def __init__(self, in_dim: int, out_dim: int, activation: str = "relu", weight_init: str = "he"):
        self.in_dim, self.out_dim = in_dim, out_dim
        self.act_name = activation
        self.f, self.fprime = _ACT[activation]

        if weight_init == "he":
            self.W = np.random.randn(in_dim, out_dim) * np.sqrt(2.0 / in_dim)
        elif weight_init == "glorot":
            lim = np.sqrt(6.0 / (in_dim + out_dim))
            self.W = np.random.uniform(-lim, lim, size=(in_dim, out_dim))
        else:
            self.W = np.random.randn(in_dim, out_dim) * 0.01

        self.b = np.zeros((1, out_dim))

        self._x = None
        self._z = None
        self._drop_mask = None
        self.dropout_prob = 0.0 

    def forward(self, x: np.ndarray, train: bool = False) -> np.ndarray:
        self._x = x
        self._z = x @ self.W + self.b
        h = self.f(self._z)

        # Apply dropout if in training mode
        if train and self.dropout_prob > 0.0:
            h, mask = dropout(h, self.dropout_prob)
            self._drop_mask = mask
        else:
            self._drop_mask = None

        return h

    def backward(self, grad_out: np.ndarray, l2: float = 0.0):
        if self._x is None or self._z is None:
            raise ValueError("Dense.backward called before forward.")

        # Apply dropout mask on backward
        if self._drop_mask is not None:
            grad_out = grad_out * self._drop_mask / (1.0 - self.dropout_prob)

        grad_z = grad_out * self.fprime(self._z)
        dW = self._x.T @ grad_z + l2 * self.W
        db = np.sum(grad_z, axis=0, keepdims=True)
        dx = grad_z @ self.W.T
        return dx, {"W": dW, "b": db}


class FFNN:
    def __init__(self, dims: List[int], activations: List[str], weight_init: str = "he"):
        assert len(dims) - 1 == len(activations), "Provide activations per layer."
        self.layers = [
            Dense(dims[i], dims[i + 1], activation=activations[i], weight_init=weight_init)
            for i in range(len(activations))
        ]

    def forward(self, x: np.ndarray, train: bool = False) -> np.ndarray:
        h = x
        for L in self.layers:
            h = L.forward(h, train=train)
        return h

    def backward(self, grad_last: np.ndarray, l2: float = 0.0):
        grads = {}
        g = grad_last
        for i in reversed(range(len(self.layers))):
            g, g_i = self.layers[i].backward(g, l2=l2)
            grads[f"W{i}"] = g_i["W"]
            grads[f"b{i}"] = g_i["b"]
        return grads

    @property
    def params(self):
        P = {}
        for i, L in enumerate(self.layers):
            P[f"W{i}"] = L.W
            P[f"b{i}"] = L.b
        return P
