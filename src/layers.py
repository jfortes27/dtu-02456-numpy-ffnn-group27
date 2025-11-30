import numpy as np
from typing import List
from .activations import relu, relu_grad, tanh, tanh_grad

_ACT = {
    "relu": (relu, relu_grad),
    "tanh": (tanh, tanh_grad),
    "linear": (lambda x: x, lambda x: np.ones_like(x)),
}

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

    def forward(self, x: np.ndarray) -> np.ndarray:
        self._x = x
        self._z = x @ self.W + self.b
        return self.f(self._z)

    def backward(self, grad_out: np.ndarray, l2: float = 0.0):
        if self._x is None or self._z is None:
            raise ValueError("Dense.backward called before forward; no cached input/activation found. Run forward(...) before backward(...).")
        grad_z = grad_out * self.fprime(self._z)
        dW = self._x.T @ grad_z + l2 * self.W
        db = np.sum(grad_z, axis=0, keepdims=True)
        dx = grad_z @ self.W.T
        return dx, {"W": dW, "b": db}

class FFNN:
    
    def __init__(self, dims: List[int], activations: List[str], weight_init: str = "he",dropout_rate: float = 0.0):
        assert len(dims) - 1 == len(activations), "Provide activations per layer."
        self.layers = []
        L = len(dims) - 1
        for i in range(L):
            #Dense layer
            self.layers.append(
                Dense(
                    dims[i],
                    dims[i+1],
                    activation=activations[i],
                    weight_init=weight_init
                )
            )
            # Dropout between hidden layers (NOT after last layer)
            if dropout_rate > 0 and i < L - 1:
                self.layers.append(Dropout(p=dropout_rate))
            
    def forward(self, x: np.ndarray) -> np.ndarray:
        h = x
        for L in self.layers:
            h = L.forward(h)
        return h
    

    def backward(self, grad_last: np.ndarray, l2: float = 0.0):
        grads = {}
        g = grad_last
        for i in reversed(range(len(self.layers))):
            g, g_i = self.layers[i].backward(g, l2=l2)
            # Only Dense layers have weights
            if "W" in g_i:
                grads[f"W{i}"] = g_i["W"]
                grads[f"b{i}"] = g_i["b"]
        return grads
    
    def train_mode(self):
        for L in self.layers:
            if hasattr(L, "training"):
                L.training = True

    def eval_mode(self):
        for L in self.layers:
            if hasattr(L, "training"):
                L.training = False

    @property
    def params(self):
        P = {}
        for i, L in enumerate(self.layers):
            P[f"W{i}"] = L.W
            P[f"b{i}"] = L.b
        return P
    
    

class Dropout:
    def __init__(self, p=0.5):
        self.p = p
        self.mask = None
        self.training = True

    def forward(self, x):
        if self.training:
            self.mask = (np.random.rand(*x.shape) > self.p).astype(x.dtype)
            return x * self.mask / (1.0 - self.p)
        else:
            return x

    def backward(self, grad_out, l2=0.0):
        if self.training:
            return grad_out * self.mask / (1.0 - self.p), {}
        else:
            return grad_out, {}

    def parameters(self):
        return {}
