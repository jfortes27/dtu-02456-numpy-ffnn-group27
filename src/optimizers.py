import numpy as np

class SGD:
    def __init__(self, lr: float = 1e-2, momentum: float = 0.0):
        self.lr = lr
        self.momentum = momentum
        self.v = {}  # dict of parameter-name -> velocity array

    def step(self, params: dict[str, np.ndarray], grads: dict[str, np.ndarray]) -> None:
        for k in params:
            g = grads[k]
            if self.momentum > 0.0:
                v = self.v.get(k, np.zeros_like(params[k]))
                v = self.momentum * v - self.lr * g
                params[k] += v
                self.v[k] = v
            else:
                params[k] -= self.lr * g

class Adam:
    def __init__(self, lr: float = 1e-3, b1: float = 0.9, b2: float = 0.999, eps: float = 1e-8):
        self.lr, self.b1, self.b2, self.eps = lr, b1, b2, eps
        self.m, self.v = {}, {}
        self.t = 0

    def step(self, params: dict[str, np.ndarray], grads: dict[str, np.ndarray]) -> None:
        self.t += 1
        for k in params:
            g = grads[k]
            m = self.m.get(k, np.zeros_like(g))
            v = self.v.get(k, np.zeros_like(g))
            m = self.b1 * m + (1.0 - self.b1) * g
            v = self.b2 * v + (1.0 - self.b2) * (g * g)
            m_hat = m / (1.0 - self.b1 ** self.t)
            v_hat = v / (1.0 - self.b2 ** self.t)
            params[k] -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
            self.m[k], self.v[k] = m, v
