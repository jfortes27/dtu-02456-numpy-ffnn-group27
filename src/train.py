import numpy as np
import os
from typing import Dict, Any

try:
    import wandb
    _WANDB = True
except Exception:
    _WANDB = False

from .layers import FFNN
from .activations import softmax
from .losses import cross_entropy
from .optimizers import SGD, Adam
from .utils import one_hot, batch_iter

def train(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    config: Dict[str, Any],
):
    """
    Train a FFNN classifier with softmax + cross-entropy.
    config keys:
        hidden: list[int]
        activations: list[str] (same length as hidden + ["linear"] implied as last)
        optimizer: "adam" | "sgd"
        lr: float
        momentum: float (if sgd)
        l2: float
        batch_size: int
        epochs: int
        weight_init: "he" | "glorot"
        wandb_project: optional str
        seed: int
    """
    seed = int(config.get("seed", 42))
    np.random.seed(seed)

    n_in = X_train.shape[1]
    n_out = int(y_train.max()) + 1

    dims = [n_in] + list(config["hidden"]) + [n_out]
    # final layer should be linear; softmax is applied outside
    activs = list(config["activations"]) + ["linear"]
    weight_init = config.get("weight_init", "he")

    net = FFNN(dims, activs, weight_init=weight_init)

    if config.get("optimizer", "adam").lower() == "adam":
        opt = Adam(lr=config["lr"])
    else:
        opt = SGD(lr=config["lr"], momentum=float(config.get("momentum", 0.0)))

    # W&B (optional)
    if _WANDB and config.get("wandb_project"):
        run = wandb.init(project=config["wandb_project"], config=config)
    else:
        run = None

    def evaluate(X, y):
        logits = net.forward(X)
        probs = softmax(logits)
        y_pred = probs.argmax(axis=1)
        acc = float((y_pred == y).mean())
        y_oh = one_hot(y, n_out)
        loss = cross_entropy(probs, y_oh)
        return acc, loss

    for epoch in range(int(config["epochs"])):
        # training epoch
        for xb, yb in batch_iter(X_train, y_train, int(config["batch_size"]), shuffle=True, seed=epoch + seed):
            yb_oh = one_hot(yb, n_out)
            logits = net.forward(xb)
            probs = softmax(logits)
            # CE + L2
            l2 = float(config.get("l2", 0.0))
            ce = cross_entropy(probs, yb_oh)
            l2_term = 0.5 * l2 * sum((L.W**2).sum() for L in net.layers)
            loss = ce + l2_term

            # gradient at logits for softmax+CE: (p - y)/N
            grad_logits = (probs - yb_oh) / len(xb)
            grads = net.backward(grad_logits, l2=l2)

            # pack params in a dict to pass to optimizer
            params = {}
            for i, L in enumerate(net.layers):
                params[f"W{i}"] = L.W
                params[f"b{i}"] = L.b
            opt.step(params, grads)

        train_acc, train_loss = evaluate(X_train, y_train)
        val_acc, val_loss = evaluate(X_val, y_val)

        if run:
            wandb.log(
                {
                    "epoch": epoch,
                    "train_acc": train_acc,
                    "train_loss": train_loss,
                    "val_acc": val_acc,
                    "val_loss": val_loss,
                }
            )
        print(
            f"epoch {epoch:03d} | train_acc={train_acc:.3f} val_acc={val_acc:.3f} "
            f"train_loss={train_loss:.3f} val_loss={val_loss:.3f}"
        )

    # --- after final epoch ---
    from .utils import confusion_matrix_from_probs

    probs_val = softmax(net.forward(X_val))
    cm = confusion_matrix_from_probs(probs_val, y_val)
    print("\nConfusion matrix on validation set:")
    print(cm)


    if run:
        run.finish()
    return net

