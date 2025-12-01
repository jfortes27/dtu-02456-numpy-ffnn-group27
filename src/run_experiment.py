import argparse, os
import numpy as np
from .train import train
from .data_loaders import load_fashion_mnist
from .activations import softmax
from .losses import cross_entropy
from .utils import one_hot


def evaluate_test_set(model, X_test, y_test):
    """Evaluate accuracy and loss on the test set."""
    logits = model.forward(X_test)
    probs = softmax(logits)

    y_pred = probs.argmax(axis=1)
    acc = float((y_pred == y_test).mean())

    y_oh = one_hot(y_test, int(y_test.max()) + 1)
    loss = cross_entropy(probs, y_oh)

    return acc, loss

def main():
    p = argparse.ArgumentParser(description="Run NumPy FFNN experiment")
    p.add_argument("--dataset", type=str, default="fashion", choices=["fashion", "cifar10"], help="Dataset to use")
    p.add_argument("--hidden", type=int, nargs="+", default=[128], help="Hidden layer sizes, e.g. --hidden 256 128")
    p.add_argument("--activations", type=str, nargs="+", default=["relu"], help="Activations for hidden layers")
    p.add_argument("--optimizer", type=str, default="adam", choices=["adam", "sgd"])
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--dropout_rate", type=float, default=0.0, help="Dropout rate between hidden layers")
    p.add_argument("--l2", type=float, default=0.0)
    p.add_argument("--weight_init", type=str, default="he", choices=["he", "glorot"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--wandb_project", type=str, default="numpy-ffnn-group27")
    args = p.parse_args()

    np.random.seed(args.seed)

    if args.dataset == "fashion":
        X_tr, y_tr, X_val, y_val, X_test, y_test = load_fashion_mnist(
            val_ratio=0.2, test_ratio=0.1, seed=args.seed
        )
    elif args.dataset == "cifar10":
        from .data_loaders import load_cifar10
        X_tr, y_tr, X_val, y_val, X_test, y_test = load_cifar10(
            val_ratio=0.2, seed=args.seed
        )
    else:
        raise ValueError("Unsupported dataset")

    config = {
        "hidden": args.hidden,
        "activations": args.activations,
        "optimizer": args.optimizer,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "l2": args.l2,
        "dropout_rate": args.dropout_rate,
        "weight_init": args.weight_init,
        "seed": args.seed,
        "wandb_project": args.wandb_project,
    }

    model = train(X_tr, y_tr, X_val, y_val, config)

    # TEST SET EVALUATION
    test_acc, test_loss = evaluate_test_set(model, X_test, y_test)

    print("\n==== FINAL TEST EVALUATION ====")
    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"Test Loss:     {test_loss:.4f}")

    # Log to wandb if used
    try:
        import wandb
        wandb.log({
            "test_acc": test_acc,
            "test_loss": test_loss
        })
    except:
        pass
if __name__ == "__main__":
    main()
