import argparse, os
import numpy as np
from .train import train
from .data_loaders import load_fashion_mnist

def main():
    p = argparse.ArgumentParser(description="Run NumPy FFNN experiment")
    p.add_argument("--dataset", type=str, default="fashion", choices=["fashion"], help="Dataset to use")
    p.add_argument("--hidden", type=int, nargs="+", default=[128], help="Hidden layer sizes, e.g. --hidden 256 128")
    p.add_argument("--activations", type=str, nargs="+", default=["relu"], help="Activations for hidden layers")
    p.add_argument("--optimizer", type=str, default="adam", choices=["adam", "sgd"])
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--l2", type=float, default=1e-4)
    p.add_argument("--weight_init", type=str, default="he", choices=["he", "glorot"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--wandb_project", type=str, default="numpy-ffnn-group27")
    args = p.parse_args()

    np.random.seed(args.seed)

    if args.dataset == "fashion":
        X_tr, y_tr, X_va, y_va = load_fashion_mnist(val_ratio=0.2, seed=args.seed)
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
        "weight_init": args.weight_init,
        "seed": args.seed,
        "wandb_project": args.wandb_project,
    }

    _ = train(X_tr, y_tr, X_va, y_va, config)

if __name__ == "__main__":
    main()
