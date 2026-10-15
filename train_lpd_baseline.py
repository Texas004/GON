"""Train the author's baseline LPD structure with the nonlinear EIT operator.

Run from the project root:
    python train_lpd_baseline.py

This is the baseline stage only; LSTM-RMA and GRU-RMA are intentionally omitted.
"""
from __future__ import annotations

import argparse
import os
import random

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from config import DATA_DIR, PROCESSED_DIR, CHECKPOINT_DIR
from dataset import EITDataset
from eit_operator_nonlinear import NonlinearEITOperator
from lpd_baseline_model import LearnedPrimalDualBaseline


def parse_args():
    parser = argparse.ArgumentParser(description="Train author-style baseline LPD for EIT")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num-iter", type=int, default=10)
    parser.add_argument("--tv-weight", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=2022)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument(
        "--data-path",
        default=os.path.join(PROCESSED_DIR, "eit_dataset.npz"),
    )
    parser.add_argument(
        "--hypers-path",
        default=os.path.join(DATA_DIR, "hypers.db"),
    )
    parser.add_argument(
        "--save-dir",
        default=CHECKPOINT_DIR,
    )
    return parser.parse_args()


def tv_loss_flat_mesh(x: torch.Tensor) -> torch.Tensor:
    """Same flattened-neighbor TV expression used in author's a_utils.py."""
    x3 = x.unsqueeze(1) if x.ndim == 2 else x
    return torch.sum(torch.abs(x3[:, :, 1:] - x3[:, :, :-1]))


def main():
    args = parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.num_iter < 1:
        raise ValueError("epochs, batch-size and num-iter must all be positive.")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        if args.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("--device cuda was requested but CUDA is not available.")
        device = torch.device(args.device)

    if not os.path.isfile(args.data_path):
        raise FileNotFoundError(f"Dataset not found: {args.data_path}")
    if not os.path.exists(args.hypers_path):
        # shelve databases can be represented by platform-dependent sidecar files;
        # check the base path plus common DBM suffixes.
        candidates = [args.hypers_path + suffix for suffix in (".db", ".dat", ".dir", ".bak")]
        found = [p for p in candidates if os.path.exists(p)]
        if not found:
            raise FileNotFoundError(f"Hyperparameter shelf not found: {args.hypers_path}")

    dataset = EITDataset(args.data_path)
    if len(dataset) == 0:
        raise ValueError("The EIT dataset contains no samples.")
    loader = DataLoader(
        dataset,
        batch_size=min(args.batch_size, len(dataset)),
        shuffle=True,
        num_workers=0,
        drop_last=False,
    )

    operator = NonlinearEITOperator(args.hypers_path, device="cpu")
    model = LearnedPrimalDualBaseline(operator, num_iter=args.num_iter).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.99))
    mse = nn.MSELoss()

    os.makedirs(args.save_dir, exist_ok=True)
    checkpoint_path = os.path.join(args.save_dir, "best_lpd_baseline.pth")
    best_loss = float("inf")

    print(f"Device: {device}")
    print(f"Samples: {len(dataset)}")
    print(f"Epochs: {args.epochs}; batch size: {loader.batch_size}; iterations: {args.num_iter}")
    print(f"Data: {os.path.abspath(args.data_path)}")
    print(f"Hyperparameters: {os.path.abspath(args.hypers_path)}")
    print(f"Checkpoint: {os.path.abspath(checkpoint_path)}")
    print("Note: this dataset is currently used for training-loss monitoring only; "
          "the reported training loss is not a held-out generalization metric.")

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        n_seen = 0

        for batch in loader:
            x0 = batch["x0"].to(device=device, dtype=torch.float32)
            y = batch["y"].to(device=device, dtype=torch.float32)
            target = batch["target"].to(device=device, dtype=torch.float32)

            optimizer.zero_grad(set_to_none=True)
            prediction = model(x0, y)
            loss_mse = mse(prediction, target)
            loss = loss_mse
            if args.tv_weight != 0.0:
                loss = loss + args.tv_weight * tv_loss_flat_mesh(prediction)

            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"Non-finite loss at epoch {epoch + 1}: {loss.item()}"
                )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            batch_n = target.shape[0]
            total_loss += loss.item() * batch_n
            n_seen += batch_n

        avg_loss = total_loss / max(n_seen, 1)
        print(f"Epoch [{epoch + 1:03d}/{args.epochs:03d}] loss={avg_loss:.8e}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(
                {
                    "model": model.state_dict(),
                    "epoch": epoch + 1,
                    "loss": best_loss,
                    "num_iter": args.num_iter,
                    "n_primal": 5,
                    "n_dual": 5,
                    "n_elements": int(dataset[0]["target"].numel()),
                    "n_measurements": int(dataset[0]["y"].numel()),
                    "model_type": "author_baseline_lpd_nonlinear_eit",
                    "seed": args.seed,
                    "args": vars(args),
                },
                checkpoint_path,
            )
            print("  Saved new best checkpoint.")

    print(f"Training finished. Best training loss: {best_loss:.8e}")
    print("This is the baseline LPD stage. It does not yet include RMA/LSTM.")


if __name__ == "__main__":
    main()
