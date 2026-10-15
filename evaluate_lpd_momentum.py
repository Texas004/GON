"""Evaluate the explicit-momentum nonlinear EIT LPD checkpoint.

Run from the project root:
    python evaluate_lpd_momentum.py
"""
from __future__ import annotations

import os
import numpy as np
import torch
from torch.utils.data import DataLoader

from config import DATA_DIR, PROCESSED_DIR, CHECKPOINT_DIR
from dataset import EITDataset
from eit_operator_nonlinear import NonlinearEITOperator
from lpd_momentum_model import LearnedPrimalDualMomentum


def main():
    data_path = os.path.join(PROCESSED_DIR, "eit_dataset.npz")
    hypers_path = os.path.join(DATA_DIR, "hypers.db")
    checkpoint_path = os.path.join(CHECKPOINT_DIR, "best_lpd_momentum.pth")

    if not os.path.isfile(data_path):
        raise FileNotFoundError(f"Dataset not found: {data_path}")
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}. Run train_lpd_momentum.py first."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = EITDataset(data_path)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    operator = NonlinearEITOperator(hypers_path, device="cpu")

    try:
        checkpoint = torch.load(
            checkpoint_path, map_location=device, weights_only=False
        )
    except TypeError:  # compatibility with older PyTorch versions
        checkpoint = torch.load(checkpoint_path, map_location=device)

    num_iter = int(checkpoint["num_iter"])
    model = LearnedPrimalDualMomentum(
        operator,
        num_iter=num_iter,
        momentum_decay=float(checkpoint.get("momentum_decay", 0.8)),
        momentum_step=float(checkpoint.get("momentum_step", 1e-3)),
    ).to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()

    gn_errors = []
    momentum_errors = []

    with torch.no_grad():
        for i, batch in enumerate(loader):
            x0 = batch["x0"].to(device=device, dtype=torch.float32)
            y = batch["y"].to(device=device, dtype=torch.float32)
            target = batch["target"].to(device=device, dtype=torch.float32)
            prediction = model(x0, y)

            denominator = torch.linalg.norm(target).clamp_min(1e-12)
            gn_err = (torch.linalg.norm(x0 - target) / denominator).item()
            momentum_err = (
                torch.linalg.norm(prediction - target) / denominator
            ).item()
            gn_errors.append(gn_err)
            momentum_errors.append(momentum_err)
            print(
                f"Sample {i:03d}: GN relative error={gn_err:.8f}; "
                f"Momentum-LPD relative error={momentum_err:.8f}"
            )

    print("\n===== Summary =====")
    print(
        f"GN mean relative error:       {np.mean(gn_errors):.8f} "
        f"+/- {np.std(gn_errors):.8f}"
    )
    print(
        f"Momentum-LPD mean relative error: {np.mean(momentum_errors):.8f} "
        f"+/- {np.std(momentum_errors):.8f}"
    )
    print(
        "Momentum-LPD better than GN on "
        f"{sum(m < g for m, g in zip(momentum_errors, gn_errors))} "
        f"/ {len(gn_errors)} samples."
    )
    print(
        "Reminder: if these samples were used for training, this is not a "
        "held-out generalization result."
    )


if __name__ == "__main__":
    main()
