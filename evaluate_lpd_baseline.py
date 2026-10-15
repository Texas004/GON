"""Evaluate a trained author-style baseline LPD checkpoint on the processed set.

Run from the project root:
    python evaluate_lpd_baseline.py

By default this reports per-sample and mean relative conductivity errors. If the
same samples were used for training, these are training-set diagnostics, not a
held-out test result.
"""
from __future__ import annotations

import os
import numpy as np
import torch
from torch.utils.data import DataLoader

from config import DATA_DIR, PROCESSED_DIR, CHECKPOINT_DIR
from dataset import EITDataset
from eit_operator_nonlinear import NonlinearEITOperator
from lpd_baseline_model import LearnedPrimalDualBaseline, PrimalDual


def main():
    data_path = os.path.join(PROCESSED_DIR, "eit_dataset.npz")
    hypers_path = os.path.join(DATA_DIR, "hypers.db")
    checkpoint_path = os.path.join(CHECKPOINT_DIR, "best_lpd_baseline.pth")

    if not os.path.isfile(data_path):
        raise FileNotFoundError(f"Dataset not found: {data_path}")
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = EITDataset(data_path)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    operator = NonlinearEITOperator(hypers_path, device="cpu")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    num_iter = int(checkpoint["num_iter"])
    model = LearnedPrimalDualBaseline(operator, num_iter=num_iter).to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    model_pd = PrimalDual(operator, num_iter=10, tau=1e-3, sigma=1e-3).to(device)

    gn_errors = []
    lpd_errors = []

    with torch.no_grad():
        for i, batch in enumerate(loader):
            x0 = batch["x0"].to(device=device, dtype=torch.float32)
            y = batch["y"].to(device=device, dtype=torch.float32)
            target = batch["target"].to(device=device, dtype=torch.float32)
            prediction = model(x0, y)
            prediction_pd = model_pd(x0, y)

            denominator = torch.linalg.norm(target).clamp_min(1e-12)
            gn_err = (torch.linalg.norm(x0 - target) / denominator).item()
            pd_err = (torch.linalg.norm(prediction_pd - target) / denominator).item()
            lpd_err = (torch.linalg.norm(prediction - target) / denominator).item()
            gn_errors.append(gn_err)
            lpd_errors.append(lpd_err)
            print(f"Sample {i:03d}: GN relative error={gn_err:.8f}; "
                  f"PD relative error={pd_err:.8f}; "
                  f"LPD relative error={lpd_err:.8f}")

    print("\n===== Summary =====")
    print(f"GN mean relative error:  {np.mean(gn_errors):.8f} "
          f"+/- {np.std(gn_errors):.8f}")
    print(f"LPD mean relative error: {np.mean(lpd_errors):.8f} "
          f"+/- {np.std(lpd_errors):.8f}")
    print(f"LPD better than GN on {sum(l < g for l, g in zip(lpd_errors, gn_errors))} "
          f"/ {len(gn_errors)} samples.")
    print("Reminder: if these samples were used for training, these metrics are not held-out generalization results.")


if __name__ == "__main__":
    main()
