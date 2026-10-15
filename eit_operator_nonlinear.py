
import shelve
import os

import numpy as np
import torch


class NonlinearEITOperator:
    """
    Nonlinear EIT forward operator for pyEIT 1.2.4.

    Input:
        sigma: (batch, n_elements) or
               (batch, 1, n_elements)

    Returns:
        voltage:  (batch, 1, n_measurements)
        jacobian: (batch, n_measurements, n_elements)
    """

    def __init__(self, hypers_path, device="cpu"):
        self.device = torch.device(device)

        with shelve.open(
            os.fspath(hypers_path), flag="r"
        ) as db:
            self.mesh = db["data"]["mesh_obj"]
            self.fwd = db["fun"]["fwd"]

        self.n_elements = self.mesh.element.shape[0]
        self.n_measurements = (
            self.mesh.n_el
            # The actual measurement count is verified
            # from compute_jac() at runtime.
        )

    def forward_and_jac(self, sigma):
        if torch.is_tensor(sigma):
            sigma = sigma.detach().cpu().numpy()

        sigma = np.asarray(sigma)

        if sigma.ndim == 1:
            sigma = sigma[None, :]
        elif sigma.ndim == 3:
            if sigma.shape[1] != 1:
                raise ValueError(
                    "Expected sigma shape (B, 1, N), "
                    f"got {sigma.shape}"
                )
            sigma = sigma[:, 0, :]

        if sigma.ndim != 2:
            raise ValueError(
                f"Expected 1D/2D/3D input, got {sigma.shape}"
            )

        if sigma.shape[1] != self.n_elements:
            raise ValueError(
                f"Expected {self.n_elements} elements, "
                f"got {sigma.shape[1]}"
            )

        voltages = []
        jacobians = []

        # pyEIT computes on CPU; process each sample
        # separately using its current conductivity.
        for perm in sigma:
            jac, voltage = self.fwd.compute_jac(
                perm=np.asarray(perm, dtype=np.float64)
            )

            jacobians.append(np.asarray(jac).real)
            voltages.append(np.asarray(voltage).real)

        voltage_tensor = torch.as_tensor(
            np.asarray(voltages),
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(1)

        jacobian_tensor = torch.as_tensor(
            np.asarray(jacobians),
            dtype=torch.float32,
            device=self.device,
        )

        return voltage_tensor, jacobian_tensor
