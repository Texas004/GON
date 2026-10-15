"""Author-structure baseline LPD adapted to the nonlinear EIT operator.

This implements the author's ``grad_type='baseline'`` path from primaldual.py:
- independent dual/primal CNNs per iteration;
- nonlinear forward evaluation A(x_k);
- conductivity-dependent Jacobian J(x_k);
- primal and dual state channels.

The recurrent momentum (LSTM/GRU) paths are intentionally not included yet.
"""
from __future__ import annotations

from typing import Sequence

import torch
from torch import nn


class ConcatenateLayer(nn.Module):
    """Concatenate tensors along the channel dimension."""

    def forward(self, *tensors: torch.Tensor) -> torch.Tensor:
        if not tensors:
            raise ValueError("ConcatenateLayer requires at least one tensor.")
        return torch.cat(list(tensors), dim=1)


class DualNet(nn.Module):
    """Author-style dual update network (1D convolutions over measurements)."""

    def __init__(self, n_dual: int = 5) -> None:
        super().__init__()
        self.n_dual = n_dual
        self.input_concat_layer = ConcatenateLayer()
        self.block = nn.Sequential(
            nn.Conv1d(n_dual + 2, 32, kernel_size=3, padding=1),
            nn.PReLU(),
            nn.Conv1d(32, 32, kernel_size=3, padding=1),
            nn.PReLU(),
            nn.Conv1d(32, n_dual, kernel_size=3, padding=1),
        )

    def forward(
        self,
        h: torch.Tensor,
        op_values: Sequence[torch.Tensor],
        measurements: torch.Tensor,
    ) -> torch.Tensor:
        # h: (B, n_dual, M), op_values[0]: (B, 1, M),
        # measurements: (B, 1, M)
        z = self.input_concat_layer(h, *op_values, measurements)
        return h + self.block(z)


class PrimalNet(nn.Module):
    """Author-style primal update network (1D convolutions over mesh elements)."""

    def __init__(self, n_primal: int = 5, n_gradient_channels: int = 1) -> None:
        super().__init__()
        self.n_primal = n_primal
        self.input_concat_layer = ConcatenateLayer()
        self.block = nn.Sequential(
            nn.Conv1d(n_primal + n_gradient_channels, 32, kernel_size=3, padding=1),
            nn.PReLU(),
            nn.Conv1d(32, 32, kernel_size=3, padding=1),
            nn.PReLU(),
            nn.Conv1d(32, 32, kernel_size=3, padding=1),
            nn.PReLU(),
            nn.Conv1d(32, n_primal, kernel_size=3, padding=1),
        )

    def forward(
        self,
        f: torch.Tensor,
        adjoint_gradients: Sequence[torch.Tensor],
    ) -> torch.Tensor:
        z = self.input_concat_layer(f, *adjoint_gradients)
        return f + self.block(z)


def _as_channel_tensor(
    value: torch.Tensor,
    name: str,
) -> torch.Tensor:
    """Normalize (B,N) or (B,1,N) inputs to (B,1,N)."""
    if value.ndim == 2:
        return value.unsqueeze(1)
    if value.ndim == 3 and value.shape[1] == 1:
        return value
    raise ValueError(
        f"{name} must have shape (B,N) or (B,1,N); got {tuple(value.shape)}"
    )


class LearnedPrimalDualBaseline(nn.Module):
    """Nonlinear EIT baseline LPD using the author's network structure.

    Args:
        operator: object with ``forward_and_jac(sigma)`` returning
            voltage ``(B,1,M)`` and Jacobian ``(B,M,N)``.
        num_iter: number of unrolled iterations (author default is 10).
        n_primal: number of primal state channels (author default is 5).
        n_dual: number of dual state channels (author default is 5).

    The public output is (B,N), matching this project's EITDataset.
    Internally, tensors use (B,channels,length), as in the author's code.
    """

    def __init__(
        self,
        operator,
        num_iter: int = 10,
        n_primal: int = 5,
        n_dual: int = 5,
        expected_measurements: int = 208,
    ) -> None:
        super().__init__()
        if num_iter < 1:
            raise ValueError("num_iter must be at least 1.")
        self.operator = operator
        self.num_iter = int(num_iter)
        self.n_primal = int(n_primal)
        self.n_dual = int(n_dual)
        self.expected_measurements = int(expected_measurements)

        # The author creates independent networks for each unrolled iteration.
        self.primal_nets = nn.ModuleList(
            [PrimalNet(n_primal=self.n_primal, n_gradient_channels=1)
             for _ in range(self.num_iter)]
        )
        self.dual_nets = nn.ModuleList(
            [DualNet(n_dual=self.n_dual) for _ in range(self.num_iter)]
        )

    def _operator_call(self, sigma: torch.Tensor):
        """Call the CPU pyEIT operator and move outputs back to model device."""
        voltage, jacobian = self.operator.forward_and_jac(sigma)
        voltage = voltage.to(device=sigma.device, dtype=sigma.dtype)
        jacobian = jacobian.to(device=sigma.device, dtype=sigma.dtype)

        if voltage.ndim != 3 or voltage.shape[1] != 1:
            raise RuntimeError(
                "operator.forward_and_jac must return voltage shaped (B,1,M); "
                f"got {tuple(voltage.shape)}"
            )
        if jacobian.ndim != 3:
            raise RuntimeError(
                "operator.forward_and_jac must return Jacobian shaped (B,M,N); "
                f"got {tuple(jacobian.shape)}"
            )
        if voltage.shape[0] != sigma.shape[0] or jacobian.shape[0] != sigma.shape[0]:
            raise RuntimeError("Operator output batch size does not match its input.")
        if voltage.shape[-1] != self.expected_measurements:
            raise RuntimeError(
                f"Expected {self.expected_measurements} measurements, "
                f"got {voltage.shape[-1]}"
            )
        if jacobian.shape[1] != voltage.shape[-1] or jacobian.shape[2] != sigma.shape[-1]:
            raise RuntimeError(
                "Jacobian shape must be (B, number_of_measurements, number_of_elements); "
                f"got {tuple(jacobian.shape)} for sigma {tuple(sigma.shape)}"
            )
        return voltage, jacobian

    def forward(self, x_gn: torch.Tensor, measurements: torch.Tensor) -> torch.Tensor:
        x_gn = _as_channel_tensor(x_gn, "x_gn")
        measurements = _as_channel_tensor(measurements, "measurements")

        if x_gn.shape[0] != measurements.shape[0]:
            raise ValueError("x_gn and measurements must have the same batch size.")
        if measurements.shape[-1] != self.expected_measurements:
            raise ValueError(
                f"Expected {self.expected_measurements} measurement values, "
                f"got {measurements.shape[-1]}"
            )

        batch_size, _, n_elements = x_gn.shape
        device, dtype = x_gn.device, x_gn.dtype
        measurements = measurements.to(device=device, dtype=dtype)

        # Author's initial primal state: repeat the GN reconstruction over
        # n_primal channels; dual state starts at zero.
        f = x_gn.repeat(1, self.n_primal, 1)
        h = torch.zeros(
            batch_size,
            self.n_dual,
            self.expected_measurements,
            device=device,
            dtype=dtype,
        )

        for i in range(self.num_iter):
            # Dual update uses A(f[:,1]) and the measured boundary data.
            f_dual = f[:, 1:2, :]
            op_f, _ = self._operator_call(f_dual)
            h = self.dual_nets[i](h, [op_f], measurements)

            # Baseline (non-RMA) uses the current adjoint gradient J(f[:,0])^T h.
            f_primal = f[:, 0:1, :]
            _, jacobian = self._operator_call(f_primal)
            h_first_channel = h[:, 0:1, :]
            adjoint_gradient = torch.bmm(h_first_channel, jacobian)

            f = self.primal_nets[i](f, [adjoint_gradient])

        # Match this project's dataset shapes: target and prediction are (B,N).
        result = f[:, 0, :]
        if result.shape[-1] != n_elements:
            raise RuntimeError("Unexpected change in primal element dimension.")
        return result



class PrimalDual(nn.Module):
    """Classical (non-learned) primal-dual baseline for nonlinear EIT.

    This is the fully algorithmic counterpart to
    :class:`LearnedPrimalDualBaseline`. Instead of learned networks for the
    primal / dual updates, it uses fixed step-size gradient-style updates:

        h^{k+1} = h^k + sigma * (A(f^k) - y)
        f^{k+1} = f^k - tau  * J(f^k)^T h^{k+1}

    The public input/output shapes are identical to
    :class:`LearnedPrimalDualBaseline`, so the two are drop-in interchangeable:

        x_gn         : (B, N)      or (B, 1, N)
        measurements : (B, M)      or (B, 1, M)
        output       : (B, N)

    Args mirror :class:`LearnedPrimalDualBaseline` (num_iter / n_primal /
    n_dual / expected_measurements) so existing training/evaluation scripts
    do not need any change. ``n_primal`` and ``n_dual`` are accepted for
    API compatibility; the classical algorithm uses a single primal and a
    single dual state channel.
    """

    def __init__(
        self,
        operator,
        num_iter: int = 10,
        n_primal: int = 5,
        n_dual: int = 5,
        expected_measurements: int = 208,
        tau: float = 1e-3,
        sigma: float = 1e-3,
    ) -> None:
        super().__init__()
        if num_iter < 1:
            raise ValueError("num_iter must be at least 1.")

        self.operator = operator
        self.num_iter = int(num_iter)
        self.n_primal = int(n_primal)
        self.n_dual = int(n_dual)
        self.expected_measurements = int(expected_measurements)
        self.tau = float(tau)
        self.sigma = float(sigma)

    # ------------------------------------------------------------------
    # operator helper (same contract as in LearnedPrimalDualBaseline)
    # ------------------------------------------------------------------
    def _operator_call(self, sigma: torch.Tensor):
        """Call the CPU pyEIT operator and move outputs back to model device."""
        voltage, jacobian = self.operator.forward_and_jac(sigma)
        voltage = voltage.to(device=sigma.device, dtype=sigma.dtype)
        jacobian = jacobian.to(device=sigma.device, dtype=sigma.dtype)

        if voltage.ndim != 3 or voltage.shape[1] != 1:
            raise RuntimeError(
                "operator.forward_and_jac must return voltage shaped (B,1,M); "
                f"got {tuple(voltage.shape)}"
            )
        if jacobian.ndim != 3:
            raise RuntimeError(
                "operator.forward_and_jac must return Jacobian shaped (B,M,N); "
                f"got {tuple(jacobian.shape)}"
            )
        if voltage.shape[0] != sigma.shape[0] or jacobian.shape[0] != sigma.shape[0]:
            raise RuntimeError("Operator output batch size does not match its input.")
        if voltage.shape[-1] != self.expected_measurements:
            raise RuntimeError(
                f"Expected {self.expected_measurements} measurements, "
                f"got {voltage.shape[-1]}"
            )
        if jacobian.shape[1] != voltage.shape[-1] or jacobian.shape[2] != sigma.shape[-1]:
            raise RuntimeError(
                "Jacobian shape must be (B, number_of_measurements, "
                "number_of_elements); "
                f"got {tuple(jacobian.shape)} for sigma {tuple(sigma.shape)}"
            )
        return voltage, jacobian

    # ------------------------------------------------------------------
    # forward
    # ------------------------------------------------------------------
    def forward(self, x_gn: torch.Tensor, measurements: torch.Tensor) -> torch.Tensor:
        x_gn = _as_channel_tensor(x_gn, "x_gn")
        measurements = _as_channel_tensor(measurements, "measurements")

        if x_gn.shape[0] != measurements.shape[0]:
            raise ValueError("x_gn and measurements must have the same batch size.")
        if measurements.shape[-1] != self.expected_measurements:
            raise ValueError(
                f"Expected {self.expected_measurements} measurement values, "
                f"got {measurements.shape[-1]}"
            )

        batch_size, _, n_elements = x_gn.shape
        device, dtype = x_gn.device, x_gn.dtype
        measurements = measurements.to(device=device, dtype=dtype)

        # Classical PD: single primal / dual state channel.
        # Primal is initialized with the GN reconstruction; dual starts at 0.
        f = x_gn  # (B, 1, N)
        h = torch.zeros(
            batch_size,
            1,
            self.expected_measurements,
            device=device,
            dtype=dtype,
        )

        for _ in range(self.num_iter):
            # Dual ascent on the data-fidelity residual:  A(f) - y
            op_f, _ = self._operator_call(f)
            h = h + self.sigma * (op_f - measurements)

            # Primal descent using the adjoint of the current Jacobian:  J(f)^T h
            _, jacobian = self._operator_call(f)
            adjoint_gradient = torch.bmm(h, jacobian)  # (B, 1, N)
            f = f - self.tau * adjoint_gradient

        result = f[:, 0, :]
        if result.shape[-1] != n_elements:
            raise RuntimeError("Unexpected change in primal element dimension.")
        return result