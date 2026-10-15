"""Author-style explicit-momentum LPD for nonlinear EIT.

This is the author's ``grad_type='momentum'`` path adapted to reuse the
validated nonlinear operator and the convolutional primal/dual networks in
``lpd_baseline_model.py``. It is the classical/exponential momentum baseline,
not the paper's LSTM-based RMA model.

The momentum coefficients follow the author's source code exactly:
    v_t = 0.8 * v_{t-1} + 1e-3 * J(x_t)^T h_t
The resulting momentum tensor is passed to the primal network in place of the
unaccelerated adjoint gradient.
"""
from __future__ import annotations

import torch

from lpd_baseline_model import (
    LearnedPrimalDualBaseline,
    _as_channel_tensor,
)


class LearnedPrimalDualMomentum(LearnedPrimalDualBaseline):
    """Nonlinear EIT LPD with the author's explicit momentum update.

    Network architecture, state channels, dynamic forward solve, dynamic
    Jacobian, and per-iteration parameterization are inherited from the
    baseline model. Only the primal input gradient is changed to the authors'
    fixed-coefficient momentum update.

    Args:
        operator: ``NonlinearEITOperator``-compatible object.
        num_iter: number of unrolled iterations (normally 10).
        n_primal: primal state channels (author uses 5).
        n_dual: dual state channels (author uses 5).
        expected_measurements: measurement count (208 for this dataset).
        momentum_decay: fixed momentum factor; author code uses 0.8.
        momentum_step: scale on the current adjoint gradient; author uses 1e-3.
    """

    def __init__(
        self,
        operator,
        num_iter: int = 10,
        n_primal: int = 5,
        n_dual: int = 5,
        expected_measurements: int = 208,
        momentum_decay: float = 0.8,
        momentum_step: float = 1e-3,
    ) -> None:
        super().__init__(
            operator=operator,
            num_iter=num_iter,
            n_primal=n_primal,
            n_dual=n_dual,
            expected_measurements=expected_measurements,
        )
        if not 0.0 <= momentum_decay < 1.0:
            raise ValueError("momentum_decay must be in [0, 1).")
        if momentum_step <= 0.0:
            raise ValueError("momentum_step must be positive.")

        # Keep these as constants, like the author's implementation, rather
        # than introducing additional trainable parameters.
        self.momentum_decay = float(momentum_decay)
        self.momentum_step = float(momentum_step)

    def forward(
        self,
        x_gn: torch.Tensor,
        measurements: torch.Tensor,
    ) -> torch.Tensor:
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

        # Same initialization as the author's LearnedPrimalDual.
        f = x_gn.repeat(1, self.n_primal, 1)
        h = torch.zeros(
            batch_size,
            self.n_dual,
            self.expected_measurements,
            device=device,
            dtype=dtype,
        )

        # Author's vt = zeros_like(f[:, 1:2]).
        momentum = torch.zeros_like(f[:, 1:2, :]).detach()

        for i in range(self.num_iter):
            # Dual step: h_{t+1} = DualNet(h_t, A(f_t), y).
            f_dual = f[:, 1:2, :]
            op_f, _ = self._operator_call(f_dual)
            h = self.dual_nets[i](h, [op_f], measurements)

            # Current adjoint gradient: J(f_t)^T h_t.
            f_primal = f[:, 0:1, :]
            _, jacobian = self._operator_call(f_primal)
            h_first_channel = h[:, 0:1, :]
            adjoint_gradient = torch.bmm(h_first_channel, jacobian)

            # Explicit momentum exactly matching primaldual.py's momentum path.
            momentum = (
                self.momentum_decay * momentum
                + self.momentum_step * adjoint_gradient
            )

            # The author's PrimalNet receives [vt], not the raw gradient.
            f = self.primal_nets[i](f, [momentum])

        result = f[:, 0, :]
        if result.shape[-1] != n_elements:
            raise RuntimeError("Unexpected change in primal element dimension.")
        return result
