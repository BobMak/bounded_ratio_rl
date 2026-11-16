"""Adaptive ratio bound (epsilon) controller for BPO-style target-ratio losses.

Framework-agnostic: feed it one scalar divergence measurement per policy update
(e.g. exact mean KL(pi_old || pi_new) over the rollout) and it returns the
bound to use for the next update. The update is multiplicative in log space,

    eps <- clip(eps * exp(rate * clip(log(target / measured), -1, 1)), eps_min, eps_max)

so the step is symmetric for over/undershoot, bounded per update (at most a
factor of e**rate), and eps stays strictly inside (0, 1) as long as eps_max < 1
(required because the target ratio is 1 - eps on its low side).
"""

from __future__ import annotations

import math


class AdaptiveClipController:
    def __init__(
        self,
        init_eps: float,
        target: float,
        rate: float = 0.5,
        eps_min: float = 0.02,
        eps_max: float = 0.9,
    ):
        if not 0.0 < eps_min <= eps_max < 1.0:
            raise ValueError(f"need 0 < eps_min <= eps_max < 1, got {eps_min}, {eps_max}")
        if target <= 0.0:
            raise ValueError(f"target divergence must be > 0, got {target}")
        if rate < 0.0:
            raise ValueError(f"rate must be >= 0, got {rate}")
        self.target = float(target)
        self.rate = float(rate)
        self.eps_min = float(eps_min)
        self.eps_max = float(eps_max)
        self.eps = min(max(float(init_eps), self.eps_min), self.eps_max)
        self.last_measured: float | None = None

    def update(self, measured: float) -> float:
        """Record a divergence measurement and return the bound for the next update."""
        if not math.isfinite(measured):
            # don't let a NaN/inf measurement poison the controller; shrink conservatively
            self.eps = max(self.eps_min, self.eps * math.exp(-self.rate))
            return self.eps
        self.last_measured = float(measured)
        err = math.log(self.target / max(measured, 1e-12))
        err = max(-1.0, min(1.0, err))
        self.eps = min(max(self.eps * math.exp(self.rate * err), self.eps_min), self.eps_max)
        return self.eps

    def state_dict(self) -> dict:
        return {
            "eps": self.eps,
            "target": self.target,
            "rate": self.rate,
            "eps_min": self.eps_min,
            "eps_max": self.eps_max,
            "last_measured": self.last_measured,
        }

    def load_state_dict(self, state: dict) -> None:
        self.eps = float(state["eps"])
        self.target = float(state["target"])
        self.rate = float(state["rate"])
        self.eps_min = float(state["eps_min"])
        self.eps_max = float(state["eps_max"])
        self.last_measured = state.get("last_measured")
