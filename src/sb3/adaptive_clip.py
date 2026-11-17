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
import random


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


class ClimbClipController:
    """"Climb until unstable" ratio bound: additive increase, multiplicative decrease.

    A constant-KL target forces eps down over training (at fixed eps the per-update KL
    grows as the policy sharpens), even when that larger KL is harmless. This controller
    instead keeps raising eps and only backs off when an update looks unstable.

    Instability is judged on the eps-normalized divergence q = KL / eps_used**kl_power.
    The per-update KL grows roughly with eps**2, so testing raw KL against its recent median
    makes the controller's own climbing look like a spike; q stays flat while eps climbs and
    jumps only when the update itself misbehaves (kl_power=0 recovers the raw-KL test).

        unstable = KL > kl_ceiling  or  q > spike_ratio * median(q over last `window` updates)
        unstable -> eps <- max(eps_min, eps * cut), velocity <- 0, hold for `cooldown` updates
                    (during the hold only the ceiling test applies)
        otherwise (not holding) -> velocity <- momentum * velocity + step; eps <- eps + velocity

    momentum: heavy-ball on climb steps only (cuts stay immediate). Uninterrupted climbing
        accelerates from `step` toward step / (1 - momentum); a cut resets it (slow start).
    noise: the eps actually used for an update is eps * exp(noise * N(0,1)), clipped to
        [eps_min, eps_max]; the controller state stays the noise-free eps. The spike test uses
        the eps actually used, so the jitter itself never looks like a spike.
    """

    def __init__(
        self,
        init_eps: float,
        step: float = 0.03,
        cut: float = 0.7,
        kl_ceiling: float = 0.5,
        spike_ratio: float = 1.5,
        window: int = 5,
        cooldown: int = 3,
        eps_min: float = 0.02,
        eps_max: float = 0.95,
        kl_power: float = 2.0,
        momentum: float = 0.0,
        noise: float = 0.0,
        seed: int | None = None,
    ):
        if not 0.0 < eps_min <= eps_max < 1.0:
            raise ValueError(f"need 0 < eps_min <= eps_max < 1, got {eps_min}, {eps_max}")
        if step < 0.0 or not 0.0 < cut < 1.0:
            raise ValueError(f"need step >= 0 and 0 < cut < 1, got {step}, {cut}")
        if kl_ceiling <= 0.0 or spike_ratio <= 1.0 or window < 1 or cooldown < 0:
            raise ValueError("need kl_ceiling > 0, spike_ratio > 1, window >= 1, cooldown >= 0")
        if kl_power < 0.0 or not 0.0 <= momentum < 1.0 or noise < 0.0:
            raise ValueError(f"need kl_power >= 0, 0 <= momentum < 1, noise >= 0, got {kl_power}, {momentum}, {noise}")
        self.step, self.cut = float(step), float(cut)
        self.kl_ceiling, self.spike_ratio = float(kl_ceiling), float(spike_ratio)
        self.window, self.cooldown = int(window), int(cooldown)
        self.eps_min, self.eps_max = float(eps_min), float(eps_max)
        self.kl_power, self.momentum, self.noise = float(kl_power), float(momentum), float(noise)
        self.eps = min(max(float(init_eps), self.eps_min), self.eps_max)
        self.velocity = 0.0
        self.history: list[float] = []  # normalized divergences q
        self.hold = 0
        self.n_cuts = 0
        self.last_measured: float | None = None
        self.last_unstable = False
        self.last_used = self.eps
        self._rng = random.Random(seed)

    def propose(self) -> float:
        """Bound to use for the coming update (eps, jittered when noise > 0)."""
        e = self.eps
        if self.noise > 0.0:
            e = min(max(e * math.exp(self.noise * self._rng.gauss(0.0, 1.0)), self.eps_min), self.eps_max)
        self.last_used = e
        return e

    def _normalized(self, measured: float, eps_used: float) -> float:
        return measured / max(eps_used, 1e-8) ** self.kl_power

    def _unstable(self, measured: float, q: float) -> bool:
        if not math.isfinite(measured) or measured > self.kl_ceiling:
            return True
        # during the post-cut hold only the hard ceiling can cut again: the spike reference
        # (pre-cut median) is stale, so one spike would otherwise ratchet eps down repeatedly
        if self.hold == 0 and len(self.history) >= self.window:
            recent = sorted(self.history[-self.window:])
            median = recent[len(recent) // 2]
            return q > self.spike_ratio * median
        return False

    def update(self, measured: float, eps_used: float | None = None) -> float:
        """Record the divergence of an update made with bound `eps_used` (default: the last
        proposed bound) and return the noise-free bound for the next update."""
        eps_used = self.last_used if eps_used is None else float(eps_used)
        q = self._normalized(measured, eps_used) if math.isfinite(measured) else float("inf")
        self.last_unstable = self._unstable(measured, q)
        if self.last_unstable:
            self.eps = max(self.eps_min, self.eps * self.cut)
            self.velocity = 0.0
            self.hold = self.cooldown
            self.n_cuts += 1
        elif self.hold > 0:
            self.hold -= 1
        else:
            self.velocity = self.momentum * self.velocity + self.step
            self.eps = min(self.eps_max, self.eps + self.velocity)
            if self.eps >= self.eps_max:
                self.velocity = 0.0
        if math.isfinite(measured):
            self.last_measured = float(measured)
            self.history.append(q)
            del self.history[: -max(self.window, 1)]
        self.last_used = self.eps
        return self.eps

    def state_dict(self) -> dict:
        return {k: getattr(self, k) for k in (
            "eps", "step", "cut", "kl_ceiling", "spike_ratio", "window", "cooldown",
            "eps_min", "eps_max", "kl_power", "momentum", "noise", "velocity",
            "history", "hold", "n_cuts", "last_measured", "last_unstable", "last_used")}

    def load_state_dict(self, state: dict) -> None:
        for k, v in state.items():
            setattr(self, k, list(v) if k == "history" else v)
