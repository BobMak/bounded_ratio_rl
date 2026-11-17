import warnings
from typing import Any, ClassVar, Optional, TypeVar, Union

import numpy as np
import torch as th
from gymnasium import spaces

# from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm
from stable_baselines3.common.policies import (
    ActorCriticCnnPolicy,
    ActorCriticPolicy,
    BasePolicy,
    MultiInputActorCriticPolicy,
)
from .policy import BPOPolicy, BPOCnnPolicy
from .adaptive_clip import AdaptiveClipController
from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.distributions import CategoricalDistribution, DiagGaussianDistribution
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from torch.nn import functional as F

from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.utils import obs_as_tensor

SelfBPO = TypeVar("SelfBPO", bound="BPO")


class BPO(OnPolicyAlgorithm):
    """
    Bounded Policy Optimization algorithm (BPO) (clip version)

    Paper: https://arxiv.org/abs/2604.18578v1

    :param policy: The policy model to use (MlpPolicy, CnnPolicy, ...)
    :param env: The environment to learn from (if registered in Gym, can be str)
    :param learning_rate: The learning rate, it can be a function
        of the current progress remaining (from 1 to 0)
    :param n_steps: The number of steps to run for each environment per update
        (i.e. rollout buffer size is n_steps * n_envs where n_envs is number of environment copies running in parallel)
        NOTE: n_steps * n_envs must be greater than 1 (because of the advantage normalization)
        See https://github.com/pytorch/pytorch/issues/29372
    :param batch_size: Minibatch size
    :param n_epochs: Number of epoch when optimizing the surrogate loss
    :param gamma: Discount factor
    :param gae_lambda: Factor for trade-off of bias vs variance for Generalized Advantage Estimator
    :param lambda: The temperature parameter
    :param loss_type: type of loss function, include Total Variation (TV), Mean Square (MSE), Forward KL (FKL) and Chi-squared (Chi)
    :param normalize_advantage: Whether to normalize or not the advantage
    :param ent_coef: Entropy coefficient for the loss calculation
    :param vf_coef: Value function coefficient for the loss calculation
    :param u_coef: median value function coefficient for the loss calculation
    :param use_median: Whether to use median value function for the loss calculation
    :param clip_ratio: The clipping ratio, it can be a function
        of the current progress remaining (from 1 to 0)
    :param adaptive_clip: If True, adapt clip_ratio after every update so that the exact
        mean KL(pi_old || pi_new) over the rollout tracks ``clip_target_kl``. The initial
        value is ``clip_ratio`` (or its schedule at progress 1); the schedule is otherwise ignored.
    :param clip_target_kl: Target per-update mean KL divergence for the adaptive bound
    :param clip_adapt_rate: Log-space step size of the controller (max factor e**rate per update)
    :param clip_min: Lower limit for the adaptive bound
    :param clip_max: Upper limit for the adaptive bound (must be < 1)
    :param controlled_eps: "Controlled eps" variant, so that the ratio bound (not gradient
        clipping) is what limits the policy step: (1) the |Q - V| weight of the policy loss is
        divided by its minibatch mean, removing its reward-scale dependence; (2) actor and
        critic gradients are norm-clipped separately (each to ``max_grad_norm``), so the large
        value/u-loss gradients no longer shrink the actor update through a joint clip.
    :param critic_type: "mean_median" (default: MSE mean critic V + separate median head u) or
        "quantile" (distributional critic: one head of ``n_quantiles`` return quantiles trained with
        the quantile-Huber loss on the GAE returns; V = mean of the atoms, u = the exact
        lambda-smoothed median of the atoms, i.e. the root of mean_j sigmoid((z_j - u)/lambda) = 1/2,
        which makes the target ratios average to 1 under pi_old)
    :param n_quantiles: Number of quantile atoms for ``critic_type="quantile"``
    :param qr_kappa: Huber threshold of the quantile-Huber loss
    :param ratio_target: "sigmoid" (default: r* = 1 - eps + 2 eps sigmoid((G - u)/lambda)) or "cdf"
        (r* = 1 - eps + 2 eps F_s(G), with F_s(G) = mean_j sigmoid((G - z_j)/lambda) the critic's
        smoothed CDF of the return at the sampled return; needs the quantile critic)
    :param dist_weight: policy-loss weight: "abs" (default: |G - V|) or "iqr" (|G - V| / IQR_s,
        divided by its minibatch mean; scale-free, needs the quantile critic)
    :param max_grad_norm: The maximum value for the gradient clipping
    :param use_sde: Whether to use generalized State Dependent Exploration (gSDE)
        instead of action noise exploration (default: False)
    :param sde_sample_freq: Sample a new noise matrix every n steps when using gSDE
        Default: -1 (only sample at the beginning of the rollout)
    :param rollout_buffer_class: Rollout buffer class to use. If ``None``, it will be automatically selected.
    :param rollout_buffer_kwargs: Keyword arguments to pass to the rollout buffer on creation
    :param target_kl: Limit the KL divergence between updates,
        because the clipping is not enough to prevent large update
        see issue #213 (cf https://github.com/hill-a/stable-baselines/issues/213)
        By default, there is no limit on the kl div.
    :param stats_window_size: Window size for the rollout logging, specifying the number of episodes to average
        the reported success rate, mean episode length, and mean reward over
    :param tensorboard_log: the log location for tensorboard (if None, no logging)
    :param policy_kwargs: additional arguments to be passed to the policy on creation. See :ref:`ppo_policies`
    :param verbose: Verbosity level: 0 for no output, 1 for info messages (such as device or wrappers used), 2 for
        debug messages
    :param seed: Seed for the pseudo random generators
    :param device: Device (cpu, cuda, ...) on which the code should be run.
        Setting it to auto, the code will be run on the GPU if possible.
    :param _init_setup_model: Whether or not to build the network at the creation of the instance
    """

    policy_aliases: ClassVar[dict[str, type[BasePolicy]]] = {
        "MlpPolicy": ActorCriticPolicy,
        "CnnPolicy": ActorCriticCnnPolicy,
        "MultiInputPolicy": MultiInputActorCriticPolicy,
        "BPOPolicy": BPOPolicy,
        "BPOCnnPolicy": BPOCnnPolicy,
    }

    def __init__(
        self,
        policy: Union[str, type[BPOPolicy]],
        env: Union[GymEnv, str],
        learning_rate: Union[float, Schedule] = 3e-4,
        n_steps: int = 2048,
        batch_size: int = 64,
        n_epochs: int = 10,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        lambda_: float = 0.001,
        ent_coef: float = 0.0,
        vf_coef: float = 0.5,
        u_coef: float = 0.5,
        use_median: bool = True,
        clip_ratio: Union[float, Schedule] = 0.3,
        adaptive_clip: bool = False,
        clip_target_kl: float = 0.02,
        clip_adapt_rate: float = 0.5,
        clip_min: float = 0.02,
        clip_max: float = 0.9,
        controlled_eps: bool = False,
        state_clip: Optional[str] = None,
        state_kl_max: float = 0.05,
        critic_type: str = "mean_median",
        n_quantiles: int = 32,
        qr_kappa: float = 1.0,
        ratio_target: str = "sigmoid",
        dist_weight: str = "abs",
        max_grad_norm: float = 1.0,
        use_sde: bool = False,
        sde_sample_freq: int = -1,
        rollout_buffer_class: Optional[type[RolloutBuffer]] = None,
        rollout_buffer_kwargs: Optional[dict[str, Any]] = None,
        stats_window_size: int = 100,
        tensorboard_log: Optional[str] = None,
        policy_kwargs: Optional[dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        _init_setup_model: bool = True,
    ):
        if critic_type not in ("mean_median", "quantile"):
            raise ValueError(f"critic_type must be 'mean_median' or 'quantile', got {critic_type!r}")
        if ratio_target not in ("sigmoid", "cdf"):
            raise ValueError(f"ratio_target must be 'sigmoid' or 'cdf', got {ratio_target!r}")
        if dist_weight not in ("abs", "iqr"):
            raise ValueError(f"dist_weight must be 'abs' or 'iqr', got {dist_weight!r}")
        if critic_type != "quantile" and (ratio_target == "cdf" or dist_weight == "iqr"):
            raise ValueError("ratio_target='cdf' and dist_weight='iqr' need critic_type='quantile'")
        if critic_type == "quantile":
            if n_quantiles < 2:
                raise ValueError(f"n_quantiles must be >= 2, got {n_quantiles}")
            policy_kwargs = {**(policy_kwargs or {}), "n_quantiles": int(n_quantiles)}
        self.critic_type = critic_type
        self.n_quantiles = int(n_quantiles)
        self.qr_kappa = float(qr_kappa)
        self.ratio_target = ratio_target
        self.dist_weight = dist_weight
        super().__init__(
            policy,
            env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            gamma=gamma,
            gae_lambda=gae_lambda,
            ent_coef=ent_coef,
            vf_coef=vf_coef,
            max_grad_norm=max_grad_norm,
            use_sde=use_sde,
            sde_sample_freq=sde_sample_freq,
            rollout_buffer_class=rollout_buffer_class,
            rollout_buffer_kwargs=rollout_buffer_kwargs,
            stats_window_size=stats_window_size,
            tensorboard_log=tensorboard_log,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            device=device,
            seed=seed,
            _init_setup_model=False,
            supported_action_spaces=(
                spaces.Box,
                spaces.Discrete,
                spaces.MultiDiscrete,
                spaces.MultiBinary,
            ),
        )

        # Sanity check, otherwise it will lead to noisy gradient and NaN
        # because of the advantage normalization


        if self.env is not None:
            # Check that `n_steps * n_envs > 1` to avoid NaN
            # when doing advantage normalization
            buffer_size = self.env.num_envs * self.n_steps
            # Check that the rollout buffer size is a multiple of the mini-batch size
            untruncated_batches = buffer_size // batch_size
            if buffer_size % batch_size > 0:
                warnings.warn(
                    f"You have specified a mini-batch size of {batch_size},"
                    f" but because the `RolloutBuffer` is of size `n_steps * n_envs = {buffer_size}`,"
                    f" after every {untruncated_batches} untruncated mini-batches,"
                    f" there will be a truncated mini-batch of size {buffer_size % batch_size}\n"
                    f"We recommend using a `batch_size` that is a factor of `n_steps * n_envs`.\n"
                    f"Info: (n_steps={self.n_steps} and n_envs={self.env.num_envs})"
                )
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.lambda_ = lambda_
        self.use_median = use_median
        # Convert clip_ratio to schedule if it's a float
        if isinstance(clip_ratio, (int, float)):
            # Constant schedule
            self.clip_ratio_schedule = lambda _: float(clip_ratio)
        else:
            # Callable
            self.clip_ratio_schedule = clip_ratio
        self.u_coef = u_coef

        self.controlled_eps = controlled_eps
        if state_clip not in (None, "mask", "scale"):
            raise ValueError(f"state_clip must be None, 'mask' or 'scale', got {state_clip!r}")
        if state_clip is not None and state_kl_max <= 0:
            raise ValueError(f"state_kl_max must be > 0, got {state_kl_max}")
        self.state_clip = state_clip
        self.state_kl_max = float(state_kl_max)
        self.adaptive_clip = adaptive_clip
        self.clip_controller: Optional[AdaptiveClipController] = None
        if adaptive_clip:
            self.clip_controller = AdaptiveClipController(
                init_eps=self.clip_ratio_schedule(1.0),
                target=clip_target_kl,
                rate=clip_adapt_rate,
                eps_min=clip_min,
                eps_max=clip_max,
            )

        if _init_setup_model:
            self._setup_model()

    def _setup_model(self) -> None:
        super()._setup_model()
        # actor-only parameters (policy MLP, action head, log_std); everything else
        # (critic MLP, value/u heads, any shared feature extractor) is the critic group
        actor_mods = [self.policy.mlp_extractor.policy_net, self.policy.action_net]
        actor_ids = {id(p) for m in actor_mods for p in m.parameters()}
        if isinstance(getattr(self.policy, "log_std", None), th.nn.Parameter):
            actor_ids.add(id(self.policy.log_std))
        self._actor_params = [p for p in self.policy.parameters() if id(p) in actor_ids]
        self._critic_params = [p for p in self.policy.parameters() if id(p) not in actor_ids]

    @staticmethod
    @th.no_grad()
    def _total_grad_norm(params: list) -> th.Tensor:
        """L2 norm over all grads, without syncing (th.nn.utils.get_total_norm needs torch>=2.6)."""
        grads = [p.grad for p in params if p.grad is not None]
        if hasattr(th, "_foreach_norm"):
            norms = th._foreach_norm(grads)  # one fused kernel instead of one per parameter
        else:
            norms = [g.norm() for g in grads]
        return th.linalg.vector_norm(th.stack(norms))

    @staticmethod
    def _detached_dist(dist: th.distributions.Distribution) -> Optional[th.distributions.Distribution]:
        """Frozen copy of a torch distribution (Normal / Categorical), or None if unsupported."""
        if isinstance(dist, th.distributions.Normal):
            return th.distributions.Normal(dist.loc.detach(), dist.scale.detach())
        if isinstance(dist, th.distributions.Categorical):
            return th.distributions.Categorical(logits=dist.logits.detach())
        return None

    @th.no_grad()
    def _policy_dists(self, obs: th.Tensor, chunk: int = 4096) -> list:
        """Detached action distributions of the current policy over obs, in fixed chunks."""
        out = []
        for i in range(0, obs.shape[0], chunk):
            dist = self.policy.get_distribution(obs[i : i + chunk]).distribution
            out.append(self._detached_dist(dist) if not isinstance(dist, list) else None)
        return out

    @th.no_grad()
    def _measure_update(self, obs, actions, old_log_prob, old_dists, chunk: int = 4096) -> dict:
        """Divergence between the pre-update and post-update policy on the rollout."""
        new_dists = self._policy_dists(obs, chunk)
        # log-probs straight from the new distribution (saves a second full forward pass
        # through evaluate_actions); only where that matches SB3's log_prob exactly
        fast_lp = isinstance(
            self.policy.action_dist, (DiagGaussianDistribution, CategoricalDistribution)
        ) and not self.policy.squash_output
        kls, log_probs = [], []
        for k, i in enumerate(range(0, obs.shape[0], chunk)):
            o, n = old_dists[k], new_dists[k]
            if o is not None and n is not None:
                kl = th.distributions.kl_divergence(o, n)
                if isinstance(n, th.distributions.Normal):
                    kl = kl.sum(-1)
                kls.append(kl.flatten())
            a = actions[i : i + chunk]
            if fast_lp and n is not None:
                lp = n.log_prob(a)
                if isinstance(n, th.distributions.Normal):
                    lp = lp.sum(-1)
            else:
                lp = self.policy.evaluate_actions(obs[i : i + chunk], a)[2]
            log_probs.append(lp.flatten())
        log_ratio = th.cat(log_probs) - old_log_prob.flatten()
        ratio = th.exp(log_ratio)
        stats = {
            # k3 estimator of KL(old||new), always available
            "kl_k3": th.mean((ratio - 1) - log_ratio).item(),
            # sample estimate of TV(old, new) = 0.5 E_old|r - 1|
            "tv_sample": 0.5 * th.mean(th.abs(ratio - 1)).item(),
        }
        stats["kl_exact"] = th.cat(kls).mean().item() if kls and len(kls) == len(old_dists) else stats["kl_k3"]
        return stats

    def _quantile_huber_loss(self, quantiles: th.Tensor, target: th.Tensor) -> th.Tensor:
        """Quantile-Huber loss of atoms [B,N] (tau_i = (2i+1)/2N) against scalar targets [B]."""
        n = quantiles.shape[-1]
        tau = (th.arange(n, device=quantiles.device, dtype=quantiles.dtype) + 0.5) / n
        u = target.unsqueeze(-1) - quantiles
        k = self.qr_kappa
        huber = th.where(u.abs() <= k, 0.5 * u.pow(2), k * (u.abs() - 0.5 * k))
        return (th.abs(tau - (u.detach() < 0).float()) * huber / k).sum(-1).mean()

    @th.no_grad()
    def _smoothed_median(self, atoms: th.Tensor, temp: float, iters: int = 40) -> th.Tensor:
        """Root u of mean_j sigmoid((z_j - u)/temp) = 1/2 per row of atoms [B,N] (bisection)."""
        lo, hi = atoms.min(-1).values, atoms.max(-1).values
        for _ in range(iters):
            mid = 0.5 * (lo + hi)
            # f(mid) decreasing in mid; f > 1/2 -> root is to the right
            right = th.sigmoid((atoms - mid.unsqueeze(-1)) / temp).mean(-1) > 0.5
            lo = th.where(right, mid, lo)
            hi = th.where(right, hi, mid)
        return 0.5 * (lo + hi)

    @staticmethod
    def _iqr(atoms: th.Tensor) -> th.Tensor:
        """Interquartile range [B] of quantile atoms [B,N] at midpoints tau_i = (2i+1)/2N."""
        z, n = th.sort(atoms, dim=-1).values, atoms.shape[-1]

        def q(tau: float) -> th.Tensor:
            pos = min(max(tau * n - 0.5, 0.0), n - 1.0)
            i0 = int(np.floor(pos))
            i1, w = min(i0 + 1, n - 1), pos - i0
            return (1 - w) * z[:, i0] + w * z[:, i1]

        return q(0.75) - q(0.25)

    def train(self) -> None:
        """
        Update policy using the currently gathered rollout buffer.
        """
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)
        # Update optimizer learning rate
        self._update_learning_rate(self.policy.optimizer)

        # Get current clip_ratio from schedule
        # Calculate progress_remaining: 1.0 at start, 0.0 at end
        if hasattr(self, "_total_timesteps") and self._total_timesteps > 0:
            progress_remaining = 1.0 - (self.num_timesteps / self._total_timesteps)
        else:
            # Fallback if total_timesteps not available yet
            progress_remaining = 1.0
        if self.clip_controller is not None:
            clip_ratio = self.clip_controller.eps
        else:
            clip_ratio = self.clip_ratio_schedule(progress_remaining)

        # Snapshot the pre-update policy on the whole rollout (fixed order) so the
        # realized divergence of this update can be measured exactly afterwards.
        # The snapshot is taken in the buffer's native (flattened) order so minibatches can
        # index into it; minibatches are drawn exactly like RolloutBuffer.get does.
        rb = self.rollout_buffer
        n_samples = rb.buffer_size * rb.n_envs
        if not rb.generator_ready:
            next(rb.get(n_samples))  # flattens the buffer in place
        full = rb._get_samples(np.arange(n_samples))
        full_actions = full.actions
        if isinstance(self.action_space, spaces.Discrete):
            full_actions = full_actions.long().flatten()
        old_dists = self._policy_dists(full.observations)

        # per-state bound: frozen pre-update distribution parameters for every sample
        old_params = None
        if self.state_clip is not None:
            if any(d is None for d in old_dists):
                raise ValueError("state_clip needs a Normal or Categorical action distribution")
            if isinstance(old_dists[0], th.distributions.Normal):
                old_params = ("normal", th.cat([d.loc for d in old_dists]), th.cat([d.scale for d in old_dists]))
            else:
                old_params = ("categorical", th.cat([d.logits for d in old_dists]))
        state_kls, state_clip_fracs, state_eps = [], [], []

        # last-epoch diagnostics of how well the ratio reaches the target
        attain, direction_ok, target_mean, ratio_dev, grad_norms, pi_grad_norms = [], [], [], [], [], []

        entropy_losses = []
        pg_losses, value_losses, u_losses = [], [], []
        # distributional-critic diagnostics (last epoch)
        cdf_vals, iqr_vals, median_gap = [], [], []

        continue_training = True
        self.max_adv_abs = th.tensor(0.0, device=self.device)
        # train for n_epochs epochs
        for epoch in range(self.n_epochs):
            approx_kl_divs = []
            # First train values

            perm = np.random.permutation(n_samples)
            for start in range(0, n_samples, self.batch_size):
                batch_inds = perm[start : start + self.batch_size]
                rollout_data = rb._get_samples(batch_inds)
                actions = rollout_data.actions
                if isinstance(self.action_space, spaces.Discrete):
                    # Convert discrete action from float to long
                    actions = rollout_data.actions.long().flatten()

                values, u_values, log_prob, entropy, quantiles = self.policy.evaluate_actions(
                    rollout_data.observations, actions, return_quantiles=True
                )
                # SB3 distributions return themselves, so this is this minibatch's pi_theta
                cur_dist = self.policy.action_dist.distribution


                values = values.flatten()
                u_values = u_values.flatten()

                # value loss function
                q_values = rollout_data.returns.flatten()
                target_values = rollout_data.returns.flatten()
                if quantiles is not None:
                    # distributional critic: V and u both come from the quantile atoms
                    value_loss = self._quantile_huber_loss(quantiles, target_values)
                else:
                    value_loss = F.mse_loss(target_values, values)
                value_losses.append(value_loss.detach())

                # u loss
                if quantiles is not None:
                    atoms = quantiles.detach()
                    u_smooth = self._smoothed_median(atoms, self.lambda_)
                    adv = (q_values - u_smooth) / self.lambda_
                    u_loss = th.zeros_like(value_loss)
                    u_losses.append(u_loss.detach())
                elif self.use_median:
                    adv = (q_values - u_values) / self.lambda_
                    u_loss = th.mean(
                        th.nn.functional.softplus(adv * self.lambda_)
                        - 0.5 * adv * self.lambda_
                    )
                    u_losses.append(u_loss.detach())
                else:
                    adv = (q_values - values) / self.lambda_
                    u_loss = th.zeros_like(value_loss)
                    u_losses.append(u_loss.detach())

                # policz loss
                with th.no_grad():
                    adv_d = adv.detach()
                    # per-state bound: exact KL(pi_old(s) || pi_theta(s)) for each state
                    state_w, eps_s = None, clip_ratio
                    if old_params is not None:
                        idx = th.as_tensor(batch_inds, device=self.device)
                        if old_params[0] == "normal":
                            old_d = th.distributions.Normal(old_params[1][idx], old_params[2][idx])
                            kl_s = th.distributions.kl_divergence(old_d, cur_dist).sum(-1)
                        else:
                            old_d = th.distributions.Categorical(logits=old_params[1][idx])
                            kl_s = th.distributions.kl_divergence(old_d, cur_dist)
                        kl_s = kl_s.flatten()
                        over = kl_s > self.state_kl_max
                        if self.state_clip == "mask":
                            # state has used up its divergence budget: stop pushing it
                            state_w = (~over).float()
                        else:
                            # shrink this state's bound; KL ~ eps^2, hence the sqrt
                            eps_s = clip_ratio * th.clamp(
                                th.sqrt(self.state_kl_max / kl_s.clamp_min(1e-12)), max=1.0
                            )
                        if epoch == self.n_epochs - 1:
                            state_kls.append(kl_s)
                            state_clip_fracs.append(over.float().mean())
                            state_eps.append(th.as_tensor(eps_s, dtype=th.float32).mean())
                if self.ratio_target == "cdf":
                    # critic's smoothed CDF of the return at the sampled return:
                    # uniform when G ~ Z(s), so E[target_ratio] = 1 at any reward scale
                    p_up = th.sigmoid((q_values.unsqueeze(-1) - atoms) / self.lambda_).mean(-1)
                else:
                    p_up = th.nn.functional.sigmoid(adv_d)
                target_ratio = 1 - eps_s + 2 * eps_s * p_up
                target_log_ratio = th.log(target_ratio)
                log_ratio = log_prob.flatten() - rollout_data.old_log_prob.flatten()
                ratio = th.exp(log_ratio)
                if epoch == self.n_epochs - 1:
                    with th.no_grad():
                        r, rt = ratio.detach(), target_ratio.flatten()
                        attain.append((th.abs(r - rt) < 0.1 * clip_ratio).float().mean())
                        direction_ok.append(
                            (th.sign(r - 1) == th.sign(rt - 1)).float().mean()
                        )
                        target_mean.append(rt.mean())
                        ratio_dev.append(th.abs(r - 1).mean())
                pg_weight = th.abs(q_values - values.detach())
                if self.dist_weight == "iqr":
                    iqr = self._iqr(atoms)
                    # per-state scale-free weight; the floor keeps near-deterministic states
                    # from dominating, the mean-normalization removes the remaining scale
                    pg_weight = pg_weight / (iqr + 0.1 * iqr.mean() + 1e-8)
                    pg_weight = pg_weight / (pg_weight.mean() + 1e-8)
                if quantiles is not None and epoch == self.n_epochs - 1:
                    with th.no_grad():
                        cdf_vals.append(th.sigmoid((q_values.unsqueeze(-1) - atoms) / self.lambda_).mean(-1))
                        iqr_vals.append(self._iqr(atoms).mean())
                        median_gap.append((u_smooth - values.detach()).mean())
                if self.controlled_eps:
                    pg_weight = pg_weight / (pg_weight.mean() + 1e-8)
                if state_w is not None:
                    # mean over all samples (not just unmasked) so the loss scale doesn't
                    # jump as states drop out, like PPO's clipped samples
                    pg_weight = pg_weight * state_w
                policy_loss = th.mean(th.abs((ratio - target_ratio).flatten()) * pg_weight)
                pg_losses.append(policy_loss.detach())

                if entropy is None:
                    # Approximate entropy when no analytical form
                    entropy_loss = -th.mean(-log_prob)
                else:
                    entropy_loss = -th.mean(entropy)

                entropy_losses.append(entropy_loss.detach())

                p_coef = 1.0
                loss = p_coef * policy_loss + self.vf_coef * value_loss + self.u_coef * u_loss + self.ent_coef * entropy_loss

                # Calculate approximate form of reverse KL Divergence for early stopping
                with th.no_grad():
                    log_ratio = log_prob - rollout_data.old_log_prob
                    approx_kl_divs.append(th.mean((th.exp(log_ratio) - 1) - log_ratio))

                # Optimization step
                self.policy.optimizer.zero_grad()
                loss.backward()
                # Clip grad norm
                if self.controlled_eps:
                    gn_pi = th.nn.utils.clip_grad_norm_(self._actor_params, self.max_grad_norm)
                    gn_vf = th.nn.utils.clip_grad_norm_(self._critic_params, self.max_grad_norm)
                    grad_norms.append(gn_pi.detach())
                else:
                    gn_pi = self._total_grad_norm(self._actor_params)
                    grad_norms.append(
                        th.nn.utils.clip_grad_norm_(
                            self.policy.parameters(), self.max_grad_norm
                        ).detach()
                    )
                pi_grad_norms.append(gn_pi.detach())
                self.policy.optimizer.step()

            self._n_updates += 1
            if not continue_training:
                break

        # single device->host sync for all per-minibatch metrics
        _mean = lambda xs: th.stack(xs).float().mean().item() if xs else float("nan")
        self.logger.record("train/entropy_loss", _mean(entropy_losses))
        self.logger.record("train/policy_gradient_loss", _mean(pg_losses))
        self.logger.record("train/value_loss", _mean(value_losses))
        self.logger.record("train/u_loss", _mean(u_losses))
        self.logger.record("train/approx_kl", _mean(approx_kl_divs))
        self.logger.record("train/loss", loss.item())
        self.logger.record("train/lambda", self.lambda_)
        self.logger.record("train/clip_ratio", clip_ratio)

        div = self._measure_update(full.observations, full_actions, full.old_log_prob, old_dists)
        for k, v in div.items():
            self.logger.record(f"train/{k}", v)
        self.logger.record("train/target_attainment", _mean(attain))
        self.logger.record("train/target_direction_ok", _mean(direction_ok))
        self.logger.record("train/target_ratio_mean", _mean(target_mean))
        self.logger.record("train/ratio_abs_dev", _mean(ratio_dev))
        grad_norms_t = th.stack(grad_norms).float()
        self.logger.record("train/grad_norm_preclip", grad_norms_t.mean().item())
        # grad_clip_frac: fraction of minibatches where the clip that governs the actor fired
        # (joint clip by default, actor-only clip with controlled_eps)
        self.logger.record("train/grad_clip_frac", (grad_norms_t > self.max_grad_norm).float().mean().item())
        self.logger.record("train/grad_norm_pi_preclip", _mean(pi_grad_norms))
        if state_kls:
            kl_all = th.cat(state_kls).float()
            self.logger.record("train/state_kl_mean", kl_all.mean().item())
            self.logger.record("train/state_kl_p90", th.quantile(kl_all, 0.9).item())
            self.logger.record("train/state_kl_max", kl_all.max().item())
            self.logger.record("train/state_clip_frac", _mean(state_clip_fracs))
            self.logger.record("train/state_eps_mean", _mean(state_eps))
        if cdf_vals:
            cdf_all = th.cat(cdf_vals).float()
            # PIT check: ~0.5 mean and ~1/12 = 0.083 variance if the critic is calibrated
            self.logger.record("train/cdf_mean", cdf_all.mean().item())
            self.logger.record("train/cdf_var", cdf_all.var().item())
            self.logger.record("train/critic_iqr", _mean(iqr_vals))
            self.logger.record("train/median_minus_mean", _mean(median_gap))
        if self.clip_controller is not None:
            next_eps = self.clip_controller.update(div["kl_exact"])
            self.logger.record("train/clip_ratio_next", next_eps)
        if hasattr(self.policy, "log_std"):
            self.logger.record("train/std", th.exp(self.policy.log_std).mean().item())

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")

    def collect_rollouts(
        self,
        env: VecEnv,
        callback: BaseCallback,
        rollout_buffer: RolloutBuffer,
        n_rollout_steps: int,
    ) -> bool:
        """
        Collect experiences using the current policy and fill a ``RolloutBuffer``.
        The term rollout here refers to the model-free notion and should not
        be used with the concept of rollout used in model-based RL or planning.

        :param env: The training environment
        :param callback: Callback that will be called at each step
            (and at the beginning and end of the rollout)
        :param rollout_buffer: Buffer to fill with rollouts
        :param n_rollout_steps: Number of experiences to collect per environment
        :return: True if function returned with at least `n_rollout_steps`
            collected, False if callback terminated rollout prematurely.
        """
        assert self._last_obs is not None, "No previous observation was provided"
        # Switch to eval mode (this affects batch norm / dropout)
        self.policy.set_training_mode(False)

        n_steps = 0
        rollout_buffer.reset()
        # Sample new weights for the state dependent exploration
        if self.use_sde:
            self.policy.reset_noise(env.num_envs)

        callback.on_rollout_start()

        while n_steps < n_rollout_steps:
            if (
                self.use_sde
                and self.sde_sample_freq > 0
                and n_steps % self.sde_sample_freq == 0
            ):
                # Sample a new noise matrix
                self.policy.reset_noise(env.num_envs)

            with th.no_grad():
                # Convert to pytorch tensor or to TensorDict
                obs_tensor = obs_as_tensor(self._last_obs, self.device)
                actions, values, u_values, log_probs = self.policy(obs_tensor)
            actions = actions.cpu().numpy()

            # Rescale and perform action
            clipped_actions = actions

            if isinstance(self.action_space, spaces.Box):
                if self.policy.squash_output:
                    # Unscale the actions to match env bounds
                    # if they were previously squashed (scaled in [-1, 1])
                    clipped_actions = self.policy.unscale_action(clipped_actions)
                else:
                    # Otherwise, clip the actions to avoid out of bound error
                    # as we are sampling from an unbounded Gaussian distribution
                    clipped_actions = np.clip(
                        actions, self.action_space.low, self.action_space.high
                    )

            new_obs, rewards, dones, infos = env.step(clipped_actions)

            self.num_timesteps += env.num_envs

            # Give access to local variables
            callback.update_locals(locals())
            if not callback.on_step():
                return False

            self._update_info_buffer(infos, dones)
            n_steps += 1

            if isinstance(self.action_space, spaces.Discrete):
                # Reshape in case of discrete action
                actions = actions.reshape(-1, 1)

            # Handle timeout by bootstrapping with value function
            # see GitHub issue #633
            for idx, done in enumerate(dones):
                if (
                    done
                    and infos[idx].get("terminal_observation") is not None
                    and infos[idx].get("TimeLimit.truncated", False)
                ):
                    terminal_obs = self.policy.obs_to_tensor(
                        infos[idx]["terminal_observation"]
                    )[0]
                    with th.no_grad():
                        terminal_value = self.policy.predict_values(terminal_obs)[0]  # type: ignore[arg-type]
                    rewards[idx] += self.gamma * terminal_value

            rollout_buffer.add(
                self._last_obs,  # type: ignore[arg-type]
                actions,
                rewards,
                self._last_episode_starts,  # type: ignore[arg-type]
                values,
                log_probs,
            )
            self._last_obs = new_obs  # type: ignore[assignment]
            self._last_episode_starts = dones

        with th.no_grad():
            # Compute value for the last timestep
            values = self.policy.predict_values(obs_as_tensor(new_obs, self.device))  # type: ignore[arg-type]

        rollout_buffer.compute_returns_and_advantage(last_values=values, dones=dones)

        callback.update_locals(locals())

        callback.on_rollout_end()

        return True

    def learn(
        self: SelfBPO,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 1,
        tb_log_name: str = "BPO",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
    ) -> SelfBPO:
        return super().learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=log_interval,
            tb_log_name=tb_log_name,
            reset_num_timesteps=reset_num_timesteps,
            progress_bar=progress_bar,
        )
