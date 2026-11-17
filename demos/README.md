# Visual Demos: Policy Gradient Collapse, TRPO, and PPO

This directory contains standalone, reproducible demonstrations illustrating the **unstable gradient update failure mode (policy collapse)** in vanilla policy gradients and how **TRPO** (Trust Region Policy Optimization) and **PPO** (Proximal Policy Optimization) fix it.

---

## 1. Demo 1: Analytical 2D Landscape & Geometry Mismatch (`visualize_landscape.py`)

### The Concept
Supervised learning loss landscapes are defined over fixed datasets with Euclidean distances. In Reinforcement Learning:
1. **The geometry is Riemannian (KL divergence)**: A step $\Delta \theta$ in parameter space does not correspond to a uniform shift in action probabilities. For a Gaussian policy $\pi_\theta(a) = \mathcal{N}(\mu, \sigma^2)$, the Fisher Information Matrix is $F = \text{diag}(1/\sigma^2, 2/\sigma^2)$. As confidence increases ($\sigma \to 0$), the curvature along $\mu$ explodes ($1/\sigma^2$).
2. **The "Cliff" Collapse**: Standard gradient updates assume Euclidean geometry (a circle in parameter space). A fixed learning rate causes $\mu$ to overshoot narrow reward ridges and fall into catastrophic failure regions where gradients vanish.

### What it Generates
- **`demos/plots/landscape_2d_comparison.png`**:
  - *Left*: 2D contour map of $J(\mu, \sigma)$ comparing the circular Euclidean step (Vanilla PG) with the elliptical KL trust region (TRPO).
  - *Right*: Multi-step optimization trajectories showing Vanilla PG falling off the cliff vs. TRPO and PPO climbing smoothly to the global optimum.
- **`demos/plots/surrogate_vs_true_return.png`**:
  - 1D slice along the gradient direction comparing True Return $J(\theta)$, the unconstrained linear surrogate $L_{\pi_{\text{old}}}(\theta)$ (which promises infinite reward into the abyss), and the PPO clipped surrogate $L^{CLIP}$ (which plateaus safely).

### How to Run
```bash
.venv/bin/python demos/visualize_landscape.py
```

---

## 2. Demo 2: CartPole-v1 Multi-Epoch Overfitting & Policy Collapse (`cartpole_collapse_demo.py`)

### The Concept
Modern deep RL algorithms (like PPO) achieve sample efficiency by running multiple epochs of gradient updates on a single rollout batch. 
- **Vanilla PG (Unclipped Surrogate)**: When optimizing $\mathbb{E}\left[\frac{\pi_\theta(a|s)}{\pi_{\text{old}}(a|s)} \hat{A}\right]$ over multiple epochs, the optimizer overfits catastrophically to the sample batch. The importance ratio explodes, the KL divergence spikes ($>10^1$), and the policy's real-world balancing performance plummets from 500 to ~9.
- **PPO (Clipped Surrogate)**: Clips the probability ratio $r_t \in [1-\epsilon, 1+\epsilon]$, preventing excessive policy distortion and maintaining stable 500 returns across all epochs.
- **TRPO (Trust Region)**: Enforces hard constraint $D_{KL} \le \delta$ via Conjugate Gradient, preventing large policy shifts.

### What it Generates
- **`demos/plots/cartpole_multi_epoch_collapse.png`**:
  - **Panel A**: Evaluation Return vs. Optimization Epochs (Vanilla crashes, PPO stays stable).
  - **Panel B**: Mean KL Divergence $D_{KL}(\pi_{\text{old}} \parallel \pi_{\text{new}})$ (Vanilla explodes exponentially).
  - **Panel C**: Max Importance Ratio $\max_t r_t(\theta)$.
  - **Panel D**: Policy Action Probabilities $P(\text{Push Right})$ across pole angle (Vanilla saturates at 100% in one direction regardless of angle).

### How to Run
```bash
.venv/bin/python demos/cartpole_collapse_demo.py
```
