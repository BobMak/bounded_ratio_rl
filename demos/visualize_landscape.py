"""
Visual Demo 1: 2D Policy Landscape, Euclidean vs. KL Geometry, and TRPO/PPO Stability
=====================================================================================
This script provides an exact analytical / numerical demonstration of why Vanilla
Policy Gradient fails due to the mismatch between parameter space and distribution
space, and how TRPO (Trust Region) and PPO (Clipped Surrogate) resolve the issue.

Environment: 1D Continuous Gaussian Action Bandit with a Reward Cliff
- Policy: a ~ N(mu, sigma^2), parameterized by theta = (mu, sigma)
- Target reward: High return near a* = 0.6, but a dangerous cliff at a > 1.0 (penalty -5.0).
- Curvature: As sigma becomes small, Fisher Information along mu scales as 1/sigma^2.
"""

import os
os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib_cache"
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Circle
import numpy as np

# Set style
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 11

def reward_fn(a):
    """
    Reward function with a sweet spot near a = 0.6 and a steep penalty cliff for a > 1.0.
    """
    # Smooth bell-shaped reward peaked at 0.6
    base_reward = 2.0 * np.exp(-((a - 0.6) ** 2) / (2 * 0.25 ** 2))
    # Dangerous cliff: huge penalty if a exceeds 1.0
    cliff_penalty = np.where(a > 1.0, -5.0, 0.0)
    return base_reward + cliff_penalty

# Quadrature grid for computing expectations over actions
_A_GRID = np.linspace(-3.0, 3.0, 1000)
_DA = _A_GRID[1] - _A_GRID[0]
_R_GRID = reward_fn(_A_GRID)

def normal_pdf(a, mu, sigma):
    return (1.0 / (np.sqrt(2 * np.pi) * sigma)) * np.exp(-0.5 * ((a - mu) / sigma) ** 2)

def expected_return(mu, sigma):
    """Computes J(mu, sigma) = E_{a ~ N(mu, sigma^2)} [ R(a) ]"""
    pdf = normal_pdf(_A_GRID, mu, sigma)
    return np.sum(pdf * _R_GRID) * _DA

def compute_gradient(mu, sigma, eps=1e-5):
    """Numerical gradient of J with respect to (mu, sigma)."""
    j_base = expected_return(mu, sigma)
    grad_mu = (expected_return(mu + eps, sigma) - j_base) / eps
    grad_sigma = (expected_return(mu, sigma + eps) - j_base) / eps
    return np.array([grad_mu, grad_sigma])

def kl_divergence(mu0, sigma0, mu1, sigma1):
    """Exact KL divergence between N(mu0, sigma0^2) and N(mu1, sigma1^2)."""
    return np.log(sigma1 / sigma0) + (sigma0 ** 2 + (mu0 - mu1) ** 2) / (2 * sigma1 ** 2) - 0.5

def fisher_matrix(sigma):
    """Fisher Information Matrix for Gaussian policy in (mu, sigma) space: F = diag(1/sigma^2, 2/sigma^2)"""
    return np.diag([1.0 / (sigma ** 2), 2.0 / (sigma ** 2)])

def surrogate_objective(mu, sigma, mu0, sigma0):
    """Unclipped surrogate: E_{pi0} [ (pi / pi0) * (R - J0) ]"""
    pdf0 = normal_pdf(_A_GRID, mu0, sigma0)
    pdf = normal_pdf(_A_GRID, mu, sigma)
    ratio = pdf / (pdf0 + 1e-12)
    j0 = expected_return(mu0, sigma0)
    adv = _R_GRID - j0
    return np.sum(pdf0 * ratio * adv) * _DA

def ppo_clipped_surrogate(mu, sigma, mu0, sigma0, epsilon=0.2):
    """PPO clipped surrogate: E_{pi0} [ min(r * A, clip(r, 1-eps, 1+eps) * A) ]"""
    pdf0 = normal_pdf(_A_GRID, mu0, sigma0)
    pdf = normal_pdf(_A_GRID, mu, sigma)
    ratio = pdf / (pdf0 + 1e-12)
    j0 = expected_return(mu0, sigma0)
    adv = _R_GRID - j0
    surr1 = ratio * adv
    surr2 = np.clip(ratio, 1.0 - epsilon, 1.0 + epsilon) * adv
    return np.sum(pdf0 * np.minimum(surr1, surr2)) * _DA


def plot_2d_landscape(output_path="demos/plots/landscape_2d_comparison.png"):
    """
    Plots the 2D optimization landscape J(mu, sigma), contrasting:
    - Euclidean step (Vanilla PG)
    - KL Trust Region ellipse (TRPO)
    - PPO Clipped Surrogate step
    """
    mu_range = np.linspace(-0.2, 1.6, 180)
    sigma_range = np.linspace(0.08, 0.7, 150)
    MU, SIGMA = np.meshgrid(mu_range, sigma_range)
    
    J = np.zeros_like(MU)
    for i in range(MU.shape[0]):
        for j in range(MU.shape[1]):
            J[i, j] = expected_return(MU[i, j], SIGMA[i, j])

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # -------------------------------------------------------------
    # Subplot 1: Geometry & Single Step Comparison
    # -------------------------------------------------------------
    ax = axes[0]
    cp = ax.contourf(MU, SIGMA, J, levels=35, cmap='viridis', alpha=0.9)
    plt.colorbar(cp, ax=ax, label=r'Expected Return $J(\mu, \sigma)$')

    # Mark the optimal region
    max_idx = np.unravel_index(np.argmax(J), J.shape)
    opt_mu, opt_sigma = MU[max_idx], SIGMA[max_idx]
    ax.scatter(opt_mu, opt_sigma, color='gold', s=120, marker='*', zorder=6, label=f'Global Optima ({opt_mu:.2f}, {opt_sigma:.2f})')

    # Starting policy: Confident policy near cliff (mu=0.75, sigma=0.18)
    theta0 = np.array([0.75, 0.18])
    mu0, sigma0 = theta0
    ax.scatter([mu0], [sigma0], color='white', edgecolor='black', s=100, zorder=6, label=r'Initial Policy $\theta_0$')

    # Compute Gradient & Fisher at theta0
    grad = compute_gradient(mu0, sigma0)
    grad_norm = grad / (np.linalg.norm(grad) + 1e-12)
    F = fisher_matrix(sigma0)
    inv_F = np.linalg.inv(F)

    # 1. Euclidean Step (Vanilla PG)
    # A standard learning rate in parameter space takes a step in direction of grad
    lr_vanilla = 0.35
    theta_vanilla = theta0 + lr_vanilla * grad_norm
    ax.annotate('', xy=theta_vanilla, xytext=theta0,
                arrowprops=dict(facecolor='red', edgecolor='red', width=2.5, headwidth=10), zorder=5)
    ax.scatter([theta_vanilla[0]], [theta_vanilla[1]], color='red', s=90, zorder=6,
               label=r'Vanilla PG Step (Over-shoots cliff into $J \approx -5$)')

    # Draw Euclidean "Trust Region" (Circle in parameter space)
    circle = Circle((mu0, sigma0), radius=0.25, color='red', fill=False, linestyle='--', linewidth=2,
                    label=r'Euclidean Ball $\|\Delta \theta\|_2 \leq r$ (Ignores curvature)')
    ax.add_patch(circle)

    # 2. TRPO Step (KL Constrained)
    # Natural gradient: inv_F @ grad
    nat_grad = inv_F @ grad
    delta_kl = 0.05
    # Step size: sqrt(2 * delta / (nat_grad @ F @ nat_grad))
    step_scale = np.sqrt(2.0 * delta_kl / (nat_grad @ F @ nat_grad + 1e-12))
    theta_trpo = theta0 + step_scale * nat_grad
    ax.annotate('', xy=theta_trpo, xytext=theta0,
                arrowprops=dict(facecolor='cyan', edgecolor='black', width=2.5, headwidth=10), zorder=5)
    ax.scatter([theta_trpo[0]], [theta_trpo[1]], color='cyan', edgecolor='black', s=90, zorder=6,
               label=r'TRPO Step (KL-bounded, safe along $\mu$)')

    # Draw KL Divergence Ellipse: Delta_theta^T F Delta_theta = 2 * delta_kl
    # For diagonal F: (Delta_mu / (sigma0 * sqrt(2*delta)))^2 + (Delta_sigma / (sigma0 * sqrt(delta)))^2 = 1
    width_mu = 2.0 * sigma0 * np.sqrt(2.0 * delta_kl)
    height_sigma = 2.0 * sigma0 * np.sqrt(delta_kl)
    kl_ellipse = Ellipse((mu0, sigma0), width=width_mu, height=height_sigma,
                         color='cyan', fill=False, linestyle='-', linewidth=2.5,
                         label=r'TRPO Trust Region $D_{KL} \leq 0.05$ (Ellipse)')
    ax.add_patch(kl_ellipse)

    # Annotate the cliff boundary
    ax.axvline(x=1.0, color='crimson', linestyle=':', linewidth=2, label='Cliff Edge ($a > 1.0$)')

    ax.set_title("Mismatch: Euclidean Step vs. KL Trust Region", fontsize=13, fontweight='bold')
    ax.set_xlabel(r"Mean Action $\mu$", fontsize=12)
    ax.set_ylabel(r"Policy Standard Deviation $\sigma$", fontsize=12)
    ax.set_xlim(-0.1, 1.5)
    ax.set_ylim(0.08, 0.65)
    ax.legend(loc='upper left', framealpha=0.9, fontsize=9)

    # -------------------------------------------------------------
    # Subplot 2: Multi-Step Optimization Trajectories
    # -------------------------------------------------------------
    ax2 = axes[1]
    cp2 = ax2.contourf(MU, SIGMA, J, levels=35, cmap='viridis', alpha=0.9)
    plt.colorbar(cp2, ax=ax2, label=r'Expected Return $J(\mu, \sigma)$')

    # Run multi-step simulation for:
    # 1. Vanilla PG (fixed lr=0.1)
    # 2. TRPO (exact KL-bounded natural gradient)
    # 3. PPO (Clipped surrogate ascent)
    n_steps = 35
    start = np.array([0.72, 0.16])

    # 1. Vanilla PG
    traj_vanilla = [start.copy()]
    cur = start.copy()
    for _ in range(n_steps):
        g = compute_gradient(cur[0], cur[1])
        cur = cur + 0.08 * g
        cur[1] = np.clip(cur[1], 0.08, 0.65)
        cur[0] = np.clip(cur[0], -0.2, 1.5)
        traj_vanilla.append(cur.copy())
    traj_vanilla = np.array(traj_vanilla)

    # 2. TRPO (Natural gradient with line search / KL constraint delta=0.03)
    traj_trpo = [start.copy()]
    cur = start.copy()
    delta = 0.03
    for _ in range(n_steps):
        g = compute_gradient(cur[0], cur[1])
        F_mat = fisher_matrix(cur[1])
        inv_F_mat = np.linalg.inv(F_mat)
        ng = inv_F_mat @ g
        denom = ng @ F_mat @ ng
        if denom > 1e-12:
            step = np.sqrt(2.0 * delta / denom) * ng
            cur = cur + step
            cur[1] = np.clip(cur[1], 0.08, 0.65)
            cur[0] = np.clip(cur[0], -0.2, 1.5)
        traj_trpo.append(cur.copy())
    traj_trpo = np.array(traj_trpo)

    # 3. PPO (Clipped surrogate ascent)
    traj_ppo = [start.copy()]
    cur = start.copy()
    for _ in range(n_steps):
        # Sample mini-batch of actions from pi_old
        a_samples = np.random.normal(cur[0], cur[1], size=128)
        adv_samples = reward_fn(a_samples) - expected_return(cur[0], cur[1])
        pdf_old = normal_pdf(a_samples, cur[0], cur[1])
        
        # Take 5 mini-epochs with clipping
        cur_sub = cur.copy()
        for _ in range(5):
            # Finite diff gradient on clipped surrogate
            eps_diff = 1e-4
            def ppo_loss(th):
                p_new = normal_pdf(a_samples, th[0], th[1])
                r = p_new / (pdf_old + 1e-10)
                clipped_r = np.clip(r, 1.0 - 0.2, 1.0 + 0.2)
                return np.mean(np.minimum(r * adv_samples, clipped_r * adv_samples))
            
            l_base = ppo_loss(cur_sub)
            g_mu = (ppo_loss([cur_sub[0] + eps_diff, cur_sub[1]]) - l_base) / eps_diff
            g_sig = (ppo_loss([cur_sub[0], cur_sub[1] + eps_diff]) - l_base) / eps_diff
            cur_sub = cur_sub + 0.015 * np.array([g_mu, g_sig])
            cur_sub[1] = np.clip(cur_sub[1], 0.08, 0.65)
        cur = cur_sub
        traj_ppo.append(cur.copy())
    traj_ppo = np.array(traj_ppo)

    # Plot trajectories
    ax2.plot(traj_vanilla[:, 0], traj_vanilla[:, 1], 'r.-', linewidth=2.5, markersize=8,
             label='Vanilla PG: Catastrophic Cliff Collapse', zorder=5)
    ax2.plot(traj_trpo[:, 0], traj_trpo[:, 1], 'c.-', linewidth=2.5, markersize=8,
             label='TRPO: Stable Monotonic Ascent', zorder=6)
    ax2.plot(traj_ppo[:, 0], traj_ppo[:, 1], 'm.-', linewidth=2.5, markersize=8,
             label='PPO: Controlled Ascent (Clipped)', zorder=6)
    ax2.scatter([start[0]], [start[1]], color='white', edgecolor='black', s=120, zorder=7)
    ax2.axvline(x=1.0, color='crimson', linestyle=':', linewidth=2, label='Cliff Edge ($a > 1.0$)')

    ax2.set_title("Multi-Step Optimization Paths on Non-Euclidean Surface", fontsize=13, fontweight='bold')
    ax2.set_xlabel(r"Mean Action $\mu$", fontsize=12)
    ax2.set_ylabel(r"Policy Standard Deviation $\sigma$", fontsize=12)
    ax2.set_xlim(-0.1, 1.5)
    ax2.set_ylim(0.08, 0.65)
    ax2.legend(loc='lower left', framealpha=0.9, fontsize=9.5)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"[Done] 2D Landscape comparison saved to {output_path}")


def plot_surrogate_vs_true_return(output_path="demos/plots/surrogate_vs_true_return.png"):
    """
    Plots a 1D slice along the gradient direction showing:
    1. True return J(theta + alpha * d)
    2. Unconstrained linear surrogate L(theta + alpha * d)
    3. PPO clipped surrogate L_clip(theta + alpha * d)
    4. TRPO Trust Region bounds [alpha_min, alpha_max]
    """
    theta0 = np.array([0.70, 0.18])
    mu0, sigma0 = theta0
    grad = compute_gradient(mu0, sigma0)
    # Direction d along gradient
    d = grad / (np.linalg.norm(grad) + 1e-12)

    alphas = np.linspace(-0.3, 0.6, 200)
    true_J = []
    linear_surr = []
    ppo_surr = []
    kl_vals = []

    j0 = expected_return(mu0, sigma0)
    for a in alphas:
        th = theta0 + a * d
        mu, sigma = th[0], th[1]
        true_J.append(expected_return(mu, sigma))
        linear_surr.append(surrogate_objective(mu, sigma, mu0, sigma0) + j0)
        ppo_surr.append(ppo_clipped_surrogate(mu, sigma, mu0, sigma0, epsilon=0.2) + j0)
        kl_vals.append(kl_divergence(mu0, sigma0, mu, sigma))

    true_J = np.array(true_J)
    linear_surr = np.array(linear_surr)
    ppo_surr = np.array(ppo_surr)
    kl_vals = np.array(kl_vals)

    # TRPO boundary where KL = 0.03
    delta = 0.03
    valid_kl = np.where(kl_vals <= delta)[0]
    alpha_min, alpha_max = alphas[valid_kl[0]], alphas[valid_kl[-1]]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True, gridspec_kw={'height_ratios': [2.2, 1]})

    # Top plot: Performance & Surrogates
    ax1.plot(alphas, true_J, 'black', linewidth=3, label=r'True Return $J(\theta_0 + \alpha d)$')
    ax1.plot(alphas, linear_surr, 'r--', linewidth=2.5, label=r'Unconstrained Linear Surrogate $L_{\pi_0}$')
    ax1.plot(alphas, ppo_surr, 'blue', linewidth=2.5, label=r'PPO Clipped Surrogate $L^{CLIP}_{\pi_0}$ ($\epsilon=0.2$)')

    # Shaded TRPO trust region
    ax1.axvspan(alpha_min, alpha_max, color='cyan', alpha=0.25, label=rf'TRPO Trust Region ($D_{{KL}} \leq {delta}$)')
    ax1.axvline(alpha_min, color='cyan', linestyle='--', linewidth=1.5)
    ax1.axvline(alpha_max, color='cyan', linestyle='--', linewidth=1.5)

    # Highlight where Vanilla PG overshoots
    vanilla_alpha = 0.45
    ax1.scatter([vanilla_alpha], [linear_surr[np.argmin(np.abs(alphas - vanilla_alpha))]], color='red', s=80, zorder=5)
    ax1.scatter([vanilla_alpha], [true_J[np.argmin(np.abs(alphas - vanilla_alpha))]], color='black', s=80, zorder=5)
    ax1.annotate('Linear surrogate\npromises huge gain!',
                 xy=(vanilla_alpha, linear_surr[np.argmin(np.abs(alphas - vanilla_alpha))]),
                 xytext=(vanilla_alpha - 0.25, 2.8),
                 arrowprops=dict(arrowstyle='->', color='red', lw=1.5), fontsize=10, color='red', fontweight='bold')
    ax1.annotate('True return falls\noff the cliff!',
                 xy=(vanilla_alpha, true_J[np.argmin(np.abs(alphas - vanilla_alpha))]),
                 xytext=(vanilla_alpha - 0.05, -3.0),
                 arrowprops=dict(arrowstyle='->', color='black', lw=1.5), fontsize=10, fontweight='bold')

    ax1.set_ylabel("Expected Return / Objective Value", fontsize=11)
    ax1.set_title("1D Optimization Slice: Why Unconstrained Policy Gradient Collapses", fontsize=13, fontweight='bold')
    ax1.legend(loc='lower left', framealpha=0.9, fontsize=10)
    ax1.grid(True, alpha=0.3)

    # Bottom plot: KL Divergence
    ax2.plot(alphas, kl_vals, 'darkgreen', linewidth=2.2, label=r'$D_{KL}(\pi_0 \parallel \pi_{\theta})$')
    ax2.axhline(delta, color='cyan', linestyle='--', linewidth=1.5, label=rf'TRPO Bound $\delta = {delta}$')
    ax2.axvspan(alpha_min, alpha_max, color='cyan', alpha=0.25)
    ax2.set_xlabel(r"Step Size along Gradient Direction $\alpha$", fontsize=11)
    ax2.set_ylabel(r"$D_{KL}$", fontsize=11)
    ax2.set_yscale('log')
    ax2.legend(loc='upper right', framealpha=0.9, fontsize=10)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"[Done] 1D Surrogate vs True Return saved to {output_path}")

if __name__ == "__main__":
    plot_2d_landscape()
    plot_surrogate_vs_true_return()
