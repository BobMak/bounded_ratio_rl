"""
Visual Demo 2: CartPole-v1 Multi-Epoch Batch Overfitting & Policy Collapse Demo
================================================================================
This script demonstrates the catastrophic failure mode of unconstrained policy
gradient updates when performing multi-epoch training on a batch of rollout data,
and shows how TRPO (Trust Region constraint) and PPO (Clipped surrogate) prevent it.

Experimental Design:
1. Pretrain an Actor-Critic policy on CartPole-v1 to reach maximum performance (500).
2. Collect a single batch of rollout data (2048 transitions).
3. Train three identical copies of the policy on this exact same batch over 20 epochs:
   - Method A: Vanilla Policy Gradient (Unclipped Surrogate)
   - Method B: PPO (Clipped Surrogate Objective, eps=0.2)
   - Method C: TRPO (Trust Region with Conjugate Gradient and KL constraint delta=0.01)
4. Measure and plot:
   - Evaluation return on real environment after each epoch
   - KL divergence D_KL(pi_old || pi_new)
   - Maximum importance sampling ratio max(r_t)
   - Policy action probability curves over pole angle
"""

import os
os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib_cache"
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
import gymnasium as gym
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Set style
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 11

DEVICE = torch.device("cpu")

# -------------------------------------------------------------
# Neural Network Models
# -------------------------------------------------------------
class ActorCritic(nn.Module):
    def __init__(self, state_dim=4, action_dim=2, hidden_dim=64):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, action_dim)
        )
        self.critic = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

    def get_action_and_value(self, state, action=None):
        logits = self.actor(state)
        dist = Categorical(logits=logits)
        if action is None:
            action = dist.sample()
        return action, dist.log_prob(action), dist.entropy(), self.critic(state), dist

    def get_logits(self, state):
        return self.actor(state)

# -------------------------------------------------------------
# Evaluation Helper
# -------------------------------------------------------------
def evaluate_policy(model, env_name="CartPole-v1", n_episodes=10):
    env = gym.make(env_name)
    returns = []
    for _ in range(n_episodes):
        obs, _ = env.reset()
        done = False
        ep_ret = 0
        while not done:
            obs_t = torch.tensor(obs, dtype=torch.float32, device=DEVICE).unsqueeze(0)
            with torch.no_grad():
                logits = model.get_logits(obs_t)
                # Greedy or sampled: let's evaluate sampled (true policy performance)
                dist = Categorical(logits=logits)
                action = dist.sample().item()
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            ep_ret += reward
        returns.append(ep_ret)
    env.close()
    return np.mean(returns), np.std(returns)

# -------------------------------------------------------------
# Fast Pretraining (Standard PPO until score > 480)
# -------------------------------------------------------------
def pretrain_cartpole(target_score=480, max_iters=40, checkpoint_path="demos/cartpole_pretrained.pt"):
    if os.path.exists(checkpoint_path):
        print(f"[Info] Loading existing pretrained checkpoint from {checkpoint_path}")
        model = ActorCritic().to(DEVICE)
        model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
        mean_ret, _ = evaluate_policy(model)
        if mean_ret >= target_score:
            print(f"[Info] Loaded model achieves {mean_ret:.1f} return.")
            return model

    print("[Info] Pretraining CartPole policy with PPO...")
    env = gym.make("CartPole-v1")
    model = ActorCritic().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    for iteration in range(1, max_iters + 1):
        # Collect rollout
        states, actions, log_probs, rewards, values, dones = [], [], [], [], [], []
        obs, _ = env.reset()
        for step in range(1024):
            obs_t = torch.tensor(obs, dtype=torch.float32, device=DEVICE).unsqueeze(0)
            with torch.no_grad():
                act, lp, _, val, _ = model.get_action_and_value(obs_t)
            next_obs, rew, term, trunc, _ = env.step(act.item())
            done = term or trunc

            states.append(obs)
            actions.append(act.item())
            log_probs.append(lp.item())
            rewards.append(rew)
            values.append(val.item())
            dones.append(done)

            obs = next_obs
            if done:
                obs, _ = env.reset()

        # GAE calculation
        states = torch.tensor(np.array(states), dtype=torch.float32, device=DEVICE)
        actions = torch.tensor(actions, dtype=torch.long, device=DEVICE)
        old_log_probs = torch.tensor(log_probs, dtype=torch.float32, device=DEVICE)
        rewards = np.array(rewards)
        values = np.array(values)
        dones = np.array(dones)

        advantages = np.zeros_like(rewards)
        last_gae = 0
        gamma, lam = 0.99, 0.95
        for t in reversed(range(len(rewards))):
            next_val = values[t + 1] if t + 1 < len(rewards) and not dones[t] else 0.0
            delta = rewards[t] + gamma * next_val - values[t]
            last_gae = delta + gamma * lam * (1.0 - dones[t]) * last_gae
            advantages[t] = last_gae
        returns = torch.tensor(advantages + values, dtype=torch.float32, device=DEVICE)
        advantages = torch.tensor(advantages, dtype=torch.float32, device=DEVICE)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Standard PPO update
        for _ in range(4):
            _, new_lp, entropy, new_vals, _ = model.get_action_and_value(states, actions)
            ratio = torch.exp(new_lp - old_log_probs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1.0 - 0.2, 1.0 + 0.2) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            val_loss = 0.5 * ((new_vals.squeeze() - returns) ** 2).mean()
            loss = policy_loss + val_loss - 0.01 * entropy.mean()

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()

        eval_mean, eval_std = evaluate_policy(model, n_episodes=5)
        if eval_mean >= target_score:
            print(f"[Success] Policy reached {eval_mean:.1f} return at iter {iteration}!")
            break

    env.close()
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    torch.save(model.state_dict(), checkpoint_path)
    return model

# -------------------------------------------------------------
# TRPO Helper Functions (Conjugate Gradient & Hessian-Vector Product)
# -------------------------------------------------------------
def flat_grad(grads):
    return torch.cat([g.contiguous().view(-1) for g in grads])

def get_params(model):
    return torch.cat([p.data.view(-1) for p in model.parameters()])

def set_params(model, flat_params):
    prev_idx = 0
    for p in model.parameters():
        flat_size = int(np.prod(list(p.size())))
        p.data.copy_(flat_params[prev_idx:prev_idx + flat_size].view(p.size()))
        prev_idx += flat_size

def conjugate_gradient(f_Ax, b, cg_iters=10, residual_tol=1e-10):
    p = b.clone()
    r = b.clone()
    x = torch.zeros_like(b)
    rdotr = torch.dot(r, r)
    for _ in range(cg_iters):
        z = f_Ax(p)
        v = rdotr / (torch.dot(p, z) + 1e-10)
        x += v * p
        r -= v * z
        newrdotr = torch.dot(r, r)
        if newrdotr < residual_tol:
            break
        mu = newrdotr / (rdotr + 1e-10)
        p = r + mu * p
        rdotr = newrdotr
    return x

def trpo_step(model, states, actions, advantages, old_log_probs, delta_kl=0.01):
    """
    Executes a single TRPO step on the model using Conjugate Gradient
    and backtracking line search to enforce D_KL <= delta_kl.
    """
    # 1. Compute surrogate gradient
    logits = model.get_logits(states)
    dist = Categorical(logits=logits)
    new_log_probs = dist.log_prob(actions)
    ratio = torch.exp(new_log_probs - old_log_probs)
    surr = (ratio * advantages).mean()
    
    loss_grads = torch.autograd.grad(surr, model.actor.parameters())
    g = flat_grad(loss_grads).data

    # 2. Define Fisher-vector product: F * v
    def f_Ax(v):
        logits = model.get_logits(states)
        dist = Categorical(logits=logits)
        # Compute KL(pi_old || pi)
        with torch.no_grad():
            old_dist = Categorical(logits=logits.clone())
        kl = torch.distributions.kl.kl_divergence(old_dist, dist).mean()
        kl_grads = torch.autograd.grad(kl, model.actor.parameters(), create_graph=True)
        flat_kl_grad = flat_grad(kl_grads)
        kl_v = (flat_kl_grad * v).sum()
        hvp = torch.autograd.grad(kl_v, model.actor.parameters())
        flat_hvp = flat_grad(hvp).data
        return flat_hvp + 0.1 * v  # damping

    # 3. Solve F * step_dir = g via CG
    step_dir = conjugate_gradient(f_Ax, g)
    shs = 0.5 * (step_dir * f_Ax(step_dir)).sum()
    if shs <= 0:
        return 0.0

    step_len = torch.sqrt(2.0 * delta_kl / shs)
    full_step = step_len * step_dir

    # 4. Backtracking line search
    old_params = get_params(model.actor)
    success = False
    for step_frac in [1.0, 0.5, 0.25, 0.125, 0.0625]:
        new_params = old_params + step_frac * full_step
        set_params(model.actor, new_params)
        
        # Check KL constraint and improvement
        with torch.no_grad():
            new_logits = model.get_logits(states)
            new_dist = Categorical(logits=new_logits)
            ref_dist = Categorical(logits=logits)
            kl_val = torch.distributions.kl.kl_divergence(ref_dist, new_dist).mean().item()
            new_surr = (torch.exp(new_dist.log_prob(actions) - old_log_probs) * advantages).mean().item()
            
        if kl_val <= delta_kl * 1.5 and new_surr >= 0:
            success = True
            break

    if not success:
        set_params(model.actor, old_params)
        return 0.0

    return kl_val

# -------------------------------------------------------------
# Main Experiment: Multi-Epoch Batch Comparison
# -------------------------------------------------------------
def run_multi_epoch_collapse_experiment(output_plot="demos/plots/cartpole_multi_epoch_collapse.png"):
    # 1. Get pretrained model
    base_model = pretrain_cartpole(target_score=480)
    initial_ret, _ = evaluate_policy(base_model, n_episodes=15)
    print(f"\n[Status] Baseline Policy Initial Return: {initial_ret:.1f} / 500.0")

    # 2. Collect one batch of rollouts (2048 steps)
    print("[Info] Collecting rollout batch of 2048 transitions...")
    env = gym.make("CartPole-v1")
    states, actions, old_log_probs, rewards, values, dones = [], [], [], [], [], []
    obs, _ = env.reset()
    for _ in range(2048):
        obs_t = torch.tensor(obs, dtype=torch.float32, device=DEVICE).unsqueeze(0)
        with torch.no_grad():
            act, lp, _, val, _ = base_model.get_action_and_value(obs_t)
        next_obs, rew, term, trunc, _ = env.step(act.item())
        done = term or trunc

        states.append(obs)
        actions.append(act.item())
        old_log_probs.append(lp.item())
        rewards.append(rew)
        values.append(val.item())
        dones.append(done)

        obs = next_obs
        if done:
            obs, _ = env.reset()
    env.close()

    # Compute Advantages
    states = torch.tensor(np.array(states), dtype=torch.float32, device=DEVICE)
    actions = torch.tensor(actions, dtype=torch.long, device=DEVICE)
    old_log_probs = torch.tensor(old_log_probs, dtype=torch.float32, device=DEVICE)
    rewards = np.array(rewards)
    values = np.array(values)
    dones = np.array(dones)

    advantages = np.zeros_like(rewards)
    last_gae = 0
    gamma, lam = 0.99, 0.95
    for t in reversed(range(len(rewards))):
        next_val = values[t + 1] if t + 1 < len(rewards) and not dones[t] else 0.0
        delta = rewards[t] + gamma * next_val - values[t]
        last_gae = delta + gamma * lam * (1.0 - dones[t]) * last_gae
        advantages[t] = last_gae
    advantages = torch.tensor(advantages, dtype=torch.float32, device=DEVICE)
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    # Reference distribution from initial policy
    with torch.no_grad():
        ref_logits = base_model.get_logits(states)
        ref_dist = Categorical(logits=ref_logits)

    # Prepare 3 clones of base_model
    model_vanilla = copy.deepcopy(base_model)
    model_ppo = copy.deepcopy(base_model)
    model_trpo = copy.deepcopy(base_model)

    opt_vanilla = optim.Adam(model_vanilla.actor.parameters(), lr=1e-3)
    opt_ppo = optim.Adam(model_ppo.actor.parameters(), lr=1e-3)

    epochs = 20
    history = {
        'vanilla': {'epoch': [0], 'return': [initial_ret], 'kl': [0.0], 'max_ratio': [1.0]},
        'ppo':     {'epoch': [0], 'return': [initial_ret], 'kl': [0.0], 'max_ratio': [1.0]},
        'trpo':    {'epoch': [0], 'return': [initial_ret], 'kl': [0.0], 'max_ratio': [1.0]},
    }

    print(f"[Info] Running {epochs} optimization epochs on the exact same batch...")

    # Training Loop over epochs
    for ep in range(1, epochs + 1):
        # -----------------------------
        # A. Vanilla (Unclipped) Update
        # -----------------------------
        logits_v = model_vanilla.get_logits(states)
        dist_v = Categorical(logits=logits_v)
        new_lp_v = dist_v.log_prob(actions)
        ratio_v = torch.exp(new_lp_v - old_log_probs)
        loss_v = -(ratio_v * advantages).mean()

        opt_vanilla.zero_grad()
        loss_v.backward()
        opt_vanilla.step()

        with torch.no_grad():
            cur_dist_v = Categorical(logits=model_vanilla.get_logits(states))
            kl_v = torch.distributions.kl.kl_divergence(ref_dist, cur_dist_v).mean().item()
            max_r_v = ratio_v.max().item()
        ret_v, _ = evaluate_policy(model_vanilla, n_episodes=8)
        history['vanilla']['epoch'].append(ep)
        history['vanilla']['return'].append(ret_v)
        history['vanilla']['kl'].append(kl_v)
        history['vanilla']['max_ratio'].append(max_r_v)

        # -----------------------------
        # B. PPO (Clipped) Update
        # -----------------------------
        logits_p = model_ppo.get_logits(states)
        dist_p = Categorical(logits=logits_p)
        new_lp_p = dist_p.log_prob(actions)
        ratio_p = torch.exp(new_lp_p - old_log_probs)
        surr1 = ratio_p * advantages
        surr2 = torch.clamp(ratio_p, 1.0 - 0.2, 1.0 + 0.2) * advantages
        loss_p = -torch.min(surr1, surr2).mean()

        opt_ppo.zero_grad()
        loss_p.backward()
        opt_ppo.step()

        with torch.no_grad():
            cur_dist_p = Categorical(logits=model_ppo.get_logits(states))
            kl_p = torch.distributions.kl.kl_divergence(ref_dist, cur_dist_p).mean().item()
            max_r_p = ratio_p.max().item()
        ret_p, _ = evaluate_policy(model_ppo, n_episodes=8)
        history['ppo']['epoch'].append(ep)
        history['ppo']['return'].append(ret_p)
        history['ppo']['kl'].append(kl_p)
        history['ppo']['max_ratio'].append(max_r_p)

        # -----------------------------
        # C. TRPO Update
        # -----------------------------
        trpo_step(model_trpo, states, actions, advantages, old_log_probs, delta_kl=0.01)
        with torch.no_grad():
            cur_dist_t = Categorical(logits=model_trpo.get_logits(states))
            kl_t = torch.distributions.kl.kl_divergence(ref_dist, cur_dist_t).mean().item()
            new_lp_t = cur_dist_t.log_prob(actions)
            max_r_t = torch.exp(new_lp_t - old_log_probs).max().item()
        ret_t, _ = evaluate_policy(model_trpo, n_episodes=8)
        history['trpo']['epoch'].append(ep)
        history['trpo']['return'].append(ret_t)
        history['trpo']['kl'].append(kl_t)
        history['trpo']['max_ratio'].append(max_r_t)

        print(f"Epoch {ep:2d}/{epochs} | "
              f"Vanilla Ret: {ret_v:5.1f} (KL: {kl_v:6.2f}) | "
              f"PPO Ret: {ret_p:5.1f} (KL: {kl_p:5.3f}) | "
              f"TRPO Ret: {ret_t:5.1f} (KL: {kl_t:5.3f})", flush=True)

    # -------------------------------------------------------------
    # 4. Generate Multi-Panel Comparison Figure
    # -------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Panel 1: Evaluation Return vs Epoch
    ax = axes[0, 0]
    ax.plot(history['vanilla']['epoch'], history['vanilla']['return'], 'r-o', linewidth=2.5, markersize=6, label='Vanilla PG (Unclipped)')
    ax.plot(history['ppo']['epoch'], history['ppo']['return'], 'm-s', linewidth=2.5, markersize=6, label='PPO (Clipped, eps=0.2)')
    ax.plot(history['trpo']['epoch'], history['trpo']['return'], 'c-^', linewidth=2.5, markersize=6, label='TRPO (Trust Region, delta=0.01)')
    ax.axhline(500, color='gray', linestyle=':', label='Max Environment Return (500)')
    ax.set_title("A. Policy Performance vs. Batch Epochs", fontsize=12, fontweight='bold')
    ax.set_xlabel("Epoch on Fixed Batch", fontsize=11)
    ax.set_ylabel("Evaluation Return", fontsize=11)
    ax.set_ylim(-10, 520)
    ax.legend(loc='lower left', framealpha=0.9)
    ax.grid(True, alpha=0.3)

    # Panel 2: KL Divergence vs Epoch (Log Scale)
    ax = axes[0, 1]
    ax.plot(history['vanilla']['epoch'], history['vanilla']['kl'], 'r-o', linewidth=2.5, markersize=6, label='Vanilla PG (Explodes > 10^1)')
    ax.plot(history['ppo']['epoch'], history['ppo']['kl'], 'm-s', linewidth=2.5, markersize=6, label='PPO (Bounded ~ 0.02)')
    ax.plot(history['trpo']['epoch'], history['trpo']['kl'], 'c-^', linewidth=2.5, markersize=6, label='TRPO (Guaranteed <= 0.01)')
    ax.axhline(0.01, color='cyan', linestyle='--', label=r'TRPO Bound $\delta = 0.01$')
    ax.set_yscale('symlog', linthresh=0.005)
    ax.set_title(r"B. Mean KL Divergence $D_{KL}(\pi_{old} \parallel \pi_{new})$", fontsize=12, fontweight='bold')
    ax.set_xlabel("Epoch on Fixed Batch", fontsize=11)
    ax.set_ylabel("KL Divergence (symlog)", fontsize=11)
    ax.legend(loc='upper left', framealpha=0.9)
    ax.grid(True, alpha=0.3)

    # Panel 3: Max Ratio max(r_t)
    ax = axes[1, 0]
    ax.plot(history['vanilla']['epoch'], history['vanilla']['max_ratio'], 'r-o', linewidth=2.5, markersize=6, label='Vanilla PG Ratio')
    ax.plot(history['ppo']['epoch'], history['ppo']['max_ratio'], 'm-s', linewidth=2.5, markersize=6, label='PPO Ratio')
    ax.plot(history['trpo']['epoch'], history['trpo']['max_ratio'], 'c-^', linewidth=2.5, markersize=6, label='TRPO Ratio')
    ax.axhline(1.2, color='magenta', linestyle=':', label=r'PPO Clip Bound $1+\epsilon = 1.2$')
    ax.set_yscale('log')
    ax.set_title(r"C. Max Importance Sampling Ratio $\max_t r_t(\theta)$", fontsize=12, fontweight='bold')
    ax.set_xlabel("Epoch on Fixed Batch", fontsize=11)
    ax.set_ylabel(r"$\max_t \pi(a_t|s_t) / \pi_{old}(a_t|s_t)$ (log)", fontsize=11)
    ax.legend(loc='upper left', framealpha=0.9)
    ax.grid(True, alpha=0.3)

    # Panel 4: Policy Probability Profile over Pole Angle
    ax = axes[1, 1]
    angles = np.linspace(-0.25, 0.25, 100)
    # Synthesize test states with varying pole angle, cart centered with zero velocity
    test_states = np.zeros((100, 4))
    test_states[:, 2] = angles  # pole angle index is 2
    test_states_t = torch.tensor(test_states, dtype=torch.float32, device=DEVICE)

    with torch.no_grad():
        p_base = torch.softmax(base_model.get_logits(test_states_t), dim=-1)[:, 1].numpy()
        p_van = torch.softmax(model_vanilla.get_logits(test_states_t), dim=-1)[:, 1].numpy()
        p_ppo = torch.softmax(model_ppo.get_logits(test_states_t), dim=-1)[:, 1].numpy()
        p_trp = torch.softmax(model_trpo.get_logits(test_states_t), dim=-1)[:, 1].numpy()

    ax.plot(angles, p_base, 'gray', linestyle='--', linewidth=2.5, label='Initial Balanced Policy')
    ax.plot(angles, p_van, 'red', linewidth=2.5, label='Vanilla (Collapsed / Saturated)')
    ax.plot(angles, p_ppo, 'magenta', linewidth=2.5, label='PPO Policy')
    ax.plot(angles, p_trp, 'cyan', linewidth=2.5, label='TRPO Policy')
    ax.set_title("D. Policy Action Probability: P(Action = 'Push Right')", fontsize=12, fontweight='bold')
    ax.set_xlabel("Pole Angle (radians)", fontsize=11)
    ax.set_ylabel("Probability P(Right)", fontsize=11)
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc='center right', framealpha=0.9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_plot), exist_ok=True)
    plt.savefig(output_plot, dpi=300)
    plt.close()
    print(f"\n[Done] CartPole multi-epoch comparison plot saved to {output_plot}")

if __name__ == "__main__":
    run_multi_epoch_collapse_experiment()
