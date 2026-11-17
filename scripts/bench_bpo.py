"""Wall-clock benchmark: pre-change BPO (sb3._BPO_old) vs current BPO on a MuJoCo task.

Usage: python -m scripts.bench_bpo [--env Ant-v5] [--updates 4] [--parallel N]
Each variant runs in its own process, one after another, so they don't contend.
Temporary — delete together with src/sb3/_BPO_old.py.
"""
import argparse
import json
import os
import statistics
import subprocess
import sys
import time

VARIANTS = {
    # name: (code, device, torch threads; 0 = torch default)
    "old_auto_default": ("old", "auto", 0),  # what runs today
    "old_cpu_1t": ("old", "cpu", 1),  # isolates change 1
    "new_auto_default": ("new", "auto", 0),  # isolates changes 2+3 on the default device
    "new_cpu_1t": ("new", "cpu", 1),  # full new config
}


def child(variant: str, env_id: str, updates: int) -> None:
    import torch as th

    code, device, threads = VARIANTS[variant]
    if threads:
        th.set_num_threads(threads)
    sys.path.insert(0, "src")
    if code == "old":
        from sb3._BPO_old import BPO
    else:
        from sb3.BPO import BPO
    from sb3.env_factory import create_mujoco_env

    sync = th.cuda.synchronize if th.cuda.is_available() else (lambda: None)
    roll_t, train_t = [], []

    class Timed(BPO):
        def collect_rollouts(self, *a, **k):
            t = time.perf_counter()
            r = super().collect_rollouts(*a, **k)
            sync()
            roll_t.append(time.perf_counter() - t)
            return r

        def train(self):
            t = time.perf_counter()
            super().train()
            sync()
            train_t.append(time.perf_counter() - t)

    n_envs, n_steps = int(os.environ.get("BENCH_ENVS", 8)), int(os.environ.get("BENCH_STEPS", 2048))  # cfgs/env/mujoco/ant.yaml + cfgs/algo/bpo/mujoco.yaml
    env = create_mujoco_env(env_id, n_envs=n_envs, seed=0)
    m = Timed(
        "BPOPolicy", env, n_steps=n_steps, batch_size=64, n_epochs=10, device=device,
        seed=0, verbose=0,
        policy_kwargs=dict(ortho_init=True, activation_fn=th.nn.ReLU,
                           net_arch=dict(pi=[256, 256], vf=[256, 256]), log_std_init=0.0),
    )
    t0 = time.perf_counter()
    m.learn(updates * n_envs * n_steps)
    total = time.perf_counter() - t0
    # drop the first update (CUDA init / allocator warm-up)
    r, t = roll_t[1:] or roll_t, train_t[1:] or train_t
    print("RESULT " + json.dumps(dict(
        variant=variant, device=str(m.device), threads=th.get_num_threads(),
        rollout_s=statistics.median(r), train_s=statistics.median(t),
        update_s=statistics.median(r) + statistics.median(t),
        fps=n_envs * n_steps / (statistics.median(r) + statistics.median(t)),
        total_s=total,
    )), flush=True)


def run(variant, env_id, updates, copies=1):
    cmd = [sys.executable, "-m", "scripts.bench_bpo", "--child", variant,
           "--env", env_id, "--updates", str(updates)]
    t = time.perf_counter()
    procs = [subprocess.Popen(cmd, stdout=subprocess.PIPE, text=True) for _ in range(copies)]
    outs = [p.communicate()[0] for p in procs]
    wall = time.perf_counter() - t
    res = [json.loads(l[7:]) for o in outs for l in o.splitlines() if l.startswith("RESULT ")]
    return res, wall


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", default="Ant-v5")
    ap.add_argument("--updates", type=int, default=4)
    ap.add_argument("--parallel", type=int, default=0, help="also run N concurrent copies (0 = nproc//2, max 8)")
    ap.add_argument("--child")
    a = ap.parse_args()
    if a.child:
        return child(a.child, a.env, a.updates)

    print(f"host cores: {os.cpu_count()}  env: {a.env}  updates/variant: {a.updates}", flush=True)
    rows = {}
    for v in VARIANTS:
        res, _ = run(v, a.env, a.updates)
        if not res:
            print(f"{v}: FAILED", flush=True)
            continue
        rows[v] = res[0]
        print("  " + json.dumps(res[0]), flush=True)

    base = rows.get("old_auto_default")
    print(f"\n{'variant':<18}{'device':<8}{'thr':>4}{'rollout s':>11}{'train s':>9}{'update s':>10}{'fps':>8}{'speedup':>9}")
    for v, r in rows.items():
        sp = base["update_s"] / r["update_s"] if base else float("nan")
        print(f"{v:<18}{r['device']:<8}{r['threads']:>4}{r['rollout_s']:>11.2f}{r['train_s']:>9.2f}"
              f"{r['update_s']:>10.2f}{r['fps']:>8.0f}{sp:>8.2f}x")

    # aggregate throughput when running several seeds at once (the real use case)
    n = a.parallel or min(8, max(1, (os.cpu_count() or 2) // 2))
    print(f"\nconcurrent runs: {n} copies each", flush=True)
    for v in ("old_auto_default", "new_cpu_1t"):
        res, wall = run(v, a.env, a.updates, copies=n)
        if len(res) != n:
            print(f"{v} x{n}: {len(res)}/{n} succeeded", flush=True)
            continue
        agg = sum(r["fps"] for r in res)
        print(f"{v:<18} x{n}: per-run fps {statistics.mean(r['fps'] for r in res):7.0f}  "
              f"aggregate fps {agg:7.0f}  wall {wall:6.1f}s", flush=True)


if __name__ == "__main__":
    main()
