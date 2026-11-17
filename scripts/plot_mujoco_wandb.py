"""Build reproducible offline figures and an interactive report from a W&B snapshot.

Requires numpy, pandas, matplotlib, plotly. Run pull_mujoco_wandb.py first,
then with --histories. No W&B writes are performed by either script.
"""
import argparse
import html
import json
import os
from collections import defaultdict
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR', '/tmp/bpo-matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

METRIC = 'rollout/ep_rew_mean'
HORIZON = 1_015_808  # First complete rollout after the requested 1M-step budget.
LABELS = ['Original BPO (paper proxy)', 'Tuned PPO', 'Tuned BPO', 'Best completed adaptive']
COLORS = ['#65758B', '#2378BC', '#D38A16', '#A94391']


def norm_name(r):
    e = r['config']['env']
    if not e.get('normalize', False):
        return 'raw'
    return ('obs+reward norm' if e.get('norm_obs', True) and e.get('norm_reward', True)
            else 'obs norm' if e.get('norm_obs', True) else 'reward norm')


def details(r):
    a = r['config']['algo']
    text = norm_name(r)
    if a.get('adaptive_clip'):
        text += f", KL target={a['clip_target_kl']:g}"
    elif 'clip_ratio' in a:
        text += f", eps={a['clip_ratio']:g}, lambda={a['lambda_']:g}"
    if a.get('state_clip'):
        text += f", state {a['state_clip']}={a['state_kl_max']:g}"
    return text


def load_runs(root):
    inventory = json.loads((root / 'inventory.json').read_text())
    rows = []
    for r in inventory['runs']:
        path = root / 'histories' / (r['id'] + '.json')
        if not path.exists():
            continue
        h = pd.DataFrame(json.loads(path.read_text()))
        if h.empty:
            continue
        h = h.dropna(subset=['global_step', METRIC]).sort_values('_step')
        h = h[np.isfinite(h['global_step']) & np.isfinite(h[METRIC])]
        h = h.drop_duplicates('global_step', keep='last').sort_values('global_step')
        if h.empty:
            continue
        r['x'] = h.global_step.to_numpy(dtype=float)
        r['y'] = h[METRIC].to_numpy(dtype=float)
        r['env'] = r['config']['env']['env_id']
        r['seed'] = r['config']['training']['seed']
        rows.append(r)
    return inventory, rows


def config_key(r):
    # Keep experiment groups separate: code revisions and durations may differ.
    a = dict(r['config']['algo'])
    for k in ['device', 'verbose', '_convert_', 'per_env']:
        a.pop(k, None)
    e = dict(r['config']['env'])
    for k in ['seed', 'render_mode']:
        e.pop(k, None)
    return (r['group'], json.dumps(dict(algo=a, env=e), sort_keys=True))


def candidates(runs, predicate, require_finished=True):
    groups = defaultdict(list)
    for r in runs:
        if predicate(r) and (not require_finished or
                            (r['state'] == 'finished' and r['x'][-1] >= HORIZON)):
            groups[config_key(r)].append(r)
    out = []
    for group in groups.values():
        # Retried runs do not count as independent seeds; take latest eligible run.
        seeds = {}
        for r in sorted(group, key=lambda x: x['created_at']):
            seeds[r['seed']] = r
        if len(seeds) >= 3:
            out.append(sorted(seeds.values(), key=lambda x: x['seed']))
    return out


def score(group, horizon=HORIZON):
    return float(np.mean([r['y'][(r['x'] >= .9 * horizon) & (r['x'] <= horizon)].mean()
                          for r in group]))


def aggregate(group, cap):
    end = min(cap, min(r['x'][-1] for r in group))
    start = max(r['x'][0] for r in group)
    x = np.unique(np.concatenate([r['x'][(r['x'] >= start) & (r['x'] <= end)] for r in group]))
    ys = np.array([np.interp(x, r['x'], r['y']) for r in group])
    return x, ys.mean(axis=0), ys.std(axis=0, ddof=1)


def figure(panels, title, subtitle, stem, root, columns=3, cap=HORIZON):
    nrows = int(np.ceil(len(panels) / columns))
    fig, axes = plt.subplots(nrows, columns, figsize=(5.3 * columns, 3.8 * nrows + .7), squeeze=False)
    interactive = make_subplots(rows=nrows, cols=columns,
                                subplot_titles=[p[0] for p in panels], vertical_spacing=.14)
    displayed = set()
    for i, (env, series) in enumerate(panels):
        ax = axes.flat[i]
        row, col = i // columns + 1, i % columns + 1
        for label, color, group, dash in series:
            x, mean, sd = aggregate(group, cap)
            ax.fill_between(x / 1e6, mean - sd, mean + sd, color=color, alpha=.13, lw=0)
            ax.plot(x / 1e6, mean, color=color, lw=2, ls='--' if dash else '-', label=f'{label} (n={len(group)})')
            for bound_index, y in enumerate([mean - sd, mean + sd]):
                interactive.add_trace(go.Scatter(x=x / 1e6, y=y, mode='lines',
                    line=dict(width=0), fill='tonexty' if bound_index == 1 else None,
                    fillcolor=f'rgba({int(color[1:3],16)},{int(color[3:5],16)},{int(color[5:7],16)},0.12)',
                    hoverinfo='skip', showlegend=False, legendgroup=label), row=row, col=col)
            interactive.add_trace(go.Scatter(x=x / 1e6, y=mean, name=label,
                legendgroup=label, showlegend=label not in displayed,
                line=dict(color=color, width=2.5, dash='dash' if dash else 'solid'),
                customdata=np.column_stack([sd, np.full(len(x),len(group))]),
                hovertemplate='Steps: %{x:.3f}M<br>Mean return: %{y:,.1f}<br>SD: %{customdata[0]:,.1f}<br>Seeds: %{customdata[1]:.0f}<extra>%{fullData.name}</extra>'), row=row, col=col)
            displayed.add(label)
        ax.set(title=env, xlabel='Environment steps (millions)', ylabel='Episode mean return')
        ax.grid(alpha=.2)
        ax.spines[['top','right']].set_visible(False)
        ax.legend(loc='best', fontsize=8, framealpha=.9)
        ax.set_xlim(left=0)
        interactive.update_xaxes(title_text='Environment steps (millions)', row=row, col=col)
        interactive.update_yaxes(title_text='Episode mean return', row=row, col=col)
    for ax in list(axes.flat)[len(panels):]:
        ax.set_visible(False)
    fig.suptitle(title, fontsize=19, y=.98, fontweight='bold')
    fig.text(.5, .88 if nrows == 1 else .925, subtitle, ha='center', fontsize=10, color='#475569')
    fig.tight_layout(rect=(0, 0, 1, .85 if nrows == 1 else .905))
    fig.savefig(root / f'{stem}.png', dpi=180, bbox_inches='tight')
    fig.savefig(root / f'{stem}.pdf', bbox_inches='tight')
    plt.close(fig)
    interactive.update_layout(template='plotly_white', height=400*nrows+90,
        title=dict(text=title+'<br><sup>'+subtitle+'</sup>',font=dict(size=21)),
        margin=dict(t=125,b=90), legend=dict(orientation='h',y=-.17), hovermode='closest')
    return interactive


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=Path('results/mujoco_training_curves'))
    args = parser.parse_args()
    root = args.output
    inventory, runs = load_runs(root)
    environments = sorted(set(r['env'] for r in runs))
    panels, selected, ranking = [], [], []
    for env in environments:
        rs = [r for r in runs if r['env'] == env]
        baseline_group = 'lambda_norm_ant_v1' if env.startswith('Ant-') else 'lambda_norm_mujoco_v1'
        predicates = [
            lambda r: r['group'] == baseline_group and norm_name(r) == 'raw' and r['config']['algo'].get('lambda_') == .001,
            lambda r: r['config']['algo']['_target_'].endswith('.PPO'),
            lambda r: r['config']['algo']['_target_'] == 'sb3.BPO' and not r['config']['algo'].get('adaptive_clip',False)
                      and not r['config']['algo'].get('controlled_eps',False) and r['group'] != 'dist_critic_v1',
            lambda r: r['config']['algo'].get('adaptive_clip',False) and not r['config']['algo'].get('controlled_eps',False),
        ]
        series = []
        for label, color, predicate in zip(LABELS, COLORS, predicates):
            options = candidates(rs, predicate)
            if not options:
                raise ValueError(f'No eligible three-seed configuration: {env} / {label}')
            options.sort(key=score, reverse=True)
            group = options[0]
            series.append((label,color,group,False))
            for rank, option in enumerate(options,1):
                ranking.append(dict(env=env,category=label,rank=rank,group=option[0]['group'],
                    settings=details(option[0]),n=len(option),late_mean=score(option),
                    run_ids=','.join(r['id'] for r in option)))
            x, mean, sd = aggregate(group,HORIZON)
            selected.append(dict(env=env,category=label,group=group[0]['group'],settings=details(group[0]),
                n=len(group),last_step=int(x[-1]),final_mean=float(mean[-1]),final_sd=float(sd[-1]),
                late_mean=score(group),runs=[{k:r[k] for k in ['id','name','url','seed','state','config']} for r in group]))
        panels.append((env,series))
    pd.DataFrame(ranking).to_csv(root/'candidate_rankings.csv',index=False)
    (root/'selection.json').write_text(json.dumps(selected,indent=2))
    summary = pd.DataFrame([{k:v for k,v in row.items() if k!='runs'} for row in selected])
    summary.to_csv(root/'summary.csv',index=False)
    tidy = []
    for env, series in panels:
        for label, _, group, _ in series:
            for r in group:
                tidy.extend(dict(env=env,category=label,run_id=r['id'],seed=r['seed'],global_step=int(x),episode_mean_return=float(y))
                            for x,y in zip(r['x'],r['y']))
    pd.DataFrame(tidy).to_csv(root/'selected_histories.csv',index=False)
    plots = [figure(panels,'MuJoCo training returns · completed comparisons',
        'Logged returns, no extra smoothing · mean ±1 sample SD · configurations selected by mean return over final 10% of shared budget',
        'mujoco_comparison',root)]

    latest_panels, latest_summary = [], []
    for env, series in panels:
        options = candidates([r for r in runs if r['env']==env],lambda r:r['group']=='adaptive_clip_best_v2',False)
        if not options:
            continue
        # The same cutoff is used for ranking all latest settings on this environment.
        cutoff = min(r['x'][-1] for group in options for r in group)
        latest_series = [series[2],series[3]]
        palette = ['#45A2A4','#4A65D1','#E46F44','#BD56BF','#653DA8']
        options.sort(key=lambda g:(bool(g[0]['config']['algo'].get('adaptive_clip')),g[0]['config']['algo'].get('clip_target_kl') if g[0]['config']['algo'].get('adaptive_clip') else g[0]['config']['algo']['clip_ratio']))
        for group,color in zip(options,palette):
            a=group[0]['config']['algo']
            label = f"Latest adaptive KL={a['clip_target_kl']:g}" if a.get('adaptive_clip') else f"Latest fixed eps={a['clip_ratio']:g}"
            latest_series.append((label,color,group,True))
            latest_summary.append(dict(env=env,category=label,n=len(group),common_comparison_step=int(cutoff),
                late_mean_at_common_budget=score(group,cutoff),min_available_step=int(min(r['x'][-1] for r in group)),
                states=','.join(r['state'] for r in group),run_ids=','.join(r['id'] for r in group)))
        latest_panels.append((env,latest_series))
    if latest_panels:
        plots.append(figure(latest_panels,'Latest clipping sweep · partial training curves',
            'Dashed: v2 sweep · solid: completed references · every curve stops at the shortest run in its seed group',
            'mujoco_latest',root))
    pd.DataFrame(latest_summary).to_csv(root/'latest_summary.csv',index=False)

    long_panels=[]
    for env in environments:
        rs=[r for r in runs if r['env']==env]
        opts=candidates(rs,lambda r:r['group']=='long_5m_v1')
        if not opts:
            continue
        long_series=[('BPO '+norm_name(g[0]),COLORS[0] if norm_name(g[0])=='raw' else COLORS[2],g,False) for g in opts]
        ppo=candidates(rs,lambda r:r['config']['algo']['_target_'].endswith('.PPO') and r['x'][-1]>=5e6)
        long_series.extend(('PPO '+norm_name(g[0]),'#77B7E8' if norm_name(g[0])=='raw' else COLORS[1],g,False) for g in ppo)
        long_panels.append((env,long_series))
    if long_panels:
        plots.append(figure(long_panels,'Longer training · all available 5M-step baselines',
            'Mean ±1 sample SD · raw and normalized settings shown separately · adaptive runs currently have shorter budgets',
            'mujoco_long',root,columns=2,cap=5_013_504))

    note = ('Original BPO is a provisional paper-baseline proxy: fixed eps=0.3, lambda=0.001, '
            'no normalization, adaptive clipping off. No runs explicitly identify themselves as paper reproductions. '
            'The checked-in paper sweep settings differ in rollout/optimizer settings, so this is not a verified paper reproduction.')
    methods = ('Source: vladmak/BPO. Metric: rollout/ep_rew_mean, versus global_step (environment transitions across all parallel environments). '
        'These are training returns, not evaluation returns; Monitor records original rewards before VecNormalize. '
        'No additional smoothing. Shading is ±1 sample standard deviation across seeds, not a confidence interval. '
        'Main comparison uses finished configurations with at least three distinct seeds reaching 1,015,808 steps. '
        'Each tuned/adaptive configuration is selected by highest seed-averaged return over the final 10% of that common budget. '
        'Selection and visualization use the same data, so best-of-sweep curves are exploratory, not unbiased held-out estimates. '
        'PPO tuning available here is normalization selection; this project does not show a broad optimizer-parameter PPO sweep. '
        'Tuned BPO candidates include normalization, lambda, fixed epsilon, and per-state clipping parameters; controlled-epsilon and distributional-critic variants are excluded. '
        'Groups are kept separate and retries deduplicated by seed. Longer runs are truncated for main comparisons. '
        'Latest v2 curves are snapshots, can be incomplete, and are never extrapolated beyond any contributing seed.')
    run_lines=[]
    for s in selected:
        run_lines.append(f"- **{s['env']} / {s['category']}** — {s['settings']}; n={s['n']}; " + ', '.join(f"[{r['name']}]({r['url']})" for r in s['runs']))
    readme = '# MuJoCo W&B training curves\n\n'+note+'\n\n'+methods+'\n\n'
    readme += f"Inventory fetched at {inventory['fetched_at']}. Histories were fetched immediately afterward; running runs can advance between calls.\n\n"
    readme += 'Open `report.html` for interactive plots. Static figures are PNG and PDF. `summary.csv`, `selection.json`, `candidate_rankings.csv`, and `selected_histories.csv` document the selection and data. `histories/` preserves all downloaded histories.\n\n'
    readme += '## Selected runs\n\n'+'\n'.join(run_lines)
    readme += '\n\n## Reproduce\n\n```bash\npython scripts/pull_mujoco_wandb.py\npython scripts/pull_mujoco_wandb.py --histories\npython scripts/plot_mujoco_wandb.py\n```\n\nThe pull uses WANDB_API_KEY or the existing api.wandb.ai netrc entry. Credentials are never exported. Existing history files are cached; remove the output histories directory to refresh a snapshot. Plotting requires numpy, pandas, matplotlib, and plotly.\n'
    (root/'README.md').write_text(readme)
    report = '<!doctype html><html><head><meta charset="utf-8"><title>MuJoCo training returns</title><style>body{font:16px system-ui;margin:30px;color:#16243a;background:#fafbfd}p{max-width:1200px;line-height:1.6}.note{padding:18px;background:#fff3d9;border-radius:10px}table{border-collapse:collapse;font-size:13px}td,th{padding:9px;border-bottom:1px solid #dde3eb;text-align:left}h1{font-size:32px}</style></head><body><h1>MuJoCo training returns</h1>'
    report += '<p>Source: <a href="https://wandb.ai/vladmak/BPO">vladmak/BPO</a> · snapshot '+html.escape(inventory['fetched_at'])+'</p>'
    report += '<p class="note">'+html.escape(note)+'</p><p>'+html.escape(methods)+'</p>'
    for i, plot in enumerate(plots):
        report += plot.to_html(full_html=False,include_plotlyjs=True if i==0 else False)
    report += '<h2>Selected configurations</h2>'+summary.round(2).to_html(index=False)
    report += '<h2>W&B source runs</h2><ul>'
    for s in selected:
        report += '<li>'+html.escape(s['env']+' / '+s['category'])+': '+', '.join('<a href="'+html.escape(r['url'])+'">'+html.escape(r['name'])+'</a>' for r in s['runs'])+'</li>'
    report += '</ul></body></html>'
    (root/'report.html').write_text(report)
    print(summary[['env','category','settings','n','final_mean','late_mean']].to_string(index=False))
    print('Wrote report:',root/'report.html')


if __name__ == '__main__':
    main()
