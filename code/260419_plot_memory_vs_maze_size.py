#!/usr/bin/env python3
"""
260419_plot_memory_vs_maze_size.py

Analyze how the critical memory horizon (RecurrentSAC reset_interval)
relates to maze size (accessible states) on a per-session basis.

Produces:
  1. RPA vs reset_interval, split by maze size (12 vs 16 nodes)
  2. RPA vs K/N (memory fraction of maze), per session
  3. Per-session heatmap: session x reset_interval

Usage:
    conda activate latMaz_RL
    python code/260419_plot_memory_vs_maze_size.py
"""
import os, sys
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.dirname(__file__))

from experiment_config import load_yoking_df, load_reward_df, build_runner, filter_sessions

# ============================================================================
# Load data
# ============================================================================
ABLATION_CSV = 'data_out/rl_sims/c260419-143647_recsac_memory_reset_ablation.csv'
assert os.path.exists(ABLATION_CSV)

abl = pd.read_csv(ABLATION_CSV)
abl['rpa'] = abl['total_reward'] / abl['n_actions']

yoking_df = load_yoking_df()
rwd_df    = load_reward_df()
runner    = build_runner(yoking_df, rwd_df)
sessions  = sorted(filter_sessions(yoking_df))

# Get per-session maze info
sess_info = []
for exp_mom in sessions:
    yoke_row = yoking_df[yoking_df['exp_moment'] == exp_mom].iloc[0]
    adj_mat, st_pos = runner._load_maze(yoke_row['adj_file'], yoke_row['st_pos_file'])
    connected = int((adj_mat.sum(axis=1) > 0).sum())
    adj_name = yoke_row['adj_file']
    if 'latMaz100' in adj_name:
        maze = 'latMaz100'
    elif 'latMaz101' in adj_name:
        maze = 'latMaz101'
    elif 'latMaz103' in adj_name:
        maze = 'latMaz103'
    else:
        maze = 'unknown'
    n_actions = int(yoke_row['n_states_visited']) - 1
    n_edges = int(adj_mat.sum() // 2)
    sess_info.append({
        'exp_moment': exp_mom, 'maze': maze,
        'n_nodes': connected, 'n_edges': n_edges, 'n_actions': n_actions,
    })

info = pd.DataFrame(sess_info)
abl = abl.merge(info, on='exp_moment', how='left')

# Convert reset_interval to numeric (None → large number for sorting)
abl['K'] = abl['reset_interval'].apply(lambda x: 9999 if (pd.isna(x) or str(x) == 'None') else int(float(x)))
K_VALUES = sorted(abl['K'].unique())

ts = 'c' + datetime.now().strftime('%y%m%d-%H%M%S')
FIG_DIR = os.path.join('reports', 'figures')
os.makedirs(FIG_DIR, exist_ok=True)

# ============================================================================
# Figure 1: RPA vs K, split by maze size
# ============================================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

# Panel A: by maze topology
ax = axes[0]
for maze, color, marker in [('latMaz100', '#2196F3', 'o'),
                              ('latMaz101', '#FF9800', 's'),
                              ('latMaz103', '#4CAF50', '^')]:
    sub = abl[abl['maze'] == maze]
    n_nodes = sub['n_nodes'].iloc[0]
    means, sems = [], []
    k_plot = []
    for k in K_VALUES:
        vals = sub[sub['K'] == k].groupby('exp_moment')['rpa'].mean()
        if len(vals) > 0:
            means.append(vals.mean())
            sems.append(vals.std() / np.sqrt(len(vals)))
            k_plot.append(k)
    k_labels = [str(k) if k < 9999 else '∞' for k in k_plot]
    ax.errorbar(range(len(k_plot)), means, yerr=sems,
                label=f'{maze} (N={n_nodes})', color=color, marker=marker,
                capsize=3, linewidth=1.5, markersize=6)

ax.set_xticks(range(len(K_VALUES)))
ax.set_xticklabels([str(k) if k < 9999 else '∞' for k in K_VALUES], fontsize=9)
ax.set_xlabel('Memory reset interval K (steps)', fontsize=11)
ax.set_ylabel('Mean RPA (± SEM)', fontsize=11)
ax.set_title('(A) RPA vs memory horizon, by maze', fontsize=12)
ax.legend(fontsize=9, loc='lower left')
ax.axhline(y=0.385, color='gray', linestyle='--', alpha=0.5, label='Forward-random')
ax.axhline(y=0.253, color='green', linestyle='--', alpha=0.5, label='Mouse')
ax.set_ylim(0.2, 0.9)
ax.grid(True, alpha=0.3)

# Panel B: by K/N ratio (memory as fraction of maze)
ax = axes[1]
for maze, color, marker in [('latMaz100', '#2196F3', 'o'),
                              ('latMaz101', '#FF9800', 's'),
                              ('latMaz103', '#4CAF50', '^')]:
    sub = abl[abl['maze'] == maze]
    n_nodes = sub['n_nodes'].iloc[0]
    ratios, means, sems = [], [], []
    for k in K_VALUES:
        if k == 9999:
            continue  # skip infinity for ratio plot
        ratio = k / n_nodes
        vals = sub[sub['K'] == k].groupby('exp_moment')['rpa'].mean()
        if len(vals) > 0:
            ratios.append(ratio)
            means.append(vals.mean())
            sems.append(vals.std() / np.sqrt(len(vals)))
    ax.errorbar(ratios, means, yerr=sems,
                label=f'{maze} (N={n_nodes})', color=color, marker=marker,
                capsize=3, linewidth=1.5, markersize=6)

ax.set_xlabel('K / N (memory steps / maze nodes)', fontsize=11)
ax.set_title('(B) RPA vs memory-to-maze ratio K/N', fontsize=12)
ax.legend(fontsize=9, loc='lower right')
ax.axhline(y=0.385, color='gray', linestyle='--', alpha=0.5)
ax.axhline(y=0.253, color='green', linestyle='--', alpha=0.5)
ax.set_ylim(0.2, 0.9)
ax.axvline(x=1.0, color='red', linestyle=':', alpha=0.4, linewidth=1.5)
ax.text(1.05, 0.85, 'K=N', color='red', fontsize=9, alpha=0.6)
ax.set_xscale('log')
ax.grid(True, alpha=0.3)

plt.tight_layout()
fig1_path = os.path.join(FIG_DIR, f'{ts}_memory_ablation_by_maze_size.png')
plt.savefig(fig1_path, dpi=150, bbox_inches='tight')
plt.close()
print(f'Saved: {fig1_path}')

# ============================================================================
# Figure 2: Per-session heatmap (session x K), sorted by maze size then budget
# ============================================================================
# Average RPA across seeds per (session, K)
pivot_data = abl.groupby(['exp_moment', 'K'])['rpa'].mean().reset_index()
pivot = pivot_data.pivot(index='exp_moment', columns='K', values='rpa')

# Sort sessions: by n_nodes (ascending) then n_actions (ascending)
sort_order = info.sort_values(['n_nodes', 'n_actions'])['exp_moment']
pivot = pivot.reindex(sort_order)

# Normalize each row by its K=inf value
baseline = pivot[9999].values[:, None]
pivot_norm = pivot.div(pivot[9999], axis=0)

fig, axes = plt.subplots(1, 2, figsize=(14, 8))

# Panel A: raw RPA
ax = axes[0]
im = ax.imshow(pivot.values, aspect='auto', cmap='RdYlGn', vmin=0.1, vmax=0.9)
ax.set_xticks(range(len(K_VALUES)))
ax.set_xticklabels([str(k) if k < 9999 else '∞' for k in K_VALUES], fontsize=9)
ax.set_xlabel('Reset interval K', fontsize=11)
ax.set_ylabel('Session (sorted by maze size, then budget)', fontsize=11)
ax.set_title('(A) Raw RPA per session', fontsize=12)
plt.colorbar(im, ax=ax, label='RPA', shrink=0.7)

# Add maze size annotations
n_nodes_sorted = info.set_index('exp_moment').reindex(sort_order)['n_nodes'].values
boundaries = []
for i in range(1, len(n_nodes_sorted)):
    if n_nodes_sorted[i] != n_nodes_sorted[i-1]:
        boundaries.append(i)
        ax.axhline(y=i - 0.5, color='white', linewidth=2)
# Label groups
if boundaries:
    ax.text(-0.8, boundaries[0] / 2, f'N={n_nodes_sorted[0]}', fontsize=9,
            ha='right', va='center', color='blue', fontweight='bold')
    ax.text(-0.8, (boundaries[0] + len(n_nodes_sorted)) / 2, f'N={n_nodes_sorted[-1]}', fontsize=9,
            ha='right', va='center', color='blue', fontweight='bold')

# Panel B: normalized (fraction of full-memory RPA)
ax = axes[1]
im2 = ax.imshow(pivot_norm.values, aspect='auto', cmap='RdYlGn', vmin=0.3, vmax=1.05)
ax.set_xticks(range(len(K_VALUES)))
ax.set_xticklabels([str(k) if k < 9999 else '∞' for k in K_VALUES], fontsize=9)
ax.set_xlabel('Reset interval K', fontsize=11)
ax.set_title('(B) Fraction of full-memory RPA', fontsize=12)
plt.colorbar(im2, ax=ax, label='RPA / RPA(K=∞)', shrink=0.7)
for b in boundaries:
    ax.axhline(y=b - 0.5, color='white', linewidth=2)

plt.tight_layout()
fig2_path = os.path.join(FIG_DIR, f'{ts}_memory_ablation_heatmap.png')
plt.savefig(fig2_path, dpi=150, bbox_inches='tight')
plt.close()
print(f'Saved: {fig2_path}')

# ============================================================================
# Figure 3: RPA retention curve with K/N annotations
# ============================================================================
fig, ax = plt.subplots(figsize=(8, 5))

# Pool across all sessions: mean ± SEM
for maze, color, marker in [('latMaz100', '#2196F3', 'o'),
                              ('latMaz101', '#FF9800', 's'),
                              ('latMaz103', '#4CAF50', '^')]:
    sub = abl[abl['maze'] == maze]
    n_nodes = sub['n_nodes'].iloc[0]
    k_real = [k for k in K_VALUES if k < 9999]
    retention = []
    for k in k_real:
        # Per-session: RPA(K) / RPA(inf)
        sess_means_k = sub[sub['K'] == k].groupby('exp_moment')['rpa'].mean()
        sess_means_inf = sub[sub['K'] == 9999].groupby('exp_moment')['rpa'].mean()
        merged = pd.DataFrame({'k': sess_means_k, 'inf': sess_means_inf}).dropna()
        ratio = (merged['k'] / merged['inf']).values
        retention.append((k, ratio.mean(), ratio.std() / np.sqrt(len(ratio)), k / n_nodes))

    ks = [r[0] for r in retention]
    means = [r[1] for r in retention]
    sems = [r[2] for r in retention]
    kn = [r[3] for r in retention]

    ax.errorbar(ks, means, yerr=sems, label=f'{maze} (N={n_nodes})',
                color=color, marker=marker, capsize=3, linewidth=2, markersize=7)

    # Annotate K/N at a few points
    for i, (k, m, _, r) in enumerate(retention):
        if k in [5, 10, 20, 50]:
            ax.annotate(f'K/N={r:.1f}', (k, m), textcoords='offset points',
                        xytext=(8, -12 if maze == 'latMaz100' else 8),
                        fontsize=7, color=color, alpha=0.8)

ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.3)
ax.axhline(y=0.95, color='orange', linestyle=':', alpha=0.5, linewidth=1)
ax.text(2, 0.955, '95% retention', fontsize=8, color='orange', alpha=0.7)
ax.set_xlabel('Reset interval K (steps)', fontsize=12)
ax.set_ylabel('RPA retention (RPA(K) / RPA(∞))', fontsize=12)
ax.set_title('Memory retention curve by maze size', fontsize=13)
ax.set_xscale('log')
ax.legend(fontsize=10)
ax.set_ylim(0.4, 1.05)
ax.grid(True, alpha=0.3)

plt.tight_layout()
fig3_path = os.path.join(FIG_DIR, f'{ts}_memory_retention_curve.png')
plt.savefig(fig3_path, dpi=150, bbox_inches='tight')
plt.close()
print(f'Saved: {fig3_path}')

# ============================================================================
# Summary stats
# ============================================================================
print('\n=== Per-maze summary ===')
for maze in ['latMaz100', 'latMaz101', 'latMaz103']:
    sub = info[info['maze'] == maze]
    n = sub['n_nodes'].iloc[0]
    n_sess = len(sub)
    print(f'{maze}: N={n} nodes, {n_sess} sessions, budget={sub["n_actions"].mean():.0f} mean')

    sub_abl = abl[abl['maze'] == maze]
    for k in [1, 5, 10, 20, 50, 100, 200, 9999]:
        vals = sub_abl[sub_abl['K'] == k].groupby('exp_moment')['rpa'].mean()
        k_label = '∞' if k == 9999 else str(k)
        print(f'  K={k_label:>4s}: RPA={vals.mean():.3f} ± {vals.std():.3f}  (K/N={k/n:.1f})')
