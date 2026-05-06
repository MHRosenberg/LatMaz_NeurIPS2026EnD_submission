"""
260504_peak_mouse_distribution.py

Supplementary figure: per-session mouse RPA distribution on canonical-60,
overlaid with key reference agent NNS lines. Motivates the "peak mouse
performance is far higher than mean" framing — mice CAN solve these mazes
when they engage; the mean is depressed by short / low-motivation sessions.

Outputs:
  paper/figures/c{ts}_peak_mouse_distribution.{pdf,png}
"""
import sys
import os
from datetime import datetime

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

PROJECT_ROOT = os.environ.get('LATMAZ_REPO_ROOT', os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'code'))

from experiment_config import load_data

OUT_DIR = os.path.join(PROJECT_ROOT, 'paper', 'figures')
TS = 'c' + datetime.now().strftime('%y%m%d-%H%M%S')

# Canonical-60 reference points (from paper_values.tex).
RANDOM_RPA = 0.178
GREEDY_RPA = 0.878
DENOM = GREEDY_RPA - RANDOM_RPA

REFERENCE_LINES = [
    ('Random floor',       0.178, 'gray',  ':'),
    ('Mouse mean',         0.270, '#444', '-'),
    ('Mouse median',       0.276, '#444', '--'),
    ('Random_forward',     0.385, '#666', '-.'),
    ('FSC_bio',            0.503, '#3b8db8', '--'),
    ('DQN pretrained',     0.527, '#2ca25f', '--'),
    ('NoBk+FullFwd',       0.576, '#888888', '--'),
    ('RecSAC (causal)',    0.555, '#2ca25f', ':'),
    ('Greedy oracle',      0.878, 'black',  ':'),
]


def main():
    yo, _, sessions = load_data()
    # Restrict to the canonical 60-session set (the same 60 sessions used
    # for HPO + Table 1). Mouse data has grown since first generation;
    # without this restriction the figure would show all 166 currently
    # filterable sessions, contradicting the "canonical 60-session set"
    # caption (PDF review #98 fix; cld 2026-05-06).
    canonical60_path = os.path.join(
        PROJECT_ROOT, 'data_released', 'results',
        'c260316-171035_hpo_tuning_FIXED.csv')
    canonical60 = set(pd.read_csv(canonical60_path, low_memory=False,
                                   usecols=['exp_moment'])['exp_moment'].dropna().unique())
    canonical_sessions = set(sessions) & canonical60
    print(f'load_data() returned {len(sessions)} sessions; '
          f'restricting to {len(canonical_sessions)} canonical-60 intersect.', flush=True)
    sub = yo[yo['exp_moment'].isin(canonical_sessions)].copy()
    sub['rpa'] = sub['n_rewards'] / (sub['n_states_visited'] - 1).clip(lower=1)
    sub['nns'] = 100.0 * (sub['rpa'] - RANDOM_RPA) / DENOM
    print(f'n sessions = {len(sub)}')
    print(f'mouse mean RPA = {sub.rpa.mean():.3f} (NNS = {sub.nns.mean():.1f}%)')
    print(f'mouse  p90 RPA = {sub.rpa.quantile(0.9):.3f} (NNS = {sub.nns.quantile(0.9):.1f}%)')
    print(f'mouse  max RPA = {sub.rpa.max():.3f} (NNS = {sub.nns.max():.1f}%)')

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2),
                                    gridspec_kw={'width_ratios': [3, 2]})

    # --- left panel: per-session RPA dot plot, ranked descending ---
    rpa_sorted = sub['rpa'].sort_values(ascending=False).values
    x_idx = np.arange(len(rpa_sorted))
    # color by animal
    aid_sorted = sub.sort_values('rpa', ascending=False)['animal_ID'].values
    colors = ['#1f77b4' if a == '031' else '#d62728' for a in aid_sorted]
    ax1.scatter(x_idx, rpa_sorted, c=colors, s=18, edgecolors='black', linewidths=0.4, zorder=3)

    # Reference horizontal lines
    for label, rpa, color, ls in REFERENCE_LINES:
        ax1.axhline(rpa, linestyle=ls, color=color, linewidth=0.9, alpha=0.7, zorder=1)
        # right-side label
        ax1.text(len(rpa_sorted) - 0.5, rpa, '  ' + label, va='center',
                 ha='left', fontsize=7, color=color)

    ax1.set_xlabel('Session rank (descending RPA)', fontsize=10)
    ax1.set_ylabel('Reward Per Action (RPA)', fontsize=10)
    # Wider right margin so reference labels fit alongside dots without
    # competing with the upper-right legend (PDF review #98 fix).
    ax1.set_xlim(-1, len(rpa_sorted) + 26)
    ax1.set_ylim(0, 0.95)
    ax1.set_title('Per-session mouse RPA (canonical-60), ranked', fontsize=10)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.grid(axis='y', linestyle='-', linewidth=0.3, alpha=0.4, zorder=0)
    # Animal legend — moved to lower-right so it does not cover the
    # high-RPA reference labels (Greedy 0.878, NoBk+FullFwd 0.576,
    # RecSAC-causal 0.555). Lower-right is empty: only Random floor at
    # 0.178 lives that low and its label is to the right of the legend.
    from matplotlib.patches import Patch
    ax1.legend(handles=[
        Patch(color='#1f77b4', label='a031'),
        Patch(color='#d62728', label='a033'),
    ], loc='lower right', fontsize=8, frameon=True,
       bbox_to_anchor=(1.0, 0.02))

    # --- right panel: NNS distribution histogram + percentile markers ---
    nns_vals = sub['nns'].values
    ax2.hist(nns_vals, bins=20, color='#888', edgecolor='black', alpha=0.6)
    pct_marks = [
        ('mean', np.mean(nns_vals), '#444', '-'),
        ('median', np.median(nns_vals), '#444', '--'),
        ('p90', np.quantile(nns_vals, 0.90), '#a50f15', '-'),
        ('p95', np.quantile(nns_vals, 0.95), '#a50f15', '--'),
        ('max', np.max(nns_vals), '#a50f15', ':'),
    ]
    # Stagger label y-positions so close-by percentiles (mean/median,
    # p90/p95) do not stack on top of each other (PDF review #98 fix).
    # Rotate to vertical and add a white background box so labels never
    # sit on top of histogram bars.
    y_levels = [0.95, 0.78, 0.95, 0.78, 0.95]
    for (label, x, color, ls), y_frac in zip(pct_marks, y_levels):
        ax2.axvline(x, linestyle=ls, color=color, linewidth=1.2, zorder=3)
        y_top = ax2.get_ylim()[1]
        ax2.text(x, y_top * y_frac, f'{label}={x:.0f}%',
                 rotation=90, va='top', ha='right', fontsize=7, color=color,
                 bbox=dict(facecolor='white', edgecolor='none',
                           alpha=0.85, pad=0.5))
    ax2.set_xlabel('Normalized Navigation Score (NNS, %)', fontsize=10)
    ax2.set_ylabel('Session count', fontsize=10)
    ax2.set_title('Mouse NNS distribution', fontsize=10)
    # Slightly wider x-range so vertical labels at the right edge (max)
    # are not clipped by the spine (PDF review #98 fix).
    ax2.set_xlim(-15, 65)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.grid(axis='y', linestyle='-', linewidth=0.3, alpha=0.4, zorder=0)

    plt.tight_layout()
    out_pdf = os.path.join(OUT_DIR, f'{TS}_peak_mouse_distribution.pdf')
    out_png = os.path.join(OUT_DIR, f'{TS}_peak_mouse_distribution.png')
    plt.savefig(out_pdf, bbox_inches='tight', dpi=150)
    plt.savefig(out_png, bbox_inches='tight', dpi=150)
    print(f'Saved: {out_pdf}')
    print(f'Saved: {out_png}')


if __name__ == '__main__':
    main()
