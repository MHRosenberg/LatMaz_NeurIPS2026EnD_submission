"""
260504_actions_vs_rewards_scatter.py

Per Author's directive 2026-05-04: scatter plot of n_actions (= n_states_visited - 1)
vs n_rewards for all sessions in the yoking dataframe. Two figures:
  (a) all-animals overlay, points colored by chronological position.
  (b) per-animal facet panels, each colored by chronological position
      within that animal's own training arc.

Output:
  paper/figures/c{ts}_actions_vs_rewards_all_animals.{pdf,png}
  paper/figures/c{ts}_actions_vs_rewards_per_animal.{pdf,png}
"""
import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap

PROJECT_ROOT = '<REPO_ROOT>'
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'code'))

from experiment_config import load_yoking_df

OUT_DIR = os.path.join(PROJECT_ROOT, 'paper', 'figures')
TS = 'c' + datetime.now().strftime('%y%m%d-%H%M%S')


def _datetime_from_exp_moment(s):
    """exp_moment is yymmdd-hhmmss → datetime."""
    return datetime.strptime(s, '%y%m%d-%H%M%S') if isinstance(s, str) else None


def main():
    yo = load_yoking_df()
    yo = yo.dropna(subset=['exp_moment', 'n_rewards', 'n_states_visited']).copy()
    yo['n_actions'] = (yo['n_states_visited'] - 1).clip(lower=1)
    yo['datetime'] = yo['exp_moment'].apply(_datetime_from_exp_moment)
    yo = yo.dropna(subset=['datetime']).sort_values('datetime')
    yo['rpa'] = yo['n_rewards'] / yo['n_actions']
    print(f'Total sessions in yoking df: {len(yo)}')
    print(f'Animals: {sorted(yo["animal_ID"].unique())}')

    # ------ (a) all-animals overlay ------
    fig, ax = plt.subplots(figsize=(9, 6))
    n = len(yo)
    chronological_idx = np.arange(n)  # already sorted
    # Animal-colored markers; chronological position via marker SIZE
    animals = sorted(yo['animal_ID'].unique())
    palette = get_cmap('tab10')
    color_for = {a: palette(i % 10) for i, a in enumerate(animals)}
    for a in animals:
        sub = yo[yo['animal_ID'] == a]
        # Color by chronological order using viridis brightness within animal
        order = (sub['datetime'].rank() - 1) / max(1, len(sub) - 1)
        ax.scatter(sub['n_actions'], sub['n_rewards'],
                   c=order, cmap='viridis', vmin=0, vmax=1,
                   s=30, edgecolors=color_for[a], linewidths=1.0, alpha=0.85,
                   label=f'a{a} (n={len(sub)})')
    ax.set_xlabel('n actions per session', fontsize=11)
    ax.set_ylabel('n rewards per session', fontsize=11)
    ax.set_title('Per-session actions vs rewards (all animals; '
                 'color = chronological order within animal, edge = animal)',
                 fontsize=10)

    # Reference lines
    xx = np.array([0, yo['n_actions'].max()])
    for rpa, label, color in [
        (0.178, 'Random floor (RPA=0.18)', 'gray'),
        (0.270, 'Mouse mean (RPA=0.27)', 'black'),
        (0.385, 'Random_forward (RPA=0.39)', '#666'),
        (0.576, 'NoBk+FullFwd (RPA=0.58)', 'darkgray'),
        (0.878, 'Greedy oracle (RPA=0.88)', 'red'),
    ]:
        ax.plot(xx, xx * rpa, linestyle=':', color=color, linewidth=0.8, alpha=0.6)
        ax.text(xx[1] * 0.98, xx[1] * rpa, '  ' + label, va='center',
                ha='right', fontsize=7, color=color)
    ax.legend(loc='upper left', fontsize=8, frameon=True, ncol=2)
    ax.set_xlim(0, None)
    ax.set_ylim(0, None)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    out_a = os.path.join(OUT_DIR, f'{TS}_actions_vs_rewards_all_animals')
    plt.savefig(out_a + '.pdf', dpi=150, bbox_inches='tight')
    plt.savefig(out_a + '.png', dpi=150, bbox_inches='tight')
    print(f'Saved: {out_a}.pdf')
    plt.close()

    # ------ (b) per-animal facets ------
    n_animals = len(animals)
    ncols = min(4, n_animals)
    nrows = (n_animals + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(3 * ncols, 2.8 * nrows),
                              sharex=False, sharey=False)
    axes = np.atleast_1d(axes).flatten()
    for i, a in enumerate(animals):
        ax = axes[i]
        sub = yo[yo['animal_ID'] == a].sort_values('datetime')
        order = (sub['datetime'].rank() - 1) / max(1, len(sub) - 1)
        sc = ax.scatter(sub['n_actions'], sub['n_rewards'],
                        c=order, cmap='viridis', vmin=0, vmax=1,
                        s=30, edgecolors='black', linewidths=0.4, alpha=0.9)
        ax.set_title(f'a{a} (n={len(sub)})', fontsize=9)
        # Reference RPA lines for this panel
        xmax = max(50, sub['n_actions'].max() * 1.05)
        xx = np.array([0, xmax])
        for rpa, color in [(0.178, 'gray'), (0.270, 'black'),
                            (0.576, 'darkgray'), (0.878, 'red')]:
            ax.plot(xx, xx * rpa, linestyle=':', color=color, linewidth=0.6, alpha=0.5)
        ax.set_xlim(0, xmax)
        ymax = max(10, sub['n_rewards'].max() * 1.05)
        ax.set_ylim(0, ymax)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(labelsize=8)
        if i % ncols == 0:
            ax.set_ylabel('n rewards', fontsize=9)
        if i // ncols == nrows - 1:
            ax.set_xlabel('n actions', fontsize=9)

    # Hide unused panels
    for j in range(n_animals, len(axes)):
        axes[j].set_visible(False)

    # Colorbar
    cbar_ax = fig.add_axes([0.93, 0.15, 0.015, 0.7])
    cbar = fig.colorbar(sc, cax=cbar_ax)
    cbar.set_label('chronological order (within animal)', fontsize=8)
    cbar.ax.tick_params(labelsize=7)

    fig.suptitle('Per-animal actions vs rewards (color = chronological order within animal)',
                 fontsize=10, y=0.99)
    plt.subplots_adjust(left=0.06, right=0.91, top=0.92, bottom=0.10,
                         hspace=0.4, wspace=0.3)
    out_b = os.path.join(OUT_DIR, f'{TS}_actions_vs_rewards_per_animal')
    plt.savefig(out_b + '.pdf', dpi=150, bbox_inches='tight')
    plt.savefig(out_b + '.png', dpi=150, bbox_inches='tight')
    print(f'Saved: {out_b}.pdf')


if __name__ == '__main__':
    main()
