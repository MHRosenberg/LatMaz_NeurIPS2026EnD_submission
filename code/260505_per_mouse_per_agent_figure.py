"""
260505_per_mouse_per_agent_figure.py

Per-mouse panels: one panel per mouse, one bar per agent (all available
in expanded cohort), color-coded by Fig-3 category. Replaces the old
n=5-mouse pre-rebait-fix Workshop Fig.~2 (paper figures/wkshp_fig2.pdf)
that lived in the now-deleted Appendix U "Outdated Workshop Figures".

Per Author PDF c260505-163800 review (annotation #9 / now #10): "Regenerate
these figures mouse vs simulation figures for all mice all simulations
(one plot per mouse). Include all 20 simulations (recheck that 20
number)."

The "20 simulations" recheck: this script plots up to 22 agent variants
(8 RL from rerun-452 + 14 no-training from 152-sess unified + extras
for a004/a007/a008 from the 260505 fill CSV); the abstract's "20"
estimate is slightly conservative.

Data sources (same as 260505_plot_per_animal_categories.py):
- yoking df: yoked_dfs/c260504-143724_animal_to_agent_yoking_info-chronological.csv
- 152-sess unified: data_released/results/c260419-143924_152sess_14agents_unified.csv
- rerun-452 corrected: data_released/results/c260505-120513_rerun452_corrected_8agents.csv
- a004/a007/a008 no-training fill: data_out/rl_sims/c260505-113331_a004_a007_a008_no_training_agents.csv

Outputs:
  paper/figures/c{ts}_per_mouse_per_agent.{pdf,png}    # multi-panel overview (12 mice)
  reports/figures/c{ts}_per_mouse_per_agent_<aid>.{pdf,png}  # per-mouse stand-alone (12 files)
"""
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

PROJECT_ROOT = Path('<REPO_ROOT>')
PAPER_FIG_DIR = PROJECT_ROOT / 'paper' / 'figures'
REPORT_FIG_DIR = PROJECT_ROOT / 'reports' / 'figures'
PAPER_FIG_DIR.mkdir(parents=True, exist_ok=True)
REPORT_FIG_DIR.mkdir(parents=True, exist_ok=True)
TS = 'c' + datetime.now().strftime('%y%m%d-%H%M%S')

# ---- Fig-3 category mapping (verbatim from the categories overview script) ----
CAT_COLORS = {
    'fixed':      '#888888',
    'online':     '#3b8db8',
    'pretrained': '#2ca25f',
    'privileged': '#a50f15',
}
CAT_ORDER = ['privileged', 'pretrained', 'online', 'fixed']
COLOR_MOUSE = '#1a1a1a'

AGENT_TO_CAT = {
    'POMCP':              'privileged',
    'Greedy_oracle':      'privileged',
    'FSC_bio':            'online',
    'POMCP_bio':          'online',
    'DQN':                'online',
    'DRQN_seq':           'online',
    'DRQN_rand':          'online',
    'QRDQN':              'online',
    'PPO':                'online',
    'A2C':                'online',
    'TRPO':               'online',
    'RecurrentPPO':       'online',
    'OOI':                'online',
    'VarMarkov':          'online',
    'Random':             'fixed',
    'Random_forward':     'fixed',
    'FullFwd':            'fixed',
    'NoBkFullFwd':        'fixed',
    'Clone_ego':          'fixed',
    'Clone_allo_real':    'fixed',
    'Clone_allo_latent':  'fixed',
    'Clone_conditional':  'fixed',
}
AGENT_RENAME = {
    'Mouse-bias_ego': 'Clone_ego',
    'Mouse-bias_allo_real': 'Clone_allo_real',
    'Mouse-bias_allo_latent': 'Clone_allo_latent',
    'ConditionalClone_ego': 'Clone_conditional',
}
# Display order within each category (RPA-descending headlines from canonical-60)
DISPLAY_ORDER = [
    # privileged
    'Greedy_oracle', 'POMCP',
    # online
    'FSC_bio', 'DRQN_seq', 'DQN', 'DRQN_rand', 'QRDQN', 'TRPO',
    'POMCP_bio', 'PPO', 'A2C', 'RecurrentPPO', 'OOI', 'VarMarkov',
    # fixed
    'NoBkFullFwd', 'Random_forward', 'Clone_ego', 'FullFwd',
    'Random', 'Clone_allo_latent', 'Clone_allo_real', 'Clone_conditional',
]
DISPLAY_LABELS = {
    'POMCP': 'POMCP', 'Greedy_oracle': 'Greedy oracle',
    'FSC_bio': 'FSC_bio', 'POMCP_bio': 'Frontier-Plan',
    'DQN': 'DQN', 'DRQN_seq': 'DRQN_seq', 'DRQN_rand': 'DRQN_rand',
    'QRDQN': 'QRDQN', 'PPO': 'PPO', 'A2C': 'A2C', 'TRPO': 'TRPO',
    'RecurrentPPO': 'RecurrentPPO',
    'OOI': 'OOI', 'VarMarkov': 'VarMarkov',
    'Random': 'Random', 'Random_forward': 'Random_forward',
    'FullFwd': 'FullFwd', 'NoBkFullFwd': 'NoBk+FullFwd',
    'Clone_ego': 'Mouse-bias (ego)',
    'Clone_allo_real': 'Mouse-bias (allo-real)',
    'Clone_allo_latent': 'Mouse-bias (allo-latent)',
    'Clone_conditional': 'Mouse-bias (cond.)',
}


# ---- Allocentric exclusion (same as categories overview) ----
def is_allocentric_session(exp_moment, animal_ID):
    if animal_ID == '044':
        return True
    if animal_ID == '033' and isinstance(exp_moment, str) and exp_moment >= '260314':
        return True
    return False


def load_data():
    yk = pd.read_csv(PROJECT_ROOT / 'yoked_dfs' / 'c260504-143724_animal_to_agent_yoking_info-chronological.csv')
    yk['exp_moment'] = yk['csv_data_path'].str.extract(r'(\d{6}-\d{6})')[0]
    yk['animal_ID']  = yk['animal_ID'].astype(str).str.zfill(3)
    yk['n_actions']  = yk['n_states_visited'] - 1
    yk['rpa_mouse']  = yk['n_rewards'] / yk['n_actions']
    yk = yk[~yk.apply(lambda r: is_allocentric_session(r['exp_moment'], r['animal_ID']), axis=1)]
    mouse = yk[['exp_moment', 'animal_ID', 'rpa_mouse']].dropna().drop_duplicates('exp_moment')

    parts = []

    # 152-sess unified (no-training agents)
    unif = pd.read_csv(PROJECT_ROOT / 'data_released' / 'results' / 'c260419-143924_152sess_14agents_unified.csv')
    unif['animal_ID'] = unif['animal_ID'].astype(str).str.zfill(3)
    unif['model'] = unif['model'].replace(AGENT_RENAME)
    unif = unif[unif['model'].isin(AGENT_TO_CAT)][['exp_moment', 'animal_ID', 'model', 'rpa']]
    unif = unif.rename(columns={'rpa': 'rpa_agent'})
    parts.append(unif)

    # rerun-452 corrected (8 RL agents)
    rerun = pd.read_csv(PROJECT_ROOT / 'data_released' / 'results' / 'c260505-120513_rerun452_corrected_8agents.csv')
    rerun['animal_ID'] = rerun['animal_ID'].astype(str).str.zfill(3)
    rerun['model'] = rerun['model'].replace(AGENT_RENAME)
    rerun = rerun[rerun['model'].isin(AGENT_TO_CAT)][['exp_moment', 'animal_ID', 'model', 'rpa']]
    rerun = rerun.rename(columns={'rpa': 'rpa_agent'})
    parts.append(rerun)

    # a004/a007/a008 fill
    extra = pd.read_csv(PROJECT_ROOT / 'data_out' / 'rl_sims' / 'c260505-113331_a004_a007_a008_no_training_agents.csv')
    extra['animal_ID'] = extra['animal_ID'].astype(str).str.zfill(3)
    extra['model'] = extra['model'].replace(AGENT_RENAME)
    extra = extra[extra['model'].isin(AGENT_TO_CAT)][['exp_moment', 'animal_ID', 'model', 'rpa']]
    extra = extra.rename(columns={'rpa': 'rpa_agent'})
    parts.append(extra)

    agents_long = pd.concat(parts, ignore_index=True)
    # Drop allocentric rows
    agents_long = agents_long[
        ~agents_long.apply(lambda r: is_allocentric_session(r['exp_moment'], r['animal_ID']), axis=1)
    ]
    return mouse, agents_long


def per_mouse_per_agent_table(mouse, agents_long):
    """Per-(mouse, agent) mean RPA across that mouse's egocentric sessions."""
    cell = agents_long.groupby(['animal_ID', 'model'])['rpa_agent'].mean().reset_index()
    cell = cell.pivot(index='animal_ID', columns='model', values='rpa_agent')
    mouse_per_animal = mouse.groupby('animal_ID')['rpa_mouse'].mean()
    cell['Mouse'] = mouse_per_animal
    return cell


def plot_overview(cell, out_path):
    animals = [a for a in cell.index if a != '044']  # 12 egocentric mice
    animals = sorted(animals)
    n = len(animals)
    ncols = 4
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4.0, nrows * 2.6),
                             sharex=False, sharey=True)
    axes = np.atleast_2d(axes)
    for i, aid in enumerate(animals):
        ax = axes[i // ncols][i % ncols]
        plot_one_mouse(ax, cell.loc[aid], aid, show_xticks=(i // ncols == nrows - 1))
    # Hide unused subplots
    for j in range(n, nrows * ncols):
        axes[j // ncols][j % ncols].axis('off')
    # Legend
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=CAT_COLORS[c]) for c in CAT_ORDER
    ]
    handles.append(plt.Rectangle((0, 0), 1, 1, color=COLOR_MOUSE))
    labels = [c.capitalize() for c in CAT_ORDER] + ['Mouse']
    fig.legend(handles, labels, loc='lower center', ncol=5, frameon=False,
               bbox_to_anchor=(0.5, -0.005))
    fig.suptitle('Per-mouse RPA: each mouse vs all agent variants (egocentric main benchmark; 12 mice)',
                 fontsize=11, y=1.0)
    fig.tight_layout(rect=[0, 0.02, 1, 0.97])
    fig.savefig(out_path.with_suffix('.pdf'), bbox_inches='tight')
    fig.savefig(out_path.with_suffix('.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Wrote: {out_path.with_suffix(".pdf")}')
    print(f'Wrote: {out_path.with_suffix(".png")}')


def plot_one_mouse(ax, row, aid, show_xticks=True):
    bars = []
    for agent in DISPLAY_ORDER:
        v = row.get(agent, np.nan)
        if pd.isna(v):
            continue
        bars.append((DISPLAY_LABELS[agent], CAT_COLORS[AGENT_TO_CAT[agent]], v))
    # Mouse bar
    bars.append(('Mouse', COLOR_MOUSE, row.get('Mouse', np.nan)))
    xs = np.arange(len(bars))
    vals = [b[2] for b in bars]
    cols = [b[1] for b in bars]
    labels = [b[0] for b in bars]
    ax.bar(xs, vals, color=cols, edgecolor='black', linewidth=0.4)
    ax.set_title(f'a{aid}', fontsize=9, fontweight='bold')
    ax.set_ylim(0, max(0.95, np.nanmax(vals) + 0.05))
    ax.set_ylabel('RPA', fontsize=8)
    ax.tick_params(axis='y', labelsize=7)
    if show_xticks:
        ax.set_xticks(xs)
        ax.set_xticklabels(labels, rotation=85, fontsize=6)
    else:
        ax.set_xticks([])
    ax.grid(axis='y', linestyle=':', alpha=0.4)


def plot_one_mouse_standalone(cell, aid, out_dir):
    fig, ax = plt.subplots(figsize=(8.5, 3.2))
    plot_one_mouse(ax, cell.loc[aid], aid, show_xticks=True)
    ax.set_title(f'a{aid}: per-agent RPA (all available variants)', fontsize=10)
    fig.tight_layout()
    out = out_dir / f'{TS}_per_mouse_per_agent_a{aid}'
    fig.savefig(out.with_suffix('.pdf'), bbox_inches='tight')
    fig.savefig(out.with_suffix('.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)


def main():
    mouse, agents_long = load_data()
    cell = per_mouse_per_agent_table(mouse, agents_long)
    print(f'Cell shape: {cell.shape} (animals x agents+Mouse)')
    print(f'Animals included: {sorted(cell.index.tolist())}')
    # Multi-panel overview into paper/figures
    out = PAPER_FIG_DIR / f'{TS}_per_mouse_per_agent'
    plot_overview(cell, out)
    # Per-mouse standalone PNG/PDF into reports/figures
    for aid in cell.index:
        if aid == '044':
            continue
        plot_one_mouse_standalone(cell, aid, REPORT_FIG_DIR)
    print(f'Wrote per-mouse standalone figures into {REPORT_FIG_DIR}')


if __name__ == '__main__':
    main()
