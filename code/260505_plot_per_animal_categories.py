"""
260505_plot_per_animal_categories.py

Per-animal "Mouse vs each Fig-3 category of agents" overview plot.
Per Author 2026-05-05: replace the existing 'all mice vs all simulations'
overview with a Fig-3-aligned 4-category breakdown (privileged,
pretrained, online learning, fixed policy) plus mouse.

Data sources:
- yoking df (mouse RPAs per session)
- data_released/results/c260419-143924_152sess_14agents_unified.csv
  (no-training/cloning/heuristic/POMCP/FSC_bio agents on 152 sessions)
- data_released/results/c260505-120513_rerun452_corrected_8agents.csv
  (8 RL agents on 452 sessions; rerun-452 corrected, supersedes the buggy
   c260504-143902 version archived in z_legacy_outdated_reference/)

Outputs in reports/figures/:
  c{ts}_per_animal_categories_overview.{pdf,png}
  c{ts}_per_animal_categories_<aid>_sorted.{pdf,png} for each animal
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
FIG_DIR = PROJECT_ROOT / 'reports' / 'figures'
FIG_DIR.mkdir(parents=True, exist_ok=True)
TS = 'c' + datetime.now().strftime('%y%m%d-%H%M%S')

# ----------------------------------------------------------------------
# Fig 3 category mapping (verbatim from 260504_fig3_performance_hierarchy_NNS.py)
# ----------------------------------------------------------------------
CAT_COLORS = {
    'fixed':      '#888888',
    'online':     '#3b8db8',
    'pretrained': '#2ca25f',
    'privileged': '#a50f15',
}
CAT_LABELS = {
    'privileged': 'Privileged',
    'pretrained': 'Pretrained',
    'online':     'Online learning',
    'fixed':      'Fixed policy',
}
CAT_ORDER = ['privileged', 'pretrained', 'online', 'fixed']
COLOR_MOUSE = '#1a1a1a'

AGENT_TO_CAT = {
    # Privileged
    'POMCP':              'privileged',
    'POMCP/fewer_sims':   'privileged',
    'Greedy_oracle':      'privileged',
    # Online (no-training online: FSC_bio, Frontier-Plan)
    'FSC_bio':            'online',
    'POMCP_bio':          'online',
    # Online (model-free RL — from rerun-452)
    'DQN':                'online',
    'DRQN_seq':           'online',
    'DRQN_rand':          'online',
    'QRDQN':              'online',
    'PPO':                'online',
    'A2C':                'online',
    'TRPO':               'online',
    'RecurrentPPO':       'online',
    # Fixed
    'Random':             'fixed',
    'Random_forward':     'fixed',
    'FullFwd':            'fixed',
    'NoBkFullFwd':        'fixed',
    'Clone_ego':          'fixed',
    'Clone_allo_real':    'fixed',
    'Clone_allo_latent':  'fixed',
}
# Note: 'pretrained' category cannot be filled in this run because
# unified152 does not include DQN_pretrained / DRQN_*_pretrained per-animal
# (those CSVs cover only canonical-60). Plot will show empty pretrained
# bars per animal and a note in the caption.

# ----------------------------------------------------------------------
# Load data
# ----------------------------------------------------------------------

def is_allocentric_session(exp_moment, animal_ID):
    """Allocentric subset (set aside, paper §3 / §6):
    - a044 entirely (apparatus redesigned 2026-04-02 onward)
    - a033 from 2026-03-14 onward (last 5 sessions only)
    """
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
    # Exclude allocentric sessions — those go into separate appendix coverage / future work
    yk = yk[~yk.apply(lambda r: is_allocentric_session(r['exp_moment'], r['animal_ID']), axis=1)]
    mouse = yk[['exp_moment', 'animal_ID', 'rpa_mouse']].dropna().drop_duplicates('exp_moment')

    unif = pd.read_csv(PROJECT_ROOT / 'data_released' / 'results' / 'c260419-143924_152sess_14agents_unified.csv')
    unif['animal_ID'] = unif['animal_ID'].astype(str).str.zfill(3)
    # Map Mouse-bias_* (rerun + new fills naming) ↔ Clone_* (unified152 naming) so they share one cat label
    AGENT_RENAME = {
        'Mouse-bias_ego': 'Clone_ego',
        'Mouse-bias_allo_real': 'Clone_allo_real',
        'Mouse-bias_allo_latent': 'Clone_allo_latent',
    }
    unif = unif[unif['model'].isin(AGENT_TO_CAT)]
    unif = unif.rename(columns={'rpa': 'rpa_agent'})

    # Fill in a004/a007/a008/a044 no-training agents (single CSV after env fix)
    extra_csvs = [
        PROJECT_ROOT / 'data_out' / 'rl_sims' / 'c260505-113331_a004_a007_a008_no_training_agents.csv',
    ]
    extras = []
    for csv in extra_csvs:
        if csv.exists():
            ex = pd.read_csv(csv)
            ex['animal_ID'] = ex['animal_ID'].astype(str).str.zfill(3)
            ex['model'] = ex['model'].replace(AGENT_RENAME)
            ex = ex[ex['model'].isin(AGENT_TO_CAT)]
            ex = ex.rename(columns={'rpa': 'rpa_agent'})
            extras.append(ex[['exp_moment', 'animal_ID', 'model', 'rpa_agent']])
    if extras:
        unif = pd.concat([unif[['exp_moment', 'animal_ID', 'model', 'rpa_agent']]] + extras, ignore_index=True)

    rerun = pd.read_csv(PROJECT_ROOT / 'data_released' / 'results' / 'c260505-120513_rerun452_corrected_8agents.csv')
    rerun['animal_ID'] = rerun['animal_ID'].astype(str).str.zfill(3)
    rerun = rerun[rerun['model'].isin(AGENT_TO_CAT)]
    rerun = rerun[['exp_moment', 'animal_ID', 'model', 'seed', 'rpa']]
    rerun = rerun.rename(columns={'rpa': 'rpa_agent'})

    agents = pd.concat([
        unif[['exp_moment', 'animal_ID', 'model', 'rpa_agent']],
        rerun[['exp_moment', 'animal_ID', 'model', 'rpa_agent']],
    ], ignore_index=True)
    # Drop allocentric sessions from agent rows too (a044 entirely + a033 post-260314)
    allo_mask = agents.apply(lambda r: is_allocentric_session(r['exp_moment'], r['animal_ID']), axis=1)
    agents = agents[~allo_mask]
    agents['cat'] = agents['model'].map(AGENT_TO_CAT)

    return mouse, agents


def per_animal_category_means(mouse_df, agents_df):
    """For each (animal, category): mean RPA over (sessions × seeds × agents in cat).
    For mouse: mean RPA over sessions per animal."""
    # Mouse
    m_per_animal = mouse_df.groupby('animal_ID')['rpa_mouse'].agg(['mean', 'std', 'count']).reset_index()
    m_per_animal.columns = ['animal_ID', 'mouse_mean', 'mouse_sd', 'mouse_n']

    # Agent: per-(session, agent) mean (collapse seeds), then per-(animal, cat) mean
    sess_agent = agents_df.groupby(['animal_ID', 'exp_moment', 'cat', 'model'])['rpa_agent'].mean().reset_index()
    cat_per_animal = sess_agent.groupby(['animal_ID', 'cat'])['rpa_agent'].agg(['mean', 'std', 'count']).reset_index()
    return m_per_animal, cat_per_animal


def plot_overview(mouse_per_animal, cat_per_animal):
    # Drop animals with no agent data at all (a000, a999 have only mouse rows in yoking df)
    animals = sorted(cat_per_animal['animal_ID'].unique())
    # Drop categories that are entirely empty (e.g., pretrained — no per-animal pretrained runs available)
    cats_present = [c for c in CAT_ORDER if not cat_per_animal[cat_per_animal['cat'] == c]['mean'].isna().all()]
    n = len(animals)
    n_cats = len(cats_present)
    n_bars = 1 + n_cats  # Mouse + categories
    width = 0.85 / n_bars  # leave 15% group gap
    x = np.arange(n)
    # Center the bar group around each x: offsets span [-(n_bars-1)/2, +(n_bars-1)/2] * width
    bar_offsets = np.arange(n_bars) - (n_bars - 1) / 2.0

    fig, ax = plt.subplots(figsize=(max(9, 1.05 * n), 5.0))

    def bar_for(cat, idx):
        means = []
        for a in animals:
            if cat == 'mouse':
                row = mouse_per_animal[mouse_per_animal['animal_ID'] == a]
                v = row['mouse_mean'].iloc[0] if len(row) else np.nan
            else:
                row = cat_per_animal[(cat_per_animal['animal_ID'] == a) & (cat_per_animal['cat'] == cat)]
                v = row['mean'].iloc[0] if len(row) else np.nan
            means.append(float(v) if not pd.isna(v) else 0.0)  # NaN → 0-height (visually gap)
        color = COLOR_MOUSE if cat == 'mouse' else CAT_COLORS[cat]
        label = 'Mouse' if cat == 'mouse' else CAT_LABELS[cat]
        return ax.bar(x + bar_offsets[idx] * width, means, width=width, color=color,
                      label=label, edgecolor='black', linewidth=0.4)

    bar_for('mouse', 0)
    for i, cat in enumerate(cats_present):
        bar_for(cat, i + 1)

    ax.set_xticks(x)
    ax.set_xticklabels([f'a{a}' for a in animals], fontsize=10)
    ax.set_ylabel('Reward per Action (RPA)', fontsize=11)
    ax.set_title('Per-animal: Mouse vs Fig 3 agent categories', fontsize=12)
    ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.20), ncol=n_bars, fontsize=9, frameon=False)
    ax.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.5)
    ax.set_ylim(0, max(0.9, ax.get_ylim()[1]))
    plt.tight_layout()
    out_pdf = FIG_DIR / f'{TS}_per_animal_categories_overview.pdf'
    out_png = FIG_DIR / f'{TS}_per_animal_categories_overview.png'
    fig.savefig(out_pdf, bbox_inches='tight', dpi=180)
    fig.savefig(out_png, bbox_inches='tight', dpi=180)
    plt.close(fig)
    print(f'Saved overview: {out_pdf.name}')


def plot_per_animal_sorted(mouse_per_animal, agents_df):
    """For each animal, a horizontal bar chart sorted by RPA descending,
    colored by Fig 3 category. Mouse shown as a special black bar."""
    animals = sorted(agents_df['animal_ID'].unique())
    for aid in animals:
        sub = agents_df[agents_df['animal_ID'] == aid]
        if sub.empty:
            continue
        per_agent = sub.groupby(['model', 'cat'])['rpa_agent'].agg(['mean', 'std', 'count']).reset_index()
        per_agent = per_agent.sort_values('mean', ascending=True)  # ascending so largest at top in barh
        mouse_row = mouse_per_animal[mouse_per_animal['animal_ID'] == aid]
        mouse_mean = mouse_row['mouse_mean'].squeeze() if not mouse_row.empty else np.nan

        fig, ax = plt.subplots(figsize=(7, max(3, 0.30 * len(per_agent) + 1.5)))
        ys = np.arange(len(per_agent))
        colors = [CAT_COLORS[c] for c in per_agent['cat']]
        ax.barh(ys, per_agent['mean'], color=colors, edgecolor='black', linewidth=0.5)
        ax.set_yticks(ys)
        ax.set_yticklabels(per_agent['model'], fontsize=9)
        ax.set_xlabel('RPA', fontsize=11)
        ax.set_title(f'Animal a{aid}: per-agent RPA, coloured by Fig 3 category', fontsize=11)
        if not np.isnan(mouse_mean):
            ax.axvline(mouse_mean, linestyle='--', color=COLOR_MOUSE, linewidth=1.5, label=f'Mouse mean ({mouse_mean:.2f})')
            ax.legend(loc='lower right', fontsize=9, frameon=False)
        ax.grid(axis='x', alpha=0.3, linestyle='--', linewidth=0.5)
        plt.tight_layout()
        out_pdf = FIG_DIR / f'{TS}_per_animal_categories_a{aid}_sorted.pdf'
        out_png = FIG_DIR / f'{TS}_per_animal_categories_a{aid}_sorted.png'
        fig.savefig(out_pdf, bbox_inches='tight', dpi=180)
        fig.savefig(out_png, bbox_inches='tight', dpi=180)
        plt.close(fig)
    print(f'Saved per-animal sorted figs for {len(animals)} animals')


def main():
    print(f'Timestamp: {TS}')
    mouse, agents = load_data()
    print(f'Mouse rows: {len(mouse)}, animals: {sorted(mouse.animal_ID.unique())}')
    print(f'Agents rows: {len(agents)}, animals: {sorted(agents.animal_ID.unique())}')
    print(f'Agent models present: {sorted(agents.model.unique())}')

    m_per_animal, cat_per_animal = per_animal_category_means(mouse, agents)
    print()
    print('=== per-animal category means ===')
    pivot = cat_per_animal.pivot(index='animal_ID', columns='cat', values='mean')
    pivot = pivot.reindex(columns=CAT_ORDER)
    print(pivot.round(3).to_string())

    plot_overview(m_per_animal, cat_per_animal)
    plot_per_animal_sorted(m_per_animal, agents)


if __name__ == '__main__':
    main()
