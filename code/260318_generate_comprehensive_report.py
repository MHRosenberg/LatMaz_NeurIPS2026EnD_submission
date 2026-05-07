#!/usr/bin/env python3
"""Generate comprehensive benchmark report with fully corrected rebait data.

Outputs:
  reports/<ts>_comprehensive_benchmark.pdf  — Multi-page PDF with figures
  reports/<ts>_comprehensive_benchmark.md   — Markdown summary
  reports/<ts>_comprehensive_benchmark.html — Standalone HTML with embedded plots

Data sources (all with corrected rebait, 260315+ code):
  - 260316-171035_*_FIXED.csv  (master rerun: HPO, pretrain, POMCP, scaling, etc.)
  - 260315-*_pomcp_c1_full.csv (POMCP c=1.0)
  - 260315-*_recurrent_sac_d32_benchmark.csv (RecurrentSAC)
  - 260217_baseline_heuristics.csv (baselines, rerun 260318)
  - 260211_mouse_rewards_filtered41.csv (mouse performance)

Usage:
    conda activate latMaz_RL
    python 260318_generate_comprehensive_report.py
"""
import base64
import io
import os
import sys
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

from datetime import datetime

import numpy as np
import pandas as pd
from scipy import stats

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.patches as mpatches

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.normpath(os.path.join(SCRIPT_DIR, '..', '..'))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'code', 'code/'))
sys.path.insert(0, SCRIPT_DIR)

DATA_DIR = os.path.join(PROJECT_ROOT, 'data_out', 'rl_sims')
REPORT_DIR = os.path.join(PROJECT_ROOT, 'reports')
FIGURE_DIR = os.path.join(REPORT_DIR, 'figures')
os.makedirs(REPORT_DIR, exist_ok=True)
os.makedirs(FIGURE_DIR, exist_ok=True)

TS = datetime.now().strftime('%y%m%d-%H%M%S')
DATE_STR = datetime.now().strftime('%Y-%m-%d')

from experiment_config import load_data
from utils_latMaz import get_most_recent_file

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
yoking_df, rwd_df, sessions = load_data()
N_SESSIONS = len(sessions)
print(f"Filtered sessions: {N_SESSIONS}")

# Master rerun (260316-171035, c-prefixed as of 260327 rename convention)
MASTER_TS = 'c260316-171035'
hpo_df = pd.read_csv(os.path.join(DATA_DIR, f'{MASTER_TS}_hpo_tuning_FIXED.csv'))
pretrain_df = pd.read_csv(os.path.join(DATA_DIR, f'{MASTER_TS}_pretrain_FIXED.csv'))
pomcp_oracle_df = pd.read_csv(os.path.join(DATA_DIR, f'{MASTER_TS}_pomcp_full41_FIXED.csv'))
pomcp_bio_df = pd.read_csv(os.path.join(DATA_DIR, f'{MASTER_TS}_pomcp_bio_FIXED.csv'))
target_pretrain_df = pd.read_csv(os.path.join(DATA_DIR, f'{MASTER_TS}_target_session_pretrain_FIXED.csv'))
cross_maze_df = pd.read_csv(os.path.join(DATA_DIR, f'{MASTER_TS}_cross_maze_pretrain_FIXED.csv'))
scaling_df = pd.read_csv(os.path.join(DATA_DIR, f'{MASTER_TS}_maze_scaling_FIXED.csv'))

# Belief oracle (c260327 rerun with rebait fix)
belief_oracle_df_path = get_most_recent_file(os.path.join(DATA_DIR, 'c*_online_belief_bfs.csv'))
if belief_oracle_df_path is None:
    belief_oracle_df_path = os.path.join(DATA_DIR, 'c260316-162337_belief_oracle_FIXED.csv')
belief_oracle_df = pd.read_csv(belief_oracle_df_path)

# RecurrentSAC + POMCP c=1.0 (c-prefixed)
rsac_csv = get_most_recent_file(os.path.join(DATA_DIR, 'c*_recurrent_sac_d32_benchmark.csv'))
if rsac_csv is None:
    rsac_csv = get_most_recent_file(os.path.join(DATA_DIR, 'c260315*_recurrent_sac_d32_benchmark.csv'))
pomcp_c1_csv = get_most_recent_file(os.path.join(DATA_DIR, 'c*_pomcp_c1_full.csv'))
if pomcp_c1_csv is None:
    pomcp_c1_csv = get_most_recent_file(os.path.join(DATA_DIR, 'c260315*_pomcp_c1_full.csv'))
rsac_df = pd.read_csv(rsac_csv)
pomcp_c1_df = pd.read_csv(pomcp_c1_csv)

# Baselines (c-prefixed rerun 260318)
baseline_df = pd.read_csv(os.path.join(DATA_DIR, 'c260318-144507_baseline_heuristics.csv'))

# Mouse
mouse_df = pd.read_csv(os.path.join(DATA_DIR, 'c260211-130712_mouse_rewards_filtered41.csv'))
MOUSE_RPA = mouse_df['mouse_rpa'].mean()

# MOP + intermediate agents (try loading if available)
mop_csv = get_most_recent_file(os.path.join(DATA_DIR, 'c*_mop_reward_weight_sweep.csv'))
inter_csv = get_most_recent_file(os.path.join(DATA_DIR, 'c*_intermediate_agents.csv'))
HAS_MOP = mop_csv is not None
HAS_INTERMEDIATE = inter_csv is not None
if HAS_MOP:
    mop_df = pd.read_csv(mop_csv)
    print(f"MOP sweep: {mop_csv}")
if HAS_INTERMEDIATE:
    inter_df = pd.read_csv(inter_csv)
    print(f"Intermediate agents: {inter_csv}")

# Filter all to the canonical sessions
def filter_df(df, sessions=sessions):
    if 'exp_moment' in df.columns:
        return df[df['exp_moment'].isin(sessions)]
    return df

hpo_df = filter_df(hpo_df)
pretrain_df = filter_df(pretrain_df)
pomcp_oracle_df = filter_df(pomcp_oracle_df)
pomcp_bio_df = filter_df(pomcp_bio_df)
target_pretrain_df = filter_df(target_pretrain_df)
cross_maze_df = filter_df(cross_maze_df)
belief_oracle_df = filter_df(belief_oracle_df)
rsac_df = filter_df(rsac_df)
pomcp_c1_df = filter_df(pomcp_c1_df)
baseline_df = filter_df(baseline_df)

# ---------------------------------------------------------------------------
# Reference values
# ---------------------------------------------------------------------------
random_rpa = baseline_df[baseline_df['model'] == 'Random']['total_reward'].sum() / \
             baseline_df[baseline_df['model'] == 'Random']['n_actions'].sum()
fwd_random_rpa = baseline_df[baseline_df['model'] == 'Random_forward']['total_reward'].sum() / \
                 baseline_df[baseline_df['model'] == 'Random_forward']['n_actions'].sum()
greedy_rpa = baseline_df[baseline_df['model'] == 'Greedy_oracle']['total_reward'].sum() / \
             baseline_df[baseline_df['model'] == 'Greedy_oracle']['n_actions'].sum()

print(f"\nReference values:")
print(f"  Random:         {random_rpa:.4f}")
print(f"  Forward_random: {fwd_random_rpa:.4f}")
print(f"  Greedy_oracle:  {greedy_rpa:.4f}")
print(f"  Mouse:          {MOUSE_RPA:.4f}")

def pct_ceiling(rpa):
    return 100.0 * (rpa - random_rpa) / (greedy_rpa - random_rpa)

# ---------------------------------------------------------------------------
# Build agent results table
# ---------------------------------------------------------------------------
def compute_rpa(df, model_col='model', model_name=None):
    if model_name:
        df = df[df[model_col] == model_name]
    return df['total_reward'].sum() / df['n_actions'].sum()

def compute_rpa_std(df, model_col='model', model_name=None):
    if model_name:
        df = df[df[model_col] == model_name]
    session_rpas = df.groupby('exp_moment').apply(
        lambda g: g['total_reward'].sum() / g['n_actions'].sum())
    return session_rpas.std()

# HPO-best per model
hpo_best = {}
for model in hpo_df['model'].unique():
    mdf = hpo_df[hpo_df['model'] == model]
    best_cfg, best_rpa = '', 0
    for cfg in mdf['hpo_config'].unique():
        sub = mdf[mdf['hpo_config'] == cfg]
        rpa = sub['total_reward'].sum() / sub['n_actions'].sum()
        if rpa > best_rpa:
            best_rpa, best_cfg = rpa, cfg
    hpo_best[model] = (best_cfg, best_rpa)

# Pretrain results
pretrain_best = {}
for model in pretrain_df['model'].unique():
    for pt in pretrain_df['pretrain_type'].unique():
        sub = pretrain_df[(pretrain_df['model'] == model) & (pretrain_df['pretrain_type'] == pt)]
        if len(sub) > 0:
            rpa = sub['total_reward'].sum() / sub['n_actions'].sum()
            pretrain_best[f"{model}_{pt}"] = rpa

# Build master table
agents = []
agents.append(('Greedy_oracle', greedy_rpa, 'Heuristic (oracle)', 'Oracle'))
agents.append(('POMCP c=20 (fewer_sims)', hpo_best.get('POMCP', ('', 0))[1], 'Planning (oracle)', 'Planning'))
agents.append(('RecurrentSAC D=32', compute_rpa(rsac_df), 'Meta-RL (GRU+SAC)', 'Meta-RL'))
agents.append(('POMCP c=20 (full41)', compute_rpa(pomcp_oracle_df), 'Planning (oracle)', 'Planning'))
agents.append(('POMCP c=1.0', compute_rpa(pomcp_c1_df), 'Planning (oracle)', 'Planning'))
agents.append(('DQN pretrained', pretrain_best.get('DQN_prior_sessions_same_maze', 0), 'Value + pretrain', 'Pretrained'))
agents.append(('DRQN_seq pretrained', pretrain_best.get('DRQN_seq_prior_sessions_same_maze', 0), 'Recurrent + pretrain', 'Pretrained'))
agents.append(('DRQN_rand pretrained', pretrain_best.get('DRQN_rand_prior_sessions_same_maze', 0), 'Recurrent + pretrain', 'Pretrained'))
agents.append(('Random_forward', fwd_random_rpa, 'Heuristic', 'Heuristic'))
agents.append(('DRQN_seq (HPO-best)', hpo_best.get('DRQN_seq', ('', 0))[1], 'Recurrent', 'Model-free'))
agents.append(('DQN (HPO-best)', hpo_best.get('DQN', ('', 0))[1], 'Value-based', 'Model-free'))
agents.append(('DRQN_rand (HPO-best)', hpo_best.get('DRQN_rand', ('', 0))[1], 'Recurrent', 'Model-free'))
agents.append(('Mouse', MOUSE_RPA, 'Biological', 'Biological'))
agents.append(('QRDQN (HPO-best)', hpo_best.get('QRDQN', ('', 0))[1], 'Value-based', 'Model-free'))
agents.append(('TRPO (HPO-best)', hpo_best.get('TRPO', ('', 0))[1], 'Policy-gradient', 'Model-free'))
agents.append(('BeliefOracle', compute_rpa(belief_oracle_df), 'Belief state (oracle)', 'Planning'))
agents.append(('POMCP_bio', compute_rpa(pomcp_bio_df), 'Planning (bio)', 'Planning'))
agents.append(('PPO (HPO-best)', hpo_best.get('PPO', ('', 0))[1], 'Policy-gradient', 'Model-free'))
agents.append(('A2C (HPO-best)', hpo_best.get('A2C', ('', 0))[1], 'Policy-gradient', 'Model-free'))
agents.append(('Random', random_rpa, 'Heuristic', 'Heuristic'))
agents.append(('RecurrentPPO (HPO-best)', hpo_best.get('RecurrentPPO', ('', 0))[1], 'Recurrent', 'Model-free'))

# Add MOP if available
if HAS_MOP:
    mop_f = filter_df(mop_df)
    for model in sorted(mop_f['model'].unique()):
        rpa = compute_rpa(mop_f, model_name=model)
        agents.append((model, rpa, 'MOP', 'MOP'))

# Add intermediate agents if available
if HAS_INTERMEDIATE:
    inter_f = filter_df(inter_df)
    for model in sorted(inter_f['model'].unique()):
        rpa = compute_rpa(inter_f, model_name=model)
        agents.append((model, rpa, 'Intermediate', 'Intermediate'))

# Sort by RPA descending
agents.sort(key=lambda x: -x[1])

# ---------------------------------------------------------------------------
# Colors by family
# ---------------------------------------------------------------------------
FAMILY_COLORS = {
    'Oracle': '#2ca02c',
    'Planning': '#1f77b4',
    'Meta-RL': '#9467bd',
    'Pretrained': '#17becf',
    'Model-free': '#ff7f0e',
    'Heuristic': '#8c564b',
    'Biological': '#e377c2',
    'MOP': '#d62728',
    'Intermediate': '#7f7f7f',
}

# ---------------------------------------------------------------------------
# Figure 1: Full Agent Landscape (horizontal bar chart)
# ---------------------------------------------------------------------------
def fig_full_landscape():
    fig, ax = plt.subplots(figsize=(14, max(8, len(agents) * 0.4)))
    names = [a[0] for a in agents]
    rpas = [a[1] for a in agents]
    families = [a[3] for a in agents]
    colors = [FAMILY_COLORS.get(f, '#999') for f in families]

    bars = ax.barh(range(len(names)), rpas, color=colors,
                   edgecolor='black', linewidth=0.5, height=0.7)

    for i, (rpa, name) in enumerate(zip(rpas, names)):
        pct = pct_ceiling(rpa)
        ax.text(rpa + 0.008, i, f'{rpa:.3f} ({pct:.0f}%)',
                va='center', fontsize=8)

    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel('Reward Per Action (RPA)', fontsize=12)
    ax.set_title(f'Full Agent Benchmark — {N_SESSIONS} Sessions, Corrected Rebait\n'
                 f'Random={random_rpa:.3f}, Mouse={MOUSE_RPA:.3f}, Greedy={greedy_rpa:.3f}',
                 fontsize=13, fontweight='bold')
    ax.invert_yaxis()
    ax.set_xlim(0, 1.1)

    ax.axvline(random_rpa, color='gray', linestyle=':', alpha=0.5, label=f'Random ({random_rpa:.3f})')
    ax.axvline(MOUSE_RPA, color='#e377c2', linestyle='--', alpha=0.5, label=f'Mouse ({MOUSE_RPA:.3f})')
    ax.axvline(fwd_random_rpa, color='brown', linestyle=':', alpha=0.5, label=f'Fwd-random ({fwd_random_rpa:.3f})')

    handles = [mpatches.Patch(color=c, label=f) for f, c in FAMILY_COLORS.items()
               if any(a[3] == f for a in agents)]
    ax.legend(handles=handles, loc='lower right', fontsize=8)
    fig.tight_layout()
    return fig

# ---------------------------------------------------------------------------
# Figure 2: Box plots per agent (session-level RPA distributions)
# ---------------------------------------------------------------------------
def fig_rpa_distributions():
    # Compute session-mean RPA for key agents
    agent_data = {}

    # Baselines
    for model in ['Random', 'Random_forward', 'Greedy_oracle']:
        sub = baseline_df[baseline_df['model'] == model]
        session_rpas = sub.groupby('exp_moment').apply(
            lambda g: g['total_reward'].sum() / g['n_actions'].sum())
        agent_data[model] = session_rpas.values

    # HPO-best for key agents
    for model in ['DQN', 'DRQN_seq', 'A2C', 'PPO']:
        cfg = hpo_best[model][0]
        sub = hpo_df[(hpo_df['model'] == model) & (hpo_df['hpo_config'] == cfg)]
        session_rpas = sub.groupby('exp_moment').apply(
            lambda g: g['total_reward'].sum() / g['n_actions'].sum())
        agent_data[f'{model} (HPO)'] = session_rpas.values

    # POMCP
    session_rpas = pomcp_oracle_df.groupby('exp_moment').apply(
        lambda g: g['total_reward'].sum() / g['n_actions'].sum())
    agent_data['POMCP c=20'] = session_rpas.values

    session_rpas = pomcp_c1_df.groupby('exp_moment').apply(
        lambda g: g['total_reward'].sum() / g['n_actions'].sum())
    agent_data['POMCP c=1.0'] = session_rpas.values

    # RecurrentSAC
    session_rpas = rsac_df.groupby('exp_moment').apply(
        lambda g: g['total_reward'].sum() / g['n_actions'].sum())
    agent_data['RecurrentSAC'] = session_rpas.values

    # Pretrained DQN
    sub = pretrain_df[(pretrain_df['model'] == 'DQN') &
                      (pretrain_df['pretrain_type'] == 'prior_sessions_same_maze')]
    if len(sub) > 0:
        session_rpas = sub.groupby('exp_moment').apply(
            lambda g: g['total_reward'].sum() / g['n_actions'].sum())
        agent_data['DQN pretrained'] = session_rpas.values

    # Order by median
    ordered = sorted(agent_data.items(), key=lambda x: np.median(x[1]), reverse=True)
    labels = [k for k, v in ordered]
    data = [v for k, v in ordered]

    fig, ax = plt.subplots(figsize=(14, 7))
    bp = ax.boxplot(data, vert=True, patch_artist=True, labels=labels,
                    widths=0.6, showfliers=True, flierprops={'markersize': 3})
    for patch in bp['boxes']:
        patch.set_facecolor('#4a86c8')
        patch.set_alpha(0.6)

    ax.axhline(random_rpa, color='gray', linestyle=':', alpha=0.7, label=f'Random ({random_rpa:.3f})')
    ax.axhline(fwd_random_rpa, color='brown', linestyle=':', alpha=0.7, label=f'Fwd-random ({fwd_random_rpa:.3f})')
    ax.axhline(MOUSE_RPA, color='#e377c2', linestyle='--', alpha=0.7, label=f'Mouse ({MOUSE_RPA:.3f})')

    ax.set_ylabel('RPA (session mean)', fontsize=12)
    ax.set_title(f'RPA Distribution by Agent ({N_SESSIONS} Sessions)', fontsize=13, fontweight='bold')
    ax.legend(loc='upper right', fontsize=9)
    ax.tick_params(axis='x', rotation=30)
    fig.tight_layout()
    return fig

# ---------------------------------------------------------------------------
# Figure 3: Target-session pretraining dose-response
# ---------------------------------------------------------------------------
def fig_target_pretrain():
    fig, ax = plt.subplots(figsize=(10, 6))
    colors_agent = {'DQN': '#1f77b4', 'DRQN_seq': '#ff7f0e', 'DRQN_rand': '#2ca02c'}

    for model in ['DQN', 'DRQN_seq', 'DRQN_rand']:
        sub = target_pretrain_df[target_pretrain_df['model'] == model]
        grouped = sub.groupby('n_pretrain_episodes').apply(
            lambda g: g['total_reward'].sum() / g['n_actions'].sum()).reset_index()
        grouped.columns = ['n_pretrain_episodes', 'rpa']
        grouped = grouped.sort_values('n_pretrain_episodes')
        ax.plot(grouped['n_pretrain_episodes'], grouped['rpa'],
                '-o', label=model, color=colors_agent.get(model, '#999'),
                markersize=6, linewidth=2)

    ax.axhline(random_rpa, color='gray', linestyle=':', alpha=0.7, label=f'Random ({random_rpa:.3f})')
    ax.axhline(fwd_random_rpa, color='brown', linestyle=':', alpha=0.7, label=f'Fwd-random ({fwd_random_rpa:.3f})')
    ax.axhline(MOUSE_RPA, color='#e377c2', linestyle='--', alpha=0.7, label=f'Mouse ({MOUSE_RPA:.3f})')

    ax.set_xlabel('Pretrain Episodes (same session)', fontsize=12)
    ax.set_ylabel('Mean RPA', fontsize=12)
    ax.set_title('Target-Session Pretraining: Dose-Response', fontsize=13, fontweight='bold')
    ax.legend(loc='lower right', fontsize=9)
    ax.set_xlim(-0.5, 21)
    fig.tight_layout()
    return fig

# ---------------------------------------------------------------------------
# Figure 4: HPO config comparison per model
# ---------------------------------------------------------------------------
def fig_hpo_comparison():
    models = ['DQN', 'DRQN_seq', 'DRQN_rand', 'A2C', 'PPO', 'QRDQN', 'TRPO', 'RecurrentPPO']
    models = [m for m in models if m in hpo_df['model'].unique()]

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(models))
    width = 0.15
    configs_all = sorted(hpo_df['hpo_config'].unique())

    for i, cfg in enumerate(configs_all):
        rpas = []
        for model in models:
            sub = hpo_df[(hpo_df['model'] == model) & (hpo_df['hpo_config'] == cfg)]
            if len(sub) > 0:
                rpas.append(sub['total_reward'].sum() / sub['n_actions'].sum())
            else:
                rpas.append(0)
        offset = (i - len(configs_all)/2) * width + width/2
        ax.bar(x + offset, rpas, width, label=cfg, alpha=0.8)

    ax.axhline(random_rpa, color='gray', linestyle=':', alpha=0.7)
    ax.axhline(fwd_random_rpa, color='brown', linestyle=':', alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=15)
    ax.set_ylabel('RPA', fontsize=12)
    ax.set_title('HPO Config Comparison (Model-Free Agents)', fontsize=13, fontweight='bold')
    ax.legend(loc='upper right', fontsize=7, ncol=2)
    fig.tight_layout()
    return fig

# ---------------------------------------------------------------------------
# Figure 5: Repoke interval split
# ---------------------------------------------------------------------------
def fig_repoke_split():
    repoke_csv = os.path.join(DATA_DIR, '260311_session_repoke_mapping.csv')
    if not os.path.exists(repoke_csv):
        return None
    repoke_map = pd.read_csv(repoke_csv, dtype={'animal_id': str})
    # Column is 'repoke_interval' per actual CSV header
    repoke_map['repoke_type'] = repoke_map['repoke_interval'].apply(
        lambda x: 'finite' if x < 1000 else 'infinite')

    # Merge with baseline + mouse
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Panel A: Mouse RPA by repoke type
    mouse_repoke = mouse_df.merge(repoke_map[['exp_moment', 'repoke_type']], on='exp_moment', how='left')
    if 'repoke_type' not in mouse_repoke.columns or mouse_repoke['repoke_type'].isna().all():
        return None

    for ax_idx, (title, df_plot, rpa_col) in enumerate([
        ('Mouse', mouse_repoke, 'mouse_rpa'),
    ]):
        finite = df_plot[df_plot['repoke_type'] == 'finite'][rpa_col]
        infinite = df_plot[df_plot['repoke_type'] == 'infinite'][rpa_col]
        ax = axes[0]
        bp = ax.boxplot([finite.values, infinite.values], labels=['Finite (≤90s)', 'Infinite'],
                        patch_artist=True, widths=0.5)
        bp['boxes'][0].set_facecolor('#3498db')
        bp['boxes'][1].set_facecolor('#e74c3c')
        ax.set_ylabel('RPA')
        ax.set_title(f'Mouse RPA by Repoke Interval\n'
                     f'Finite: {finite.mean():.3f} (n={len(finite)}), '
                     f'Inf: {infinite.mean():.3f} (n={len(infinite)})')

        if len(finite) > 1 and len(infinite) > 1:
            t_stat, p_val = stats.ttest_ind(finite, infinite)
            ax.text(0.5, 0.02, f'p={p_val:.4f}', transform=ax.transAxes,
                    ha='center', fontsize=10)

    # Panel B: Agent comparison by repoke type
    ax = axes[1]
    agent_names = ['Random', 'Random_forward', 'Greedy_oracle']
    finite_rpas, inf_rpas = [], []
    for model in agent_names:
        sub = baseline_df[baseline_df['model'] == model].merge(
            repoke_map[['exp_moment', 'repoke_type']], on='exp_moment', how='left')
        for rtype, container in [('finite', finite_rpas), ('infinite', inf_rpas)]:
            rsub = sub[sub['repoke_type'] == rtype]
            if len(rsub) > 0:
                container.append(rsub['total_reward'].sum() / rsub['n_actions'].sum())
            else:
                container.append(0)

    x = np.arange(len(agent_names))
    ax.bar(x - 0.2, finite_rpas, 0.35, label='Finite (≤90s)', color='#3498db')
    ax.bar(x + 0.2, inf_rpas, 0.35, label='Infinite', color='#e74c3c')
    ax.set_xticks(x)
    ax.set_xticklabels(agent_names, rotation=15)
    ax.set_ylabel('RPA')
    ax.set_title('Baseline Agents by Repoke Interval')
    ax.legend()

    fig.suptitle('Repoke Interval Analysis', fontsize=14, fontweight='bold', y=1.02)
    fig.tight_layout()
    return fig

# ---------------------------------------------------------------------------
# Figure 6: Maze scaling
# ---------------------------------------------------------------------------
def fig_maze_scaling():
    if len(scaling_df) == 0:
        return None

    agent_col = 'agent' if 'agent' in scaling_df.columns else 'model'
    fig, ax = plt.subplots(figsize=(10, 6))

    for agent in sorted(scaling_df[agent_col].unique()):
        sub = scaling_df[scaling_df[agent_col] == agent]
        grouped = sub.groupby('size')['rpa'].agg(['mean', 'std']).reset_index()
        ax.errorbar(grouped['size'], grouped['mean'], yerr=grouped['std'],
                    fmt='-o', label=agent, markersize=5, linewidth=1.5, capsize=3)

    ax.set_xlabel('Maze Size (N×N)', fontsize=12)
    ax.set_ylabel('RPA', fontsize=12)
    ax.set_title('Maze Scaling: Agent Performance vs Maze Size', fontsize=13, fontweight='bold')
    ax.legend(loc='upper right', fontsize=8)
    fig.tight_layout()
    return fig

# ---------------------------------------------------------------------------
# Generate all figures
# ---------------------------------------------------------------------------
print("\nGenerating figures...")
figures = {}
figures['landscape'] = fig_full_landscape()
print("  Fig 1: Full landscape")
figures['distributions'] = fig_rpa_distributions()
print("  Fig 2: RPA distributions")
figures['target_pretrain'] = fig_target_pretrain()
print("  Fig 3: Target pretrain")
figures['hpo_comparison'] = fig_hpo_comparison()
print("  Fig 4: HPO comparison")
fig5 = fig_repoke_split()
if fig5:
    figures['repoke_split'] = fig5
    print("  Fig 5: Repoke split")
fig6 = fig_maze_scaling()
if fig6:
    figures['maze_scaling'] = fig6
    print("  Fig 6: Maze scaling")

# Save individual PNGs
for name, fig in figures.items():
    path = os.path.join(FIGURE_DIR, f'{TS}_benchmark_{name}.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')

# ---------------------------------------------------------------------------
# Statistical tests
# ---------------------------------------------------------------------------
stat_results = []

# Mouse vs Random
mouse_rpas = mouse_df['mouse_rpa'].values
random_session = baseline_df[baseline_df['model'] == 'Random'].groupby('exp_moment').apply(
    lambda g: g['total_reward'].sum() / g['n_actions'].sum())
# Align sessions
common = set(mouse_df['exp_moment']) & set(random_session.index)
m_vals = mouse_df.set_index('exp_moment').loc[list(common), 'mouse_rpa'].values
r_vals = random_session.loc[list(common)].values
if len(m_vals) > 1:
    t, p = stats.ttest_rel(m_vals, r_vals)
    d = (m_vals - r_vals).mean() / (m_vals - r_vals).std()
    stat_results.append(('Mouse vs Random', t, p, d, len(common)))

# Mouse vs Forward_random
fwd_session = baseline_df[baseline_df['model'] == 'Random_forward'].groupby('exp_moment').apply(
    lambda g: g['total_reward'].sum() / g['n_actions'].sum())
common2 = set(mouse_df['exp_moment']) & set(fwd_session.index)
m2 = mouse_df.set_index('exp_moment').loc[list(common2), 'mouse_rpa'].values
f2 = fwd_session.loc[list(common2)].values
if len(m2) > 1:
    t, p = stats.ttest_rel(m2, f2)
    d = (m2 - f2).mean() / (m2 - f2).std()
    stat_results.append(('Mouse vs Forward_random', t, p, d, len(common2)))


def _paired_rpa_test(label, agent_df, mouse_df, model_name=None):
    """Paired t-test comparing agent session-level RPA against mouse RPA."""
    if model_name:
        adf = agent_df[agent_df['model'] == model_name]
    else:
        adf = agent_df
    if len(adf) == 0:
        return
    a_sess = adf.groupby('exp_moment').apply(
        lambda g: g['total_reward'].sum() / g['n_actions'].sum())
    m_sess = mouse_df.set_index('exp_moment')['mouse_rpa']
    common = a_sess.index.intersection(m_sess.index)
    if len(common) < 2:
        return
    a_vals = a_sess.loc[common].values
    m_vals = m_sess.loc[common].values
    diff = a_vals - m_vals
    t, p = stats.ttest_rel(a_vals, m_vals)
    d = diff.mean() / diff.std() if diff.std() > 0 else 0.0
    stat_results.append((label, t, p, d, len(common)))


def _paired_agent_test(label, df1, df2, model1=None, model2=None):
    """Paired t-test comparing two agents on common sessions."""
    a1 = (df1[df1['model'] == model1] if model1 else df1).groupby('exp_moment').apply(
        lambda g: g['total_reward'].sum() / g['n_actions'].sum())
    a2 = (df2[df2['model'] == model2] if model2 else df2).groupby('exp_moment').apply(
        lambda g: g['total_reward'].sum() / g['n_actions'].sum())
    common = a1.index.intersection(a2.index)
    if len(common) < 2:
        return
    v1 = a1.loc[common].values
    v2 = a2.loc[common].values
    diff = v1 - v2
    t, p = stats.ttest_rel(v1, v2)
    d = diff.mean() / diff.std() if diff.std() > 0 else 0.0
    stat_results.append((label, t, p, d, len(common)))


# Mouse vs DQN (HPO-best)
dqn_best_cfg = hpo_best.get('DQN', ('',))[0]
_paired_rpa_test('Mouse vs DQN_HPO',
                 hpo_df[(hpo_df['model'] == 'DQN') & (hpo_df['hpo_config'] == dqn_best_cfg)],
                 mouse_df)

# Mouse vs DRQN_seq (HPO-best)
dsq_best_cfg = hpo_best.get('DRQN_seq', ('',))[0]
_paired_rpa_test('Mouse vs DRQN_seq_HPO',
                 hpo_df[(hpo_df['model'] == 'DRQN_seq') & (hpo_df['hpo_config'] == dsq_best_cfg)],
                 mouse_df)

# Mouse vs RecurrentSAC D=32
_paired_rpa_test('Mouse vs RecurrentSAC', rsac_df, mouse_df)

# RecurrentSAC vs POMCP c=20 (oracle)
_paired_agent_test('RecurrentSAC vs POMCP_c20', rsac_df, pomcp_oracle_df)

# DQN pretrained vs DQN baseline (unpretrained), same sessions
dqn_pretrained = pretrain_df[(pretrain_df['model'] == 'DQN') & (pretrain_df['pretrain_type'] != 'none')]
dqn_baseline = pretrain_df[(pretrain_df['model'] == 'DQN') & (pretrain_df['pretrain_type'] == 'none')]
_paired_agent_test('DQN_pretrained vs DQN_baseline', dqn_pretrained, dqn_baseline)

# Bonferroni correction: adjust p-values for all tests
n_tests = len(stat_results)
stat_results_corrected = [
    (name, t, min(p * n_tests, 1.0), d, n)
    for name, t, p, d, n in stat_results
]

print(f"\nStatistical tests (raw p-values, Bonferroni corrected p in brackets, n_tests={n_tests}):")
for (name, t, p, d, n), (_, _, p_corr, _, _) in zip(stat_results, stat_results_corrected):
    sig = '***' if p_corr < 0.001 else ('**' if p_corr < 0.01 else ('*' if p_corr < 0.05 else 'ns'))
    print(f"  {name}: t={t:.2f}, p={p:.2e} [p_bonf={p_corr:.2e} {sig}], d={d:.2f}, n={n}")

# ---------------------------------------------------------------------------
# PDF report
# ---------------------------------------------------------------------------
pdf_path = os.path.join(REPORT_DIR, f'{TS}_comprehensive_benchmark.pdf')
with PdfPages(pdf_path) as pdf:
    for name, fig in figures.items():
        pdf.savefig(fig, bbox_inches='tight')
print(f"\nSaved PDF: {pdf_path}")

# ---------------------------------------------------------------------------
# Markdown report
# ---------------------------------------------------------------------------
md_lines = [
    f"# Comprehensive Agent Benchmark Report",
    f"",
    f"**Date:** {DATE_STR}",
    f"**Dataset:** {N_SESSIONS} filtered sessions, fully corrected rebait (corrected-rebait release)",
    f"**Reference:** Random={random_rpa:.3f}, Forward={fwd_random_rpa:.3f}, "
    f"Greedy={greedy_rpa:.3f}, Mouse={MOUSE_RPA:.3f}",
    f"",
    f"## Agent Rankings",
    f"",
    f"| Rank | Agent | RPA | % Ceiling | Family |",
    f"|------|-------|-----|-----------|--------|",
]
for i, (name, rpa, family, _) in enumerate(agents):
    md_lines.append(f"| {i+1} | {name} | {rpa:.3f} | {pct_ceiling(rpa):.1f}% | {family} |")

md_lines += [
    f"",
    f"## Key Findings",
    f"",
    f"1. **Greedy oracle sets the ceiling** at RPA={greedy_rpa:.3f}. Simple BFS to nearest reward.",
    f"2. **RecurrentSAC D=32 is the top learning agent** ({pct_ceiling(compute_rpa(rsac_df)):.0f}% ceiling).",
    f"3. **POMCP c=20 (fewer_sims) reaches {pct_ceiling(hpo_best['POMCP'][1]):.0f}%** — best planning agent.",
    f"4. **Pretraining is essential for model-free RL** — DQN jumps from {hpo_best['DQN'][1]:.3f} to {pretrain_best.get('DQN_prior_sessions_same_maze', 0):.3f} with same-maze pretraining.",
    f"5. **Mouse at {pct_ceiling(MOUSE_RPA):.0f}%** — barely above random, below forward-random.",
    f"6. **Most model-free agents cluster near random** (0-8% ceiling).",
]

if stat_results:
    md_lines += [f"", f"## Statistical Tests", f"",
                 f"Bonferroni correction applied (n_tests={n_tests}).",
                 f"",
                 f"| Comparison | t | p (raw) | p (Bonferroni) | sig | d | n |",
                 f"|-----------|---|---------|----------------|-----|---|---|"]
    for (name, t, p, d, n), (_, _, p_corr, _, _) in zip(stat_results, stat_results_corrected):
        sig = '***' if p_corr < 0.001 else ('**' if p_corr < 0.01 else ('*' if p_corr < 0.05 else 'ns'))
        md_lines.append(f"| {name} | {t:.2f} | {p:.2e} | {p_corr:.2e} | {sig} | {d:.2f} | {n} |")

md_lines += [f"", f"---", f"*Generated by `260318_generate_comprehensive_report.py`*"]

md_path = os.path.join(REPORT_DIR, f'{TS}_comprehensive_benchmark.md')
with open(md_path, 'w') as f:
    f.write('\n'.join(md_lines))
print(f"Saved markdown: {md_path}")

# ---------------------------------------------------------------------------
# HTML report (standalone, base64-embedded figures)
# ---------------------------------------------------------------------------
def fig_to_base64(fig, dpi=120):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=dpi, bbox_inches='tight')
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')

html_lines = [
    '<!DOCTYPE html>',
    '<html><head>',
    f'<title>Comprehensive Benchmark — {DATE_STR}</title>',
    '<meta charset="utf-8">',
    '<style>',
    'body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;',
    '       max-width: 1200px; margin: 0 auto; padding: 20px; background: #fafafa; color: #333; }',
    'h1 { border-bottom: 2px solid #333; padding-bottom: 8px; }',
    'h2 { color: #1a5276; margin-top: 40px; }',
    'table { border-collapse: collapse; margin: 15px 0; width: 100%; }',
    'th, td { border: 1px solid #ddd; padding: 8px 12px; text-align: left; }',
    'th { background: #2c3e50; color: white; }',
    'tr:nth-child(even) { background: #f2f2f2; }',
    'tr:hover { background: #ddd; }',
    '.figure { text-align: center; margin: 20px 0; }',
    '.figure img { max-width: 100%; border: 1px solid #ccc; border-radius: 4px; }',
    '.highlight { background: #d4efdf; font-weight: bold; }',
    '.warn { background: #fadbd8; }',
    '.mouse-row { background: #f5eef8; }',
    'code { background: #eee; padding: 2px 5px; border-radius: 3px; font-size: 0.9em; }',
    '.summary-box { background: #eaf2f8; border-left: 4px solid #2980b9;',
    '               padding: 15px; margin: 15px 0; border-radius: 4px; }',
    '.warning-box { background: #fdf2e9; border-left: 4px solid #e67e22;',
    '               padding: 15px; margin: 15px 0; border-radius: 4px; }',
    '</style>',
    '</head><body>',
    f'<h1>Comprehensive Agent Benchmark Report</h1>',
    f'<p><strong>Date:</strong> {DATE_STR} &nbsp;|&nbsp; '
    f'<strong>Sessions:</strong> {N_SESSIONS} &nbsp;|&nbsp; '
    f'<strong>Code:</strong> fully corrected rebait (corrected-rebait release)</p>',
    '',
    '<div class="summary-box">',
    f'<strong>Key result:</strong> RecurrentSAC D=32 (RPA={compute_rpa(rsac_df):.3f}, '
    f'{pct_ceiling(compute_rpa(rsac_df)):.0f}% ceiling) is the top learning agent. '
    f'POMCP c=20 reaches {pct_ceiling(hpo_best["POMCP"][1]):.0f}% with oracle knowledge. '
    f'Mouse performance ({MOUSE_RPA:.3f}) is only {pct_ceiling(MOUSE_RPA):.0f}% of ceiling — '
    f'below the trivial forward-random heuristic ({fwd_random_rpa:.3f}).',
    '</div>',
    '',
]

# Figure 1
html_lines += [
    '<h2>1. Full Agent Landscape</h2>',
    f'<div class="figure"><img src="data:image/png;base64,{fig_to_base64(figures["landscape"])}" '
    f'alt="Full landscape"></div>',
    '',
]

# Results table
html_lines += [
    '<h2>2. Agent Rankings</h2>',
    '<table>',
    '<tr><th>#</th><th>Agent</th><th>RPA</th><th>% Ceiling</th><th>Family</th></tr>',
]
for i, (name, rpa, family, fam_cat) in enumerate(agents):
    pct = pct_ceiling(rpa)
    cls = ''
    if pct >= 80: cls = ' class="highlight"'
    elif name == 'Mouse': cls = ' class="mouse-row"'
    elif pct < 0: cls = ' class="warn"'
    html_lines.append(f'<tr{cls}><td>{i+1}</td><td>{name}</td><td>{rpa:.3f}</td>'
                       f'<td>{pct:.1f}%</td><td>{family}</td></tr>')
html_lines.append('</table>')

# Figure 2
html_lines += [
    '<h2>3. RPA Distributions</h2>',
    f'<div class="figure"><img src="data:image/png;base64,{fig_to_base64(figures["distributions"])}" '
    f'alt="RPA distributions"></div>',
    '',
]

# Figure 3
html_lines += [
    '<h2>4. Target-Session Pretraining</h2>',
    '<p>All 3 agents (DQN, DRQN_seq, DRQN_rand) are trained on the same session for N episodes '
    'before evaluation. Shows the value of repeated exposure to the same maze.</p>',
    f'<div class="figure"><img src="data:image/png;base64,{fig_to_base64(figures["target_pretrain"])}" '
    f'alt="Target pretrain"></div>',
    '',
]

# Figure 4
html_lines += [
    '<h2>5. HPO Configuration Comparison</h2>',
    f'<div class="figure"><img src="data:image/png;base64,{fig_to_base64(figures["hpo_comparison"])}" '
    f'alt="HPO comparison"></div>',
    '',
]

# HPO best configs table
html_lines += [
    '<h3>Best HPO Config per Agent</h3>',
    '<table>',
    '<tr><th>Agent</th><th>Best Config</th><th>RPA</th><th>% Ceiling</th></tr>',
]
for model in sorted(hpo_best.keys()):
    cfg, rpa = hpo_best[model]
    pct = pct_ceiling(rpa)
    html_lines.append(f'<tr><td>{model}</td><td>{cfg}</td><td>{rpa:.3f}</td><td>{pct:.1f}%</td></tr>')
html_lines.append('</table>')

# Figure 5 (repoke)
if 'repoke_split' in figures:
    html_lines += [
        '<h2>6. Repoke Interval Analysis</h2>',
        f'<div class="figure"><img src="data:image/png;base64,{fig_to_base64(figures["repoke_split"])}" '
        f'alt="Repoke split"></div>',
        '',
    ]

# Figure 6 (scaling)
if 'maze_scaling' in figures:
    html_lines += [
        '<h2>7. Maze Scaling</h2>',
        f'<div class="figure"><img src="data:image/png;base64,{fig_to_base64(figures["maze_scaling"])}" '
        f'alt="Maze scaling"></div>',
        '',
    ]

# Stats
if stat_results:
    html_lines += [
        '<h2>Statistical Tests</h2>',
        f'<p>Bonferroni correction applied (n_tests={n_tests}).</p>',
        '<table>',
        '<tr><th>Comparison</th><th>t-stat</th><th>p (raw)</th><th>p (Bonferroni)</th>'
        '<th>sig</th><th>Cohen d</th><th>n</th></tr>',
    ]
    for (name, t, p, d, n), (_, _, p_corr, _, _) in zip(stat_results, stat_results_corrected):
        sig_str = '***' if p_corr < 0.001 else ('**' if p_corr < 0.01 else ('*' if p_corr < 0.05 else 'ns'))
        row_cls = ' class="highlight"' if p_corr < 0.05 else ''
        html_lines.append(f'<tr{row_cls}><td>{name}</td><td>{t:.2f}</td><td>{p:.2e}</td>'
                          f'<td>{p_corr:.2e}</td><td>{sig_str}</td>'
                          f'<td>{d:.2f}</td><td>{n}</td></tr>')
    html_lines.append('</table>')

# Data provenance
html_lines += [
    '',
    '<div class="warning-box">',
    '<strong>Data provenance:</strong> All results use fully corrected rebait logic '
    '(corrected-rebait release on 2026-03-15). The 260316-171035 master rerun replaced all prior '
    '260308 FIXED CSVs which had a double-+1 off-by-one bug. '
    'RecurrentSAC and POMCP c=1.0 use 260315 CSVs. '
    'Baselines rerun 2026-03-18.',
    '</div>',
    '',
    f'<p><em>Generated by <code>260318_generate_comprehensive_report.py</code> '
    f'at {TS}</em></p>',
    '</body></html>',
]

html_path = os.path.join(REPORT_DIR, f'{TS}_comprehensive_benchmark.html')
with open(html_path, 'w') as f:
    f.write('\n'.join(html_lines))
print(f"Saved HTML: {html_path}")

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print(f"\n{'='*60}")
print(f"  Report generated successfully!")
print(f"  PDF:  {pdf_path}")
print(f"  MD:   {md_path}")
print(f"  HTML: {html_path}")
print(f"{'='*60}")
