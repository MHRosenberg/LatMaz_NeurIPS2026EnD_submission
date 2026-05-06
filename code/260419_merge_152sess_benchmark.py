#!/usr/bin/env python3
"""
260419_merge_152sess_benchmark.py

Merge all no-training agent results into a unified 152-session × 14-agent CSV
(13 original + Clone_conditional). Validate against PNG ground truth.
Compute per-animal statistics.

Usage:
    conda activate latMaz_RL
    python code/260419_merge_152sess_benchmark.py
"""
import os, sys
from datetime import datetime

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.dirname(__file__))

from experiment_config import load_yoking_df, filter_sessions, output_path

# ============================================================================
# 1. Load all source CSVs
# ============================================================================
DATA_DIR = 'data_out/rl_sims'

SOURCES = {
    # Canonical 60 sessions
    'baselines':  f'{DATA_DIR}/c260318-144507_baseline_heuristics.csv',
    'cloning':    f'{DATA_DIR}/c260413-103634_cloning_agents.csv',
    'pomcp':      f'{DATA_DIR}/c260316-171035_pomcp_full41_FIXED.csv',
    'pomcp_bio':  f'{DATA_DIR}/c260316-171035_pomcp_bio_FIXED.csv',
    'fsc_bio':    f'{DATA_DIR}/c260328-174258_fsc_bio_paper_params.csv',
    'varmarkov_ooi': f'{DATA_DIR}/c260419-143442_varmarkov_ooi_canonical60.csv',
    # Expanded 92 sessions
    'expanded92': f'{DATA_DIR}/c260414-155810_expanded92_no_training_agents.csv',
    # Conditional clone (all 152)
    'cond_clone': f'{DATA_DIR}/c260419-143812_conditional_clone_152sess.csv',
}

# Check all files exist
for name, path in SOURCES.items():
    assert os.path.exists(path), f'{name}: {path} not found'
    print(f'  {name}: {path}')

# Load
dfs = {name: pd.read_csv(path) for name, path in SOURCES.items()}

# ============================================================================
# 2. Standardize columns and merge
# ============================================================================
STD_COLS = ['exp_moment', 'animal_ID', 'model', 'seed', 'total_reward', 'n_actions']

def standardize(df, source_name):
    """Ensure standard columns, compute rpa."""
    df = df.copy()
    # Standardize animal_ID to zero-padded string
    if 'animal_ID' in df.columns:
        df['animal_ID'] = df['animal_ID'].astype(str).str.zfill(3)
    # Compute n_actions if missing
    if 'n_actions' not in df.columns and 'n_states_visited' in df.columns:
        df['n_actions'] = df['n_states_visited'] - 1
    # Compute rpa
    if 'rpa' not in df.columns:
        df['rpa'] = df['total_reward'] / df['n_actions']
    # Ensure seed column
    if 'seed' not in df.columns:
        df['seed'] = df.get('repeat_idx', 0)
    return df

for name in dfs:
    dfs[name] = standardize(dfs[name], name)

# Baselines have 15 reps — subsample to 5
baselines = dfs['baselines']
baselines_5 = baselines[baselines['repeat_idx'] < 5].copy()
dfs['baselines'] = baselines_5

# ============================================================================
# 3. Session sets
# ============================================================================
yoking_df = load_yoking_df()
sess60 = set(filter_sessions(yoking_df))
fbr_sess = set(pd.read_csv('data_out/rl_sims/c260331-182639_fbr_152sessions.csv')['exp_moment'].unique())
sess92 = fbr_sess - sess60

print(f'\nCanonical 60: {len(sess60)}, Older 92: {len(sess92)}, Total: {len(fbr_sess)}')

# ============================================================================
# 4. Build unified DataFrame: 13 no-training agents + Clone_conditional
# ============================================================================
NO_TRAINING_AGENTS = [
    'Random', 'Random_forward', 'Greedy_oracle',
    'FullFwd', 'NoBkFullFwd',
    'Clone_ego', 'Clone_allo_real', 'Clone_allo_latent',
    'VarMarkov', 'OOI', 'FSC_bio',
    'POMCP', 'POMCP_bio',
]

# Canonical 60: collect from multiple CSVs
canon_parts = []
for src in ['baselines', 'cloning', 'pomcp', 'pomcp_bio', 'fsc_bio', 'varmarkov_ooi']:
    df = dfs[src]
    df = df[df['exp_moment'].isin(sess60)]
    canon_parts.append(df)
canon60 = pd.concat(canon_parts, ignore_index=True)

# Filter to only no-training agents
canon60 = canon60[canon60['model'].isin(NO_TRAINING_AGENTS)]

# Expanded 92: already has all 13
exp92 = dfs['expanded92'].copy()
exp92 = exp92[exp92['model'].isin(NO_TRAINING_AGENTS)]

# Merge
unified = pd.concat([canon60, exp92], ignore_index=True)

# Add conditional clone
cond_clone = dfs['cond_clone'].copy()
unified = pd.concat([unified, cond_clone], ignore_index=True)

# Ensure rpa
unified['rpa'] = unified['total_reward'] / unified['n_actions']

# Add animal_ID from yoking for any missing
yoking_lookup = yoking_df.set_index('exp_moment')['animal_ID'].to_dict()
mask = unified['animal_ID'].isna() | (unified['animal_ID'] == '') | (unified['animal_ID'] == 'nan')
unified.loc[mask, 'animal_ID'] = unified.loc[mask, 'exp_moment'].map(
    lambda x: str(yoking_lookup.get(x, '')).zfill(3))

print(f'\nUnified: {len(unified)} rows')
print(f'  Sessions: {unified["exp_moment"].nunique()}')
print(f'  Agents: {sorted(unified["model"].unique())}')
print(f'  Animals: {sorted(unified["animal_ID"].unique())}')

# Save
ts = 'c' + datetime.now().strftime('%y%m%d-%H%M%S')
out_path = output_path(f'{ts}_152sess_14agents_unified.csv')
unified.to_csv(out_path, index=False)
print(f'Saved -> {out_path}')

# ============================================================================
# 5. PNG validation
# ============================================================================
print('\n=== PNG Ground Truth Validation ===')
png_path = 'data_out/260304-233949_full_png_reward_extraction.csv'
if os.path.exists(png_path):
    png_df = pd.read_csv(png_path)
    png_sessions = set(png_df['exp_moment'].values)
    our_sessions = set(unified['exp_moment'].unique())

    covered = our_sessions & png_sessions
    missing = our_sessions - png_sessions
    print(f'  Our 152 sessions in PNG data: {len(covered)}/{len(our_sessions)}')

    if missing:
        print(f'  MISSING from PNG: {sorted(missing)[:10]}')

    # Check for anomalies
    anomalies = png_df[png_df['exp_moment'].isin(our_sessions) & (png_df['status'] != 'ok')]
    if len(anomalies) > 0:
        print(f'  ANOMALOUS sessions: {len(anomalies)}')
        for _, row in anomalies.iterrows():
            print(f'    {row["exp_moment"]}: status={row["status"]}')
    else:
        print(f'  All {len(covered)} sessions have status=ok')
else:
    print(f'  PNG file not found: {png_path}')

# ============================================================================
# 6. Per-animal statistics
# ============================================================================
print('\n=== Per-Animal Statistics (13 no-training agents) ===')

# Mouse RPA from yoking
mouse_rpa = {}
for exp_mom in fbr_sess:
    row = yoking_df[yoking_df['exp_moment'] == exp_mom]
    if len(row) == 0:
        continue
    row = row.iloc[0]
    n_actions = int(row['n_states_visited']) - 1
    # Mouse RPA = n_rewards / n_actions (from yoking n_rewards field)
    rpa_mouse = float(row.get('n_rewards', 0)) / n_actions if n_actions > 0 else 0
    animal = str(row['animal_ID']).zfill(3)
    mouse_rpa[exp_mom] = {'rpa': rpa_mouse, 'animal_ID': animal}

mouse_df = pd.DataFrame(mouse_rpa).T.reset_index().rename(columns={'index': 'exp_moment'})
mouse_df['rpa'] = mouse_df['rpa'].astype(float)

# Agent pooled mean per session (13 no-training agents only)
agent_only = unified[unified['model'].isin(NO_TRAINING_AGENTS)]
agent_pool = agent_only.groupby('exp_moment')['rpa'].mean().reset_index()
agent_pool.columns = ['exp_moment', 'agent_rpa_pooled']

# Merge
compare = mouse_df.merge(agent_pool, on='exp_moment', how='inner')

print(f'\nSessions with both mouse and agent data: {len(compare)}')
print(f'Animals: {sorted(compare["animal_ID"].unique())}')

# Per-animal paired t-tests
print(f'\n{"Animal":>8s} {"n":>4s} {"Mouse RPA":>10s} {"Agent (pooled)":>14s} {"t":>8s} {"p (raw)":>12s} {"p (Bonf)":>12s} {"sig":>5s}')
print('-' * 80)

n_animals = compare['animal_ID'].nunique()
for animal in ['ALL'] + sorted(compare['animal_ID'].unique()):
    if animal == 'ALL':
        g = compare
    else:
        g = compare[compare['animal_ID'] == animal]

    if len(g) < 2:
        print(f'{animal:>8s} {len(g):>4d}   insufficient data')
        continue

    mv = g['rpa'].values
    av = g['agent_rpa_pooled'].values
    t_stat, p_val = stats.ttest_rel(mv, av)
    p_bonf = min(p_val * n_animals, 1.0)
    sig = '***' if p_bonf < 0.001 else ('**' if p_bonf < 0.01 else ('*' if p_bonf < 0.05 else 'ns'))
    print(f'{animal:>8s} {len(g):>4d} {mv.mean():>10.3f} {av.mean():>14.3f} {t_stat:>8.2f} {p_val:>12.2e} {p_bonf:>12.2e} {sig:>5s}')

# Per-agent summary across all 152 sessions
print('\n=== Per-Agent RPA Summary (all 152 sessions) ===')
all_agents = sorted(unified['model'].unique())
for agent in all_agents:
    ag = unified[unified['model'] == agent]
    sess_rpa = ag.groupby('exp_moment')['rpa'].mean()
    print(f'  {agent:25s}: RPA={sess_rpa.mean():.4f} ± {sess_rpa.std():.4f}  (n={len(sess_rpa)} sessions)')

print(f'\nMouse (from yoking): RPA={mouse_df["rpa"].mean():.4f} ± {mouse_df["rpa"].std():.4f}  (n={len(mouse_df)} sessions)')
