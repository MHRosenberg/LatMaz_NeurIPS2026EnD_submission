#!/usr/bin/env python3
"""
POMCP simulation-count sweep at c=1.0 to investigate the c=20 paradox.

Hypothesis
----------
With high c=20 (strong exploration pressure), more simulations cause the MCTS tree to
spread extremely thin — each branch is visited fewer times, degrading value estimates at the
root. Fewer sims with the same high c still explore well but concentrate more visits per
branch. At c=1.0 (Silver & Veness default), exploration pressure is moderate and we expect
more sims to either match or improve performance.

This script runs c=1.0 at 200, 500, and 1000 sims and compares to existing results
(c=20 fewer_sims, c=20 more_sims, c=1.0 500-sims from 260315 run).

Usage:
    conda run --no-capture-output -n latMaz_RL python -u \
        code/260326_pomcp_sim_sweep.py
"""
import os
import sys
import time
import glob
from datetime import datetime

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.dirname(__file__))

from yoked_rl_runner import AlgoConfig
from experiment_config import (
    load_data, build_runner, pomcp_obs_kwargs, output_path,
)
from utils_latMaz import get_most_recent_file

# =============================================================================
# User-configurable parameters
# =============================================================================
EXPLORATION_CONSTS = [1.0]          # just c=1.0 (c=20 results already exist)
SIM_COUNTS = [200, 500, 1000]       # sweep
N_REPEATS = 5
MAX_DEPTH = 80                       # same as existing more_sims config

GREEDY_RPA  = 0.878
RANDOM_RPA  = 0.178

# =============================================================================
# Print hypothesis
# =============================================================================
print("=" * 72)
print("POMCP SIM-COUNT SWEEP — c=1.0")
print("=" * 72)
print("""
Hypothesis
----------
With high c=20 (strong exploration pressure), more simulations cause the MCTS tree to
spread extremely thin — each branch is visited fewer times, degrading value estimates at
the root. Fewer sims with the same high c still explore well but concentrate more visits
per branch. At c=1.0 (Silver & Veness default), exploration pressure is moderate and we
expect more sims to either match or improve performance.
""")
print(f"Sweep:  c={EXPLORATION_CONSTS}, num_sims={SIM_COUNTS}, n_repeats={N_REPEATS}, max_depth={MAX_DEPTH}")
print()

# =============================================================================
# Load data
# =============================================================================
yoking_df, rwd_df, sessions = load_data()
n_sessions = len(sessions)
print(f"Filtered sessions: {n_sessions}")

_obs = pomcp_obs_kwargs()
timestamp = 'c' + datetime.now().strftime('%y%m%d-%H%M%S')

# =============================================================================
# Identify which (c, num_sims) combos already have existing data to skip
# =============================================================================
# c=1.0 / 500 sims is in 260315-*_pomcp_c1_full.csv → do NOT re-run those
EXISTING_COMBOS = {}  # (c, num_sims) -> label string for display

try:
    c1_csv = get_most_recent_file(output_path('*pomcp_c1_full.csv'))
except (IndexError, TypeError):
    c1_csv = None
if c1_csv and os.path.exists(c1_csv):
    # c260315 file used config 'c1_d100_s1000' → num_sims=1000
    EXISTING_COMBOS[(1.0, 1000)] = c1_csv
    print(f"  Found c=1.0 / 1000-sims: {os.path.basename(c1_csv)} — will skip re-run")

# =============================================================================
# Run sweep
# =============================================================================
results = []
t0_all = time.time()

for c in EXPLORATION_CONSTS:
    for num_sims in SIM_COUNTS:
        if (c, num_sims) in EXISTING_COMBOS:
            print(f"\n[c={c}, sims={num_sims}] Skipping — existing data in "
                  f"{os.path.basename(EXISTING_COMBOS[(c, num_sims)])}")
            continue

        label = f"c={c}, sims={num_sims}"
        total_runs = n_sessions * N_REPEATS
        print(f"\n{'='*60}")
        print(f"Running POMCP  {label}  ({n_sessions} sessions × {N_REPEATS} repeats = {total_runs} runs)")
        print(f"{'='*60}")

        algo = AlgoConfig()
        algo.POMCP = {
            'max_depth': MAX_DEPTH,
            'num_sims': num_sims,
            'exploration_const': float(c),
            'discount': 0.99,
        }

        runner = build_runner(yoking_df, rwd_df)
        runner.algo_config = algo

        combo_results = []

        for i, exp_mom in enumerate(sessions):
            yoke_row = yoking_df[yoking_df['exp_moment'] == exp_mom].iloc[0]
            adj_mat, st_positions = runner._load_maze(
                yoke_row['adj_file'], yoke_row['st_pos_file'])
            rewards, reset_val = runner._get_rewarded_states(exp_mom, len(st_positions))
            n_actions = int(yoke_row['n_states_visited']) - 1
            start_node = int(yoke_row['start_state'])

            for rep in range(N_REPEATS):
                seed = 42 + rep
                t0 = time.time()
                reward = runner.run_single(
                    model_name='POMCP',
                    adj_mat=adj_mat,
                    st_positions=st_positions,
                    start_node=start_node,
                    rewards=rewards,
                    n_actions=n_actions,
                    **_obs,
                    min_allowed_rewarded_states=reset_val,
                    seed=seed,
                )
                elapsed = time.time() - t0
                rpa = reward / n_actions if n_actions > 0 else 0.0

                row = {
                    'exp_moment': exp_mom,
                    'model': 'POMCP',
                    'c': float(c),
                    'num_sims': num_sims,
                    'total_reward': reward,
                    'n_actions': n_actions,
                    'rpa': rpa,
                    'seed': seed,
                    'elapsed_s': elapsed,
                }
                combo_results.append(row)
                results.append(row)

            done = (i + 1) * N_REPEATS
            session_rpas = [r['rpa'] for r in combo_results[-N_REPEATS:]]
            elapsed_total = time.time() - t0_all
            print(f"  [{done:4d}/{total_runs}] {exp_mom}: "
                  f"rpa={np.mean(session_rpas):.3f}  "
                  f"({elapsed_total:.0f}s total)")

            # Periodic partial save
            if (i + 1) % 5 == 0:
                pd.DataFrame(results).to_csv(
                    output_path(f'{timestamp}_pomcp_sim_sweep_PARTIAL.csv'), index=False)

# =============================================================================
# Save new results
# =============================================================================
if results:
    df_new = pd.DataFrame(results)
    csv_path = output_path(f'{timestamp}_pomcp_sim_sweep.csv')
    df_new.to_csv(csv_path, index=False)
    print(f"\nSaved {len(df_new)} new results to {csv_path}")
else:
    df_new = pd.DataFrame(columns=['exp_moment', 'model', 'c', 'num_sims',
                                   'total_reward', 'n_actions', 'rpa', 'seed', 'elapsed_s'])
    csv_path = None
    print("\nNo new runs performed (all combos had existing data).")

# Clean up partial
partial = output_path(f'{timestamp}_pomcp_sim_sweep_PARTIAL.csv')
if os.path.exists(partial):
    os.remove(partial)

# =============================================================================
# Load and assemble existing reference data
# =============================================================================
DATA_DIR = output_path('')  # = data_out/rl_sims/

# c=20 fewer_sims — from HPO tuning FIXED file
hpo_csv = get_most_recent_file(os.path.join(DATA_DIR, '*hpo_tuning_FIXED*.csv'))
df_c20_fewer = pd.DataFrame()
df_c20_more  = pd.DataFrame()
if hpo_csv and os.path.exists(hpo_csv):
    hpo_df = pd.read_csv(hpo_csv)
    pomcp_hpo = hpo_df[hpo_df['model'] == 'POMCP'].copy()
    if 'rpa' not in pomcp_hpo.columns:
        pomcp_hpo['rpa'] = pomcp_hpo['total_reward'] / pomcp_hpo['n_actions']
    df_c20_fewer = pomcp_hpo[pomcp_hpo['hpo_config'] == 'fewer_sims'].copy()
    df_c20_more  = pomcp_hpo[pomcp_hpo['hpo_config'] == 'more_sims'].copy()
    print(f"\nLoaded HPO file: {os.path.basename(hpo_csv)}")
    print(f"  c=20 fewer_sims rows: {len(df_c20_fewer)}")
    print(f"  c=20 more_sims  rows: {len(df_c20_more)}")
else:
    print("\nWARNING: HPO tuning FIXED file not found — c=20 reference data unavailable.")

# c=1.0 / 1000 sims — from pomcp_c1 file (c1_d100_s1000 config)
df_c1_1000 = pd.DataFrame()
if c1_csv and os.path.exists(c1_csv):
    df_c1_1000 = pd.read_csv(c1_csv)
    if 'rpa' not in df_c1_1000.columns:
        df_c1_1000['rpa'] = df_c1_1000['total_reward'] / df_c1_1000['n_actions']
    print(f"Loaded c=1.0/1000 sims: {os.path.basename(c1_csv)} ({len(df_c1_1000)} rows)")

# =============================================================================
# Summary table
# =============================================================================
def rpa_from_df(df):
    """Macro-RPA: mean of per-session RPA (weighted by action count)."""
    if df is None or len(df) == 0:
        return np.nan
    return df['total_reward'].sum() / df['n_actions'].sum()


def pct_ceiling(rpa):
    return 100.0 * (rpa - RANDOM_RPA) / (GREEDY_RPA - RANDOM_RPA)


print("\n" + "=" * 72)
print("SUMMARY — Mean RPA and % Ceiling  (Greedy=0.878, Random=0.178)")
print("=" * 72)
print(f"{'Config':<30} {'c':>5} {'sims':>6} {'mean_RPA':>10} {'%_ceiling':>10}  {'source'}")
print("-" * 72)

# --- Existing reference rows ---
rows_existing = [
    ("c=20 fewer_sims (existing)",  20.0,  200, df_c20_fewer, "260316-171035_hpo_tuning_FIXED"),
    ("c=20 more_sims (existing)",   20.0, 1000, df_c20_more,  "260316-171035_hpo_tuning_FIXED"),
    ("c=1.0 / 1000 sims (existing)", 1.0, 1000,  df_c1_1000,   "260315-*_pomcp_c1_full"),
]
for label, c_val, n_s, df_ref, src in rows_existing:
    rpa = rpa_from_df(df_ref)
    pct = pct_ceiling(rpa) if not np.isnan(rpa) else float('nan')
    rpa_str = f"{rpa:.3f}" if not np.isnan(rpa) else "N/A"
    pct_str = f"{pct:.1f}%" if not np.isnan(pct) else "N/A"
    print(f"  {label:<28} {c_val:>5.1f} {n_s:>6}    {rpa_str:>8}   {pct_str:>8}  {src}")

# --- New sweep rows ---
if len(df_new) > 0:
    print()
    for c_val in sorted(df_new['c'].unique()):
        for n_s in sorted(df_new[df_new['c'] == c_val]['num_sims'].unique()):
            sub = df_new[(df_new['c'] == c_val) & (df_new['num_sims'] == n_s)]
            rpa = rpa_from_df(sub)
            pct = pct_ceiling(rpa)
            label = f"c={c_val:.1f} / {n_s} sims (NEW)"
            print(f"  {label:<28} {c_val:>5.1f} {n_s:>6}    {rpa:.3f}     {pct:.1f}%  this run")

print()
print("Notes:")
print(f"  GREEDY_RPA = {GREEDY_RPA}  (oracle ceiling)")
print(f"  RANDOM_RPA = {RANDOM_RPA}  (random baseline)")
if csv_path:
    print(f"  New data saved to: {csv_path}")
print("=" * 72)
