"""Run FSCBioAgent with correct paper params: exploit_timeout=1, exploit_greediness=3.

Prior run (260328, greediness=0) → RPA=0.421.  WRONG params.
oTmac paper params (timeout=1, greediness=3)  → RPA=0.573 (56% ceiling).
This script verifies the oTmac result on ptonMac's canonical 60-session dataset.

See internal full-context notes (not in public release).

Usage:
    conda run --no-capture-output -n latMaz_RL python -u \
        code/260331_fsc_bio_correct_params.py
"""
import os
import sys
import time
from datetime import datetime

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.dirname(__file__))

from yoked_rl_runner import AlgoConfig
from experiment_config import (
    load_data, build_runner, pomcp_obs_kwargs, output_path, DEFAULT_N_REPEATS,
)

ts = 'c' + datetime.now().strftime('%y%m%d-%H%M%S')

GREEDY_RPA = 0.878
RANDOM_RPA = 0.178

print("=" * 72)
print("FSC_bio — CORRECT paper params: exploit_timeout=1, exploit_greediness=3")
print("=" * 72)
print(f"Paper claim (oTmac): RPA=0.573 (56% ceiling)")
print(f"Wrong prior run (greediness=0): RPA=0.421 (34% ceiling)")
print(f"Canonical default (timeout=10, greediness=3): RPA=0.503 (46%)")
print()

yoking_df, rwd_df, sessions = load_data()
n_sessions = len(sessions)
n_repeats = DEFAULT_N_REPEATS
total = n_sessions * n_repeats
print(f"Running {n_sessions} sessions × {n_repeats} repeats = {total} sims")

runner = build_runner(yoking_df, rwd_df)

# CORRECT paper params: timeout=1 (rapid mode switching), greediness=3 (moderate exploitation)
algo = AlgoConfig()
algo.FSC_bio = {
    'exploit_timeout': 1,
    'trace_decay': 0.8,
    'pseudocount': 1.0,
    'forward_bias': 1.0,
    'prevent_backward': True,
    'exploit_greediness': 3.0,
}
runner.algo_config = algo

_obs = pomcp_obs_kwargs()
results = []
t0_all = time.time()

for i, exp_mom in enumerate(sessions):
    yoke_row = yoking_df[yoking_df['exp_moment'] == exp_mom].iloc[0]
    adj_mat, st_positions = runner._load_maze(yoke_row['adj_file'], yoke_row['st_pos_file'])
    rewards, reset_val = runner._get_rewarded_states(exp_mom, len(st_positions))
    n_actions = int(yoke_row['n_states_visited']) - 1
    start_node = int(yoke_row['start_state'])

    for rep in range(n_repeats):
        seed = 42 + rep
        t0 = time.time()
        reward = runner.run_single(
            model_name='FSC_bio',
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

        results.append({
            'exp_moment': exp_mom,
            'model': 'FSC_bio',
            'exploit_timeout': 1,
            'exploit_greediness': 3.0,
            'seed': seed,
            'total_reward': reward,
            'n_actions': n_actions,
            'rpa': rpa,
            'elapsed_s': round(elapsed, 3),
        })

    done = (i + 1) * n_repeats
    session_rpas = [r['rpa'] for r in results[-n_repeats:]]
    elapsed_total = time.time() - t0_all
    print(f"  [{done:4d}/{total}] {exp_mom}: rpa={np.mean(session_rpas):.3f}  ({elapsed_total:.0f}s total)")

    if (i + 1) % 10 == 0:
        pd.DataFrame(results).to_csv(
            output_path(f'{ts}_fsc_bio_correct_params_PARTIAL.csv'), index=False)

df = pd.DataFrame(results)
out_path = output_path(f'{ts}_fsc_bio_correct_params.csv')
df.to_csv(out_path, index=False)
print(f"\nSaved {len(df)} results to {out_path}")

rpa_overall = df['total_reward'].sum() / df['n_actions'].sum()
pct_ceil = 100 * (rpa_overall - RANDOM_RPA) / (GREEDY_RPA - RANDOM_RPA)
print(f"\nFSC_bio (timeout=1, greediness=3):")
print(f"  Macro-RPA  = {rpa_overall:.4f}")
print(f"  %-ceiling   = {pct_ceil:.1f}%  (Greedy={GREEDY_RPA}, Random={RANDOM_RPA})")
print(f"  Paper claim: 0.573 (56%)")
print(f"  Wrong prior: 0.421 (34%)")
print(f"  Default cfg: 0.503 (46%)")

# Clean up partial
partial = output_path(f'{ts}_fsc_bio_correct_params_PARTIAL.csv')
if os.path.exists(partial):
    os.remove(partial)
