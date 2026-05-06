#!/usr/bin/env python3
"""
260419_recsac_memory_reset_ablation.py

Evaluate the trained RecurrentSAC D=32 model with periodic memory resets.
Tests how much episodic memory (GRU hidden state) contributes to performance.

reset_interval=None  → full memory (baseline, ~0.796 RPA)
reset_interval=1     → memoryless (should approach feedforward/random level)

Usage:
    conda activate latMaz_RL
    python code/260419_recsac_memory_reset_ablation.py
"""
import os, sys, time
from datetime import datetime

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.dirname(__file__))

from recurrent_sac_agent import RecurrentSACAgent
from experiment_config import (
    load_yoking_df, load_reward_df, build_runner, filter_sessions, output_path,
)

N_SEEDS = 5
SEED_BASE = 42
RESET_INTERVALS = [None, 200, 100, 50, 20, 10, 5, 1]

WEIGHTS_PATH = os.path.join(
    os.path.dirname(__file__), '..', '..', 'data_out', 'rl_sims',
    'recurrent_sac_weights', 'recurrent_sac_d32_ptonMac.pt',
)
assert os.path.exists(WEIGHTS_PATH), f'Weights not found: {WEIGHTS_PATH}'

yoking_df = load_yoking_df()
rwd_df    = load_reward_df()
sessions  = sorted(filter_sessions(yoking_df))
runner    = build_runner(yoking_df, rwd_df)

agent = RecurrentSACAgent(hidden_dim=32, weights_path=WEIGHTS_PATH, prevent_backward=True)
print(f'Loaded RecurrentSAC D=32 from {WEIGHTS_PATH}')
print(f'Sessions: {len(sessions)}, seeds: {N_SEEDS}, intervals: {RESET_INTERVALS}')

results = []
t_start = time.time()

for ri_idx, reset_interval in enumerate(RESET_INTERVALS):
    ri_label = str(reset_interval) if reset_interval is not None else 'None'
    t0 = time.time()

    for i, exp_mom in enumerate(sessions):
        yoke_row = yoking_df[yoking_df['exp_moment'] == exp_mom].iloc[0]
        adj_mat, st_positions = runner._load_maze(yoke_row['adj_file'], yoke_row['st_pos_file'])
        rewards, reset_val = runner._get_rewarded_states(exp_mom, len(st_positions))
        n_actions  = int(yoke_row['n_states_visited']) - 1
        start_node = int(yoke_row['start_state'])

        for seed_offset in range(N_SEEDS):
            total_reward = agent.run_episode(
                adj_mat=adj_mat, node_positions=st_positions,
                start_node=start_node, rewarded_nodes=rewards,
                n_actions=n_actions, min_rewarded_states=reset_val,
                seed=SEED_BASE + seed_offset,
                reset_interval=reset_interval,
            )
            results.append({
                'exp_moment': exp_mom,
                'reset_interval': ri_label,
                'seed': SEED_BASE + seed_offset,
                'total_reward': total_reward,
                'n_actions': n_actions,
            })

    elapsed = time.time() - t0
    rpa_vals = [r['total_reward'] / r['n_actions'] for r in results if r['reset_interval'] == ri_label]
    print(f'  reset_interval={ri_label:>4s}: RPA={np.mean(rpa_vals):.4f} ± {np.std(rpa_vals):.4f}  ({elapsed:.1f}s)')

df = pd.DataFrame(results)
df['rpa'] = df['total_reward'] / df['n_actions']

ts = 'c' + datetime.now().strftime('%y%m%d-%H%M%S')
out_path = output_path(f'{ts}_recsac_memory_reset_ablation.csv')
df.to_csv(out_path, index=False)
print(f'\nSaved {len(df)} rows -> {out_path}')
print(f'Total time: {time.time() - t_start:.0f}s')

print('\n--- Summary ---')
summary = df.groupby('reset_interval')['rpa'].agg(['mean', 'std']).round(4)
print(summary.to_string())
