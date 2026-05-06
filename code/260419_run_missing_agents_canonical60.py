#!/usr/bin/env python3
"""
260419_run_missing_agents_canonical60.py

Run VarMarkov + OOI on the canonical 60 sessions (these were missing from
the canonical benchmark — needed for the 13-agent common-denominator set).

Usage:
    conda activate latMaz_RL
    python code/260419_run_missing_agents_canonical60.py
"""
import os, sys, time
from datetime import datetime

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.dirname(__file__))

from intermediate_agents import VarMarkovAgent, OOIAgent
from experiment_config import load_yoking_df, load_reward_df, build_runner, filter_sessions, output_path, DEFAULT_N_REPEATS

N_REPEATS = DEFAULT_N_REPEATS  # 5
SEED_BASE = 42

yoking_df = load_yoking_df()
rwd_df    = load_reward_df()
sessions  = sorted(filter_sessions(yoking_df))
runner    = build_runner(yoking_df, rwd_df)

print(f'Canonical sessions: {len(sessions)}')

AGENTS = [
    ('VarMarkov', VarMarkovAgent()),
    ('OOI',       OOIAgent()),
]

results = []
for i, exp_mom in enumerate(sessions):
    yoke_row   = yoking_df[yoking_df['exp_moment'] == exp_mom].iloc[0]
    adj_mat, st_positions = runner._load_maze(yoke_row['adj_file'], yoke_row['st_pos_file'])
    rewards, reset_val = runner._get_rewarded_states(exp_mom, len(st_positions))
    n_actions  = int(yoke_row['n_states_visited']) - 1
    start_node = int(yoke_row['start_state'])
    animal_id  = str(yoke_row.get('animal_ID', ''))

    for agent_name, agent in AGENTS:
        for rep in range(N_REPEATS):
            try:
                total_reward = agent.run_episode(
                    adj_mat=adj_mat, node_positions=st_positions,
                    start_node=start_node, rewarded_nodes=rewards,
                    n_actions=n_actions, min_rewarded_states=reset_val,
                    seed=SEED_BASE + rep, prevent_reverse=False,
                )
            except Exception as e:
                print(f'  WARNING: {agent_name} failed on {exp_mom} rep {rep}: {e}')
                total_reward = 0
            results.append({
                'exp_moment': exp_mom, 'animal_ID': animal_id,
                'model': agent_name, 'seed': SEED_BASE + rep,
                'repeat_idx': rep, 'total_reward': total_reward,
                'n_actions': n_actions,
            })

    rpa_str = '  '.join(
        f"{m}={np.mean([r['total_reward'] for r in results if r['exp_moment']==exp_mom and r['model']==m])/n_actions:.3f}"
        for m, _ in AGENTS)
    print(f'[{i+1:3d}/{len(sessions)}] {exp_mom} (a{animal_id}): {rpa_str}')

df = pd.DataFrame(results)
df['rpa'] = df['total_reward'] / df['n_actions']

ts = 'c' + datetime.now().strftime('%y%m%d-%H%M%S')
out_path = output_path(f'{ts}_varmarkov_ooi_canonical60.csv')
df.to_csv(out_path, index=False)
print(f'\nSaved {len(df)} rows -> {out_path}')
print(df.groupby('model')['rpa'].agg(['mean', 'std']).round(4).to_string())
