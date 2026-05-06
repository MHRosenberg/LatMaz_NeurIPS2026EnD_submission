#!/usr/bin/env python3
"""
260419_run_conditional_clone.py

Run ConditionalCloningAgent (P(action|obs) from mouse trajectory) on all
152 sessions. Compare against marginal Clone_ego.

Usage:
    conda activate latMaz_RL
    python code/260419_run_conditional_clone.py
"""
import ast, os, sys, time
from datetime import datetime

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.dirname(__file__))

from intermediate_agents import ConditionalCloningAgent
from experiment_config import (
    load_yoking_df, load_reward_df, build_runner, filter_sessions, output_path,
    DEFAULT_N_REPEATS,
)

N_REPEATS = DEFAULT_N_REPEATS  # 5
SEED_BASE = 42

FBR_CSV = 'data_out/rl_sims/c260331-182639_fbr_152sessions.csv'
assert os.path.exists(FBR_CSV), f'FBR CSV not found: {FBR_CSV}'

yoking_df = load_yoking_df()
rwd_df    = load_reward_df()
runner    = build_runner(yoking_df, rwd_df)

fbr_sess = set(pd.read_csv(FBR_CSV)['exp_moment'].unique())
sessions = sorted(fbr_sess & set(yoking_df['exp_moment'].values))
print(f'Sessions: {len(sessions)}')

results = []
t_start = time.time()

for i, exp_mom in enumerate(sessions):
    yoke_row = yoking_df[yoking_df['exp_moment'] == exp_mom].iloc[0]
    adj_mat, st_positions = runner._load_maze(yoke_row['adj_file'], yoke_row['st_pos_file'])
    rewards, reset_val = runner._get_rewarded_states(exp_mom, len(st_positions))
    n_actions  = int(yoke_row['n_states_visited']) - 1
    start_node = int(yoke_row['start_state'])
    animal_id  = str(yoke_row.get('animal_ID', ''))

    # Parse states_visited from yoking
    sv_raw = yoke_row.get('states_visited', '[]')
    states_visited = ast.literal_eval(sv_raw) if isinstance(sv_raw, str) else sv_raw
    if not isinstance(states_visited, (list, np.ndarray)) or len(states_visited) < 2:
        print(f'  SKIP {exp_mom}: states_visited too short ({len(states_visited) if hasattr(states_visited, "__len__") else "?"})')
        continue

    # Build per-session conditional clone
    try:
        agent = ConditionalCloningAgent(
            states_visited=states_visited,
            adj_mat=adj_mat, node_positions=st_positions,
            start_node=start_node, rewarded_nodes=rewards,
            min_rewarded_states=reset_val,
        )
    except Exception as e:
        print(f'  WARNING: ConditionalCloningAgent init failed for {exp_mom}: {e}')
        continue

    for rep in range(N_REPEATS):
        total_reward = agent.run_episode(
            adj_mat=adj_mat, node_positions=st_positions,
            start_node=start_node, rewarded_nodes=rewards,
            n_actions=n_actions, min_rewarded_states=reset_val,
            seed=SEED_BASE + rep,
        )
        results.append({
            'exp_moment': exp_mom, 'animal_ID': animal_id,
            'model': 'Clone_conditional', 'seed': SEED_BASE + rep,
            'repeat_idx': rep, 'total_reward': total_reward,
            'n_actions': n_actions,
        })

    rpa = np.mean([r['total_reward'] for r in results[-N_REPEATS:]]) / n_actions
    if (i + 1) % 20 == 0 or i == 0:
        print(f'[{i+1:3d}/{len(sessions)}] {exp_mom} (a{animal_id}): Clone_cond={rpa:.3f}')

df = pd.DataFrame(results)
df['rpa'] = df['total_reward'] / df['n_actions']

ts = 'c' + datetime.now().strftime('%y%m%d-%H%M%S')
out_path = output_path(f'{ts}_conditional_clone_152sess.csv')
df.to_csv(out_path, index=False)
print(f'\nSaved {len(df)} rows -> {out_path}')
print(f'Total time: {time.time() - t_start:.0f}s')

print(f'\nConditional clone RPA: {df["rpa"].mean():.4f} ± {df["rpa"].std():.4f}')
print(f'Sessions covered: {df["exp_moment"].nunique()}')
