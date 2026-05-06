#!/usr/bin/env python3
"""
260419_recsac_hidden_dim_sweep.py

Train RecurrentSAC with D=4, 8, 16 (D=32 already exists) and evaluate
on canonical 60 sessions. Tests capacity-performance relationship.

Usage:
    conda activate latMaz_RL
    python code/260419_recsac_hidden_dim_sweep.py
"""
import os, sys, time, gc
from datetime import datetime

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.dirname(__file__))

from recurrent_sac_agent import train_meta_rl_sac, RecurrentSACAgent
from experiment_config import (
    load_yoking_df, load_reward_df, build_runner, filter_sessions, output_path,
)

HIDDEN_DIMS = [4, 8, 16]  # D=32 already trained
N_EPOCHS = 100
N_EVAL_SEEDS = 5
SEED_BASE = 42

yoking_df = load_yoking_df()
rwd_df    = load_reward_df()
runner    = build_runner(yoking_df, rwd_df)
sessions  = sorted(filter_sessions(yoking_df))

# Prepare session data for training
adj_mats, node_pos_list, start_nodes = [], [], []
rewarded_list, n_actions_list, min_rwd_list = [], [], []

for exp_mom in sessions:
    yoke_row = yoking_df[yoking_df['exp_moment'] == exp_mom].iloc[0]
    adj_mat, st_pos = runner._load_maze(yoke_row['adj_file'], yoke_row['st_pos_file'])
    rewards, reset_val = runner._get_rewarded_states(exp_mom, len(st_pos))
    n_actions = int(yoke_row['n_states_visited']) - 1
    start_node = int(yoke_row['start_state'])

    adj_mats.append(adj_mat)
    node_pos_list.append(st_pos)
    start_nodes.append(start_node)
    rewarded_list.append(rewards)
    n_actions_list.append(n_actions)
    min_rwd_list.append(reset_val)

print(f'Sessions: {len(sessions)}')
print(f'Hidden dims to train: {HIDDEN_DIMS}')
print(f'Epochs: {N_EPOCHS}, Eval seeds: {N_EVAL_SEEDS}')

ts = 'c' + datetime.now().strftime('%y%m%d-%H%M%S')
results = []

# Also evaluate existing D=32
existing_d32 = os.path.join(
    os.path.dirname(__file__), '..', '..', 'data_out', 'rl_sims',
    'recurrent_sac_weights', 'recurrent_sac_d32_ptonMac.pt',
)
all_dims = HIDDEN_DIMS + ([32] if os.path.exists(existing_d32) else [])

for dim in all_dims:
    print(f'\n=== Hidden dim D={dim} ===')
    t0 = time.time()

    if dim == 32 and os.path.exists(existing_d32):
        print(f'  Loading existing D=32 weights from {existing_d32}')
        agent = RecurrentSACAgent(hidden_dim=32, weights_path=existing_d32, prevent_backward=True)
    else:
        print(f'  Training D={dim} for {N_EPOCHS} epochs...')
        model, epoch_rewards = train_meta_rl_sac(
            hidden_dim=dim,
            adj_mats=adj_mats, node_positions_list=node_pos_list,
            start_nodes=start_nodes, rewarded_nodes_list=rewarded_list,
            n_actions_list=n_actions_list,
            min_rewarded_states_list=min_rwd_list,
            n_epochs=N_EPOCHS,
            lr=3e-4,
            gamma=0.99,
            tau=0.005,
            context_len=64,
            batch_size=32,
            buffer_capacity=500,
            updates_per_episode=8,
            prevent_backward=True,
            device='cpu',
            verbose=True,
            training_seed=42,
        )
        agent = RecurrentSACAgent(hidden_dim=dim, model=model, prevent_backward=True)

        # Save weights
        weights_dir = os.path.join('data_out', 'rl_sims', 'recurrent_sac_weights')
        os.makedirs(weights_dir, exist_ok=True)
        wpath = os.path.join(weights_dir, f'recurrent_sac_d{dim}_ptonMac.pt')
        torch.save(model.state_dict(), wpath)
        print(f'  Saved weights: {wpath}')

    train_time = time.time() - t0

    # Evaluate
    print(f'  Evaluating on {len(sessions)} sessions x {N_EVAL_SEEDS} seeds...')
    for i, exp_mom in enumerate(sessions):
        for seed_off in range(N_EVAL_SEEDS):
            total_reward = agent.run_episode(
                adj_mat=adj_mats[i], node_positions=node_pos_list[i],
                start_node=start_nodes[i], rewarded_nodes=rewarded_list[i],
                n_actions=n_actions_list[i], min_rewarded_states=min_rwd_list[i],
                seed=SEED_BASE + seed_off,
            )
            results.append({
                'exp_moment': exp_mom, 'hidden_dim': dim,
                'seed': SEED_BASE + seed_off,
                'total_reward': total_reward,
                'n_actions': n_actions_list[i],
            })

    rpa_vals = [r['total_reward'] / r['n_actions'] for r in results if r['hidden_dim'] == dim]
    print(f'  D={dim}: RPA={np.mean(rpa_vals):.4f} ± {np.std(rpa_vals):.4f}  (train={train_time:.0f}s)')

    # Free memory
    del agent
    gc.collect()

df = pd.DataFrame(results)
df['rpa'] = df['total_reward'] / df['n_actions']

out_path = output_path(f'{ts}_recsac_hidden_dim_sweep.csv')
df.to_csv(out_path, index=False)
print(f'\nSaved {len(df)} rows -> {out_path}')

print('\n--- Summary ---')
summary = df.groupby('hidden_dim')['rpa'].agg(['mean', 'std']).round(4)
print(summary.to_string())
