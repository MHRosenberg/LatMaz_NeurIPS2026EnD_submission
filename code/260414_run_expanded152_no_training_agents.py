#!/usr/bin/env python3
"""
260414_run_expanded152_no_training_agents.py

Run all no-training agents on the 92 older sessions that, combined with the
canonical 60-session set, make up the 152-session expanded multi-animal set.

The 92 older sessions cover animals: a001 (10), a002 (3), a003 (2), a029 (23),
a030 (19), a185 (13), a188 (22).  Session dates: Aug 2024 – Mar 2025.

No-training agents (can be applied to new sessions without retraining):
  - POMCP (planning, deterministic)
  - POMCP_bio (same)
  - FSC_bio (fixed-policy clique-based)
  - VarMarkov, OOI (fixed-policy)
  - Greedy_oracle, Random_forward, Random (heuristics)
  - FullFwd, NoBkFullFwd (heuristics)
  - Clone_ego, Clone_allo_real, Clone_allo_latent (cloning)

NOT run here (require training on specific sessions):
  - DQN, DRQN_seq, DRQN_rand, QRDQN, PPO, A2C, TRPO, RecurrentPPO (RL training)
  - RecurrentSAC (meta-RL; requires full training loop)

Output:
  data_out/rl_sims/c{ts}_expanded92_no_training_agents.csv
  (combine with canonical CSVs to produce 152-session per-animal figures)

Usage:
    conda activate latMaz_RL
    python code/260414_run_expanded152_no_training_agents.py
"""
import ast
import os
import sys
import time
from datetime import datetime

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.dirname(__file__))

from intermediate_agents import FullFwdAgent, NoBkFullFwdAgent, CloningAgent, FSCBioAgent, VarMarkovAgent, OOIAgent
from advanced_agents import ALLO_TO_EGO
from utils_latMaz import displacement_to_compass_heading, allo_actions_one_hot_dict
from experiment_config import (
    load_yoking_df, load_reward_df, build_runner, pomcp_obs_kwargs, output_path,
    DEFAULT_N_REPEATS, run_random_agent, run_forward_random_agent, run_greedy_agent,
)
from yoked_rl_runner import GraphMazeEnv, AlgoConfig

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
N_REPEATS  = DEFAULT_N_REPEATS   # 5
SEED_BASE  = 42
SAVE_EVERY = 20

# ---------------------------------------------------------------------------
# Load the 92 older sessions
# ---------------------------------------------------------------------------
FBR_CSV = 'data_out/rl_sims/c260331-182639_fbr_152sessions.csv'
assert os.path.exists(FBR_CSV), f'FBR 152-session CSV not found: {FBR_CSV}'

fbr_sess = set(pd.read_csv(FBR_CSV)['exp_moment'].unique())
yoking_df = load_yoking_df()
rwd_df    = load_reward_df()

# Canonical 60-session set
from experiment_config import filter_sessions
sess60 = set(filter_sessions(yoking_df))

# Older 92 sessions = FBR 152 minus canonical 60
older92 = fbr_sess - sess60
sessions = sorted(older92)   # sorted for reproducibility

print(f'Canonical 60 sessions: {len(sess60)}')
print(f'FBR 152 sessions:      {len(fbr_sess)}')
print(f'Older 92 sessions to run: {len(sessions)}')

# Verify all older sessions are in yoking_df
missing_in_yoke = [s for s in sessions if s not in yoking_df['exp_moment'].values]
if missing_in_yoke:
    raise RuntimeError(f'{len(missing_in_yoke)} sessions not in yoking_df: {missing_in_yoke[:5]}')

runner = build_runner(yoking_df, rwd_df)

# POMCP config
algo = AlgoConfig()
algo.POMCP = {'max_depth': 80, 'num_sims': 1000, 'exploration_const': 20.0, 'discount': 0.99}
runner.algo_config = algo
_pomcp_obs = pomcp_obs_kwargs()

ts = 'c' + datetime.now().strftime('%y%m%d-%H%M%S')
partial_path = output_path(f'{ts}_expanded92_no_training_agents_PARTIAL.csv')

# ---------------------------------------------------------------------------
# Helper: parse prob_dict column
# ---------------------------------------------------------------------------
def _parse_prob_dict(raw):
    if isinstance(raw, float): return None
    try:    return ast.literal_eval(raw)
    except: return None

def _compute_prob_dict(states_visited, node_positions):
    """Fallback: compute ego/allo marginals from trajectory."""
    ego_counts, allo_counts = {}, {}
    ego_letters = {0: 'F', 1: 'B', 2: 'L', 3: 'R'}
    heading = 0
    for i in range(len(states_visited) - 1):
        na, nb = states_visited[i], states_visited[i+1]
        disp = node_positions[nb] - node_positions[na]
        compass = displacement_to_compass_heading(disp)
        allo_idx = allo_actions_one_hot_dict[compass]
        ego_idx  = ALLO_TO_EGO[heading][allo_idx]
        heading  = allo_idx
        ego_counts[ego_letters[ego_idx]] = ego_counts.get(ego_letters[ego_idx], 0) + 1
        allo_counts[compass] = allo_counts.get(compass, 0) + 1
    t_ego  = sum(ego_counts.values()) or 1
    t_allo = sum(allo_counts.values()) or 1
    pd_ego  = {(1,1): {k: v/t_ego  for k,v in ego_counts.items()}}
    pd_allo = {(1,1): {k: v/t_allo for k,v in allo_counts.items()}}
    return pd_ego, pd_allo

# ---------------------------------------------------------------------------
# Agent groups
# ---------------------------------------------------------------------------

# Group A: gym-based heuristics (Random, Random_forward, Greedy_oracle)
GYM_AGENTS = [
    ('Random',         run_random_agent),
    ('Random_forward', run_forward_random_agent),
    ('Greedy_oracle',  run_greedy_agent),
]

# Group B: direct run_episode heuristics (no gym env needed)
DIRECT_AGENTS = [
    ('FullFwd',      FullFwdAgent()),
    ('NoBkFullFwd',  NoBkFullFwdAgent()),
]

# Group C: cloning agents (need per-session prob_dict)
CLONE_FRAMES = [
    ('Clone_ego',         'prob_dict_ego',       'ego'),
    ('Clone_allo_real',   'prob_dict_allo_real',  'allo_real'),
    ('Clone_allo_latent', 'prob_dict_allo_latent','allo_latent'),
]

# Group D: intermediate agents
INTERMEDIATE_AGENTS = [
    ('VarMarkov', VarMarkovAgent()),
    ('OOI',       OOIAgent()),
    ('FSC_bio',   FSCBioAgent()),
]

# Group E: POMCP variants (run_single via runner)
POMCP_MODELS = ['POMCP', 'POMCP_bio']

# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
results = []
n_total_sessions = len(sessions)

for i, exp_mom in enumerate(sessions):
    yoke_row = yoking_df[yoking_df['exp_moment'] == exp_mom].iloc[0]
    adj_mat, st_positions = runner._load_maze(yoke_row['adj_file'], yoke_row['st_pos_file'])
    rewards, reset_val = runner._get_rewarded_states(exp_mom, len(st_positions))
    n_actions  = int(yoke_row['n_states_visited']) - 1
    start_node = int(yoke_row['start_state'])
    animal_id  = str(yoke_row.get('animal_ID', ''))

    def _rec(model, rep, total_reward, elapsed=None):
        results.append({
            'exp_moment': exp_mom,
            'animal_ID':  animal_id,
            'model':      model,
            'seed':       SEED_BASE + rep,
            'repeat_idx': rep,
            'total_reward': total_reward,
            'n_actions':  n_actions,
        })

    # ---- Group A: gym-based heuristics ----
    def _make_env():
        return GraphMazeEnv(
            adjacency_matrix=adj_mat,
            node_xy_coordinate_list=st_positions,
            start_node=start_node,
            rwd_per_node_initial=rewards,
            n_valid_actions_per_episode=n_actions,
            action_type='ego', obs_type='ego',
            min_allowed_rewarded_states=reset_val,
        )
    for agent_name, agent_fn in GYM_AGENTS:
        for rep in range(N_REPEATS):
            env = _make_env()
            total_reward = agent_fn(env, SEED_BASE + rep)
            _rec(agent_name, rep, total_reward)
            del env

    # ---- Group B: direct heuristics ----
    for agent_name, agent in DIRECT_AGENTS:
        for rep in range(N_REPEATS):
            t0 = time.time()
            total_reward = agent.run_episode(
                adj_mat=adj_mat, node_positions=st_positions,
                start_node=start_node, rewarded_nodes=rewards,
                n_actions=n_actions, min_rewarded_states=reset_val,
                seed=SEED_BASE + rep, prevent_reverse=False,
            )
            _rec(agent_name, rep, total_reward)

    # ---- Group C: cloning agents ----
    sv_raw = yoke_row.get('states_visited', '[]')
    states_visited = ast.literal_eval(sv_raw) if isinstance(sv_raw, str) else sv_raw
    fallback_ego, fallback_allo = None, None

    for agent_name, col, frame in CLONE_FRAMES:
        prob_dict = _parse_prob_dict(yoke_row.get(col))
        if prob_dict is None:
            if fallback_ego is None:
                fallback_ego, fallback_allo = _compute_prob_dict(states_visited, st_positions)
            prob_dict = fallback_ego if frame == 'ego' else fallback_allo
        clone = CloningAgent(prob_dict=prob_dict, frame=frame)
        for rep in range(N_REPEATS):
            total_reward = clone.run_episode(
                adj_mat=adj_mat, node_positions=st_positions,
                start_node=start_node, rewarded_nodes=rewards,
                n_actions=n_actions, min_rewarded_states=reset_val,
                seed=SEED_BASE + rep, prevent_reverse=False,
            )
            _rec(agent_name, rep, total_reward)

    # ---- Group D: intermediate agents ----
    for agent_name, agent in INTERMEDIATE_AGENTS:
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
            _rec(agent_name, rep, total_reward)

    # ---- Group E: POMCP variants ----
    for model_name in POMCP_MODELS:
        for rep in range(N_REPEATS):
            t0 = time.time()
            try:
                total_reward = runner.run_single(
                    model_name=model_name,
                    adj_mat=adj_mat, st_positions=st_positions,
                    start_node=start_node, rewards=rewards,
                    n_actions=n_actions,
                    **_pomcp_obs,
                    min_allowed_rewarded_states=reset_val,
                    seed=SEED_BASE + rep,
                )
            except Exception as e:
                print(f'  WARNING: {model_name} failed on {exp_mom} rep {rep}: {e}')
                total_reward = 0
            elapsed = time.time() - t0
            _rec(model_name, rep, total_reward)

    # Progress
    n_agents = len(GYM_AGENTS) + len(DIRECT_AGENTS) + len(CLONE_FRAMES) + len(INTERMEDIATE_AGENTS) + len(POMCP_MODELS)
    sess_results = [r for r in results if r['exp_moment'] == exp_mom]
    rpa_by_model = {}
    for r in sess_results:
        rpa_by_model.setdefault(r['model'], []).append(r['total_reward'])
    rpa_str = '  '.join(f"{m}={np.mean(v)/n_actions:.3f}" for m,v in sorted(rpa_by_model.items()))
    print(f"[{i+1:3d}/{n_total_sessions}] {exp_mom} (a{animal_id}): {rpa_str}")

    if (i+1) % SAVE_EVERY == 0:
        pd.DataFrame(results).to_csv(partial_path, index=False)
        print(f'  [partial save: {len(results)} rows]')

# ---------------------------------------------------------------------------
# Final save
# ---------------------------------------------------------------------------
df = pd.DataFrame(results)
df['rpa'] = df['total_reward'] / df['n_actions']

out_path = output_path(f'{ts}_expanded92_no_training_agents.csv')
df.to_csv(out_path, index=False)
print(f'\nSaved {len(df)} rows → {out_path}')

print('\n--- Per-agent RPA summary (92 older sessions) ---')
summary = df.groupby('model')['rpa'].agg(['mean', 'std', 'count']).round(4)
print(summary.to_string())

if os.path.exists(partial_path):
    os.remove(partial_path)

print('\n=== MISSING (require training — not run on expanded sessions) ===')
print('  DQN, DRQN_seq, DRQN_rand, QRDQN, PPO, A2C, TRPO, RecurrentPPO')
print('  RecurrentSAC (meta-RL, requires full training loop per session)')
print('  To add these: retrain each model on the 92 older sessions individually.')
