"""Run FullFwd, NoBkFullFwd, and CloningAgent variants on 60 canonical sessions.

Produces canonical benchmark values for the five Table 1 rows that oTmac
imported from the 2025 NeurIPS workshop paper without backing code or data.
See GitHub issue #18 and docs/outbox/260413_cloning_agents_no_backing_data.md.

Agents implemented in intermediate_agents.py (sections 5 and 6):
  FullFwdAgent       — prefer Forward, else random valid
  NoBkFullFwdAgent   — no backward + prefer Forward, else random valid
  CloningAgent       — sample from per-session mouse marginal, restricted to
                       valid actions; frame in {'ego', 'allo_real', 'allo_latent'}

Each agent × session × N_REPEATS seeds. Per-session prob_dict columns parsed
from the yoking CSV using ast.literal_eval.
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

from intermediate_agents import FullFwdAgent, NoBkFullFwdAgent, CloningAgent
from advanced_agents import ALLO_TO_EGO
from utils_latMaz import displacement_to_compass_heading, allo_actions_one_hot_dict
from experiment_config import load_data, build_runner, output_path, DEFAULT_N_REPEATS


# ---------------------------------------------------------------------------
# Helper: compute marginal prob_dict from trajectory (used for sessions where
# the precomputed column is missing/NaN).
# For ego: track heading (starts N=0); convert allo→ego at each step.
# For allo (real or latent): use position displacements directly.
# Note: both allo variants are computed from the latent maze positions (the
# only coordinate frame available in the yoking df). For the ~19 sessions
# missing precomputed prob_dicts (Jan-Feb 2026), allo_real≈allo_latent.
# ---------------------------------------------------------------------------

def _compute_prob_dict(states_visited, node_positions):
    """Return ``{(1,1): marginal_dict}`` for ego and allo frames.

    Returns a tuple ``(prob_dict_ego, prob_dict_allo_latent)`` where each is
    ``{(1,1): {action_letter: probability}}``.
    """
    ego_counts = {}
    allo_counts = {}
    ego_letters = {0: 'F', 1: 'B', 2: 'L', 3: 'R'}
    heading = 0  # start facing N

    for i in range(len(states_visited) - 1):
        na, nb = states_visited[i], states_visited[i + 1]
        disp = node_positions[nb] - node_positions[na]
        compass = displacement_to_compass_heading(disp)
        allo_idx = allo_actions_one_hot_dict[compass]
        ego_idx  = ALLO_TO_EGO[heading][allo_idx]
        heading  = allo_idx

        ego_counts[ego_letters[ego_idx]] = ego_counts.get(ego_letters[ego_idx], 0) + 1
        allo_counts[compass] = allo_counts.get(compass, 0) + 1

    total_ego  = sum(ego_counts.values()) or 1
    total_allo = sum(allo_counts.values()) or 1
    pd_ego  = {(1, 1): {k: v / total_ego  for k, v in ego_counts.items()}}
    pd_allo = {(1, 1): {k: v / total_allo for k, v in allo_counts.items()}}
    return pd_ego, pd_allo


def _parse_prob_dict(raw):
    """Parse prob_dict column value; return dict or None."""
    if isinstance(raw, float):   # NaN
        return None
    try:
        return ast.literal_eval(raw)
    except Exception:
        return None

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
N_REPEATS = DEFAULT_N_REPEATS   # 5
SEED_BASE  = 42
SAVE_EVERY = 10  # sessions between partial saves

AGENTS_NO_PROB = [
    ('FullFwd',       FullFwdAgent()),
    ('NoBkFullFwd',   NoBkFullFwdAgent()),
]
CLONE_FRAMES = [
    ('Clone_ego',         'prob_dict_ego',       'ego'),
    ('Clone_allo_real',   'prob_dict_allo_real',  'allo_real'),
    ('Clone_allo_latent', 'prob_dict_allo_latent','allo_latent'),
]

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
yoking_df, rwd_df, sessions = load_data()
runner = build_runner(yoking_df, rwd_df)

print(f"Sessions: {len(sessions)}, repeats: {N_REPEATS}")
print(f"Agents: {[n for n,_ in AGENTS_NO_PROB] + [n for n,_,_ in CLONE_FRAMES]}")
print()

ts = 'c' + datetime.now().strftime('%y%m%d-%H%M%S')
partial_path = output_path(f'{ts}_cloning_agents_PARTIAL.csv')

results = []

for i, exp_mom in enumerate(sessions):
    yoke_row = yoking_df[yoking_df['exp_moment'] == exp_mom].iloc[0]
    adj_mat, st_positions = runner._load_maze(yoke_row['adj_file'], yoke_row['st_pos_file'])
    rewards, reset_val = runner._get_rewarded_states(exp_mom, len(st_positions))
    n_actions  = int(yoke_row['n_states_visited']) - 1
    start_node = int(yoke_row['start_state'])
    animal_id  = yoke_row.get('animal_ID', '')

    # ---- heuristic agents (no prob_dict needed) ----------------------------
    for agent_name, agent in AGENTS_NO_PROB:
        session_rewards = []
        for rep in range(N_REPEATS):
            seed = SEED_BASE + rep
            t0 = time.time()
            total_reward = agent.run_episode(
                adj_mat=adj_mat,
                node_positions=st_positions,
                start_node=start_node,
                rewarded_nodes=rewards,
                n_actions=n_actions,
                min_rewarded_states=reset_val,
                seed=seed,
                prevent_reverse=False,
            )
            elapsed = time.time() - t0
            results.append({
                'exp_moment': exp_mom,
                'model': agent_name,
                'seed': seed,
                'total_reward': total_reward,
                'n_actions': n_actions,
                'animal_ID': animal_id,
            })
            session_rewards.append(total_reward)
        rpa = np.mean(session_rewards) / n_actions
        print(f"  [{i+1:3d}/{len(sessions)}] {agent_name:15s} {exp_mom}: RPA={rpa:.3f} ({elapsed:.2f}s/sim)")

    # ---- cloning agents (per-session prob_dict) ----------------------------
    # Precompute fallback from trajectory for sessions missing precomputed dicts
    sv_raw = yoke_row.get('states_visited', '[]')
    states_visited = ast.literal_eval(sv_raw) if isinstance(sv_raw, str) else sv_raw

    fallback_ego, fallback_allo = None, None

    for agent_name, col, frame in CLONE_FRAMES:
        prob_dict = _parse_prob_dict(yoke_row.get(col))
        if prob_dict is None:
            # Compute on-the-fly from trajectory
            if fallback_ego is None:
                fallback_ego, fallback_allo = _compute_prob_dict(states_visited, st_positions)
            prob_dict = fallback_ego if frame == 'ego' else fallback_allo

        clone = CloningAgent(prob_dict=prob_dict, frame=frame)
        session_rewards = []
        for rep in range(N_REPEATS):
            seed = SEED_BASE + rep
            t0 = time.time()
            total_reward = clone.run_episode(
                adj_mat=adj_mat,
                node_positions=st_positions,
                start_node=start_node,
                rewarded_nodes=rewards,
                n_actions=n_actions,
                min_rewarded_states=reset_val,
                seed=seed,
                prevent_reverse=False,
            )
            elapsed = time.time() - t0
            results.append({
                'exp_moment': exp_mom,
                'model': agent_name,
                'seed': seed,
                'total_reward': total_reward,
                'n_actions': n_actions,
                'animal_ID': animal_id,
            })
            session_rewards.append(total_reward)
        rpa = np.mean(session_rewards) / n_actions
        print(f"  [{i+1:3d}/{len(sessions)}] {agent_name:15s} {exp_mom}: RPA={rpa:.3f} ({elapsed:.2f}s/sim)")

    if (i + 1) % SAVE_EVERY == 0:
        pd.DataFrame(results).to_csv(partial_path, index=False)
        print(f"  [partial save: {len(results)} rows]")

# ---------------------------------------------------------------------------
# Final save + summary
# ---------------------------------------------------------------------------
df = pd.DataFrame(results)
out_path = output_path(f'{ts}_cloning_agents.csv')
df.to_csv(out_path, index=False)
print(f"\nSaved {len(df)} rows → {out_path}")

df['rpa'] = df['total_reward'] / df['n_actions']
print("\n--- Per-agent RPA summary ---")
summary = df.groupby('model')['rpa'].agg(['mean', 'std']).round(4)
print(summary.to_string())

# Clean up partial
if os.path.exists(partial_path):
    os.remove(partial_path)
