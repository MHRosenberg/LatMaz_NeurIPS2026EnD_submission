"""Explicit POMCPbioAlloAgent verification run — all 60 sessions × 5 seeds.

This script is a verification run, not a bug fix. Investigation (see
internal investigation notes (not in public release))
confirmed that:

  1. POMCP_OBS_CONFIG = {'obs_type': 'ego'} has ZERO effect on POMCP_bio.
     Planning agents bypass the gym in run_single(); obs_type is never passed
     to agent.run_episode().

  2. MODEL_CLASSES['POMCP_bio'] = POMCPbioAlloAgent — ptonMac already runs
     the allocentric variant.

  3. The canonical result (c260316-171035_pomcp_bio_FIXED.csv, RPA≈0.209) IS
     from POMCPbioAlloAgent with the rebait double-increment fix.

This script instantiates POMCPbioAlloAgent *directly* (not via the POMCP_bio
key in MODEL_CLASSES) to produce an unambiguous, explicitly documented run.
Expected result: RPA ≈ 0.209, matching the canonical CSV.
"""
import os, sys, time
from datetime import datetime
from copy import deepcopy

import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.dirname(__file__))

from advanced_agents import POMCPbioAlloAgent
from yoked_rl_runner import AlgoConfig
from experiment_config import load_data, build_runner, output_path, DEFAULT_N_REPEATS

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
N_REPEATS = DEFAULT_N_REPEATS   # 5
SEED_BASE  = 42
SAVE_EVERY = 5  # sessions between partial saves

# AlgoConfig defaults for POMCP_bio (max_depth=100, num_sims=500)
AGENT_KWARGS = deepcopy(AlgoConfig().POMCP_bio)

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
yoking_df, rwd_df, sessions = load_data()
runner = build_runner(yoking_df, rwd_df)

ts = 'c' + datetime.now().strftime('%y%m%d-%H%M%S')
partial_path = output_path(f'{ts}_pomcp_bio_allo_PARTIAL.csv')

total = len(sessions) * N_REPEATS
print(f"POMCPbioAlloAgent explicit verification run")
print(f"  Sessions: {len(sessions)}, repeats: {N_REPEATS}, total sims: {total}")
print(f"  Agent kwargs: {AGENT_KWARGS}")
print()

results = []

for i, exp_mom in enumerate(sessions):
    yoke_row = yoking_df[yoking_df['exp_moment'] == exp_mom].iloc[0]
    adj_mat, st_positions = runner._load_maze(yoke_row['adj_file'], yoke_row['st_pos_file'])
    rewards, reset_val = runner._get_rewarded_states(exp_mom, len(st_positions))
    n_actions  = int(yoke_row['n_states_visited']) - 1
    start_node = int(yoke_row['start_state'])

    session_rewards = []
    for rep in range(N_REPEATS):
        seed = SEED_BASE + rep
        np.random.seed(seed)

        t0 = time.time()
        # Instantiate POMCPbioAlloAgent directly — no MODEL_CLASSES lookup
        agent = POMCPbioAlloAgent(**deepcopy(AGENT_KWARGS))
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
        del agent
        elapsed = time.time() - t0

        results.append({
            'exp_moment': exp_mom,
            'model': 'POMCP_bio_allo_explicit',
            'seed': seed,
            'total_reward': total_reward,
            'n_actions': n_actions,
            'animal_ID': yoke_row.get('animal_ID', ''),
        })
        session_rewards.append(total_reward)

    done = (i + 1) * N_REPEATS
    session_rpa = np.mean(session_rewards) / n_actions
    print(f"  [{done:4d}/{total}] {exp_mom}: RPA={session_rpa:.3f} ({elapsed:.1f}s/sim)")

    if (i + 1) % SAVE_EVERY == 0:
        pd.DataFrame(results).to_csv(partial_path, index=False)

# ---------------------------------------------------------------------------
# Final save
# ---------------------------------------------------------------------------
df = pd.DataFrame(results)
out_path = output_path(f'{ts}_pomcp_bio_allo_explicit.csv')
df.to_csv(out_path, index=False)
print(f"\nSaved {len(df)} rows to {out_path}")

df['rpa'] = df['total_reward'] / df['n_actions']
mean_rpa = df['rpa'].mean()
print(f"Overall RPA: {mean_rpa:.3f} ± {df['rpa'].std():.3f}")

# Compare with canonical
canon_path = output_path('c260316-171035_pomcp_bio_FIXED.csv')
if os.path.exists(canon_path):
    canon = pd.read_csv(canon_path)
    canon['rpa'] = canon['total_reward'] / canon['n_actions']
    canon_rpa = canon['rpa'].mean()
    print(f"Canonical RPA (c260316): {canon_rpa:.3f}")
    print(f"Delta: {mean_rpa - canon_rpa:+.4f}  (expect ≈ 0)")
else:
    print("Canonical CSV not found for comparison.")

# Clean up partial
if os.path.exists(partial_path):
    os.remove(partial_path)
