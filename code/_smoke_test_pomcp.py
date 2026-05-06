#!/usr/bin/env python3
"""
POMCP (omniscient) regression smoke test.

Runs POMCP on 2 sessions with fixed seed and checks that the reward matches
the expected value.  Takes ~30-60s.  Run after every change to advanced_agents.py
to verify that the omniscient POMCP upper-bound has not been corrupted.

Usage:
    source ~/miniconda3/etc/profile.d/conda.sh && conda activate latMaz_RL
    python code/_smoke_test_pomcp.py
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.dirname(__file__))
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from yoked_rl_runner import YokedRLRunner, AlgoConfig
from experiment_config import load_data, build_runner, pomcp_obs_kwargs

# ── Load data ──────────────────────────────────────────────────────
yoking_df, rwd_df, sessions = load_data()

algo = AlgoConfig()
runner = build_runner(yoking_df, rwd_df)

# ── Pick 2 sessions (first and last from filtered list) ───────────
test_sessions = [sessions[0], sessions[-1]]

SEED = 42

# ── Expected values (set to None to discover, then hardcode) ──────
# These were established by running POMCP once; update if the
# omniscient POMCP code is intentionally changed.
EXPECTED = {
    '260124-225237': 200.0,   # was 205.0 — re-baselined 2026-05-03 after Codex audit;
                              #   matches canonical c260316 HPO 'more_sims' config seed-mean
    '251104-230415': 69.0,
}


def run_pomcp_on_session(exp_mom):
    yoke_row = yoking_df[yoking_df['exp_moment'] == exp_mom].iloc[0]
    adj_mat, st_positions = runner._load_maze(
        yoke_row['adj_file'], yoke_row['st_pos_file'])
    rewards, reset_val = runner._get_rewarded_states(exp_mom, len(st_positions))
    n_actions = int(yoke_row['n_states_visited']) - 1
    start_node = int(yoke_row['start_state'])

    reward = runner.run_single(
        model_name='POMCP', adj_mat=adj_mat, st_positions=st_positions,
        start_node=start_node, rewards=rewards, n_actions=n_actions,
        **pomcp_obs_kwargs(),
        min_allowed_rewarded_states=reset_val, seed=SEED)
    return reward


# ── Run ───────────────────────────────────────────────────────────
print(f"POMCP Smoke Test — {len(test_sessions)} sessions, seed={SEED}")
print("=" * 60)

observed = {}
all_pass = True

for session in test_sessions:
    t0 = time.time()
    reward = run_pomcp_on_session(session)
    elapsed = time.time() - t0
    observed[session] = reward

    if EXPECTED is not None:
        expected_r = EXPECTED.get(session)
        if expected_r is not None and reward != expected_r:
            print(f"  FAIL  {session}: got {reward}, expected {expected_r}  ({elapsed:.1f}s)")
            all_pass = False
        else:
            print(f"  PASS  {session}: reward={reward}  ({elapsed:.1f}s)")
    else:
        print(f"  RUN   {session}: reward={reward}  ({elapsed:.1f}s)")

print("=" * 60)
if EXPECTED is None:
    print("Discovery mode — paste this into EXPECTED:\n")
    print("EXPECTED = {")
    for s, r in observed.items():
        print(f"    '{s}': {r},")
    print("}")
else:
    if all_pass:
        print("SMOKE TEST PASSED")
    else:
        print("SMOKE TEST FAILED")
        sys.exit(1)
