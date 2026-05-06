"""
260505_rerun_inf_repoke_prevent_reverse.py

Re-run all 8 RL agents on the inf-repoke subset of rerun-452 with
prevent_reverse=True, to bring the agent's action availability into
per-action parity with the mouse's apparatus-enforced sensor timeout
(no-revisit on the same port).

This addresses the candidate refinement flagged in paper Appendix C:
    "Re-running the canonical benchmark with prevent_reverse=True to
     bring the agent's action availability into per-action parity with
     the mouse's apparatus-enforced sensor timeout is a candidate
     refinement for future iterations of this benchmark."

Scope:
  - 48 inf-repoke sessions (a030: 7, a031: 14, a033: 27) within
    rerun-452's 452-session cohort.
  - Joined via data_out/rl_sims/c260311-200037_session_repoke_mapping.csv
    on exp_moment, where repoke_interval == 'inf' indicates the apparatus
    enforces no-revisit on the most-recently poked port (mouse cannot
    issue an immediate-reverse action; RL agent should not either).
  - 8 RL agents x 5 seeds x 48 sessions = 1,920 cells.
  - With --workers 4: ~30-60 min wall-clock on M3 Max.

Output:
  data_out/rl_sims/c{ts}_inf_repoke_prevent_reverse_8agents.csv

Usage:
  conda activate latMaz_RL
  python code/260505_rerun_inf_repoke_prevent_reverse.py --workers 4
"""
from __future__ import annotations
import argparse
import multiprocessing as mp
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path('<REPO_ROOT>')
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(PROJECT_ROOT / 'code'))

REQ_AGENTS = ['DQN', 'DRQN_seq', 'DRQN_rand', 'A2C', 'PPO', 'QRDQN', 'TRPO', 'RecurrentPPO']
SEEDS = list(range(5))
RERUN452_CSV = PROJECT_ROOT / 'data_released' / 'results' / 'c260505-120513_rerun452_corrected_8agents.csv'
REPOKE_CSV   = PROJECT_ROOT / 'data_out' / 'rl_sims' / 'c260311-200037_session_repoke_mapping.csv'


def run_one_cell_prevent_reverse(args):
    """Worker — same as run_one_cell from 260503_run_cohort_HPObest.py
    but with prevent_reverse=True passed to runner.run_single()."""
    agent, exp_mom, seed = args
    try:
        from experiment_config import load_data, build_runner, obs_kwargs, apply_best_configs
        yoking_df, rwd_df, _ = load_data()
        runner = build_runner(yoking_df, rwd_df)
        runner.algo_config = apply_best_configs([agent])
        yoke_row = yoking_df[yoking_df['exp_moment'] == exp_mom].iloc[0]
        adj_mat, st_positions = runner._load_maze(
            yoke_row['adj_file'], yoke_row['st_pos_file'])
        rewards, reset_val = runner._get_rewarded_states(exp_mom, len(st_positions))
        n_actions = int(yoke_row['n_states_visited']) - 1
        start_node = int(yoke_row['start_state'])
        t0 = time.time()
        total_reward = runner.run_single(
            model_name=agent, adj_mat=adj_mat, st_positions=st_positions,
            start_node=start_node, rewards=rewards, n_actions=n_actions,
            **obs_kwargs(),
            min_allowed_rewarded_states=int(reset_val),
            seed=seed,
            prevent_reverse=True,            # the only difference vs canonical run
        )
        elapsed = time.time() - t0
        return {
            'exp_moment': exp_mom, 'animal_ID': int(yoke_row['animal_ID']),
            'model': agent, 'seed': seed,
            'total_reward': float(total_reward), 'n_actions': n_actions,
            'rpa': float(total_reward) / max(n_actions, 1),
            'elapsed_s': round(elapsed, 2),
            'reset_val': int(reset_val), 'n_init_rewards': int((rewards > 0).sum()),
            'adj_file': yoke_row['adj_file'],
            'prevent_reverse': True,
            'status': 'ok',
            'timestamp': datetime.now().isoformat(timespec='seconds'),
        }
    except Exception as exc:
        import traceback
        return {
            'exp_moment': exp_mom, 'animal_ID': -1, 'model': agent, 'seed': seed,
            'total_reward': float('nan'), 'n_actions': -1, 'rpa': float('nan'),
            'elapsed_s': -1.0, 'reset_val': -1, 'n_init_rewards': -1,
            'adj_file': '?', 'prevent_reverse': True,
            'status': f'error:{type(exc).__name__}',
            'timestamp': datetime.now().isoformat(timespec='seconds'),
        }


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--output', type=Path, default=None)
    p.add_argument('--limit', type=int, default=None,
                   help='Smoke test: cap cells (e.g. --limit 10).')
    args = p.parse_args()

    if args.output is None:
        ts = 'c' + datetime.now().strftime('%y%m%d-%H%M%S')
        args.output = PROJECT_ROOT / 'data_out' / 'rl_sims' / f'{ts}_inf_repoke_prevent_reverse_8agents.csv'

    # Identify inf-repoke sessions inside rerun-452 cohort
    repoke = pd.read_csv(REPOKE_CSV, dtype={'animal_id': str})
    inf_sessions = set(repoke.loc[repoke['repoke_interval'].astype(str) == 'inf', 'exp_moment'])
    rerun452 = pd.read_csv(RERUN452_CSV, dtype={'animal_ID': str})
    rerun452_sessions = set(rerun452['exp_moment'].unique())
    target_sessions = sorted(inf_sessions & rerun452_sessions)

    cells = [(agent, sess, seed) for agent in REQ_AGENTS for sess in target_sessions for seed in SEEDS]
    if args.limit is not None:
        cells = cells[:args.limit]

    print(f'rerun-452 inf-repoke prevent_reverse=True sweep')
    print(f'  Sessions:   {len(target_sessions)}  (out of {len(inf_sessions)} inf-repoke total)')
    print(f'  Agents:     {len(REQ_AGENTS)}  ({", ".join(REQ_AGENTS)})')
    print(f'  Seeds:      {len(SEEDS)}')
    print(f'  Total cells:{len(cells)}')
    print(f'  Workers:    {args.workers}')
    print(f'  Output:     {args.output}')
    print()

    t_start = time.time()
    results = []

    if args.workers == 1:
        for i, c in enumerate(cells):
            results.append(run_one_cell_prevent_reverse(c))
            if (i + 1) % 50 == 0:
                pd.DataFrame(results).to_csv(args.output, index=False)
                print(f'  [{i+1}/{len(cells)}] {time.time()-t_start:.0f}s elapsed; CSV checkpoint')
    else:
        with mp.Pool(args.workers) as pool:
            for i, r in enumerate(pool.imap_unordered(run_one_cell_prevent_reverse, cells)):
                results.append(r)
                if (i + 1) % 50 == 0:
                    pd.DataFrame(results).to_csv(args.output, index=False)
                    elapsed = time.time() - t_start
                    eta = elapsed / (i + 1) * (len(cells) - i - 1)
                    print(f'  [{i+1}/{len(cells)}] {elapsed:.0f}s elapsed; ETA {eta:.0f}s')

    pd.DataFrame(results).to_csv(args.output, index=False)
    elapsed = time.time() - t_start
    df = pd.DataFrame(results)
    n_ok = (df['status'] == 'ok').sum()
    print()
    print(f'Done in {elapsed:.0f}s ({elapsed/60:.1f} min). {n_ok}/{len(cells)} cells ok.')
    print(f'Output: {args.output}')


if __name__ == '__main__':
    main()
