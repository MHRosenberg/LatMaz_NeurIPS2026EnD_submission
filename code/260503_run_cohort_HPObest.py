"""
260503_run_cohort_HPObest.py

Run HPO-best configs of selected RL agents on the 6-animal expanded
cohort (137 sessions × 5 seeds), using YokedRLRunner. Output: one CSV
per worker, merged at the end (or per-row append if --workers=1).

Designed to run identically on:
  - 16-core Mac (`--workers 4` to leave 12 free for editing)
  - University cluster login node (`--workers 16` or higher)

Resume support: skips (agent, exp_moment, seed) tuples already in
the output CSV.

Usage:
  conda activate latMaz_RL
  cd code/
  python 260503_run_cohort_HPObest.py \\
      --agents DQN \\
      --workers 4 \\
      --output ../../data_out/rl_sims/c{ts}_cohort137_DQN_HPObest.csv

Or pass multiple agents:
  python 260503_run_cohort_HPObest.py --agents DQN DRQN_seq QRDQN

Cohort definition (per Author 2026-05-03 decision):
  6 animals × 137 sessions: a033(36)+a031(24)+a029(23)+a188(22)+a030(19)+a185(13).
  Drops a001/a002/a003 (n_sess ≤ 10).
"""
from __future__ import annotations
import argparse
import os
import sys
import time
import multiprocessing as mp
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path('<REPO_ROOT>')
CODE_DIR = Path(__file__).parent
sys.path.insert(0, str(CODE_DIR))
sys.path.insert(0, str(PROJECT_ROOT / 'code'))
sys.path.insert(0, str(PROJECT_ROOT))

# ---- Cohort definition ----
# yoking_df['animal_ID'] is stored as zero-padded string per load_yoking_df dtype.
COHORT_6_ANIMALS = {'031', '033', '029', '188', '030', '185'}
CANONICAL_2_ANIMALS = {'031', '033'}

# ---- Defaults ----
DEFAULT_AGENTS = ['DQN', 'DRQN_seq', 'QRDQN', 'PPO', 'A2C', 'TRPO',
                   'RecurrentPPO', 'DRQN_rand']  # NO RecurrentSAC (heavy)
DEFAULT_SEEDS = list(range(5))


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    p.add_argument('--agents', nargs='+', default=['DQN'],
                    help='HPO-best RL agents to run. Choices: ' + ' '.join(DEFAULT_AGENTS))
    p.add_argument('--seeds', nargs='+', type=int, default=DEFAULT_SEEDS)
    p.add_argument('--workers', type=int, default=4,
                    help='Parallel worker processes. Mac: 4 (leaves 12 cores free).')
    p.add_argument('--output', type=Path, default=None,
                    help='CSV path. Default: data_out/rl_sims/c{ts}_cohort137_HPObest.csv')
    p.add_argument('--cohort-animals', nargs='+', default=list(COHORT_6_ANIMALS),
                    help='Animal IDs (zero-padded strings, e.g. 029 188) to include')
    p.add_argument('--skip-canonical', action='store_true',
                    help='Skip canonical 60 (re-runs only the cohort expansion sessions)')
    p.add_argument('--limit', type=int, default=None,
                    help='Optional: limit total (agent, session, seed) tuples for smoke test')
    return p.parse_args()


def get_cells(agents, seeds, cohort_animals, skip_canonical=False):
    """Build the list of (agent, exp_moment, seed) tuples to run."""
    from experiment_config import load_data
    yoking_df, rwd_df, sessions = load_data()
    cohort_animals = {str(a).zfill(3) for a in cohort_animals}  # normalise to '029'-style
    extra_sessions = []
    for aid in cohort_animals:
        if aid in CANONICAL_2_ANIMALS:
            continue
        aid_sub = yoking_df[yoking_df['animal_ID'] == aid]
        valid = aid_sub[pd.to_numeric(aid_sub['n_states_visited'], errors='coerce') > 50]
        extra_sessions.extend(valid['exp_moment'].dropna().tolist())
    if skip_canonical:
        cohort_sessions = sorted(set(extra_sessions))
        print(f'Cohort sessions (expansion-only): {len(cohort_sessions)} '
              f'(skipping canonical 60)')
    else:
        cohort_sessions = sorted(set(list(sessions) + extra_sessions))
        print(f'Cohort sessions: {len(cohort_sessions)} total '
              f'({len(sessions)} canonical + '
              f'{len(set(extra_sessions) - set(sessions))} new expansion)')

    cells = []
    for agent in agents:
        for exp_mom in cohort_sessions:
            for seed in seeds:
                cells.append((agent, exp_mom, int(seed)))
    return cells, yoking_df, rwd_df


def run_one_cell(args):
    """Worker function — runs one (agent, session, seed) sim."""
    agent, exp_mom, seed = args
    try:
        from experiment_config import load_data, build_runner, obs_kwargs, apply_best_configs
        yoking_df, rwd_df, _ = load_data()
        runner = build_runner(yoking_df, rwd_df)
        runner.algo_config = apply_best_configs([agent])  # apply HPO-best (FIX 260505: was runner.algo, a no-op)
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
            min_allowed_rewarded_states=int(reset_val), seed=seed)
        elapsed = time.time() - t0
        return {'exp_moment': exp_mom, 'animal_ID': int(yoke_row['animal_ID']),
                'model': agent, 'seed': seed, 'total_reward': float(total_reward),
                'n_actions': n_actions, 'rpa': float(total_reward) / max(n_actions, 1),
                'elapsed_s': round(elapsed, 2),
                'reset_val': int(reset_val), 'n_init_rewards': int((rewards > 0).sum()),
                'adj_file': yoke_row['adj_file'], 'status': 'ok',
                'timestamp': datetime.now().isoformat(timespec='seconds')}
    except Exception as exc:
        import traceback
        return {'exp_moment': exp_mom, 'animal_ID': -1, 'model': agent, 'seed': seed,
                'total_reward': float('nan'), 'n_actions': -1, 'rpa': float('nan'),
                'elapsed_s': float('nan'), 'reset_val': -1, 'n_init_rewards': -1,
                'adj_file': '', 'status': f'error: {repr(exc)[:200]}',
                'timestamp': datetime.now().isoformat(timespec='seconds')}


def main():
    args = parse_args()
    if args.output is None:
        ts = 'c' + datetime.now().strftime('%y%m%d-%H%M%S')
        args.output = PROJECT_ROOT / 'data_out' / 'rl_sims' / f'{ts}_cohort137_HPObest.csv'
    args.output.parent.mkdir(parents=True, exist_ok=True)
    print(f'Output: {args.output}')
    print(f'Agents: {args.agents}')
    print(f'Seeds: {args.seeds}')
    print(f'Workers: {args.workers}')

    cells, _, _ = get_cells(args.agents, args.seeds, args.cohort_animals, args.skip_canonical)
    if args.limit:
        cells = cells[:args.limit]
        print(f'Limited to {args.limit} cells (smoke test mode)')
    print(f'Total (agent, session, seed) cells: {len(cells)}')

    # Resume: skip cells already in output CSV
    if args.output.exists():
        done = pd.read_csv(args.output)
        done_keys = set(zip(done['model'], done['exp_moment'], done['seed']))
        cells = [c for c in cells if c not in done_keys]
        print(f'Resume: {len(done_keys)} already done; {len(cells)} remaining')

    if not cells:
        print('Nothing to do.')
        return

    t_start = time.time()
    results = []
    if args.workers == 1:
        for i, c in enumerate(cells):
            r = run_one_cell(c)
            results.append(r)
            if (i + 1) % 5 == 0:
                pd.DataFrame(results).to_csv(args.output, mode='a',
                    header=not args.output.exists() or args.output.stat().st_size == 0,
                    index=False)
                results = []
                print(f'  [{i+1}/{len(cells)}] elapsed={time.time()-t_start:.0f}s '
                       f'rate={ (i+1)/(time.time()-t_start):.2f}/s')
    else:
        with mp.Pool(args.workers) as pool:
            for i, r in enumerate(pool.imap_unordered(run_one_cell, cells, chunksize=1)):
                results.append(r)
                if (i + 1) % 10 == 0 or i == 0:
                    df = pd.DataFrame(results)
                    df.to_csv(args.output, mode='a',
                        header=not args.output.exists() or args.output.stat().st_size == 0,
                        index=False)
                    results = []
                    elapsed = time.time() - t_start
                    eta_s = (len(cells) - (i + 1)) / max((i + 1) / elapsed, 1e-6)
                    print(f'  [{i+1}/{len(cells)}] elapsed={elapsed:.0f}s '
                           f'rate={(i+1)/elapsed:.2f}/s ETA={eta_s/60:.1f}min  agent={r.get("model")} '
                           f'sess={r.get("exp_moment")} seed={r.get("seed")} status={r.get("status")[:30]}')
    if results:
        pd.DataFrame(results).to_csv(args.output, mode='a',
            header=not args.output.exists() or args.output.stat().st_size == 0,
            index=False)

    total = time.time() - t_start
    print(f'\nDone. {len(cells)} cells in {total/60:.1f}min ({total/3600:.2f}h).')
    print(f'Output: {args.output}')


if __name__ == '__main__':
    main()
