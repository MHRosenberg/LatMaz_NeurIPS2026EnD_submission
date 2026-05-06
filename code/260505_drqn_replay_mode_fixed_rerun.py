"""
260505_drqn_replay_mode_fixed_rerun.py

After fixing yoked_rl_runner.py:sample_sequences() to honor the
`replay_mode` parameter (commit pending, 2026-05-05), re-run BOTH
DRQN_seq and DRQN_rand on:
  (a) HPO sweep on canonical-60 — 4 cfg x 60 sess x 5 seeds x 2 agents
       = 2,400 cells. Re-determines per-agent HPO-best.
  (b) rerun-452 corrected re-evaluation under each agent's NEW HPO-best
       config — 452 sess x 5 seeds x 2 agents = 4,520 cells.

Smoke verified: with the fix, on 3 long-budget sessions x 3 seeds = 9
cells, DRQN_seq != DRQN_rand at every cell (range of |seq - rand|: 50
to 220 reward units). Pre-fix the two agents were bit-identical under
matched seeds.

Outputs:
  (a) data_out/rl_sims/c{ts}_drqn_replay_fix_HPO_canonical60.csv
  (b) data_out/rl_sims/c{ts}_drqn_replay_fix_rerun452_corrected.csv

Usage:
  conda activate latMaz_RL
  python code/260505_drqn_replay_mode_fixed_rerun.py --workers 4
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
sys.path.insert(0, str(PROJECT_ROOT / 'code0_init-readOnly' / 'code'))

DRQN_AGENTS = ['DRQN_seq', 'DRQN_rand']
SEEDS = list(range(5))
RERUN452_CSV = PROJECT_ROOT / 'data_released' / 'results' / 'c260505-120513_rerun452_corrected_8agents.csv'


# =============================================================================
# (a) HPO sweep on canonical-60 for DRQN_seq + DRQN_rand
# =============================================================================
def run_hpo_cell(args):
    agent, exp_mom, hpo_config_name, seed = args
    try:
        from experiment_config import load_data, build_runner, obs_kwargs
        from tune_models import HPO_CONFIGS
        from copy import deepcopy
        yoking_df, rwd_df, _ = load_data()
        runner = build_runner(yoking_df, rwd_df)
        # Apply this specific HPO config (not the BEST_CONFIGS one)
        cfg = runner.algo_config
        base = deepcopy(getattr(cfg, agent))
        overrides = HPO_CONFIGS[agent][hpo_config_name]
        base.update(overrides)
        setattr(cfg, agent, base)
        runner.algo_config = cfg

        yoke_row = yoking_df[yoking_df['exp_moment'] == exp_mom].iloc[0]
        adj_mat, st_positions = runner._load_maze(yoke_row['adj_file'], yoke_row['st_pos_file'])
        rewards, reset_val = runner._get_rewarded_states(exp_mom, len(st_positions))
        n_actions = int(yoke_row['n_states_visited']) - 1
        start_node = int(yoke_row['start_state'])
        t0 = time.time()
        total_reward = runner.run_single(
            model_name=agent, adj_mat=adj_mat, st_positions=st_positions,
            start_node=start_node, rewards=rewards, n_actions=n_actions,
            **obs_kwargs(),
            min_allowed_rewarded_states=int(reset_val), seed=seed,
        )
        elapsed = time.time() - t0
        return {
            'exp_moment': exp_mom, 'animal_ID': int(yoke_row['animal_ID']),
            'model': agent, 'hpo_config': hpo_config_name, 'seed': seed,
            'total_reward': float(total_reward), 'n_actions': n_actions,
            'rpa': float(total_reward) / max(n_actions, 1),
            'elapsed_s': round(elapsed, 2),
            'reset_val': int(reset_val), 'n_init_rewards': int((rewards > 0).sum()),
            'adj_file': yoke_row['adj_file'], 'replay_mode_fixed': True,
            'status': 'ok', 'timestamp': datetime.now().isoformat(timespec='seconds'),
        }
    except Exception as exc:
        return {
            'exp_moment': exp_mom, 'animal_ID': -1, 'model': agent,
            'hpo_config': hpo_config_name, 'seed': seed,
            'total_reward': float('nan'), 'n_actions': -1, 'rpa': float('nan'),
            'elapsed_s': -1.0, 'reset_val': -1, 'n_init_rewards': -1,
            'adj_file': '?', 'replay_mode_fixed': True,
            'status': f'error:{type(exc).__name__}:{str(exc)[:80]}',
            'timestamp': datetime.now().isoformat(timespec='seconds'),
        }


# =============================================================================
# (b) rerun-452 corrected re-eval under each agent's HPO-best (post-fix)
# =============================================================================
def run_452_cell(args):
    agent, exp_mom, seed = args
    try:
        from experiment_config import load_data, build_runner, obs_kwargs, apply_best_configs
        yoking_df, rwd_df, _ = load_data()
        runner = build_runner(yoking_df, rwd_df)
        runner.algo_config = apply_best_configs([agent])
        yoke_row = yoking_df[yoking_df['exp_moment'] == exp_mom].iloc[0]
        adj_mat, st_positions = runner._load_maze(yoke_row['adj_file'], yoke_row['st_pos_file'])
        rewards, reset_val = runner._get_rewarded_states(exp_mom, len(st_positions))
        n_actions = int(yoke_row['n_states_visited']) - 1
        start_node = int(yoke_row['start_state'])
        t0 = time.time()
        total_reward = runner.run_single(
            model_name=agent, adj_mat=adj_mat, st_positions=st_positions,
            start_node=start_node, rewards=rewards, n_actions=n_actions,
            **obs_kwargs(),
            min_allowed_rewarded_states=int(reset_val), seed=seed,
        )
        elapsed = time.time() - t0
        return {
            'exp_moment': exp_mom, 'animal_ID': int(yoke_row['animal_ID']),
            'model': agent, 'seed': seed,
            'total_reward': float(total_reward), 'n_actions': n_actions,
            'rpa': float(total_reward) / max(n_actions, 1),
            'elapsed_s': round(elapsed, 2),
            'reset_val': int(reset_val), 'n_init_rewards': int((rewards > 0).sum()),
            'adj_file': yoke_row['adj_file'], 'replay_mode_fixed': True,
            'status': 'ok', 'timestamp': datetime.now().isoformat(timespec='seconds'),
        }
    except Exception as exc:
        return {
            'exp_moment': exp_mom, 'animal_ID': -1, 'model': agent, 'seed': seed,
            'total_reward': float('nan'), 'n_actions': -1, 'rpa': float('nan'),
            'elapsed_s': -1.0, 'reset_val': -1, 'n_init_rewards': -1,
            'adj_file': '?', 'replay_mode_fixed': True,
            'status': f'error:{type(exc).__name__}:{str(exc)[:80]}',
            'timestamp': datetime.now().isoformat(timespec='seconds'),
        }


def run_pool(cells, fn, output, workers, label):
    print(f'\n=== {label} ({len(cells)} cells, {workers} workers) ===')
    print(f'    Output: {output}')
    t_start = time.time()
    results = []
    if workers == 1:
        for i, c in enumerate(cells):
            results.append(fn(c))
            if (i + 1) % 100 == 0:
                pd.DataFrame(results).to_csv(output, index=False)
                print(f'    [{i+1}/{len(cells)}] {time.time()-t_start:.0f}s')
    else:
        with mp.Pool(workers) as pool:
            for i, r in enumerate(pool.imap_unordered(fn, cells)):
                results.append(r)
                if (i + 1) % 100 == 0:
                    pd.DataFrame(results).to_csv(output, index=False)
                    elapsed = time.time() - t_start
                    eta = elapsed / (i + 1) * (len(cells) - i - 1)
                    print(f'    [{i+1}/{len(cells)}] {elapsed:.0f}s elapsed; ETA {eta:.0f}s')
    pd.DataFrame(results).to_csv(output, index=False)
    elapsed = time.time() - t_start
    df = pd.DataFrame(results)
    n_ok = (df['status'] == 'ok').sum()
    print(f'    Done in {elapsed:.0f}s ({elapsed/60:.1f} min). {n_ok}/{len(cells)} cells ok.')
    return df


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--skip-hpo', action='store_true')
    p.add_argument('--skip-452', action='store_true')
    p.add_argument('--limit', type=int, default=None,
                   help='Smoke test: cap cells per phase')
    args = p.parse_args()

    ts = 'c' + datetime.now().strftime('%y%m%d-%H%M%S')

    if not args.skip_hpo:
        # (a) HPO sweep on canonical-60
        from experiment_config import load_data
        from tune_models import HPO_CONFIGS
        yoking_df, _, sessions_60 = load_data()
        hpo_cells = []
        for agent in DRQN_AGENTS:
            for cfg_name in HPO_CONFIGS[agent]:
                for sess in sessions_60:
                    for seed in SEEDS:
                        hpo_cells.append((agent, sess, cfg_name, seed))
        if args.limit is not None:
            hpo_cells = hpo_cells[:args.limit]
        out_a = PROJECT_ROOT / 'data_out' / 'rl_sims' / f'{ts}_drqn_replay_fix_HPO_canonical60.csv'
        run_pool(hpo_cells, run_hpo_cell, out_a, args.workers, '(a) HPO canonical-60')

    if not args.skip_452:
        # (b) rerun-452 corrected re-eval under post-fix HPO-best
        rerun_df = pd.read_csv(RERUN452_CSV, dtype={'animal_ID': str})
        sessions_452 = sorted(rerun_df['exp_moment'].unique())
        cells_452 = [(agent, sess, seed)
                     for agent in DRQN_AGENTS for sess in sessions_452 for seed in SEEDS]
        if args.limit is not None:
            cells_452 = cells_452[:args.limit]
        out_b = PROJECT_ROOT / 'data_out' / 'rl_sims' / f'{ts}_drqn_replay_fix_rerun452_corrected.csv'
        run_pool(cells_452, run_452_cell, out_b, args.workers, '(b) rerun-452 corrected')


if __name__ == '__main__':
    main()
