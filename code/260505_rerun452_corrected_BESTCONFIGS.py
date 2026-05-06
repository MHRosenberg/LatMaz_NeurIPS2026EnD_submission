"""
260505_rerun452_corrected_BESTCONFIGS.py

Re-run rerun-452 for all 8 RL agents with corrected configs after the
two-issue fix (see 260505_rerun452_rpa_discrepancy_diagnosis.md):
  DQN, DRQN_seq, DRQN_rand, A2C, PPO, QRDQN, TRPO, RecurrentPPO.

Uses the *exact* 452 session set from the existing released rerun-452 CSV
(preserves paper's \\data{452} macro). Source: data_released/results/
c260505-120513_rerun452_corrected_8agents.csv (the previous c260504-143902
buggy version was archived 2026-05-05 to z_legacy_outdated_reference/).
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

# Reuse the worker function from the original cohort runner — it already
# calls apply_best_configs() which now reads the corrected BEST_CONFIGS.
from importlib import import_module
_orig = import_module('260503_run_cohort_HPObest')
run_one_cell = _orig.run_one_cell

REQ_AGENTS = ['DQN', 'DRQN_seq', 'DRQN_rand', 'A2C', 'PPO', 'QRDQN', 'TRPO', 'RecurrentPPO']
SEEDS = list(range(5))
EXISTING_RERUN_CSV = PROJECT_ROOT / 'data_released' / 'results' / 'c260505-120513_rerun452_corrected_8agents.csv'


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--output', type=Path, default=None)
    p.add_argument('--limit', type=int, default=None, help='Smoke test: cap cells')
    args = p.parse_args()

    if args.output is None:
        ts = 'c' + datetime.now().strftime('%y%m%d-%H%M%S')
        args.output = PROJECT_ROOT / 'data_out' / 'rl_sims' / f'{ts}_rerun452_corrected_8agents.csv'
    args.output.parent.mkdir(parents=True, exist_ok=True)

    existing = pd.read_csv(EXISTING_RERUN_CSV)
    sessions = sorted(existing['exp_moment'].dropna().unique().tolist())
    print(f'Sessions in existing rerun-452 CSV: {len(sessions)}')

    cells = [(a, s, seed) for a in REQ_AGENTS for s in sessions for seed in SEEDS]
    print(f'Total cells (agents × sessions × seeds): {len(cells)}')
    if args.limit:
        cells = cells[:args.limit]
        print(f'Limited to {args.limit} (smoke test)')

    if args.output.exists():
        done = pd.read_csv(args.output)
        done_keys = set(zip(done['model'], done['exp_moment'], done['seed']))
        cells = [c for c in cells if c not in done_keys]
        print(f'Resume: {len(done_keys)} already done; {len(cells)} remaining')

    if not cells:
        print('Nothing to do.')
        return

    t0 = time.time()
    results = []
    with mp.Pool(args.workers) as pool:
        for i, r in enumerate(pool.imap_unordered(run_one_cell, cells, chunksize=1)):
            results.append(r)
            if (i + 1) % 20 == 0 or i == 0:
                df = pd.DataFrame(results)
                df.to_csv(args.output, mode='a',
                          header=not args.output.exists() or args.output.stat().st_size == 0,
                          index=False)
                results = []
                el = time.time() - t0
                eta = (len(cells) - (i + 1)) / max((i + 1) / el, 1e-6)
                print(f'  [{i+1}/{len(cells)}] el={el:.0f}s rate={(i+1)/el:.2f}/s '
                       f'ETA={eta/60:.1f}min  {r.get("model")} {r.get("exp_moment")} '
                       f'seed={r.get("seed")} status={(r.get("status") or "")[:25]}')
    if results:
        pd.DataFrame(results).to_csv(args.output, mode='a',
            header=not args.output.exists() or args.output.stat().st_size == 0,
            index=False)
    total = time.time() - t0
    print(f'\nDone. {len(cells)} cells in {total/60:.1f}min ({total/3600:.2f}h)')
    print(f'Output: {args.output}')


if __name__ == '__main__':
    main()
