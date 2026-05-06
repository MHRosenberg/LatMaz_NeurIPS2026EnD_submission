"""
260504_analyze_cohort137_HPO.py

Post-sweep analysis of the local cohort-137 HPO sweep (overnight 2026-05-04).
Produces:
  (a) per-(agent, hpo_config) macro RPA + NNS table on the 137-session cohort.
  (b) HPO-best (per-agent, by global RPA on cohort-137) comparison vs the
      canonical-60 HPO-best (from c260316-171035_hpo_tuning_FIXED.csv).
  (c) generalisation-gap summary: NNS drop from canonical-60 to cohort-137
      under each agent's canonical-60 HPO-best config (the HPO-leakage diagnostic
      proposed in docs/outbox/260504_HPO_leakage_non_triviality_argument.md).

Inputs:
  - data_transfer_cld/cluster_bundle_260503/outputs/c{ts}_cohort137_8agents_HPO_mac_overnight.csv
    (incremental during the sweep; final once the sweep finishes)
  - data_out/rl_sims/c260316-171035_hpo_tuning_FIXED.csv  (canonical HPO source)

Output:
  data_out/sanity_checks/c{ts}_cohort137_HPO_analysis.csv
  reports/c{ts}_cohort137_HPO_analysis.md

Usage:
  conda activate latMaz_RL
  python code/260504_analyze_cohort137_HPO.py \
      [--cohort-csv data_transfer_cld/cluster_bundle_260503/outputs/<ts>.csv] \
      [--quick]   # use partial CSV; report partial-coverage caveat
"""
from __future__ import annotations
import argparse
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path('<REPO_ROOT>')
COHORT_DIR = PROJECT_ROOT / 'data_transfer_cld' / 'cluster_bundle_260503' / 'outputs'
HPO_CSV = PROJECT_ROOT / 'data_out' / 'rl_sims' / 'c260316-171035_hpo_tuning_FIXED.csv'
OUT_DIR = PROJECT_ROOT / 'data_out' / 'sanity_checks'
REPORT_DIR = PROJECT_ROOT / 'reports'

# Reference points for NNS (pulled from paper_values.tex; canonical-60 baselines)
RANDOM_RPA_CANON = 0.178
GREEDY_RPA_CANON = 0.878   # NNS denominator since 2026-05-04 switch


def find_latest_cohort_csv():
    csvs = sorted(COHORT_DIR.glob('c*cohort137*HPO*.csv'))
    if not csvs:
        raise FileNotFoundError(f'No cohort-137 HPO CSV found in {COHORT_DIR}')
    return csvs[-1]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--cohort-csv', type=Path, default=None)
    p.add_argument('--hpo-csv', type=Path, default=HPO_CSV)
    p.add_argument('--quick', action='store_true',
                    help='Allow partial cohort CSV (will warn about coverage).')
    return p.parse_args()


def load_canonical_hpo(hpo_csv):
    df = pd.read_csv(hpo_csv)
    df['rpa'] = df['total_reward'] / df['n_actions'].clip(lower=1)
    df = df[df['model'] != 'POMCP']  # focus on model-free
    return df


def load_cohort(cohort_csv):
    df = pd.read_csv(cohort_csv)
    if 'rpa' not in df.columns:
        df['rpa'] = df['total_reward'] / df['n_actions'].clip(lower=1)
    df = df[df['status'] == 'ok']
    return df


def hpo_best_per_agent(df):
    """Pick HPO-best config per agent by global mean RPA."""
    g = df.groupby(['model', 'hpo_config'])['rpa'].mean().reset_index()
    best = g.loc[g.groupby('model')['rpa'].idxmax()]
    return best.set_index('model')['hpo_config'].to_dict()


def main():
    args = parse_args()
    if args.cohort_csv is None:
        args.cohort_csv = find_latest_cohort_csv()
    print(f'Cohort CSV: {args.cohort_csv}')
    print(f'HPO CSV:    {args.hpo_csv}')
    print()

    cohort = load_cohort(args.cohort_csv)
    canon = load_canonical_hpo(args.hpo_csv)
    print(f'Cohort rows (status=ok): {len(cohort)}')
    print(f'Canonical HPO rows: {len(canon)}')
    print()

    n_expected = sum(
        n_cfg * 137 * 5
        for n_cfg in [4, 4, 4, 4, 3, 4, 3, 3]  # A2C, DQN, DRQN_seq/rand, PPO, QRDQN, RecurrentPPO, TRPO
    )
    coverage = len(cohort) / n_expected
    print(f'Cohort coverage: {len(cohort)} / {n_expected} = {coverage:.1%}')
    if coverage < 0.95 and not args.quick:
        print(f'WARNING: coverage < 95%; pass --quick to override or wait for sweep to finish')
        sys.exit(0)
    print()

    # --- (a) Per-(agent, hpo_config) macro RPA + NNS on cohort-137 ---
    print('=' * 60)
    print('(a) Per-(agent, hpo_config) macro RPA + NNS on cohort-137')
    print('=' * 60)
    g = cohort.groupby(['model', 'hpo_config']).agg(
        n=('rpa', 'count'),
        rpa_mean=('rpa', 'mean'),
        rpa_std=('rpa', 'std')).reset_index()
    g['nns'] = 100.0 * (g['rpa_mean'] - RANDOM_RPA_CANON) / (GREEDY_RPA_CANON - RANDOM_RPA_CANON)
    print(g.to_string(index=False))
    print()

    # --- (b) HPO-best (per-agent, by global RPA on cohort-137) ---
    print('=' * 60)
    print('(b) Cohort-137 HPO-best per agent')
    print('=' * 60)
    cohort_hpo_best = hpo_best_per_agent(cohort)
    canon_hpo_best = hpo_best_per_agent(canon)
    summary = []
    for agent in sorted(set(cohort_hpo_best) | set(canon_hpo_best)):
        canon_best = canon_hpo_best.get(agent, '?')
        cohort_best = cohort_hpo_best.get(agent, '?')
        same = '=' if canon_best == cohort_best else 'DIFFERENT'
        summary.append({
            'agent': agent,
            'canon60_HPO_best': canon_best,
            'cohort137_HPO_best': cohort_best,
            'best_match': same,
        })
    print(pd.DataFrame(summary).to_string(index=False))
    print()

    # --- (c) Generalisation-gap diagnostic ---
    print('=' * 60)
    print('(c) Canonical-60 -> cohort-137 NNS drop under canonical HPO-best')
    print('=' * 60)
    rows = []
    for agent, canon_best in canon_hpo_best.items():
        canon_rpa = canon[(canon.model == agent) & (canon.hpo_config == canon_best)]['rpa'].mean()
        canon_nns = 100.0 * (canon_rpa - RANDOM_RPA_CANON) / (GREEDY_RPA_CANON - RANDOM_RPA_CANON)
        cohort_rpa = cohort[(cohort.model == agent) & (cohort.hpo_config == canon_best)]['rpa'].mean()
        cohort_nns = 100.0 * (cohort_rpa - RANDOM_RPA_CANON) / (GREEDY_RPA_CANON - RANDOM_RPA_CANON)
        rows.append({
            'agent': agent,
            'canon_HPO_best': canon_best,
            'canon60_RPA': round(canon_rpa, 3),
            'cohort137_RPA': round(cohort_rpa, 3),
            'canon60_NNS_pct': round(canon_nns, 1),
            'cohort137_NNS_pct': round(cohort_nns, 1),
            'NNS_drop_pp': round(canon_nns - cohort_nns, 1),
        })
    gap = pd.DataFrame(rows).sort_values('NNS_drop_pp', ascending=False)
    print(gap.to_string(index=False))
    print()

    # --- Write outputs ---
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime('%y%m%d-%H%M%S')
    out_csv = OUT_DIR / f'c{ts}_cohort137_HPO_analysis.csv'
    gap.to_csv(out_csv, index=False)
    print(f'Wrote {out_csv}')

    md_path = REPORT_DIR / f'c{ts}_cohort137_HPO_analysis.md'
    with open(md_path, 'w') as f:
        f.write(f'# Cohort-137 HPO sweep analysis — {ts}\n\n')
        f.write(f'Source CSV: `{args.cohort_csv}` ({coverage:.1%} cohort coverage)\n\n')
        f.write('## Per-(agent, hpo_config) macro RPA + NNS on cohort-137\n\n')
        f.write(g.to_markdown(index=False))
        f.write('\n\n## HPO-best per agent: canonical-60 vs cohort-137\n\n')
        f.write(pd.DataFrame(summary).to_markdown(index=False))
        f.write('\n\n## Generalisation gap (HPO-leakage diagnostic)\n\n')
        f.write(gap.to_markdown(index=False))
        f.write('\n\n')
        f.write('### Interpretation\n\n')
        f.write('See `docs/outbox/260504_HPO_leakage_non_triviality_argument.md` for context.\n')
        f.write('Large NNS drops (>10 pp) indicate the canonical-60 HPO-best config does\n')
        f.write('not generalize to the wider 137-session cohort. This is consistent with\n')
        f.write('two interpretations: (a) HPO leakage (overfitting to canonical-60 quirks),\n')
        f.write('or (b) genuine task variation (different mazes, reward functions, animals\n')
        f.write('in the wider cohort). The HPO-leakage non-triviality argument reframes\n')
        f.write('the drop as evidence of benchmark structural difficulty, not as a failure\n')
        f.write('mode of RL itself.\n')
    print(f'Wrote {md_path}')


if __name__ == '__main__':
    main()
