"""
c260506-extract_latent_state_biased.py

Extract the latent_state_biased policy rows from the legacy 260204
analysis pipeline CSV (data_out/c260204_per_exp_result_df.csv) and
write a clean c-prefix CSV to data_released/results/ so the appendix
"Latent-state-biased" paragraph can cite an actual numerical RPA
instead of saying "numerical evaluation is deferred".

Per Author 2026-05-06 batch B-1 part 2.

Inputs:
  data_out/c260204_per_exp_result_df.csv  (mixed policy types, ~all rows)

Outputs:
  data_released/results/c{ts}_latent_state_biased_legacy260204.csv
  prints summary RPA stats (mean/median/p90/max) + N sessions to stdout

NOTE FOR REVIEWERS:
  This script is provided for provenance only -- it documents how the file at
  data_released/results/c260506-154615_latent_state_biased_legacy260204.csv
  was generated. The source CSV (data_out/c260204_per_exp_result_df.csv) is
  part of an older 2025-09 analysis pipeline and is NOT included in the
  public release (several hundred MB; largely superseded by the canonical-60
  and rerun-452 results in data_released/results/). The output CSV is the
  artefact of record; re-running this script is not required for replicating
  the headline results.
"""
import os
import sys
from datetime import datetime

import pandas as pd

PROJECT_ROOT = os.environ.get('LATMAZ_REPO_ROOT', os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC_CSV = os.path.join(PROJECT_ROOT, 'data_out', 'c260204_per_exp_result_df.csv')
OUT_DIR = os.path.join(PROJECT_ROOT, 'data_released', 'results')
TS = 'c' + datetime.now().strftime('%y%m%d-%H%M%S')


def main():
    if not os.path.exists(SRC_CSV):
        print(f'ERROR: source CSV not found: {SRC_CSV}')
        sys.exit(1)

    print(f'Reading {SRC_CSV} ...', flush=True)
    df = pd.read_csv(SRC_CSV, low_memory=False)
    print(f'  total rows: {len(df):,}', flush=True)
    print(f'  columns: {list(df.columns)}', flush=True)

    # Filter to latent_state_biased policy
    # Column is agent_model with the literal value 'latent state biased' (spaces, not underscores).
    if 'agent_model' in df.columns:
        sub = df[df['agent_model'].astype(str).str.strip() == 'latent state biased'].copy()
    elif 'policy_class' in df.columns:
        sub = df[df['policy_class'].astype(str).str.contains('latent_state_biased', na=False)].copy()
    else:
        print('ERROR: no agent_model or policy_class column')
        sys.exit(1)

    print(f'  latent_state_biased rows: {len(sub):,}', flush=True)
    if len(sub) == 0:
        print('ERROR: no rows after filtering')
        sys.exit(1)

    # Compute RPA
    if 'n_rewards_obtained_agent' in sub.columns:
        sub['n_rewards'] = pd.to_numeric(sub['n_rewards_obtained_agent'], errors='coerce')
    elif 'total_reward' in sub.columns:
        sub['n_rewards'] = pd.to_numeric(sub['total_reward'], errors='coerce')
    else:
        print('ERROR: no reward column found')
        sys.exit(1)

    # n_actions: use len of states_visited_agent if available, else explicit column
    def states_to_n_actions(s):
        try:
            if pd.isna(s):
                return None
            # states_visited_agent looks like "[0, 1, 2, ...]" - count entries
            s = str(s).strip()
            if s.startswith('['):
                # crude: count commas + 1
                inner = s.strip('[] \n')
                if not inner:
                    return 0
                return inner.count(',')  # n_actions = states - 1
        except Exception:
            return None

    if 'states_visited_agent' in sub.columns:
        sub['n_actions_calc'] = sub['states_visited_agent'].apply(states_to_n_actions)
    if 'n_actions' in sub.columns:
        sub['n_actions_explicit'] = pd.to_numeric(sub['n_actions'], errors='coerce')

    if 'n_actions_calc' in sub.columns:
        sub['n_actions_final'] = sub['n_actions_calc']
    elif 'n_actions_explicit' in sub.columns:
        sub['n_actions_final'] = sub['n_actions_explicit']
    else:
        print('ERROR: cannot compute n_actions')
        sys.exit(1)

    sub = sub[sub['n_actions_final'].notna() & (sub['n_actions_final'] > 0)].copy()
    sub['rpa'] = sub['n_rewards'] / sub['n_actions_final']
    print(f'  after rpa compute: {len(sub):,} valid rows', flush=True)

    # Slim output: keep identifying + result columns
    keep = []
    for col in ['exp_moment', 'animal_ID', 'csv_data_path', 'csv_filename',
                'policy_class', 'representation', 'avoid_reversal', 'sequence_length',
                'agent_model', 'repeat_idx', 'repeat_group_idx', 'seed',
                'n_rewards', 'n_actions_final', 'rpa', 'initial_rewards']:
        if col in sub.columns:
            keep.append(col)
    out = sub[keep].copy()
    out.rename(columns={'n_actions_final': 'n_actions'}, inplace=True)

    out_path = os.path.join(OUT_DIR, f'{TS}_latent_state_biased_legacy260204.csv')
    out.to_csv(out_path, index=False)
    print(f'\nSaved: {out_path}', flush=True)
    print(f'  bytes: {os.path.getsize(out_path):,}', flush=True)

    # Per-session aggregation
    if 'exp_moment' in out.columns:
        per_session = out.groupby('exp_moment')['rpa'].agg(['mean', 'count']).reset_index()
        print(f'\nPer-session aggregation: {len(per_session):,} unique sessions', flush=True)
        sess_means = per_session['mean']
        print(f'  RPA across sessions:', flush=True)
        print(f'    mean   : {sess_means.mean():.4f}', flush=True)
        print(f'    median : {sess_means.median():.4f}', flush=True)
        print(f'    p10    : {sess_means.quantile(0.10):.4f}', flush=True)
        print(f'    p90    : {sess_means.quantile(0.90):.4f}', flush=True)
        print(f'    max    : {sess_means.max():.4f}', flush=True)
        print(f'    min    : {sess_means.min():.4f}', flush=True)

        # NNS using canonical anchors
        RANDOM_RPA = 0.178
        GREEDY_RPA = 0.878
        sess_means_nns = 100.0 * (sess_means - RANDOM_RPA) / (GREEDY_RPA - RANDOM_RPA)
        print(f'  NNS across sessions (anchors: Random=0.178, Greedy=0.878):', flush=True)
        print(f'    mean   : {sess_means_nns.mean():.1f}%', flush=True)
        print(f'    median : {sess_means_nns.median():.1f}%', flush=True)
        print(f'    p90    : {sess_means_nns.quantile(0.90):.1f}%', flush=True)
        print(f'    max    : {sess_means_nns.max():.1f}%', flush=True)

    # Pooled (total/total) reference
    total_rwd = out['n_rewards'].sum()
    total_act = out['n_actions'].sum()
    pooled_rpa = total_rwd / total_act if total_act > 0 else float('nan')
    print(f'\n  Pooled RPA (sum_r / sum_a): {pooled_rpa:.4f} (n_rwd={total_rwd}, n_act={total_act})', flush=True)


if __name__ == '__main__':
    main()
