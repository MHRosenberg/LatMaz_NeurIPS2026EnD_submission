"""
260504_extend_yoking_df_path2.py

Path-2 yoking-df extension: mimics Author's marimo notebook ingestion logic
(0a_aa_251201_latMaz_dataPrep-organize_n_start_yoking-STABLE.py:470-516)
to produce yoking-df rows for all raw exp dirs not yet in the canonical
c260301_*.csv. Designed to add a044 + any other newly-imported sessions.

Constants mirror Author's notebook exactly:
  MIN_EXP_DF_LEN = 10  (action count threshold)
  WINDOW_LEN_N_SHIFT_CONFIGS = [(1, 1), (2, 1), (3, 1), (5, 1)]
  INVALID_MAZE_SPEC_FILES_EXCLUDED = ['240507_latent_maze_003b_5x9-adjacency.csv']

Output: yoked_dfs/c{ts}_animal_to_agent_yoking_info-chronological_PATH2.csv
        (suffix _PATH2 distinguishes it from Author's marimo output)
"""
from __future__ import annotations
import sys
import os
import ast
import re
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

PROJECT_ROOT = Path('<REPO_ROOT>')
sys.path.insert(0, str(PROJECT_ROOT / 'code0_init-readOnly' / 'code'))

from utils_latMaz import (
    extract_animal_IDs, convert_action_words_to_chars,
    displacement_to_compass_heading, load_maze, get_n_action_prob_dict,
)

# Mirror Author's constants
MIN_EXP_DF_LEN = 10
WINDOW_LEN_N_SHIFT_CONFIGS = [(1, 1), (2, 1), (3, 1), (5, 1)]
INVALID_MAZE_SPEC_FILES_EXCLUDED = ['240507_latent_maze_003b_5x9-adjacency.csv']

FLAT_CSV_DIR = PROJECT_ROOT / 'data_in' / '1_experiment_csvs'         # Author's marimo-canonical source
RPI_RAW_DIR = PROJECT_ROOT / 'data_in' / '0_raw_exp_dirs_from_RPi'    # session-folder structure from RPi
YOKED_DIR = PROJECT_ROOT / 'yoked_dfs'
MAZE_DIR = PROJECT_ROOT / 'data_in' / 'mazes'


def find_existing_yoking_csv():
    csvs = sorted(YOKED_DIR.glob('c*animal_to_agent_yoking_info*.csv'))
    csvs = [p for p in csvs if 'PATH2' not in p.name and 'merged' not in p.name]
    return csvs[-1] if csvs else None


def find_all_data_csvs():
    """Find all *_data.csv files in BOTH the flat 1_experiment_csvs/ dir
    AND the per-session 0_raw_exp_dirs_from_RPi/ dirs. Dedup by exp_moment;
    prefer the flat copy when both exist (matches Author's marimo source-of-truth)."""
    csvs = {}  # exp_moment -> Path
    # Flat dir first (canonical for Author's marimo)
    if FLAT_CSV_DIR.exists():
        for f in sorted(FLAT_CSV_DIR.glob('*_data.csv')):
            m = get_exp_moment(f)
            if m and m not in csvs:
                csvs[m] = f
    # Then RPi raw dirs (only fill gaps)
    if RPI_RAW_DIR.exists():
        for d in sorted(RPI_RAW_DIR.iterdir()):
            if not d.is_dir():
                continue
            # In a session dir there may be multiple data CSVs (mid-session restarts);
            # take the latest by mtime/lex.
            session_csvs = sorted(d.glob('*_data.csv'))
            if not session_csvs:
                continue
            f = session_csvs[-1]
            m = get_exp_moment(f)
            if m and m not in csvs:
                csvs[m] = f
    return list(csvs.values())


def get_exp_moment(csv_file):
    m = re.search(r'(\d{6}-\d{6})', str(csv_file))
    return m.group(1) if m else None


def _normalise_animal_id(raw):
    """extract_animal_IDs may return a list (when the path has multiple a### matches).
    Collapse to a single zero-padded 3-digit string if all elements agree."""
    if isinstance(raw, (list, tuple)):
        s = set(str(x).zfill(3) for x in raw)
        if len(s) == 1:
            return s.pop()
        # ambiguous; pick the most common
        from collections import Counter
        return Counter(str(x).zfill(3) for x in raw).most_common(1)[0][0]
    return str(raw).zfill(3)


def ingest_session(csv_file):
    """Mirror Author's marimo loop body for one session. Returns dict or None."""
    a_ID = _normalise_animal_id(extract_animal_IDs(str(csv_file)))
    d_df = pd.read_csv(csv_file)
    if len(d_df) < MIN_EXP_DF_LEN:
        return None  # too short
    if d_df['adjacency_file'].nunique() != 1:
        return None  # multiple mazes in one session, Author's pipeline rejects
    adj_file = d_df['adjacency_file'].unique()[0]
    if os.path.basename(adj_file) in INVALID_MAZE_SPEC_FILES_EXCLUDED:
        return None
    actions_ego = convert_action_words_to_chars(d_df.action.tolist())
    actions_allo_real = [ast.literal_eval(s)[0] for s in d_df.choice.tolist()]
    if len(actions_ego) != len(actions_allo_real):
        return None
    if d_df['start_state'].nunique() != 1:
        return None
    start_state = int(deepcopy(d_df['start_state'].unique()[0]))
    n_states_visited = len(d_df)
    if d_df['state'].to_list()[0] != start_state:
        states_visited = [start_state]
        n_states_visited += 1
    else:
        states_visited = []
    states_visited.extend(d_df['state'].to_list())

    st_pos_file = d_df['st_positions_file'].unique()[0]
    try:
        adj_mat, st_pos = load_maze(str(MAZE_DIR / adj_file),
                                     str(MAZE_DIR / st_pos_file))
    except Exception as e:
        print(f'  load_maze failed for {csv_file.name}: {e}')
        return None

    actions_allo_latent = []
    st_tm1 = states_visited[0]
    for st in states_visited[1:]:
        try:
            actions_allo_latent.append(
                displacement_to_compass_heading(st_pos[st] - st_pos[st_tm1]))
        except Exception:
            actions_allo_latent.append('?')
        st_tm1 = st

    if not (np.diff(d_df['n_rewards'].values) >= 0).all():
        return None  # non-monotonic rewards rejected by Author's pipeline
    n_rewards = int(d_df['n_rewards'].to_list()[-1])

    prob_dict_allo_real = {
        (wl, ns): get_n_action_prob_dict(actions_allo_real, wl, ns)
        for wl, ns in WINDOW_LEN_N_SHIFT_CONFIGS
    }
    prob_dict_allo_latent = {
        (wl, ns): get_n_action_prob_dict(actions_allo_latent, wl, ns)
        for wl, ns in WINDOW_LEN_N_SHIFT_CONFIGS
    }
    prob_dict_ego = {
        (wl, ns): get_n_action_prob_dict(actions_ego, wl, ns)
        for wl, ns in WINDOW_LEN_N_SHIFT_CONFIGS
    }

    return {
        'animal_ID': a_ID,
        'csv_data_path': str(csv_file.relative_to(PROJECT_ROOT)),
        'states_visited': states_visited,
        'adj_file': adj_file,
        'st_pos_file': st_pos_file,
        'n_rewards': n_rewards,
        'start_state': start_state,
        'n_states_visited': n_states_visited,
        'prob_dict_allo_real': prob_dict_allo_real,
        'prob_dict_allo_latent': prob_dict_allo_latent,
        'prob_dict_ego': prob_dict_ego,
    }


def main():
    existing = find_existing_yoking_csv()
    if existing is None:
        raise SystemExit('No existing yoking CSV found.')
    print(f'Existing yoking CSV: {existing}')
    yo_old = pd.read_csv(existing, dtype={'animal_ID': str})
    yo_old['animal_ID'] = yo_old['animal_ID'].str.zfill(3)
    print(f'  rows: {len(yo_old)}')
    # Dedup by exp_moment (YYMMDD-HHMMSS) extracted from csv_data_path,
    # since the path format differs between Author's marimo output and our raw dir layout.
    existing_moments = set()
    for p in yo_old['csv_data_path'].dropna():
        m = re.search(r'(\d{6}-\d{6})', str(p))
        if m:
            existing_moments.add(m.group(1))

    all_csvs = find_all_data_csvs()
    print(f'Found {len(all_csvs)} unique candidate data CSVs (across both 1_experiment_csvs/ and 0_raw_exp_dirs_from_RPi/)')

    new_rows = []
    skipped = 0
    failed = 0
    seen_new_moments = set()
    for csv in tqdm(all_csvs, desc='ingesting'):
        moment = get_exp_moment(csv)
        if moment is None:
            failed += 1
            continue
        if moment in existing_moments or moment in seen_new_moments:
            skipped += 1
            continue
        seen_new_moments.add(moment)
        try:
            row = ingest_session(csv)
            if row is None:
                failed += 1
                continue
            new_rows.append(row)
        except Exception as e:
            print(f'  err {csv.name}: {e}')
            failed += 1

    print(f'\nNew sessions: {len(new_rows)}; skipped (already in df): {skipped}; '
          f'rejected (filter / parse-error): {failed}')
    if not new_rows:
        print('No new sessions to add. Exiting.')
        return

    yo_new = pd.DataFrame(new_rows)
    # Ensure consistent animal_ID type before concat
    yo_new['animal_ID'] = yo_new['animal_ID'].astype(str).str.zfill(3)
    yo_combined = pd.concat([yo_old, yo_new], ignore_index=True)
    yo_combined['animal_ID'] = yo_combined['animal_ID'].astype(str)
    print(f'Combined yoking df: {len(yo_old)} + {len(new_rows)} = {len(yo_combined)} rows')
    print(f'  per-animal:')
    counts = yo_combined['animal_ID'].value_counts()
    for k in sorted(counts.index, key=lambda v: str(v)):
        print(f'    a{k}: {counts[k]}')

    ts = datetime.now().strftime('%y%m%d-%H%M%S')
    out_path = YOKED_DIR / f'c{ts}_animal_to_agent_yoking_info-chronological_PATH2.csv'
    yo_combined.to_csv(out_path, index=False)
    print(f'\nWrote {out_path}')


if __name__ == '__main__':
    main()
