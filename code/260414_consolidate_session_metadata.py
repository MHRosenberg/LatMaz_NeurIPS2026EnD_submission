#!/usr/bin/env python3
"""
Consolidate per-session experiment metadata into a single CSV.

Columns:
  exp_session_date        - date portion of session timestamp (YYMMDD)
  rpi_session_subdir      - immediate parent dir in 0_raw_exp_dirs_from_RPi
  animal_ID               - e.g. a002
  maze_ID                 - adjacency file stem (e.g. 240218_intuition_maze_003)
  reward_function         - from rwd_configs_df approach (matched via data CSV name)
  rewarded_states_initial - priority: png_validated > usr_params > rwd_configs > corrected
  rewarded_states_after_rebait - from usr_params REPLENISHED_REWARDED_STATES_0
  reverse_actions_allowed - from SAME_PORT_MIN_REPOKE_INTERVAL (inf=False, other=True,
                            missing=True_inferred)
  same_port_min_repoke_s  - raw numeric value (or 'inf' or 'unknown')
  rwd_initial_source      - provenance label for rewarded_states_initial
  png_validated           - True/False (from PNG extraction)
  png_sim_step_match_rate - step-match rate (or NaN)
  notes                   - flags (firstWeek, rwdSwitches, etc.)

Usage:
  conda run -n latMaz_RL python3 code/260414_consolidate_session_metadata.py

# ===========================================================================
# KNOWN COVERAGE GAPS AND NAMING DISCREPANCIES — READ BEFORE MODIFYING
# ===========================================================================
#
# GAP 1: rwd_configs_df / rwd_corr_df only cover a033 through 260124
#   Both reference tables (260202_rwd_configs_df.csv and 260202_rewarded_states_corrected.csv)
#   were generated on 260202 and include a033 sessions only up to session 260124.
#   Sessions 260201–260223 for a033 exist in raw dirs and 1_experiment_csvs but are
#   ABSENT from both lookup tables. Reward function for these sessions is derived
#   directly from usr_params (see Fallback 3 in reward_function section below).
#   → If you regenerate these CSVs to extend coverage, re-check the Fallback 3 logic.
#
# GAP 2: Underscore/dash naming discrepancy (old-format sessions)
#   Old raw session dirs (pre-240824) name their data CSVs as YYMMDD_HHMMSS_aXXX_data.csv
#   (underscore between date and time). When copied to 1_experiment_csvs and referenced
#   in rwd_configs_df / rwd_corr_df / png_extraction_csv, the first underscore becomes a
#   DASH → YYMMDD-HHMMSS_aXXX_data.csv. Both forms must be tried in all lookups.
#   The _normalise_basename() helper converts underscore→dash form.
#   → If you add new lookup tables, always try both raw and normalised basenames.
#
# ===========================================================================
"""

import ast
import os
import sys
import glob
import re
from datetime import datetime

import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', '..'))
RAW_DIR       = os.path.join(PROJECT_ROOT, 'data_in', '0_raw_exp_dirs_from_RPi')
RWD_INFO_DIR  = os.path.join(PROJECT_ROOT, 'rwd_info_dfs')
DATA_OUT      = os.path.join(PROJECT_ROOT, 'data_out')

TIMESTAMP = 'c' + datetime.now().strftime('%y%m%d-%H%M%S')

# ---------------------------------------------------------------------------
# Exclusion rules (per decisions)
# ---------------------------------------------------------------------------
EXCLUDE_STRINGS = [
    'a999',           # test/fake animal
    'a000',           # flagged in PNG extraction pipeline
    'incorrectDateTime',  # timestamp-duplicate dirs
    'crashed',        # crashed session (< 20 PNGs, skip)
    'b_firstWeekOfData',  # handled separately below (nested structure)
]
# rwdSwitches sessions: include but flag
RWD_SWITCHES_FLAG = 'NOTE: rwd_switches_location_on_consumption — Author: too few sessions to be worth considering. Drop from stats and downstream analyses.'

# ---------------------------------------------------------------------------
# Load reference tables
# ---------------------------------------------------------------------------
# rwd_configs_df (reward approach by data CSV name)
rwd_configs_path = os.path.join(RWD_INFO_DIR, '260202_rwd_configs_df.csv')
rwd_configs_df = pd.read_csv(rwd_configs_path)
# normalise csv_path to basename for matching
rwd_configs_df['csv_basename'] = rwd_configs_df['csv_path'].apply(os.path.basename)

# rewarded_states_corrected (for firstWeek sessions)
rwd_corr_path = os.path.join(RWD_INFO_DIR, '260202_rewarded_states_corrected.csv')
rwd_corr_df = pd.read_csv(rwd_corr_path)
rwd_corr_df['csv_basename'] = rwd_corr_df['exp_csv_path'].apply(os.path.basename)

# PNG reward extraction (authoritative validated initial states)
png_ext_path = os.path.join(DATA_OUT, '260305-121336_full_png_reward_extraction.csv')
png_df = pd.read_csv(png_ext_path)
# exp_moment is data-CSV timestamp (e.g. '260124-225237')
# animal_ID in png_df is e.g. 'a033' — match via exp_moment + animal_ID → csv_basename
def _png_to_basename(row):
    # data CSV format: {exp_moment}_{animal_ID}_data.csv
    # exp_moment in png_df has '-' separator: '260124-225237' → filename uses same
    return f"{row['exp_moment']}_{row['animal_ID']}_data.csv"
png_df['csv_basename'] = png_df.apply(_png_to_basename, axis=1)
# Also build normalised key (old-format raw CSVs use underscore: 240229_172154 → 240229-172154)
def _normalise_basename(b):
    """Replace first underscore in YYMMDD_HHMMSS with a dash → YYMMDD-HHMMSS."""
    m = re.match(r'^(\d{6})_(\d{6}_.+)$', b)
    return f"{m.group(1)}-{m.group(2)}" if m else b
png_df['csv_basename_norm'] = png_df['csv_basename'].apply(_normalise_basename)
png_lookup      = png_df.set_index('csv_basename')
png_lookup_norm = png_df.set_index('csv_basename_norm')

# ---------------------------------------------------------------------------
# Helper: parse Python literal or return as-is
# ---------------------------------------------------------------------------
def _chk_type(val):
    if isinstance(val, list):
        return val
    if isinstance(val, str):
        try:
            return ast.literal_eval(val)
        except Exception:
            return val
    return val

# ---------------------------------------------------------------------------
# Helper: read usr_params CSV → dict
# ---------------------------------------------------------------------------
def _load_usr_params(path):
    df = pd.read_csv(path, header=0)  # columns: (unnamed index), param, setting
    # column names vary slightly; always take cols [1] and [2]
    cols = df.columns.tolist()
    param_col  = cols[1]
    setting_col = cols[2]
    return dict(zip(df[param_col], df[setting_col]))

# ---------------------------------------------------------------------------
# Find all valid raw session dirs
# ---------------------------------------------------------------------------
def _collect_raw_dirs():
    """
    Returns list of (rpi_subdir_name, full_path, is_firstWeek, notes_prefix).
    Handles the nested b_firstWeekOfData structure.
    """
    entries = []

    for name in sorted(os.listdir(RAW_DIR)):
        full = os.path.join(RAW_DIR, name)
        if not os.path.isdir(full):
            continue

        # b_firstWeekOfData: recurse one level
        if name.startswith('b_firstWeekOfData'):
            for sub in sorted(os.listdir(full)):
                sub_full = os.path.join(full, sub)
                if not os.path.isdir(sub_full):
                    continue
                skip = any(s in sub for s in EXCLUDE_STRINGS)
                if skip:
                    continue
                entries.append((sub, sub_full, True, ''))
            continue

        # Normal dirs
        skip = any(s in name for s in EXCLUDE_STRINGS)
        if skip:
            continue
        entries.append((name, full, False, ''))

    return entries

# ---------------------------------------------------------------------------
# Main build
# ---------------------------------------------------------------------------
def build_metadata():
    raw_dirs = _collect_raw_dirs()
    rows = []
    skipped = []

    for rpi_subdir, full_path, is_firstWeek, notes_prefix in raw_dirs:

        # ---- find data CSV ----
        data_csvs = sorted(glob.glob(os.path.join(full_path, '*_data.csv')))
        if not data_csvs:
            skipped.append((rpi_subdir, 'no_data_csv'))
            continue
        if len(data_csvs) > 1:
            # keep newest (highest filename timestamp)
            data_csvs = [sorted(data_csvs)[-1]]
        data_csv_path = data_csvs[0]
        data_csv_basename = os.path.basename(data_csv_path)

        # ---- read data CSV header ----
        try:
            exp_df = pd.read_csv(data_csv_path, nrows=5)
        except Exception as e:
            skipped.append((rpi_subdir, f'read_error: {e}'))
            continue
        if len(exp_df) == 0:
            skipped.append((rpi_subdir, 'empty_data_csv'))
            continue

        # ---- session date ----
        # Timestamp format is YYMMDD_HHMMSS or YYMMDD-HHMMSS — take first 6 digits only
        try:
            ts_str = str(exp_df['time'].iloc[0])
            m_date = re.match(r'^(\d{6})', ts_str)
            exp_session_date = m_date.group(1) if m_date else 'unknown'
        except Exception:
            exp_session_date = 'unknown'

        # ---- animal ID ----
        animal_ID = 'unknown'
        try:
            m = re.search(r'_a(\d+)_data\.csv$', data_csv_basename)
            if m:
                animal_ID = f"a{m.group(1).zfill(3)}"
        except Exception:
            pass

        # ---- maze ID ----
        maze_ID = 'unknown'
        try:
            adj_col = 'adjacency_file'
            if adj_col in exp_df.columns:
                adj_val = exp_df[adj_col].dropna().iloc[0]
                maze_ID = os.path.basename(str(adj_val)).replace('-adjacency.csv', '')
        except Exception:
            pass

        # ---- usr_params ----
        usr_params_paths = sorted(glob.glob(os.path.join(full_path, '*usr_params*.csv')))
        has_usr_params = len(usr_params_paths) > 0
        usr_params = {}
        if has_usr_params:
            try:
                usr_params = _load_usr_params(usr_params_paths[-1])
            except Exception as e:
                print(f'  WARNING: could not load usr_params for {rpi_subdir}: {e}')

        # ---- reverse_actions_allowed ----
        same_port_raw = usr_params.get('SAME_PORT_MIN_REPOKE_INTERVAL', None)
        if same_port_raw is None:
            same_port_str = 'unknown'
            reverse_actions_allowed = 'True (inferred: backwards actions present in data)'
        else:
            same_port_str = str(same_port_raw).strip()
            if same_port_str.lower() == 'inf':
                reverse_actions_allowed = False
            else:
                reverse_actions_allowed = True

        # Normalised basename for matching old-format CSVs (YYMMDD_HHMMSS → YYMMDD-HHMMSS)
        data_csv_basename_norm = _normalise_basename(data_csv_basename)

        # ---- reward approach ----
        reward_function = 'unknown'
        # Try both raw and normalised basename in rwd_configs_df
        rwd_match = rwd_configs_df[rwd_configs_df['csv_basename'] == data_csv_basename]
        if len(rwd_match) == 0:
            rwd_match = rwd_configs_df[rwd_configs_df['csv_basename'] == data_csv_basename_norm]
        if len(rwd_match) == 1:
            reward_function = rwd_match.iloc[0]['approach']
        else:
            # Fallback 1: rwdSwitches flag
            if 'rwdSwitches2newLocationUponConsumption' in rpi_subdir:
                reward_function = 'rwd_switches_location_on_consumption'
            else:
                # Fallback 2: rewarded_states_corrected.rwd_approach (covers post-260124 sessions)
                corr_match = rwd_corr_df[rwd_corr_df['csv_basename'] == data_csv_basename]
                if len(corr_match) == 0:
                    corr_match = rwd_corr_df[rwd_corr_df['csv_basename'] == data_csv_basename_norm]
                if len(corr_match) == 1:
                    reward_function = corr_match.iloc[0]['rwd_approach']

        # Fallback 3: derive from usr_params directly (handles a033 260201–260223 gap and
        # any other sessions absent from both lookup tables but with usr_params present).
        # See KNOWN COVERAGE GAPS note at top of file.
        if reward_function == 'unknown' and has_usr_params:
            init_val = usr_params.get('INITIAL_REWARDED_STATES', None)
            repl_val = usr_params.get('REPLENISHED_REWARDED_STATES_0', None)
            if init_val is not None and repl_val is not None:
                init_parsed = _chk_type(init_val)
                repl_parsed = _chk_type(repl_val)
                if init_parsed == repl_parsed:
                    reward_function = 'v4-pacman: init == replenished'
                else:
                    reward_function = 'v4-different'

        # ---- PNG validation lookup ----
        png_validated = False
        png_sim_step_match_rate = float('nan')
        png_initial_from_extraction = None
        _png_key = None
        if data_csv_basename in png_lookup.index:
            _png_key = data_csv_basename
            _pl = png_lookup
        elif data_csv_basename_norm in png_lookup_norm.index:
            _png_key = data_csv_basename_norm
            _pl = png_lookup_norm
        if _png_key is not None:
            pr = _pl.loc[_png_key]
            png_validated = bool(pr['has_validated'])
            png_sim_step_match_rate = float(pr['sim_step_match_rate']) if pd.notna(pr['sim_step_match_rate']) else float('nan')
            if png_validated and pd.notna(pr['validated_initial']):
                try:
                    png_initial_from_extraction = _chk_type(pr['validated_initial'])
                except Exception:
                    pass

        # ---- rewarded_states_initial ----
        rwd_initial = 'unknown'
        rwd_initial_source = 'unknown'

        # Priority 1: PNG validated
        if png_initial_from_extraction is not None:
            rwd_initial = png_initial_from_extraction
            rwd_initial_source = 'png_validated'

        # Priority 2: usr_params INITIAL_REWARDED_STATES
        elif has_usr_params and 'INITIAL_REWARDED_STATES' in usr_params:
            val = _chk_type(usr_params['INITIAL_REWARDED_STATES'])
            if val not in ('unknown', None, ''):
                rwd_initial = val
                rwd_initial_source = 'usr_params'

        # Priority 3: rwd_configs_df first entry
        elif len(rwd_match) == 1:
            try:
                onsets = _chk_type(rwd_match.iloc[0]['rwd_config_onsets'])
                if isinstance(onsets, list) and len(onsets) > 0:
                    rwd_initial = onsets[0][1]
                    rwd_initial_source = 'rwd_configs_df_first_onset'
            except Exception:
                pass

        # Priority 4: corrected DF (try both raw and normalised basename)
        if rwd_initial == 'unknown':
            corr_match = rwd_corr_df[rwd_corr_df['csv_basename'] == data_csv_basename]
            if len(corr_match) == 0:
                corr_match = rwd_corr_df[rwd_corr_df['csv_basename'] == data_csv_basename_norm]
            if len(corr_match) == 1:
                corr_val = corr_match.iloc[0]['corrected_rewarded_states']
                if corr_val not in ('unknown', None, ''):
                    rwd_initial = _chk_type(corr_val)
                    rwd_initial_source = 'corrected_df'

        # ---- rewarded_states_after_rebait ----
        rwd_after_rebait = 'unknown'
        if has_usr_params and 'REPLENISHED_REWARDED_STATES_0' in usr_params:
            val = _chk_type(usr_params['REPLENISHED_REWARDED_STATES_0'])
            if val not in ('unknown', None, ''):
                rwd_after_rebait = val
        elif has_usr_params and 'REWARDED_STATES' in usr_params:
            # Fallback: old all-states approach
            val = usr_params.get('REWARDED_STATES', '')
            if 'all' in str(val).lower():
                rwd_after_rebait = 'all (replenish all states)'

        # ---- notes ----
        notes_parts = [notes_prefix] if notes_prefix else []
        if is_firstWeek:
            notes_parts.append('FLAG: firstWeekOfData — rewarded_states_initial may be incorrect in source CSV; corrected values used where available')
        if 'rwdSwitches2newLocationUponConsumption' in rpi_subdir:
            notes_parts.append(RWD_SWITCHES_FLAG)
        if rwd_initial_source in ('corrected_df', 'rwd_configs_df_first_onset') and is_firstWeek:
            notes_parts.append(f'rwd_initial_source={rwd_initial_source}')
        if not has_usr_params:
            notes_parts.append('no_usr_params: SAME_PORT_MIN_REPOKE_INTERVAL inferred from data')
        if 'mislabled_a185' in full_path:
            notes_parts.append('FLAG: mislabled_a185 — animal was mislabelled a185; excluded from standard pipeline (FLAGGED_STRINGS); real data but treat with caution')
        if 'lastExpWaterRestricted' in rpi_subdir:
            notes_parts.append('FLAG: lastExpWaterRestricted — final session before water restriction ended; excluded from standard pipeline')
        notes = '; '.join(notes_parts)

        rows.append({
            'exp_session_date':            exp_session_date,
            'rpi_session_subdir':          rpi_subdir,
            'animal_ID':                   animal_ID,
            'maze_ID':                     maze_ID,
            'reward_function':             reward_function,
            'rewarded_states_initial':     str(rwd_initial),
            'rewarded_states_after_rebait': str(rwd_after_rebait),
            'reverse_actions_allowed':     reverse_actions_allowed,
            'same_port_min_repoke_s':      same_port_str,
            'rwd_initial_source':          rwd_initial_source,
            'png_validated':               png_validated,
            'png_sim_step_match_rate':     png_sim_step_match_rate,
            'notes':                       notes,
        })

    df = pd.DataFrame(rows)
    df = df.sort_values(['exp_session_date', 'animal_ID']).reset_index(drop=True)

    out_path = os.path.join(DATA_OUT, f'{TIMESTAMP}_session_metadata.csv')
    df.to_csv(out_path, index=False)
    print(f'Wrote {len(df)} rows → {out_path}')

    print(f'\nSkipped {len(skipped)} dirs:')
    for name, reason in skipped:
        print(f'  {name}: {reason}')

    # Summary stats
    print('\n--- Summary ---')
    print('animal_ID counts:')
    print(df['animal_ID'].value_counts().sort_index().to_string())
    print('\nreward_function counts:')
    print(df['reward_function'].value_counts().to_string())
    print('\nreverse_actions_allowed counts:')
    print(df['reverse_actions_allowed'].value_counts().to_string())
    print('\nrwd_initial_source counts:')
    print(df['rwd_initial_source'].value_counts().to_string())
    print('\nmaze_ID counts:')
    print(df['maze_ID'].value_counts().to_string())

    return df, out_path


if __name__ == '__main__':
    os.chdir(os.path.join(PROJECT_ROOT))
    df, out_path = build_metadata()
