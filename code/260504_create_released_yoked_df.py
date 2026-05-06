"""
260504_create_released_yoked_df.py

Create the canonical released yoking dataframe at
data_released/yoked_dfs/<orig_timestamp>_yoked_sessions.csv.

Input: yoked_dfs/c260504-143724_animal_to_agent_yoking_info-chronological.csv
       (538 rows, 360 unique session basenames; 178 sessions appear twice
        because both storage locations were ingested.)

Operations:
  1. De-dup by basename. For each duplicate group, merge values column-by-
     column: take the unique non-nan value if exactly one is non-nan; if
     both rows agree, keep that value; if both differ and both non-nan,
     prefer the row whose path is data_in/1_experiment_csvs/ (curated).
  2. Drop the 1 missing-on-disk session (a000 first session).
  3. Rewrite csv_data_path to ./data_released/raw_session_csvs/<basename>.csv
     so the released df references the bundled CSVs, not the ignored
     data_in/ originals.
  4. Save with the original timestamp prefix (per Author 2026-05-04: keep
     cyymmdd-hhmmss provenance).
"""
from __future__ import annotations
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path('<REPO_ROOT>')
SOURCE = PROJECT_ROOT / 'yoked_dfs' / 'c260504-143724_animal_to_agent_yoking_info-chronological.csv'
BUNDLE_DIR = PROJECT_ROOT / 'data_released' / 'raw_session_csvs'
OUT = PROJECT_ROOT / 'data_released' / 'yoked_dfs' / 'c260504-143724_yoked_sessions.csv'

PREFERRED_PATH_SUBSTR = '1_experiment_csvs'


def merge_group(group: pd.DataFrame) -> pd.Series:
    """For one or two rows sharing a basename, build a single merged row.
    Tie-break logic (per column):
      - if all values agree, keep that value
      - elif exactly one non-nan, keep the non-nan
      - else (both non-nan and differing), prefer the curated-path row
    """
    if len(group) == 1:
        return group.iloc[0].copy()

    # Find the curated-path row (preferred for value tie-break).
    curated_mask = group['csv_data_path'].astype(str).str.contains(PREFERRED_PATH_SUBSTR)
    if curated_mask.any():
        preferred = group[curated_mask].iloc[0]
    else:
        preferred = group.iloc[0]

    merged = preferred.copy()
    for col in group.columns:
        if col == 'csv_data_path':
            continue  # rewritten downstream
        # Drop nan, keep unique stringified values
        vals = group[col].dropna().tolist()
        if not vals:
            continue
        unique_str = {str(v) for v in vals}
        if len(unique_str) == 1:
            merged[col] = vals[0]
        else:
            # Multiple non-nan values that differ — keep the preferred row's value
            merged[col] = preferred[col] if pd.notna(preferred[col]) else vals[0]
    return merged


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(SOURCE)
    print(f'Loaded {len(df)} rows from {SOURCE.name}')

    df['_basename'] = df['csv_data_path'].apply(lambda p: Path(str(p)).name)

    # Sanity: how many bundled files exist?
    bundled = {p.name for p in BUNDLE_DIR.glob('*.csv') if p.name not in
               {'raw_session_manifest.csv', 'missing_files.csv'}}
    print(f'Bundled CSVs available: {len(bundled)}')

    # Drop rows whose basename is not in the bundle (the 1 missing-on-disk session).
    pre_drop = len(df)
    df = df[df['_basename'].isin(bundled)].copy()
    print(f'Dropped {pre_drop - len(df)} rows whose basename is not in the bundle')

    # Merge duplicates per basename.
    merged_rows = []
    for bn, group in df.groupby('_basename', sort=False):
        merged_rows.append(merge_group(group))
    merged = pd.DataFrame(merged_rows).reset_index(drop=True)
    print(f'After per-basename merge: {len(merged)} unique sessions')

    # Rewrite csv_data_path to the released-bundle relative path.
    merged['csv_data_path'] = merged['_basename'].apply(
        lambda bn: f'./data_released/raw_session_csvs/{bn}'
    )
    merged = merged.drop(columns=['_basename'])

    # Order columns to match the source (csv_data_path stays in its original slot)
    merged = merged[df.drop(columns=['_basename']).columns]

    # Sanity: prob_dict completeness vs source
    for col in ('prob_dict_ego', 'prob_dict_allo_real', 'prob_dict_allo_latent'):
        n_nan = merged[col].isna().sum()
        print(f'  {col}: {n_nan} nan / {len(merged)} sessions')

    merged.to_csv(OUT, index=False)
    print(f'\nWrote {OUT.relative_to(PROJECT_ROOT)} ({len(merged)} rows)')


if __name__ == '__main__':
    main()
