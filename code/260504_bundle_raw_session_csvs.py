"""
260504_bundle_raw_session_csvs.py

Bundle the raw experiment CSVs referenced by the canonical yoking df into
data_released/raw_session_csvs/. The yoking df has 538 rows but only 360
unique sessions because 178 sessions appear twice (once with the curated
path under data_in/1_experiment_csvs/, once with the per-session-dir path
under data_in/0_raw_exp_dirs_from_RPi/<session>/<file>.csv); both rows
reference the same underlying CSV. We bundle each unique basename once and
record all original paths in the manifest.

Per Author 2026-05-04: bundle into data_released/ (drop cld_ prefix on mature
data); manifest records relative paths only.

Reads:
  yoked_dfs/c260504-143724_animal_to_agent_yoking_info-chronological.csv

Writes:
  data_released/raw_session_csvs/<basename>.csv  (~360 files, ~28 MB total)
  data_released/raw_session_csvs/raw_session_manifest.csv
  data_released/raw_session_csvs/missing_files.csv  (only if any are missing)
"""
from __future__ import annotations
import hashlib
import re
import shutil
from collections import defaultdict
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path('<REPO_ROOT>')
YOKING_DF = PROJECT_ROOT / 'yoked_dfs' / 'c260504-143724_animal_to_agent_yoking_info-chronological.csv'
OUT_DIR = PROJECT_ROOT / 'data_released' / 'raw_session_csvs'
MANIFEST = OUT_DIR / 'raw_session_manifest.csv'

EXP_MOMENT_RE = re.compile(r'^(\d{6}-\d{6})')


def relative_path(csv_data_path: str) -> str:
    s = str(csv_data_path)
    if s.startswith('.//'):
        return './' + s[3:]
    if s.startswith(str(PROJECT_ROOT)):
        return './' + s[len(str(PROJECT_ROOT)) + 1:]
    if s.startswith('./'):
        return s
    return './' + s


def absolute_path(rel: str) -> Path:
    s = rel
    if s.startswith('./'):
        s = s[2:]
    return PROJECT_ROOT / s


def sha256_of(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for chunk in iter(lambda: f.read(65536), b''):
            h.update(chunk)
    return h.hexdigest()


def derive_exp_moment(basename: str) -> str:
    m = EXP_MOMENT_RE.match(basename)
    return m.group(1) if m else ''


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(YOKING_DF)
    print(f'Loaded {len(df)} rows from {YOKING_DF.name}')

    # Group rows by basename (the unique session-CSV identifier).
    by_basename: dict[str, list[pd.Series]] = defaultdict(list)
    for _, r in df.iterrows():
        bn = Path(str(r['csv_data_path'])).name
        by_basename[bn].append(r)
    print(f'Unique basenames: {len(by_basename)}; rows: {len(df)}')

    rows = []
    missing = []
    copied = 0
    for bn, group in by_basename.items():
        # Pick the first existing source. Prefer 1_experiment_csvs if multiple.
        sorted_group = sorted(
            group,
            key=lambda r: 0 if '1_experiment_csvs' in str(r['csv_data_path']) else 1,
        )
        src = None
        original_paths = []
        for r in sorted_group:
            rel = relative_path(r['csv_data_path'])
            original_paths.append(rel)
            ap = absolute_path(rel)
            if src is None and ap.exists():
                src = ap

        animal_ids = sorted({str(r.get('animal_ID', '')).zfill(3) for r in group})
        animal_id_str = ';'.join(animal_ids)

        if src is None:
            missing.append({
                'exp_moment': derive_exp_moment(bn),
                'animal_ID': animal_id_str,
                'bundled_filename': bn,
                'original_paths': ';'.join(original_paths),
                'reason': 'all candidate paths missing on disk',
            })
            continue

        dst = OUT_DIR / bn
        shutil.copy2(src, dst)
        copied += 1

        rows.append({
            'exp_moment': derive_exp_moment(bn),
            'animal_ID': animal_id_str,
            'bundled_filename': bn,
            'original_paths': ';'.join(original_paths),
            'n_yoking_df_rows': len(group),
            'sha256': sha256_of(dst),
            'file_size_bytes': dst.stat().st_size,
        })

    manifest_df = pd.DataFrame(rows).sort_values('exp_moment').reset_index(drop=True)
    manifest_df.to_csv(MANIFEST, index=False)
    print(f'\nCopied {copied} files into {OUT_DIR}')
    print(f'Manifest: {MANIFEST.relative_to(PROJECT_ROOT)} ({len(manifest_df)} rows)')
    print(f'Total bundled size: {manifest_df["file_size_bytes"].sum()/1e6:.1f} MB')
    print(f'Mean file size: {manifest_df["file_size_bytes"].mean()/1e3:.1f} KB; '
          f'max: {manifest_df["file_size_bytes"].max()/1e3:.1f} KB')

    if missing:
        miss_df = pd.DataFrame(missing)
        miss_path = OUT_DIR / 'missing_files.csv'
        miss_df.to_csv(miss_path, index=False)
        print(f'\nMISSING ({len(missing)} unique basenames): see {miss_path.relative_to(PROJECT_ROOT)}')
        for m in missing:
            print(f'  - {m["exp_moment"]} a{m["animal_ID"]}: {m["bundled_filename"]}')


if __name__ == '__main__':
    main()
