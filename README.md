# LatMaz: A Dataset and Evaluation Benchmark for Biological and Artificial Navigation on Abstract Graphs

**NeurIPS 2026 Evaluations and Datasets Track** (under review). Anonymous mirror: <https://anonymous.4open.science/r/LatMaz_NeurIPS2026EnD_submission_055C>.

LatMaz is a yoked (per-session matched) lab-mouse vs. synthetic RL-agent benchmark for partially-observed graph navigation. Mouse data come from a freely-moving virtual-reality apparatus that instantiates arbitrary latent graphs at a single physical four-arm junction, eliminating path-integration and landmark cues. Agents inherit the mouse's exact graph topology, start state, reward configuration, rebait threshold, and action budget per session.

The release contains:

- **Mouse data**: 12 mice across 422 egocentric main-benchmark sessions; 60 canonical sessions (a031, a033) used for HPO and headline claims; 29-session in-progress allocentric subset reported separately.
- **Apparatus / training / microcontroller materials**: experiment-control scripts at `code/apparatus_control/` (`1_setup_and_calibration/`, `2_canonical_runtime/` incl. `allocentric/`, `3_maze_variants/`, `4_historical/`). Raspberry-Pi-side runtime: `code/apparatus_control/utils_latMaz_apparatus.py`. RPi-side dependencies in `requirementsRPi.txt`.
- **Animal-data analysis code**: `code/` (mouse-trajectory parsing, yoking, statistics).
- **Gymnasium-compatible graph-maze simulator + agent zoo**: `code/yoked_rl_runner.py` (GraphMazeEnv + YokedRLRunner), `intermediate_agents.py`, `advanced_agents.py`. Compatible with Stable-Baselines3.
- **Precomputed per-agent, per-session simulation CSVs**: `data_released/results/` (HPO sweep, baselines, RecurrentSAC, pretraining, cloning, rerun-452, scaling sweep, allocentric subset).
- **Generated paper-value macros**: `data_released/paper_values.tex` (regenerable via `code/260405_generate_paper_values.py`).
- **Figure-generation scripts**: alongside `code/`.
- **Croissant 1.0 metadata**: `data_released/croissant_metadata.json`.

## Quick start (replication recipe)

### 1. Environment

The pipeline is tested with Python 3.11 and the `latMaz_RL` conda environment.

```bash
# create and activate environment
conda create -n latMaz_RL python=3.11 -y
conda activate latMaz_RL
pip install -r requirements.txt
```

### 2. Reproduce paper values from released CSVs

```bash
cd code/
python 260405_generate_paper_values.py
# writes data_released/paper_values.tex (matches the canonical version bit-for-bit)
```

The script reads from `data_released/results/` by default and falls back to `data_out/rl_sims/` for in-development runs.

### 3. Re-run a single yoked simulation cell

```bash
cd code/
python -c "
from experiment_config import load_data, build_runner, apply_best_configs, obs_kwargs
yoked, rwd, sessions = load_data()
runner = build_runner(yoked, rwd)
runner.algo_config = apply_best_configs(['DQN'])
sess = sessions[0]
yoke = yoked[yoked['exp_moment']==sess].iloc[0]
adj, st = runner._load_maze(yoke['adj_file'], yoke['st_pos_file'])
rewards, reset_val = runner._get_rewarded_states(sess, len(st))
total = runner.run_single(model_name='DQN', adj_mat=adj, st_positions=st,
    start_node=int(yoke['start_state']), rewards=rewards,
    n_actions=int(yoke['n_states_visited'])-1, **obs_kwargs(),
    min_allowed_rewarded_states=int(reset_val), seed=0)
print(f'DQN seed=0 on {sess}: total_reward={total}')
"
```

### 4. Re-run the held-out cohort (rerun-452, 8 RL agents x 452 sessions x 5 seeds)

```bash
python 260505_rerun452_corrected_BESTCONFIGS.py --workers 4
# ~2 hours on a 16-core M3 Max with 4 workers; output CSV at
# data_out/rl_sims/c{ts}_rerun452_corrected_8agents.csv
```

### 5. Reproduce a figure

Each figure generator under `code/` reads released CSVs and writes to `reports/figures/c{ts}_<name>.png`. Example:

```bash
python 260505_plot_per_animal_categories.py
# writes reports/figures/c{ts}_per_animal_categories_overview.png
```

## Project structure

```

├── paper/                              # LaTeX manuscript + generated PDF + paper_values
├── code/                               # active RL code: runner, agents, experiments, figure scripts
├── data_released/                      # canonical release artifacts
│   ├── croissant_metadata.json         #   Croissant 1.0 metadata
│   ├── yoked_dfs/                      #   canonical yoked-sessions dataframe
│   ├── raw_session_csvs/, mazes/, *.zip
│   └── results/                        #   precomputed simulation CSVs (14 files)
├── reports/figures/                    # generated figures
├── data_in/, data_out/                 # working dirs (gitignored)
└── docs/                               # workflow / approach notes
```

## Replication validation across machines

The pipeline is deterministic at the per-(agent, session, seed) level. Independent replication on a second machine should match seed-for-seed:

```bash
# on the second machine, after cloning:
conda activate latMaz_RL
python code/260505_rerun452_corrected_BESTCONFIGS.py \
    --workers 4 --limit 200 \
    --output /tmp/rerun452_replication_smoke.csv
# compare /tmp/rerun452_replication_smoke.csv to
# data_released/results/c260505-120513_rerun452_corrected_8agents.csv
# for the same (model, exp_moment, seed) tuples — RPAs should match exactly.
```

## HPO replication / reviewer reproducibility test

This section is the focused recipe for a separate machine / agent to reproduce the
HPO-tuning + rerun-452 results end-to-end. It assumes the canonical artifact
in `data_released/` and the `latMaz_RL` env from §1 are in place.

### Step 1 — reproduce paper-values.tex from released CSVs (no compute)

```bash
cd code/
python 260405_generate_paper_values.py
diff -u ../data_released/paper_values.tex /dev/null   # snapshot pre-run for comparison
# script writes ../data_released/paper_values.tex; should be bit-identical to
# the committed canonical version. If it differs, the released CSVs in
# data_released/results/ have drifted from the paper claims — flag immediately.
```

The script reads from `data_released/results/` first (then falls back to
`data_out/rl_sims/`); `sessions_60` is pinned to the HPO CSV's exp_moments.

### Step 2 — single-cell sanity check (the simulator runs deterministically)

```bash
cd code/
python -c "
from experiment_config import load_data, build_runner, apply_best_configs, obs_kwargs
yoked, rwd, _ = load_data()
runner = build_runner(yoked, rwd)
runner.algo_config = apply_best_configs(['DQN'])  # NOTE: assign to algo_config, not algo
sess = '240222-151220'  # canonical example session
yoke = yoked[yoked['exp_moment']==sess].iloc[0]
adj, st = runner._load_maze(yoke['adj_file'], yoke['st_pos_file'])
rewards, reset_val = runner._get_rewarded_states(sess, len(st))
total = runner.run_single(model_name='DQN', adj_mat=adj, st_positions=st,
    start_node=int(yoke['start_state']), rewards=rewards,
    n_actions=int(yoke['n_states_visited'])-1, **obs_kwargs(),
    min_allowed_rewarded_states=int(reset_val), seed=0)
rpa = total / max(int(yoke['n_states_visited'])-1, 1)
print(f'DQN aggressive_explore on {sess} seed=0: rpa={rpa:.6f}')
# Expected (post-fix BEST_CONFIGS, 2026-05-05): rpa=0.178571 (10 / 56)
"
```

If the printed RPA matches `0.178571` (or `0.178571428...`), the simulator + HPO-best
config plumbing is reproducing seed-for-seed. Mismatch → either the env has changed
or `BEST_CONFIGS` is stale; check `experiment_config.py:38-84`.

### Step 3 — small-slice rerun-452 smoke test (~5 min on 4 cores)

```bash
cd code/
python 260505_rerun452_corrected_BESTCONFIGS.py \
    --workers 4 --limit 200 \
    --output /tmp/rerun452_replication_smoke.csv

# Compare to released file row-by-row for the same (model, exp_moment, seed):
python -c "
import pandas as pd
new = pd.read_csv('/tmp/rerun452_replication_smoke.csv')
ref = pd.read_csv('../../data_released/results/c260505-120513_rerun452_corrected_8agents.csv')
key = ['model', 'exp_moment', 'seed']
m = new.merge(ref, on=key, suffixes=('_new', '_ref'))
mismatches = m[(m['rpa_new'] - m['rpa_ref']).abs() > 1e-6]
print(f'cells compared: {len(m)}, mismatches (|Delta rpa| > 1e-6): {len(mismatches)}')
"
```

Expected: 0 mismatches.

### Step 4 (optional, heavy) — full rerun-452 reproduction (~2 hr on 4 cores, ~1 hr on 8)

```bash
python 260505_rerun452_corrected_BESTCONFIGS.py --workers 4
# writes data_out/rl_sims/c{ts}_rerun452_corrected_8agents.csv (18,080 rows)
# expected to match data_released/results/c260505-120513_*.csv seed-for-seed.
```

### Known reproducibility caveats

- **Seed offsets in HPO sweep vs rerun-452**: the original HPO sweep
  (`tune_models.py:135-160`) uses `seed_offset = hash(config_name) % 10000`,
  while the rerun wrapper uses plain `seed=0..4`. So canonical-60 macros from
  rerun-452 and from `c260316-171035_hpo_tuning_FIXED.csv` are **close
  (within ±0.07 RPA) but not identical** — this is expected seed-stochasticity
  divergence, not a bug.
- **Rebait notation drift**: paper §2 uses `N` ("rebait when fewer than $N$
  unvisited remain"); §4.7 uses `R` ("rebait at $R$ unrewarded remaining");
  the data field in `c260302_rewarded_states_df.csv` is `reset_when_n_rwds_remaining`
  ($\equiv R$, $\equiv N - 1$). All canonical-60 sessions are $N{=}2 / R{=}1$.
- **Adjacency loading**: ALWAYS use `utils_latMaz.load_maze` or
  `YokedRLRunner._load_maze`. Naive `pd.read_csv(..., header=None).values`
  without symmetrization gives wrong navigable-node counts because the CSV
  stores only the upper triangle. The canonical helper `utils_latMaz.load_maze`
  symmetrises before use; ad-hoc `pd.read_csv(adj_path, header=None).values`
  in analysis scripts is incorrect.
- **noRev asymmetry**: `GraphMazeEnv` defaults to `prevent_reverse=False`;
  RL agents can issue immediate-reverse actions that the mouse is physically
  prevented from issuing within the apparatus's 45-90s sensor timeout. Per-action
  cost is symmetric (reversal consumes one action either way). See paper App C.

## Hugging Face Datasets mirror (camera-ready)

The dataset is mirror-eligible for Hugging Face Datasets at camera-ready time.
See `docs/hf_dataset_setup.md` for the full procedure: throwaway org creation,
upload via `huggingface-cli`, dataset-card README template, Croissant
auto-render, and the migration path from anonymous-throwaway-org to canonical
post-acceptance. Not required for double-blind review (the
`anonymous.4open.science` mirror suffices).

## Citation

Anonymous Authors (2026). *LatMaz: A Dataset and Evaluation Benchmark for Biological and Artificial Navigation on Abstract Graphs.* NeurIPS 2026 Evaluations & Datasets Track (under review).

## License

- Code: MIT
- Dataset: CC-BY-4.0

See the dataset card appendix in the submitted PDF for full collection, preprocessing, intended uses, and limitations.
