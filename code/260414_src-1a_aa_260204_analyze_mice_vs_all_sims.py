import marimo

__generated_with = "0.19.1"
app = marimo.App(width="full", auto_download=["html", "ipynb"])


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 1a_aa_ analyze mice vs all sims
    """)
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _(mo):
    mo.md(r"""
    251222 Author: these versions appear to work
    - NumPy version: 2.1.2
    - Pandas version: 2.2.3
    - Matplotlib version: 3.9.2
    - Seaborn version: 0.13.2
    """)
    return


@app.cell
def _():
    import os
    import re
    import numpy as np
    import pandas as pd
    import matplotlib
    import matplotlib.pyplot as plt
    import seaborn as sns
    from pathlib import Path
    from tqdm import tqdm
    from glob import glob
    import collections
    from datetime import datetime
    import shutil
    import pdb
    from copy import deepcopy
    import gzip
    import sys
    import imageio
    import gc
    from datetime import datetime
    from time import time
    import sys
    from datetime import timedelta
    import pyarrow as pa

    sys.path.append('./')
    print('240618 Author: I assume utils_latMaz is in the same directory as the notebook')

    """ allows auto reloading the dependencies """
    import importlib
    ## 1) import the modules themselves
    import utils_latMaz
    import latMaz_policies
    import utils_latMaz_analysis
    ## 2) reload them so edits on disk are picked up
    importlib.reload(utils_latMaz)
    importlib.reload(latMaz_policies)
    importlib.reload(utils_latMaz_analysis)
    ## 3) NOW pull what you need into the notebook namespace

    from utils_latMaz import (get_most_recent_file, chk_type, get_moment_strings, load_maze, get_now_str, get_adj_states, 
        displacement_to_compass_heading, get_direction_from_radian, allo_radian_map_dict, get_ego_direction, get_verified_st_traj, move_df_col_to_leftmost,)

    from latMaz_policies import policy_random, policy_action_biased_trunc_v3_minimal, policy_obs_action_biased, \
        policy_latent_state_biased, zero_pad_numbers, generate_trajectory_v4

    # from utils_latMaz_analysis import extract_valid_moment, sanity_check_latent_vs_real_allo_actions, extract_rewarded_states_from_usr_params, get_n_rewards_obtained_by_simulations, get_n_rewards_obtained_by_simulations_fast, check_n_states_match_per_exp
    # from utils_latMaz_analysis import extract_valid_moment, sanity_check_latent_vs_real_allo_actions, extract_rewarded_states_from_usr_params, get_n_rewards_obtained_by_simulations_fast, check_n_states_match_per_exp, reformat_fixed_policy_sim_df

    from utils_latMaz_analysis import (extract_valid_moment, sanity_check_latent_vs_real_allo_actions, extract_rewarded_states_from_usr_params,
        check_n_states_match_per_exp, reformat_fixed_policy_sim_df)


    import filecmp

    # PROJECT_DIR = '<DATA_ROOT>/latent_maze-cup/a_latMaz_neurIPSworkshopPrep'
    PROJECT_DIR = '<DATA_ROOT>/latent_maze-cup/a_latMaz-checking_n_deepRL'

    # Configure plotting
    sns.set_style("whitegrid")  # Using seaborn's set_style instead of plt.style
    # %matplotlib inline

    print("250612 CRS: Environment setup complete. Using Python packages:")
    print(f"NumPy version: {np.__version__}")
    print(f"Pandas version: {pd.__version__}")
    print(f"Matplotlib version: {matplotlib.__version__}")
    print(f"Seaborn version: {sns.__version__}")
    return (
        check_n_states_match_per_exp,
        chk_type,
        extract_rewarded_states_from_usr_params,
        extract_valid_moment,
        gc,
        get_moment_strings,
        get_most_recent_file,
        get_now_str,
        get_verified_st_traj,
        glob,
        move_df_col_to_leftmost,
        np,
        os,
        pa,
        pd,
        reformat_fixed_policy_sim_df,
        sanity_check_latent_vs_real_allo_actions,
        tqdm,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 251221 GPT: notes on optional future speed improvements
    1a_ speed notes (stable but slow)

    Main bottlenecks observed
    - **CSV parsing + concat** across many sim result files (text parsing + dtype inference + object arrays).
    - **Row-wise parsing of repeated objects** (esp. `policy_dict`) via `.apply(...)` on every row.
    - **Eager parsing of huge list-like columns** (e.g., `actions_taken_*`) even when downstream only needs scalars.
    - **Slow pandas patterns**: `query(...).iterrows()` inside loops; repeated `rewarded_states_df.query(...)`.
    - **Inefficient truncation** of state/action sequences via `while pop()` loops.

    ---

    Safe + simple speed optimizations (keep semantics the same)

     1) Read less: `usecols` at CSV read time
    If the run only needs stats/plots, **don’t read** columns you won’t use (especially list-like `actions_taken_*`).
    - Good “slim” read set typically includes only:
      - keys: `csv_data_path`, `repeat_group_idx`
      - model label inputs: `policy_dict` (only if needed to build `full_sim_desc`)
      - reward extraction inputs: `states_visited` (only if reward counts are computed in 1a)

     2) Faster CSV engine + dtype backend
    Use pyarrow-backed parsing to reduce overhead and memory pressure:
    - `pd.read_csv(..., engine="pyarrow", dtype_backend="pyarrow")`
    - still combine with `usecols=[...]` for the biggest win

     3) Convert groupby keys to `category` early
    After standardization / `full_sim_desc` creation:
    - `animal_ID`, `full_sim_desc`, `policy_category`, `representation`, `training_stage` → `category`
    This improves groupby speed and memory footprint.

     4) Vectorize ID extraction (avoid Python loops)
    Replace per-row path parsing with regex-based vectorized extraction:
    - `exp_moment` from filename via `str.extract(...)`
    - `animal_ID` from filename via `str.extract(...)`
    Add asserts that extraction succeeded.

     5) Parse `policy_dict` once per **unique value**, then map back
    Instead of `sim_df.policy_dict.apply(chk_type)` across all rows:
    - compute `unique_policy_dicts = sim_df["policy_dict"].unique()`
    - parse each unique dict once
    - `sim_df["policy_dict"] = sim_df["policy_dict"].map(parsed_map)`
    Same for extracting `policy_category`, `representation`, etc.

     6) Don’t parse `actions_taken_*` in normal runs
    Only parse action lists when explicitly debugging / sanity checking:
    - add `parse_actions=False` flag to reformatting function
    - keep default off

     7) Fix `get_n_rewards_obtained_by_simulations()` hotspots
    - **Pre-merge** rewarded states onto `sim_df` once (avoid repeated `rewarded_states_df.query(...)`)
    - Iterate with `for row in grp.itertuples(index=False):` (avoid `.query(...).iterrows()`)
    - Replace `while pop()` alignment with slice-to-min-len:
      - `min_len = min(len(mouse), len(agent))`
      - `mouse = mouse[:min_len]`, `agent = agent[:min_len]`
      - track dropped counts via length differences

     8) Avoid `apply(axis=1)` for RL `full_sim_desc`
    Build `full_sim_desc` via vectorized string concatenation:
    - `"stableBaselines_"+agent_model+";action_"+action_type+";obs_"+obs_type`

    ---

     Feather cache approach (complementary, recommended)
     Why feather helps
    - Avoids CSV text parsing on repeated runs
    - Preserves chosen dtypes (category/int/string)
    - Supports **column selection at load time**:
      - `pd.read_feather(PATH, columns=[...])`
    - Caveat: feather does **not** do row-filter pushdown; filtering happens after load.

     Recommended split: “slim feather” + optional debug CSV
    **Slim feather** for downstream plotting/stats (1b):
    - only scalar columns, e.g.:
      - `animal_ID`, `exp_moment`, `csv_data_path`, `repeat_group_idx`, `full_sim_desc`
      - `n_states_visited`, `n_rewards_obtained_mouse`, `n_rewards_obtained_agent`
    - no list/dict columns

    **Debug artifact** (only when needed):
    - CSV containing `states_visited`, `rewarded_action_inds_*`, etc.

     Where to cache
    - Best immediate payoff: cache **1a outputs** as slim feather (accelerates 1b iterations).
    - Secondary: optionally convert repeated input sim CSVs → feather once, if 1a is rebuilt often.

    ---

     Standardization to move earlier (reduce glue work in 1a)
    - Adopt **one canonical time key** everywhere: `exp_moment` (consistent formatting).
    - Standardize `animal_ID` formatting at write-time in upstream stage (avoid patch logic later).
    - Normalize paths (`os.path.normpath`) upstream when writing outputs.
    - Write `full_sim_desc` upstream (0b) so 1a doesn’t need to parse `policy_dict` for labeling.
    - Consider writing both upstream outputs:
      - slim (scalar-only) artifact for analysis
      - debug (list/dict) artifact for inspection

    ---

     Notes / TODOs captured from markdown comments
    - TODO: move notebook-defined functions into a utils file.
    - TODO: unify moment string strategy so experiment filenames and usr_params filenames match exactly (same second).
    - TODO: move path standardization upstream.
    - TODO: add assert that rewarded-state definitions cover expected states (“initial rewarded states is all of the states”).
    - Note: saving is much faster if `policy_dict` is dropped (supports “slim artifact” approach).
    - Downstream (1b) TODO: move usr_param filename fix upstream into the RPi experiment code.
    """)
    return


@app.cell
def _(os):
    PARENT_DIR = '/jukebox'
    if not os.path.exists(PARENT_DIR):
        PARENT_DIR = '/Volumes'
    PARENT_DIR
    return (PARENT_DIR,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # (optional) reload yoked_df
    - 251221 Author: not necessary actually? Keep around for reference and debugging tho
    """)
    return


@app.cell
def _(
    ADJ_FILTER_STR,
    chk_type,
    extract_valid_moment,
    get_most_recent_file,
    pd,
):
    RELOAD_YOKED_DF = True

    ## only used if RELOAD_YOKED_DF = True

    # YOKED_DF_PATH = get_most_recent_file('./yoked_dfs/*animal_to_agent_yoking_info*.csv') # legacy
    YOKED_DF_PATH = get_most_recent_file('./yoked_dfs/*yoked_df-w_obs*.csv') # w/ extra observation info added

    if RELOAD_YOKED_DF:

        FILTER = False
        # ADJ_FILTER_STR = 'mop001half' # requires FILTER = True

        print(f"loading yoking_df from: {YOKED_DF_PATH}")
        yoking_df = pd.read_csv(YOKED_DF_PATH, dtype={'animal_ID': str})
        assert yoking_df.iloc[0].n_states_visited == len(chk_type(yoking_df.iloc[0].states_visited)), 'Author: mismatch suggests a clash in code versions'
        ## extract the yymmdd-hhmmss string from the csv data file name
        yoking_df['exp_moment'] = yoking_df.csv_data_path.apply(extract_valid_moment)
        ## legacy approach but with a single _ moment separator
        # yoking_df['moment'] = exp_moment
        # yoking_df.moment = [f"{e[0]}-{e[1]}" if len(e) == 2 else e for e in yoking_df.moment.to_list()]
        # yoking_df.moment = pd.to_datetime(yoking_df.moment, format='%y%m%d-%H%M%S')

        if FILTER:
            # yoking_df = yoking_df.query("moment > '2024-09-01'") # filter by date here
            yoking_df = yoking_df[yoking_df.adj_file.str.contains(ADJ_FILTER_STR, na=False)]

        yoking_df, sorted(yoking_df.exp_moment.to_list())[::-1]
    return (yoking_df,)


@app.cell
def _(yoking_df):
    yoking_df.animal_ID.unique()
    return


@app.cell
def _(mo):
    mo.md(r"""
    # (optional): combine separate fixed policy simulation results (merge all files into 1)

    - 251226 Author:
        - forgot to add this before (so results between 251221 and 251226 are stale)
        - NOTE: can be skipped if the most recent file is new enough for you
    """)
    return


@app.cell
def _(get_moment_strings, glob, np, os):
    RECOMBINE_UPSTREAM_FIXED_POLICY_SIMS = True

    if RECOMBINE_UPSTREAM_FIXED_POLICY_SIMS:
        SELECTED_SIM_PATTERN = 'latent_maze_fixed_policy_sims'
        DISCARD_SIMS_BEFORE = 251224 # Author: note we're using an int here cuz we'll compare below
        N_REPEATS_USED = np.inf # Author: set to int or np.inf (if you want to exclusion)

        assert type(DISCARD_SIMS_BEFORE) == int, f'Author: downstream code expects DISCARD_SIMS_BEFORE is int not {type(DISCARD_SIMS_BEFORE)}'

        all_sim_paths = sorted(glob('./simulations/*'))
        selected_sim_paths = [p for p in all_sim_paths if SELECTED_SIM_PATTERN in p]
        selected_sim_paths = [p for p in selected_sim_paths if int(get_moment_strings(p)) >= DISCARD_SIMS_BEFORE] # filter by moment string
        selected_sim_paths = [p for p in selected_sim_paths if int(os.path.splitext(os.path.basename(p))[0].split('repeat')[1]) < N_REPEATS_USED] # discard repeats above N_REPEATS_USED

        # all_sim_paths
        selected_sim_paths
    return RECOMBINE_UPSTREAM_FIXED_POLICY_SIMS, selected_sim_paths


@app.cell
def _(selected_sim_paths):
    selected_sim_paths
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## load and reformat the simulation result dataframe to be more convenient
    - 260107 Author: took 1.5 mins... (notebook on compute-cluster)
    - 251210 Author: took ~ 1.5 hours for notebook running on compute-cluster
    - 251204 Author: took 6 mins for 3.1 mill rows
    - 250901 Author: took ~ 45 mins for 1.6 million rows
    - 250626 Author: took ~ 5 mins for 350k rows

    - 251226 Author: 5 secs mins via tmux on compute-cluster but for only 10 repeats...
    - 8 mins via tmux on compute-cluster
    """)
    return


@app.cell
def _(pd, selected_sim_paths):
    _z_df = pd.read_csv(selected_sim_paths[0])
    # _z_df.columns
    _z_df.animal_ID.unique()
    return


@app.cell
def _(
    RECOMBINE_UPSTREAM_FIXED_POLICY_SIMS,
    gc,
    os,
    pa,
    pd,
    reformat_fixed_policy_sim_df,
    selected_sim_paths,
    tqdm,
):
    if RECOMBINE_UPSTREAM_FIXED_POLICY_SIMS:
        sim_dfs = []
        for p in tqdm(selected_sim_paths):
            print(os.path.basename(p))
            sim_df = pd.read_csv(p)
            print(f"  - {len(sim_df)} rows")
            sim_dfs.append(sim_df)

        sim_df = pd.concat(sim_dfs, ignore_index=True)
        del sim_dfs
        gc.collect()

        ## good case
        try: 
            sim_df = reformat_fixed_policy_sim_df(sim_df, check_all_sim_results_homogenous=False)
            print('Author: finished! : D phew, no deserialization issues!')

        ## darn, upstream is not compatible... but I'll try to be helpful...
        except Exception as e: 
            print(f"Author: encountered error during reformatting, likely due to a tricky datatype in the upstream dataframes"
                f"(is there a function handle? If so, drop it.): {e}")
            for c in sim_df.columns:
                try:
                    pa.array(sim_df[c])
                except Exception as e:
                    if c != 'policy_dict': # expected failure
                        print(f"Column {c!r} fails: {e}")
    return (sim_df,)


@app.cell
def _(mo):
    mo.md(r"""
    251209 Author: 3 mins for
    251204 Author: 10 mins for 3.1 mil rows
    """)
    return


@app.cell
def _(sim_df):
    type(sim_df.policy_dict.iloc[0])
    return


@app.cell
def _():
    return


@app.cell
def _(mo):
    mo.md(r"""
    251210 Author: 3 mins to save if we drop the policy dict column...
    """)
    return


@app.cell
def _(sim_df_simple):
    # sys.exit('Author wip: inspect the sim_df to make sure that we can drop the policy_dict')
    sim_df_simple.animal_ID.unique()
    return


@app.cell
def _(get_now_str, sim_df):
    sim_df_simple = sim_df.drop(columns=['policy_dict'])
    sim_df_simple.to_feather(f"./combined_sim_dfs/c{get_now_str()}_combined_sims_df-woPolicyDict.feather")
    return (sim_df_simple,)


@app.cell
def _(sim_df):
    sim_df.animal_ID.unique()
    return


@app.cell
def _(mo):
    mo.md(r"""
    251226 Author: legacy GPT reformatting, unclear if necessary/helpful
    """)
    return


@app.cell
def _():
    # ## 251225 Author: is this still necessary? 251204 Author: 7 mins for 3.1 mil rows

    # # def make_jsonable(x):
    # #     if isinstance(x, dict):
    # #         return {str(k): make_jsonable(v) for k,v in x.items()}
    # #     if isinstance(x, (list, tuple, set)):
    # #         return [make_jsonable(v) for v in x]
    # #     if isinstance(x, np.generic): return x.item()
    # #     if pd.isna(x): return None
    # #     return x

    # # # sim_df = sim_df.copy()
    # # sim_df['policy_dict'] = sim_df['policy_dict'].apply(lambda d: make_jsonable(d) if isinstance(d, dict) else d)

    # import numpy as np
    # import pandas as pd

    # def _make_jsonable(x, _depth=0, _max_depth=50, _seen=None):
    #     if _seen is None: _seen = set()

    #     # depth / cycle guards
    #     if _depth > _max_depth:
    #         # fall back to a string so we don't recurse forever
    #         return f"<max_depth_reached type={type(x).__name__}>"
    #     oid = id(x)
    #     if oid in _seen:
    #         return f"<cycle type={type(x).__name__}>"
    #     _seen.add(oid)

    #     # fast path for safe scalars
    #     if isinstance(x, (str, int, float, bool)) or x is None:
    #         return x

    #     # numpy scalars
    #     if isinstance(x, np.generic):
    #         return x.item()

    #     # numpy arrays -> list (JSONable)
    #     if isinstance(x, np.ndarray):
    #         # if these can be huge, consider truncating or summarizing instead
    #         return [_make_jsonable(v, _depth+1, _max_depth, _seen) for v in x.tolist()]

    #     # dicts
    #     if isinstance(x, dict):
    #         return {str(k): _make_jsonable(v, _depth+1, _max_depth, _seen)
    #                 for k, v in x.items()}

    #     # list / tuple / set
    #     if isinstance(x, (list, tuple, set)):
    #         seq = [_make_jsonable(v, _depth+1, _max_depth, _seen) for v in x]
    #         if isinstance(x, list):  return seq
    #         if isinstance(x, tuple): return tuple(seq)
    #         return list(seq)  # for set

    #     # pd.isna only on scalar-ish things (avoid arrays/Series)
    #     try:
    #         if pd.api.types.is_scalar(x) and pd.isna(x):
    #             return None
    #     except TypeError:
    #         pass

    #     # fallback: leave as-is; json.dumps(default=str) can handle the rest if needed
    #     return x


    # def make_jsonable(x):
    #     # tiny wrapper so callers don't have to pass seen/depth
    #     return _make_jsonable(x, _depth=0, _max_depth=50, _seen=set())


    # col = 'policy_dict'
    # mask = sim_df[col].map(lambda d: isinstance(d, dict))

    # idx = sim_df.index[mask]
    # CHUNK_SIZE = 10_000  # adjust to taste

    # for start in range(0, len(idx), CHUNK_SIZE):
    #     batch_idx = idx[start:start+CHUNK_SIZE]
    #     sim_df.loc[batch_idx, col] = (
    #         sim_df.loc[batch_idx, col].map(make_jsonable)
    #     )
    return


@app.cell
def _(mo):
    mo.md(r"""
    # (MUST RUN): load most recent combined fixed policy simulation file
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    251221 Author:
    - note: see 1a_251221_analyze_mice_vs_all_sims-STABLE_butSlow.ipynb for code to load the fixed policy and RL simulations separately (or just rewrite)
        - consolidating here via a single input file
    - loaded in 1.5 mins
    """)
    return


@app.cell
def _(get_most_recent_file, pd):
    READ_FROM_LOCAL = False # True only useful if working remotely via bad wifi or something AND you have the file on your computer already. 
    LOCAL_PARENT_DIR = '<REPO_ROOT>/data_in/' # only used if READ_FROM_LOCAL = True

    if READ_FROM_LOCAL: 
        most_recent_combined_sim_path = get_most_recent_file(f'{LOCAL_PARENT_DIR}*combined_sims_df.feather')
    else:
        most_recent_combined_sim_path = get_most_recent_file('./combined_sim_dfs/*.feather')

    sim_df_reloaded = pd.read_feather(most_recent_combined_sim_path)

    most_recent_combined_sim_path
    return (sim_df_reloaded,)


@app.cell
def _(np, pd, sim_df_reloaded):
    COLS_TO_NAN_CHECK = sim_df_reloaded.columns.to_list()

    def check_df_cols_for_nans(df, col_list): # 251226 Author naive
        nan_cols = []
        for c in col_list:
            if df[c].isna().any():
                nan_cols.append(c)
        return nan_cols

    ## 251226 Author: GPT generalization of the above
    def check_df_missing_GPT(df, col_list=None, *,
                         check_inf=False,
                         check_empty_str=False,
                         empty_str_values=("",),
                         strip_str=True,
                         return_mode="report",  # "report" | "cols"
                         sort_counts=True):
        """
        Robust-ish missing/bad-value checker.

        Always checks pandas-missing (NaN/None/pd.NA/NaT) via isna().
        Optionally checks:
          - +/-inf in numeric columns (check_inf=True)
          - empty/whitespace strings (check_empty_str=True)

        Handles missing column names (won't KeyError): returns them under "missing_cols".

        return_mode:
          - "report": dict with details + counts
          - "cols": ordered list of columns (present cols only) with any issue
        """
        if col_list is None: col_list=list(df.columns)
        else:
            seen=set()
            col_list=[c for c in col_list if not (c in seen or seen.add(c))]

        present=[c for c in col_list if c in df.columns]
        missing=[c for c in col_list if c not in df.columns]

        if len(present)==0:
            report={
                "present_cols":[],
                "missing_cols":missing,
                "bad_cols":[],
                "na_counts":pd.Series(dtype="int64"),
                "inf_counts":pd.Series(dtype="int64"),
                "empty_str_counts":pd.Series(dtype="int64"),
            }
            return report if return_mode=="report" else []

        sub=df.loc[:,present]

        na_counts=sub.isna().sum()
        na_counts=na_counts[na_counts>0]
        if sort_counts and len(na_counts): na_counts=na_counts.sort_values(ascending=False)

        inf_counts=pd.Series(dtype="int64")
        if check_inf:
            num_cols=sub.select_dtypes(include="number").columns
            if len(num_cols):
                arr=sub[num_cols].to_numpy(dtype=float, copy=False)
                counts=np.isinf(arr).sum(axis=0)
                inf_counts=pd.Series(counts, index=num_cols)
                inf_counts=inf_counts[inf_counts>0]
                if sort_counts and len(inf_counts): inf_counts=inf_counts.sort_values(ascending=False)

        empty_str_counts=pd.Series(dtype="int64")
        if check_empty_str:
            if empty_str_values is None: empty_str_values=("",)
            str_cols=sub.select_dtypes(include=["object","string"]).columns
            if len(str_cols):
                s=sub[str_cols].astype("string")
                if strip_str: s=s.str.strip()
                mask=s.isin(list(empty_str_values))
                empty_str_counts=mask.sum()
                empty_str_counts=empty_str_counts[empty_str_counts>0]
                if sort_counts and len(empty_str_counts): empty_str_counts=empty_str_counts.sort_values(ascending=False)

        bad_set=set(na_counts.index) | set(inf_counts.index) | set(empty_str_counts.index)
        bad_cols=[c for c in present if c in bad_set]  # preserve original order

        report={
            "present_cols":present,
            "missing_cols":missing,              # names in col_list not found in df
            "bad_cols":bad_cols,                 # any issue among present cols (ordered)
            "na_counts":na_counts,               # per-col NA count (>0 only)
            "inf_counts":inf_counts,             # per-col inf count (>0 only) if enabled
            "empty_str_counts":empty_str_counts, # per-col empty-string count (>0 only) if enabled
        }
        return report if return_mode=="report" else bad_cols
    ## calling side usage
    # # strict NA only (like your original, but vectorized + missing-col-safe)
    # bad=check_df_missing(sim_df_reloaded, sim_df_reloaded.columns, return_mode="cols")

    # # richer debug report (recommended when tracking down problems)
    # r=check_df_missing(sim_df_reloaded, sim_df_reloaded.columns,
    #                    check_inf=True,check_empty_str=True,return_mode="report")
    # r["missing_cols"], r["bad_cols"], r["na_counts"].head()

    assert len(check_df_cols_for_nans(sim_df_reloaded, COLS_TO_NAN_CHECK)) == 0, f"Author: found NaNs in columns: {check_df_cols_for_nans(sim_df_reloaded, COLS_TO_NAN_CHECK)}"

    check_df_missing_GPT(df=sim_df_reloaded, col_list=COLS_TO_NAN_CHECK)
    return


@app.cell
def _(sim_df_reloaded):
    sim_df_reloaded.columns
    return


@app.cell
def _():
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Action allo vs allo real sanity check on simulations
    - 251222 Author: attempting to optimize this check (run_mode := 'naive' -> 'fast'): runs in 3 mins
    - 251211 Author: tests passed in 10 mins
    - 250904 Author: all simulations passed
    """)
    return


@app.cell
def _(sim_df_reloaded):
    sim_df_reloaded.animal_ID.unique()
    return


@app.cell
def _(sim_df_reloaded):
    sim_df_reloaded.query("repeat_group_idx == 0").full_sim_desc.value_counts()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # most trusted reward extraction

    - 251213 Author: create a shared moment string for both the experiment and user parameters file names!
      - (PARTIALLY ADDRESSED) 251222 Author: should be implemented in at least some of the experiment files (EARLIEST AND LATEST NEED TO BE CHECKED!)
    - 251204 Author todo: compare to earlier reward extractions
    """)
    return


@app.cell
def _(extract_rewarded_states_from_usr_params, glob, sim_df_reloaded):
    exp_user_param_paths = sorted(glob(f"./data_in/0_raw_exp_dirs_from_RPi/**/*usr_params*.csv", recursive=True))

    rewarded_states_df = extract_rewarded_states_from_usr_params(sim_df_reloaded, exp_user_param_paths)
    return (rewarded_states_df,)


@app.cell
def _(rewarded_states_df):
    rewarded_states_df.reset_when_n_rwds_remaining.value_counts()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    251221 Author sanity check: should be 13 inputs between 251119-171644 and 251130-23221;
    - note: legacy experiment script had an edge case where the experiment data and user params have moment strings that are off by 1 second (should be fixed now)
    """)
    return


@app.cell
def _(sim_df_reloaded):
    sorted(list(sim_df_reloaded.query("animal_ID == 'a031'").moment.unique())), sim_df_reloaded.query("animal_ID == 'a031'").moment.nunique()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # filter out data with uncertain rewarded states or poor performing animals
    - 251221 Author TODO: can the conversion back be streamlined?
        - ran in 1.5 mins on compute-cluster
    """)
    return


@app.cell
def _(chk_type, sim_df_reloaded):
    ANIMAL_IDS_INCLUDED = ['a001', 'a002', 'a003', 'a007', 'a008', 'a185', 'a188', 'a029', 'a030', 'a033', 'a031']
    # ANIMAL_IDS_INCLUDED = ['a001', 'a002', 'a003', 'a185', 'a188', 'a029', 'a030', 'a033', 'a031']
    # ANIMAL_IDS_INCLUDED = ['a029', 'a030', 'a033', 'a031']
    EARLIEST_EXP_MOMENT_INCLUDED = '240828-152257' # Author: 240828-152257 is the first file to have the user parameters populated which greatly simplifies analysis; todo: investigate earlier data

    sim_df_filtered = sim_df_reloaded.query(f"animal_ID in {ANIMAL_IDS_INCLUDED} and moment >= @EARLIEST_EXP_MOMENT_INCLUDED").copy()
    sorted(list(set(sim_df_filtered.csv_data_path.to_list()))) # display list of included files

    ## convert results back into python variables
    print('converting actions back to lists from strings')

    # sim_df_filtered.states_visited = list(sim_df_filtered.states_visited.apply(chk_type)) # convert string to list
    sim_df_filtered.states_visited = sim_df_filtered.states_visited.apply(chk_type).apply(list) # convert string/numpy to list
    sim_df_filtered.actions_taken_allo_latent = sim_df_filtered.actions_taken_allo_latent.apply(chk_type) #
    sim_df_filtered.actions_taken_allo_real = sim_df_filtered.actions_taken_allo_real.apply(chk_type) #
    sim_df_filtered.actions_taken_ego = sim_df_filtered.actions_taken_ego.apply(chk_type) # convert string to list
    sim_df_filtered['policy_class'] = 'fixed policy'
    return (sim_df_filtered,)


@app.cell
def _(sim_df_reloaded):
    sim_df_reloaded.animal_ID.unique()
    return


@app.cell
def _(sim_df_filtered):
    sim_df_filtered.policy_class
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # calculate n rewards achieved for simulations
    - 251224 Author: equivalance between the fast and slow methods was verified
    - 251221 Author: attempting streamlining and optimization:
        - runs in 6 mins on compute-cluster
        - TODO: use the new reset_when_n_rwds_remaining column in rewarded_states_df
        - TODO add an assert that the initial rewarded states is all of the states
            - req's loading the maze info tho... maybe do earlier or later in the pipeline, depending on how much faster the GPT speed optimized reward extraction is
    - 251221 Author TODO: Does the RL data already have this calculated? If so, assert equality?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
 
    """)
    return


@app.cell
def _(chk_type, get_now_str, get_verified_st_traj, np, os, pd):
    # 251222 Author: new GPT approach, attempting to speed up the processing by avoiding repeated .query() calls and redundant path normalization
    def get_n_rewards_obtained_by_simulations_fast_fix(sim_df, rewarded_states_df, *, verbose=True, save_csv=True):
        def _normpath_map(series: pd.Series) -> pd.Series:
            """Normalize paths via unique-map (faster than normpath per row when many repeats)."""
            uniq = series.dropna().unique()
            m = {p: os.path.normpath(p) for p in uniq}
            return series.map(m)

        def _to_int_list(x):
            """Parse list-ish states_visited and ensure python ints."""
            if isinstance(x, list):
                return [int(v) for v in x]
            if isinstance(x, tuple):
                return [int(v) for v in x]
            if isinstance(x, np.ndarray):
                return [int(v) for v in x.tolist()]
            # fall back to your parser for strings / serialized objects
            x = chk_type(x)
            return [int(v) for v in x]

        ## ---- normalize paths once (unique-map avoids repeated normpath work) ----
        sim_df = sim_df.copy()
        rewarded_states_df = rewarded_states_df.copy()

        sim_df["csv_data_path"] = _normpath_map(sim_df["csv_data_path"])
        rewarded_states_df["csv_data_path"] = _normpath_map(rewarded_states_df["csv_data_path"])

        ## ---- cheap sanity: if there are NaNs, groupby(dropna=True) will drop rows ----
        if sim_df[["animal_ID", "csv_data_path"]].isna().any().any():
            bad = sim_df[sim_df[["animal_ID", "csv_data_path"]].isna().any(axis=1)][["animal_ID", "csv_data_path"]]
            raise AssertionError(f"Author: NaNs in group keys; groupby(dropna=True) would drop rows:\n{bad.head(20)}")

        ## ---- build rewarded-states lookup ONCE (replaces repeated .query()) ---- enforce exactly one rewarded_states row per csv_data_path
        counts = rewarded_states_df["csv_data_path"].value_counts(dropna=False)
        bad_paths = counts[counts != 1]
        if len(bad_paths):
            raise AssertionError(
                "Author: expected exactly 1 rewarded_states entry per csv_data_path; "
                f"bad counts:\n{bad_paths.head(50)}")

        reward_map = dict(zip(rewarded_states_df["csv_data_path"], rewarded_states_df["rewarded_states"]))
        reset_when_n_rwds_remaining_map = dict(zip(rewarded_states_df["csv_data_path"], rewarded_states_df["reset_when_n_rwds_remaining"]))

        ## choose moment column once
        moment_col = "moment" if "moment" in sim_df.columns else "exp_moment"
        if moment_col not in sim_df.columns:
            raise AssertionError("Author: expected either 'moment' or 'exp_moment' column in sim_df")

        per_exp_result_rows = []

        ## ---- main loop: group once; inner loop uses itertuples() over the group ----
        for (a_ID, csv_data_path), animal_x_csv_path_df in sim_df.groupby(["animal_ID", "csv_data_path"], sort=False, dropna=True):
            if verbose:
                print(f"processing sims for animal: {a_ID}; session: {os.path.basename(csv_data_path)}")

            if not os.path.exists(csv_data_path):
                raise AssertionError(f"Author: csv_data_path {csv_data_path} does not exist, check the path or the data")

            ## load mouse exp once per (animal, session)
            mouse_exp_df = pd.read_csv(csv_data_path)
            states_visited_mouse_full = get_verified_st_traj(mouse_exp_df)
            if not isinstance(states_visited_mouse_full, list):
                raise AssertionError(f"Author: get_verified_st_traj must return list; got {type(states_visited_mouse_full)}")

            rewarded_action_inds_mouse_full = mouse_exp_df.loc[mouse_exp_df["reward_time"].notna(), "action_idx"].to_list()

            ## validate mouse reward count once
            try: 
                mouse_n_rewards = int(mouse_exp_df["n_rewards"].iloc[-1])
                if len(rewarded_action_inds_mouse_full) != mouse_n_rewards:
                    raise AssertionError(
                        "Author: len(rewarded_action_inds_mouse_full) should match mouse_exp_df.n_rewards, "
                        f"but got {len(rewarded_action_inds_mouse_full)} vs {mouse_n_rewards}")
            except Exception as e:
                print(e); print('Author: skipping this one do to missing mouse n rewards info upstream; This should only occur for ignored early pilot data...'); continue


            ## fetch + parse rewarded_states_initial once per session
            try: 
                rewarded_states_initial = chk_type(reward_map[csv_data_path])
                reset_when_n_rwds_remaining = chk_type(reset_when_n_rwds_remaining_map[csv_data_path])
            except Exception as e:
                print(e); print('Author: skipping this one do to missing reward info upstream; This should only occur for ignored early pilot data...'); continue


            if (not isinstance(rewarded_states_initial, list)) and rewarded_states_initial == "tbd":
                if verbose:
                    print(f"skipping detailed rwd determination for {csv_data_path} due to tbd rewarded states")
                if a_ID in ["a031", "a033"]:
                    raise AssertionError(f"Author: unexpected tbd rewarded states for animal_ID {a_ID} for : {csv_data_path}")
                continue

            if not isinstance(rewarded_states_initial, list):
                raise AssertionError(f"Author: rewarded_states_initial should be list or 'tbd'; got {type(rewarded_states_initial)}")
            if len(rewarded_states_initial) == 0:
                raise AssertionError("Author: rewarded_states_initial was found to be an empty list...")

            ## for faster membership/remove: use set if unique, else Counter to preserve duplicates
            init_list = rewarded_states_initial
            init_is_unique = (len(set(init_list)) == len(init_list))
            assert init_is_unique, "Author: rewarded_states_initial should be unique"
            init_set = set(init_list) 
            # init_counter = Counter(init_list) 

            ## iterate rows in this group efficiently; ensure we have needed columns
            NEEDED_COLS = [moment_col, "repeat_group_idx", "full_sim_desc", "states_visited", "policy_class"]
            missing = [c for c in NEEDED_COLS if c not in animal_x_csv_path_df.columns]
            if missing:
                raise AssertionError(f"Author: missing required columns in sim_df group: {missing}")

            sb3_cols = {'start_state',  'action_type', 'n_valid_actions_per_episode', 'actions_taken_allo_real', 'adj_file', 'actions_taken_ego', 'total_reward', 'agent_model', 'states_visited', 'st_pos_file', 'actions_taken_allo_latent', 'result_creation_moment', 'obs_log_path', 'obs_type'}
            cols_we_dropped_but_might_want = sb3_cols - set(animal_x_csv_path_df.columns.to_list())
            print(f"cols_we_dropped_but_might_want: {cols_we_dropped_but_might_want}")

            ## precompute positions once per group
            cols = ["Index"] + list(animal_x_csv_path_df.columns)
            pos = {c: cols.index(c) for c in NEEDED_COLS + ["Index"]}

            ## 260107 Author: preserve all input columns (for obs_type, action_type, agent_model, etc)
            all_input_cols = list(animal_x_csv_path_df.columns)

            for row in animal_x_csv_path_df.itertuples(index=True, name=None):
                sim_idx = row[pos["Index"]]
                exp_moment = row[pos[moment_col]]
                repeat_group_idx = row[pos["repeat_group_idx"]]
                sdf = row[pos["full_sim_desc"]]
                states_visited_raw = row[pos["states_visited"]]

                if not isinstance(sdf, str):
                    raise AssertionError(f"Author: full_sim_desc should be str, got {type(sdf)}")
                if not isinstance(repeat_group_idx, (int, np.integer)):
                    raise AssertionError(f"Author: repeat_group_idx should be int, got {type(repeat_group_idx)}")

                states_visited_agent_full = _to_int_list(states_visited_raw)
                if len(states_visited_agent_full) == 0:
                    raise AssertionError(f"Author: empty states_visited_agent at row {sim_idx}")

                ## ---- compute agent reward indices (no deepcopy; reset via copy) ----
                rewarded_action_inds_agent = []

                # if init_is_unique:
                cur = init_set.copy()
                ## enumerate actions (agent) corresponds to states_visited_agent_full[1:]
                for i, s in enumerate(states_visited_agent_full[1:]):
                    if s in cur:
                        rewarded_action_inds_agent.append(i)
                        cur.remove(s)
                        if len(cur) == reset_when_n_rwds_remaining:
                            cur = init_set.copy()
                # else:
                #     cur = init_counter.copy()
                #     cur_len = sum(cur.values())
                #     for i, s in enumerate(states_visited_agent_full[1:]):
                #         cnt = cur.get(s, 0)
                #         if cnt:
                #             rewarded_action_inds_agent.append(i)
                #             if cnt == 1:
                #                 del cur[s]
                #             else:
                #                 cur[s] = cnt - 1
                #             cur_len -= 1
                #             if cur_len == reset_when_n_rwds_remaining:
                #                 cur = init_counter.copy()
                #                 cur_len = sum(cur.values())

                ## ---- align trajectories WITHOUT mutating the shared mouse list ----
                initial_n_mouse = len(states_visited_mouse_full)
                initial_n_agent = len(states_visited_agent_full)
                min_len = min(initial_n_mouse, initial_n_agent)

                ## slice (drops from end like your pop-loops)
                states_visited_mouse = states_visited_mouse_full[:min_len]
                states_visited_agent = states_visited_agent_full[:min_len]

                n_mouse_dropped = initial_n_mouse - min_len
                n_agent_dropped = initial_n_agent - min_len

                ## filter reward indices to aligned action range (actions are len(states)-1)
                max_action_idx = min_len - 2  # last valid action index
                if max_action_idx < 0:
                    raise AssertionError(f"Author: trajectory too short after alignment at row {sim_idx}: min_len={min_len}")

                if rewarded_action_inds_agent:
                    rewarded_action_inds_agent = [i for i in rewarded_action_inds_agent if i <= max_action_idx]
                if rewarded_action_inds_mouse_full:
                    rewarded_action_inds_mouse = [i for i in rewarded_action_inds_mouse_full if i <= max_action_idx]
                else:
                    rewarded_action_inds_mouse = []

                ## ---- assumption checks (kept, but made safe on empty lists) ----
                if abs(initial_n_mouse - initial_n_agent) >= 6:
                    raise AssertionError(
                        f"Author: abs(initial_n_states_visited_mouse - initial_n_states_visited_agent) == "
                        f"{abs(initial_n_mouse - initial_n_agent)}")
                if states_visited_agent[0] != states_visited_mouse[0]:
                    raise AssertionError(f"Author: first state should be the same, but got {states_visited_agent[0]} and {states_visited_mouse[0]}")

                ## last reward indices must be within range (guard empties)
                if rewarded_action_inds_agent and rewarded_action_inds_agent[-1] > max_action_idx:
                    raise AssertionError(f"Author: row {sim_idx}: last rewarded agent action index ({rewarded_action_inds_agent[-1]}) "
                        f"exceeds max_action_idx ({max_action_idx})")
                if rewarded_action_inds_mouse and rewarded_action_inds_mouse[-1] > max_action_idx:
                    raise AssertionError(f"Author: row {sim_idx}: last rewarded mouse action index ({rewarded_action_inds_mouse[-1]}) "
                        f"exceeds max_action_idx ({max_action_idx})")

                if pd.isna(exp_moment):
                    raise AssertionError("exp_moment is NaN after fallback")

                ## ============================================================
                ## 260107 Author: NEW - preserve ALL input columns + add calculated fields
                ## ============================================================

                ## Convert namedtuple row to dict, preserving all input columns
                row_dict = {}
                for i, col_name in enumerate(all_input_cols):
                    ## row[0] is Index, row[1:] are the actual column values
                    row_dict[col_name] = row[i + 1] # Author: CLD says i+1 to skip Index which is otherwise considered a column

                ## Add group keys (animal_ID, csv_data_path)
                row_dict['animal_ID'] = a_ID
                row_dict['csv_data_path'] = csv_data_path

                ## Standardize moment column name to exp_moment
                if moment_col == "moment" and "moment" in row_dict:
                    row_dict['exp_moment'] = row_dict['moment']
                    ## Keep 'moment' too for backward compatibility

                ## Update/overwrite with calculated reward fields
                row_dict.update({
                    'repeat_group_idx': int(repeat_group_idx),
                    'initial_rewards': init_list,
                    'n_states_visited': min_len,
                    'n_mouse_states_dropped_to_align': n_mouse_dropped,
                    'n_agent_states_dropped_to_align': n_agent_dropped,
                    'n_rewards_obtained_agent': len(rewarded_action_inds_agent),
                    'rewarded_action_inds_agent': rewarded_action_inds_agent,
                    'states_visited_agent': states_visited_agent,
                    'n_rewards_obtained_mouse': len(rewarded_action_inds_mouse),
                    'rewarded_action_inds_mouse': rewarded_action_inds_mouse,
                    'states_visited_mouse': states_visited_mouse,
                })

                per_exp_result_rows.append(row_dict)

        per_exp_result_df = pd.DataFrame(per_exp_result_rows)
        if len(per_exp_result_df) == 0:
            raise AssertionError("Author: no results found in per_exp_result_df")

        per_exp_result_df = per_exp_result_df.sort_values(["animal_ID", "full_sim_desc", "repeat_group_idx"])

        if save_csv:
            per_exp_result_df.to_csv(f"./data_out/c{get_now_str()}_per_exp_result_df.csv", index=False)

        return per_exp_result_df
    return (get_n_rewards_obtained_by_simulations_fast_fix,)


@app.cell
def _(sim_df_filtered):
    sim_df_filtered.animal_ID.unique()
    return


@app.cell
def _(
    check_n_states_match_per_exp,
    get_n_rewards_obtained_by_simulations_fast_fix,
    rewarded_states_df,
    sim_df_filtered,
):
    ## saves the df to ./simulations internally

    # per_exp_result_df_w_rwds_fast = get_n_rewards_obtained_by_simulations_fast(sim_df_filtered, rewarded_states_df) 
    per_exp_result_df_w_rwds_fast = get_n_rewards_obtained_by_simulations_fast_fix(sim_df_filtered, rewarded_states_df) 

    check_n_states_match_per_exp(per_exp_result_df_w_rwds_fast)
    assert 'policy_class' in per_exp_result_df_w_rwds_fast.columns, f'Author: expected policy_class column missing after reward extraction'

    ## 251224 Author: legacy slow approach: assert passed today
    # per_exp_result_df_w_rwds = get_n_rewards_obtained_by_simulations(sim_df_filtered, rewarded_states_df) # saves the df to ./simulations internally
    # check_n_states_match_per_exp(per_exp_result_df_w_rwds)
    # assert per_exp_result_df_w_rwds.equals(per_exp_result_df_w_rwds_fast), 'Author: fast vs naive reward extraction mismatch'
    return (per_exp_result_df_w_rwds_fast,)


@app.cell(hide_code=True)
def _(per_exp_result_df_w_rwds_fast):
    per_exp_result_df_w_rwds_fast.policy_class
    # sim_df_filtered.policy_class
    return


@app.cell
def _(per_exp_result_df_w_rwds_fast):
    per_exp_result_df_w_rwds_fast.animal_ID.unique()
    return


@app.cell
def _(sanity_check_latent_vs_real_allo_actions, sim_df_filtered):
    sanity_check_latent_vs_real_allo_actions(sim_df_filtered)
    return


@app.cell
def _():
    return


@app.cell
def _(per_exp_result_df_w_rwds_fast):
    model_names = sorted(per_exp_result_df_w_rwds_fast.full_sim_desc.unique())
    model_names
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # load deep RL results
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    251225 Author: takes 15 mins to load ~125 repeats
    """)
    return


@app.cell
def _(get_now_str, glob, os, pd, tqdm):
    RL_RESULT_DIR = "./latMaz_RL_sims/data_out"
    RL_RESULT_FILTER_STR = 'RL_results_df-repeat' # 251225 Author TODO: possibly relax the constraint to include the earlier versions at some point; note that some data might be invalid (step_df_path? see comment below)

    all_rl_result_paths = sorted(glob(f"{RL_RESULT_DIR}/*{RL_RESULT_FILTER_STR}*.csv"))

    read_failures = []
    agg_dfs = []
    for agg_rst_df_path in tqdm(all_rl_result_paths):
        try:
            if os.path.splitext(agg_rst_df_path)[-1] == '.feather':
                df = pd.read_feather(agg_rst_df_path)
            elif os.path.splitext(agg_rst_df_path)[-1] == '.csv':
                df = pd.read_csv(agg_rst_df_path)
        except Exception as e:
                print(f"Error reading {agg_rst_df_path}: {e}")
                read_failures.append(agg_rst_df_path)
                continue

        agg_dfs.append(df)

    agg_RL_results_df = pd.concat(agg_dfs, ignore_index=True) # Author: do not modify this df directly; do it on a copy
    agg_RL_results_df.to_feather(f"./data_out/c{get_now_str()}_agg_RL_results.feather")

    all_rl_result_paths[::-1]
    return agg_RL_results_df, read_failures


@app.cell
def _(read_failures):
    # assert len(read_failures) == 0, f"Failed to {len(read_failures)} read files: {read_failures}"
    f"Failed to {len(read_failures)} read files: {read_failures}"
    return


@app.cell
def _():
    # agg_RL_results_df.result_creation_moment.max()
    return


@app.cell
def _(mo):
    mo.md(r"""
    - 260107 Author: 15+ mins on compute-cluster
    - 251229 Author: 9 mins on compute-cluster
    """)
    return


@app.cell
def _(get_most_recent_file):
    get_most_recent_file(f"./data_out/*_RL_results_df-woRwds.feather")
    return


@app.cell
def _(
    PARENT_DIR,
    agg_RL_results_df,
    chk_type,
    extract_valid_moment,
    get_most_recent_file,
    get_now_str,
    move_df_col_to_leftmost,
    os,
    pd,
):
    RELOAD_RL_RESULTS_DF_FROM_FILE = False



    if RELOAD_RL_RESULTS_DF_FROM_FILE:
        RL_results_df = pd.read_feather(get_most_recent_file(f"{PARENT_DIR}/<DATA_ROOT>/latent_maze-cup/a_latMaz-checking_n_deepRL/latMaz_RL_sims/data_out/*_RL_results_df-woRwds.feather"))

    else:


        ## 251225 Author: retiring whatever filter this was
        # RL_results_df = agg_RL_results_df.query('not step_df_path.notna()').sort_values('result_creation_moment').reset_index(drop=True).copy()
        # RL_results_df.drop(columns=['step_df_path'], inplace=True) # discard outdated data

        RL_results_df = agg_RL_results_df.sort_values('result_creation_moment').reset_index(drop=True).copy()

        RL_results_df.animal_ID = RL_results_df.animal_ID.apply(lambda x: 'a' + str(x).zfill(3)) # match animal ID format of fixed policy sims

        RL_results_df['full_sim_desc'] = RL_results_df.apply(lambda row: f"{row.agent_model};action_{row.action_type};obs_{row.obs_type}", axis=1) 

        # 251228 moving upstream
        if 'policy_class' in RL_results_df.columns:
            assert RL_results_df.policy_class.notna().all(), f'Author: unexpected non-naN policy_class values found; should be prepopulated upstream'
        else: ## handle legacy results for a while...
            RL_results_df['policy_class'] = 'MDP: custom gym env w SB3 agents' 

        RL_results_df['exp_moment'] = RL_results_df.csv_data_path.apply(extract_valid_moment)
        # RL_results_df.rename(columns={'repeat_group_idx': 'RL_repeat_group_idx'}, inplace=True)

        RL_results_df = move_df_col_to_leftmost(RL_results_df, 'full_sim_desc')
        RL_results_df = move_df_col_to_leftmost(RL_results_df, 'exp_moment')
        RL_results_df = RL_results_df.query("exp_moment in @RL_results_df.exp_moment.to_list()").copy() # filter to only include exp_moments present in fixed policy sims
        RL_results_df['n_rewards_obtained_agent'] = RL_results_df.total_reward
        RL_results_df['n_states_visited'] = RL_results_df.states_visited.apply(lambda x: len(chk_type(x)))
        RL_results_df['csv_data_path'] = RL_results_df.csv_data_path.apply(os.path.normpath)

        RL_results_df.states_visited = RL_results_df.states_visited.apply(chk_type).apply(list) # convert string/numpy to list
        RL_results_df.actions_taken_allo_latent = RL_results_df.actions_taken_allo_latent.apply(chk_type) #
        RL_results_df.actions_taken_allo_real = RL_results_df.actions_taken_allo_real.apply(chk_type) #
        RL_results_df.actions_taken_ego = RL_results_df.actions_taken_ego.apply(chk_type) # convert string to list

        RL_results_df.sort_values('exp_moment', inplace=True)

        exp_moments_only_in_fixed_policy_sims = set(RL_results_df.exp_moment.to_list()) - set(RL_results_df.exp_moment.to_list())
        exp_moments_only_in_RL_sims = sorted(list(set(RL_results_df.exp_moment.to_list()) - set(RL_results_df.exp_moment.to_list())))

        assert len(exp_moments_only_in_fixed_policy_sims) == 0, \
            f"Author: found exp_moments only in fixed policy sims: {exp_moments_only_in_fixed_policy_sims}; recheck which data is included where carefully. It appears to have changed since 251111"
        print(exp_moments_only_in_RL_sims)

        """ drop a002 and a003 cuz they only have 4 sessions that meet the criteria """
        ## neurIPS initial submission
        RL_results_df = RL_results_df[~RL_results_df.animal_ID.isin(["a002", "a003"])]

        ## Hopfield poster submission
        RL_results_df = RL_results_df[~RL_results_df.animal_ID.isin(["a002", "a003"])]

        # RL_results_df.rename(columns={'repeat_group_idx': 'fixed_policy_repeat_group_idx'}, inplace=True)
        RL_results_df.sort_values('exp_moment', inplace=True)
        RL_results_df.to_feather(f"./data_out/c{get_now_str()}_RL_results_df-woRwds.feather")

    RL_results_df
    return (RL_results_df,)


@app.cell
def _(RL_results_df):
    RL_results_df.animal_ID.unique()
    return


@app.cell
def _():
    return


@app.cell
def _(RL_results_df):
    RL_results_df
    return


@app.cell
def _(RL_results_df):
    RL_results_df.columns
    return


@app.cell
def _(RL_results_df):
    RL_results_df.full_sim_desc.value_counts()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # (re)calculate the number of rewards obtained for the RL simulation

    - 251226 Author: TODO: reset_when_n_rwds_remaining was added to fixed policy sims but is lacking in RL
        - (this handles earlier experiments before the n rewards remaining was relaxed to 1
    """)
    return


@app.cell
def _():
    return


@app.cell
def _(
    RL_results_df,
    check_n_states_match_per_exp,
    get_n_rewards_obtained_by_simulations_fast_fix,
    rewarded_states_df,
):
    # get_n_rewards_obtained_by_simulations_fast_fix

    ## version in utils drops a bunch of columns we want... 
    # agg_RL_results_df_w_rwds_fast = get_n_rewards_obtained_by_simulations_fast(RL_results_df, rewarded_states_df) # saves the df to ./simulations internally
    agg_RL_results_df_w_rwds_fast = get_n_rewards_obtained_by_simulations_fast_fix(RL_results_df, rewarded_states_df) # saves the df to ./simulations internally

    check_n_states_match_per_exp(agg_RL_results_df_w_rwds_fast)
    assert 'policy_class' in agg_RL_results_df_w_rwds_fast.columns, f'Author: expected policy_class column missing after reward extraction'

    ## Author: legacy slow approach: assert passed 251224
    # agg_RL_results_df_w_rwds = get_n_rewards_obtained_by_simulations(RL_results_df, rewarded_states_df) # saves the df to ./simulations internally
    # assert agg_RL_results_df_w_rwds_fast.equals(agg_RL_results_df_w_rwds), 'Author: fast vs naive reward extraction mismatch for RL sims'
    return (agg_RL_results_df_w_rwds_fast,)


@app.cell
def _(agg_RL_results_df_w_rwds_fast):
    agg_RL_results_df_w_rwds_fast.columns
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # combine the fixed policy and RL simulation results into a single feather file
    """)
    return


@app.cell
def _():
    # sanity_check_latent_vs_real_allo_actions(sim_df_reloaded)
    # print(f'{get_now_str(hms=True)}: checking fixed policy sims:')
    # sanity_check_latent_vs_real_allo_actions(per_exp_result_df_w_rwds)
    # print('{get_now_str(hms=True)}: checking RL sims:')
    # sanity_check_latent_vs_real_allo_actions(RL_results_df)
    return


@app.cell
def _(mo):
    mo.md(r"""
    251224 Author: saved in 1.5 mins
    """)
    return


@app.cell
def _(
    agg_RL_results_df_w_rwds_fast,
    get_now_str,
    os,
    pd,
    per_exp_result_df_w_rwds_fast,
    sanity_check_latent_vs_real_allo_actions,
):
    sanity_check_latent_vs_real_allo_actions(per_exp_result_df_w_rwds_fast)
    sanity_check_latent_vs_real_allo_actions(agg_RL_results_df_w_rwds_fast)

    all_sims_rst_df = pd.concat([per_exp_result_df_w_rwds_fast, agg_RL_results_df_w_rwds_fast], ignore_index=True, sort=False)
    all_sims_rst_df.sort_values(['exp_moment', 'full_sim_desc'], inplace=True)
    all_sims_rst_df.reset_index(drop=True, inplace=True)

    ## swap any strings with perfect_oracle;perfect knowledge of best action;avoidReversal-False;seqLength-1 any elements where full_sim_desc is 
    STRINGS_TO_SWAP = {'perfect_oracle': 'rwd_dist_oracle', 'perfect knowledge of best action': 'select node w min median/mean distance', '_trajectory': ''}
    ## fill in code to swap the keys for the values in STRINGS_TO_SWAP

    ## swap substrings in full_sim_desc
    # COLS_TO_MODIFY = ["full_sim_desc", "obs_type", "action_type"] # 260107 Author: the obs_type didn't make it this far
    COLS_TO_MODIFY = all_sims_rst_df.columns.to_list() # 260107 Author: this is kinda suspect

    ## apply all swaps (regex=False = literal substring replace)
    for col in COLS_TO_MODIFY:
        ## ensure string dtype (handles NaN safely)
        all_sims_rst_df[col] = all_sims_rst_df[col].astype("string")

        for old, new in STRINGS_TO_SWAP.items():
            all_sims_rst_df[col] = all_sims_rst_df[col].str.replace(old, new, regex=False)

        ## optional: collapse accidental double spaces created by removals like "_trajectory" -> ""
        all_sims_rst_df[col] = all_sims_rst_df[col].str.replace(r"\s{2,}", " ", regex=True).str.strip()

    ## 260107 GPT: untested but should help corruption or type issues downstream; Author: this fails maybe that's ok
    # LIST_COLS=["states_visited_agent", "states_visited_mouse", "rewarded_action_inds_agent", "rewarded_action_inds_mouse"]
    # for _col in [_col for _col in LIST_COLS if _col in all_sims_rst_df.columns]:
    #     bad=all_sims_rst_df[_col].dropna().map(lambda x:isinstance(x,str)).any()
    #     assert not bad,f"Author: {_col} contains strings; would require downstream deserialization. Fix upstream."

    NUM_COLS=["n_states_visited","n_rewards_obtained_agent","n_rewards_obtained_mouse"]
    for _col in [_col for _col in NUM_COLS if _col in all_sims_rst_df.columns]:
        # coerce once here (better than later)
        all_sims_rst_df[_col]=pd.to_numeric(all_sims_rst_df[_col],errors="raise")

    ## 260107 GPT: trying to catch upstream issues early
    fixed=all_sims_rst_df["policy_class"].astype("string").eq("fixed")
    if fixed.any():
        assert "agent_model" in all_sims_rst_df.columns, "Author: fixed-policy rows exist but agent_model column missing"
        am=all_sims_rst_df.loc[fixed,"agent_model"].astype("string")
        bad=am.isna() | am.isin(["<NA>","nan","None",""])
        assert not bad.any(), ("GPT: refusing to write all_sims feather: fixed-policy rows have missing agent_model. "
                               f"Example full_sim_desc: {all_sims_rst_df.loc[fixed & bad,'full_sim_desc'].head(20).tolist()}")

    ## 260107 Author: creating expected column to leaving legacy column in place if necessary
    try: 
        all_sims_rst_df["exp_moment"] = all_sims_rst_df["moment"].astype("string")
    except Exception as e:
        print(e); print('Author: this might mean that the proper exp_moment column was created upstream already... ')

    os.makedirs('./data_out', exist_ok=True)
    all_sims_rst_df.to_feather(f"./data_out/c{get_now_str()}_all_sims-fixed_policy_n_stableBaselines.feather")
    all_sims_rst_df
    return (all_sims_rst_df,)


@app.cell
def _(all_sims_rst_df):
    all_sims_rst_df.animal_ID.unique()
    return


@app.cell
def _(agg_RL_results_df_w_rwds_fast):
    agg_RL_results_df_w_rwds_fast.full_sim_desc.unique()
    return


@app.cell
def _(all_sims_rst_df):
    all_sims_rst_df.columns
    return


@app.cell
def _(all_sims_rst_df):
    all_sims_rst_df.groupby(['policy_class', 'full_sim_desc']).size()
    return


@app.cell
def _(per_exp_result_df_w_rwds_fast):
    per_exp_result_df_w_rwds_fast.full_sim_desc.value_counts()
    return


@app.cell
def _(per_exp_result_df_w_rwds_fast):
    per_exp_result_df_w_rwds_fast.columns
    return


@app.cell
def _(all_sims_rst_df):
    all_sims_rst_df.full_sim_desc.value_counts()
    return


@app.cell
def _():
    return


@app.cell
def _():
    # all_sims_rst_df.to_csv(f"./data_out/c{get_now_str()}_all_sims-fixed_policy_n_stableBaselines.csv", index=False)
    return


@app.cell
def _():
    # sanity_check_latent_vs_real_allo_actions(sim_df_filtered)
    return


@app.cell
def _():
    return


@app.cell
def _():
    # sanity_check_latent_vs_real_allo_actions(RL_results_df)


    # sanity_check_latent_vs_real_allo_actions(all_sims_rst_df)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # (optional) inspect the results after reloading the file
    """)
    return


@app.cell
def _(PARENT_DIR, get_most_recent_file, os, pd):
    assert os.path.exists(PARENT_DIR), f'Author: cannot find jukebox or Volumes dirs on this computer: {PARENT_DIR}; Fill in the proper parent dir above'

    all_sims_rst_df_reloaded = pd.read_feather(get_most_recent_file(f"{PARENT_DIR}/<DATA_ROOT>/latent_maze-cup/a_latMaz-checking_n_deepRL/data_out/*_all_sims-fixed_policy_n_stableBaselines.feather"))
    all_sims_rst_df_reloaded
    return (all_sims_rst_df_reloaded,)


@app.cell
def _(all_sims_rst_df_reloaded):
    all_sims_rst_df_reloaded.groupby(['policy_class', 'full_sim_desc']).size()
    return


@app.cell
def _(all_sims_rst_df_reloaded):
    all_sims_rst_df_reloaded.policy_class.value_counts()
    return


@app.cell
def _(RL_results_df, per_exp_result_df_w_rwds_fast):
    set(per_exp_result_df_w_rwds_fast.columns) - set(RL_results_df.columns)
    return


@app.cell
def _(RL_results_df, per_exp_result_df_w_rwds_fast):
    set(RL_results_df.columns) - set(per_exp_result_df_w_rwds_fast.columns)
    return


@app.cell
def _():
    # len(sorted(list(all_sims_rst_df_reloaded.query("animal_ID == 'a031'").exp_moment.unique()))), sorted(list(all_sims_rst_df_reloaded.query("animal_ID == 'a031'").exp_moment.unique()))
    return


@app.cell
def _():
    # blach blah just trying to save. 
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    GPT code that might be helpful in aligning the data formats
    """)
    return


@app.cell
def _():
    # # ---- choose grouping key (safer if exp_moment is not globally unique) ----
    # _KEY_COLS = ["animal_ID","exp_moment"] if "animal_ID" in all_sims_df.columns else ["exp_moment"]
    # RWD_COL = "n_rewards_obtained_mouse"


    # print('251221 Author: temporary hack: assuming fixed policy sims are those with non nan n_rewards_obtained_mouse')
    # fixed_policy_sims_df = all_sims_df.dropna(subset=["exp_moment","n_rewards_obtained_mouse"]).copy()

    # rwd_nunique_by_exp = (fixed_policy_sims_df.groupby("exp_moment", sort=False)["n_rewards_obtained_mouse"]
    #                       .nunique(dropna=True)
    #                       .rename("n_unique_reward_counts"))

    # violating_exp_moments = rwd_nunique_by_exp.index[rwd_nunique_by_exp > 1]

    # violations_counts = (fixed_policy_sims_df.loc[fixed_policy_sims_df["exp_moment"].isin(violating_exp_moments)]
    #                      .groupby(["exp_moment","n_rewards_obtained_mouse"], sort=False)
    #                      .size()
    #                      .rename("n_rows")
    #                      .reset_index()
    #                      .sort_values(["exp_moment","n_rows"], ascending=[True,False]))


    # assert (rwd_nunique_by_exp <= 1).all(), f"Violating exp_moments: {list(violating_exp_moments)[:20]}"


    # # ---- build a mapping _KEY_COLS -> unique reward count (only where it's truly unique) ----
    # non_nan = all_sims_df.dropna(subset=_KEY_COLS + [RWD_COL]).copy()
    # nunique_by_key = non_nan.groupby(_KEY_COLS, sort=False)[RWD_COL].nunique(dropna=True)
    # unique_keys = nunique_by_key.index[nunique_by_key == 1]

    # # value is well-defined for unique_keys (all rows in group share the same value)
    # first_val_by_key = non_nan.groupby(_KEY_COLS, sort=False)[RWD_COL].first()
    # valid_map = first_val_by_key.loc[unique_keys].rename("fill_val").reset_index()

    # # ---- fill NaNs using the mapping ----
    # mask_nan_before = all_sims_df[RWD_COL].isna()
    # n_nan_before = int(mask_nan_before.sum())

    # nan_key_df = all_sims_df.loc[mask_nan_before, _KEY_COLS].copy()
    # nan_key_df["_orig_index"] = nan_key_df.index
    # nan_key_df = nan_key_df.merge(valid_map, on=_KEY_COLS, how="left")

    # fill_vals = nan_key_df.set_index("_orig_index")["fill_val"]
    # all_sims_df.loc[fill_vals.index, RWD_COL] = fill_vals

    # # ---- remaining failures (still NaN after attempted fill) ----
    # mask_nan_after = all_sims_df[RWD_COL].isna()
    # n_nan_after = int(mask_nan_after.sum())
    # n_filled = n_nan_before - n_nan_after

    # mouse_rwd_nan_fill_fail_df = all_sims_df.loc[mask_nan_after].copy()

    # print(f"Author: {RWD_COL} NaNs before fill: {n_nan_before}")
    # print(f"Author: {RWD_COL} filled via {_KEY_COLS} map: {n_filled}")
    # print(f"Author: {RWD_COL} remaining NaNs after fill: {n_nan_after}")
    # print(f"Author: saved remaining failures to mouse_rwd_nan_fill_fail_df (n={len(mouse_rwd_nan_fill_fail_df)})")

    # # ---- drop remaining failures from all_sims_df ----
    # all_sims_df = all_sims_df.loc[~mask_nan_after].copy()
    # print(f"Author: all_sims_df rows after dropping remaining failures: {len(all_sims_df)}")

    # # violations_counts
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
