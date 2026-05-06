import marimo

__generated_with = "0.19.1"
app = marimo.App(width="full")

with app.setup:
    # Initialization code that runs before all other cells
    pass


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 1b: plot mice vs all simulations

    Anonymous Author, Anonymous Author, Anonymous Author, 2025

    # 1. Setup and Configuration

    ## 1.1 Imports and Environment Setup


    working versions as of 251224:

    - NumPy version: 2.1.2
    - Pandas version: 2.2.3
    - Matplotlib version: 3.9.2
    - Seaborn version: 0.13.2
    """)
    return


@app.cell
def _():
    import marimo as mo

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
    import matplotlib.cm as cm
    import matplotlib.colors as mcolors
    from scipy import stats as ss
    # import scipy.stats as ss
    from pandas.api.types import is_numeric_dtype
    from matplotlib.lines import Line2D
    from matplotlib import gridspec
    import filecmp
    import matplotlib as mpl
    from matplotlib.figure import Figure
    from matplotlib.patches import Patch

    # Configure plotting
    sns.set_style("whitegrid")  # Using seaborn's set_style instead of plt.style
    # %matplotlib inline

    print("250612 CRS: Environment setup complete. Using Python packages:")
    print(f"NumPy version: {np.__version__}")
    print(f"Pandas version: {pd.__version__}")
    print(f"Matplotlib version: {matplotlib.__version__}")
    print(f"Seaborn version: {sns.__version__}")
    return (
        Line2D,
        Patch,
        cm,
        is_numeric_dtype,
        mo,
        mpl,
        np,
        os,
        pd,
        plt,
        ss,
        sys,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    251218 Author: attempting to separate from import above so that we don't end up needing re-deserialize unnecessarily if we change the imports
    """)
    return


@app.cell
def _(sys):
    ## 251218 Author: attempting to separate from import above so that we don't end up needing re-deserialize unnecessarily if we change the imports

    """ allows auto reloading the dependencies """
    ## 1) import the modules themselves
    import importlib
    import utils_latMaz
    import latMaz_policies
    import utils_latMaz_stats_n_plotting
    ## 2) reload them so edits on disk are picked up
    importlib.reload(utils_latMaz)
    importlib.reload(latMaz_policies)
    importlib.reload(utils_latMaz_stats_n_plotting)
    ## 3) NOW pull what you need into the notebook namespace

    sys.path.append('./')
    print('240618 Author: I assume utils_latMaz is in the same directory as the notebook')

    from utils_latMaz import (get_most_recent_file, deserialize_str, get_moment_strings, load_maze, get_now_str, get_adj_states, 
        displacement_to_compass_heading, get_direction_from_radian, allo_radian_map_dict, get_ego_direction, get_verified_st_traj, 
        move_df_col_to_leftmost, chk_type)

    from latMaz_policies import (policy_random, policy_action_biased_trunc_v3_minimal, policy_obs_action_biased, 
        policy_latent_state_biased, zero_pad_numbers, generate_trajectory_v4)

    from utils_latMaz_stats_n_plotting import (validate_paired_stats_inputs, get_paired_stats, assert_required_cols, make_hashable,
        warn_if_small_n_for_hedges, rank_biserial_from_diff, rrb_from_scipy_wstat, summarize_stats, run_case_rank_biserial_n_wstat_sanity_checks, 
        split_early_late, groupby_mean_MR, avg_over_MR, sanity_check_session_subset_df, sanity_check_per_model_confidence_interval_inputs,
        sanity_check_all_agents_confidence_interval_inputs, model_name_2_legend_str, plot_mice_vs_sims_per_session, 
        mean_and_ci, draw_sig_bracket, p_display_from_stats, p_from_result, stars_for_p, stats_for_df, combined_barplots_brackets_stars,
        combined_barplots_brackets_stars_2panel, p_from_row, significance2stars, prepare_table, bonferroni)

    # PROJECT_DIR = '<DATA_ROOT>/latent_maze-cup/a_latMaz_neurIPSworkshopPrep'
    PROJECT_DIR = '<DATA_ROOT>/latent_maze-cup/a_latMaz-checking_n_deepRL'
    return (
        avg_over_MR,
        bonferroni,
        combined_barplots_brackets_stars,
        combined_barplots_brackets_stars_2panel,
        deserialize_str,
        get_moment_strings,
        get_most_recent_file,
        get_now_str,
        get_paired_stats,
        mean_and_ci,
        model_name_2_legend_str,
        move_df_col_to_leftmost,
        prepare_table,
        run_case_rank_biserial_n_wstat_sanity_checks,
        sanity_check_all_agents_confidence_interval_inputs,
        sanity_check_per_model_confidence_interval_inputs,
        sanity_check_session_subset_df,
        significance2stars,
        split_early_late,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    251218 Author: attempting to hack so that plots can display in dark mode but save in light mode
    """)
    return


@app.cell
def _(mpl, plt):
    mpl.rcParams["savefig.facecolor"] = "white"
    mpl.rcParams["savefig.edgecolor"] = "white"  # or "none" (see below)
    mpl.rcParams["savefig.transparent"] = False

    print("figure.facecolor:", plt.rcParams.get("figure.facecolor"))
    print("axes.facecolor:", plt.rcParams.get("axes.facecolor"))
    print("savefig.facecolor:", plt.rcParams.get("savefig.facecolor"))
    print("savefig.edgecolor:", plt.rcParams.get("savefig.edgecolor"))
    print("savefig.transparent:", plt.rcParams.get("savefig.transparent"))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # load upstream prepared data

    NOTES/TODOs:

    - 251224 Author: irrelevant now that I combine RL and fixed policy sims?
    - 251219 Author: would a feather file make the deserializing faster? (for per_exp_result_df atleast)
    - 251215 Author: deserializing took 12 mins via compute-cluster
    - 251209 Author: make sure numpy ints aren't serialized upstream (wrote some conversion scripts to fix this but still...)
    - 251214 Author: missing data resolved via renaming usr_param files to match the exact second of the
        - todo: move this fix upstream to the experiment code on the RPi (Done for some scripts at least, up through _2?)
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # load combined fixed policy and RL simulation data
    """)
    return


@app.cell
def _(get_most_recent_file):
    ALL_SIMS_DF_PATH = get_most_recent_file(f"./data_out/*all_sims*.feather")
    ALL_SIMS_DF_PATH
    return (ALL_SIMS_DF_PATH,)


@app.cell
def _(mo):
    mo.md(r"""
    260109 Author: make sure you see the animals and models you expect here
    """)
    return


@app.cell
def _(ALL_SIMS_DF_PATH, pd):
    _all_sims_df = pd.read_feather(ALL_SIMS_DF_PATH)
    _all_sims_df.animal_ID.unique(), _all_sims_df.agent_model.unique()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    251221 Author: quick GPT hack
    - TODO: prep and parse RL data more carefully
    """)
    return


@app.cell
def _(ALL_SIMS_DF_PATH, deserialize_str, pd):
    KEY_COLS = ["animal_ID","exp_moment"] 
    RWD_COL = "n_rewards_obtained_mouse"

    all_sims_df = pd.read_feather(ALL_SIMS_DF_PATH)

    initial_n_animals = all_sims_df.animal_ID.nunique()

    all_sims_df.rewarded_action_inds_mouse = all_sims_df.rewarded_action_inds_mouse.apply(deserialize_str)
    all_sims_df.rewarded_action_inds_agent = all_sims_df.rewarded_action_inds_agent.apply(deserialize_str)

    all_sims_df.n_rewards_obtained_mouse = all_sims_df.n_rewards_obtained_mouse.apply(deserialize_str)
    all_sims_df.n_rewards_obtained_agent = all_sims_df.n_rewards_obtained_agent.apply(deserialize_str)

    all_sims_df.states_visited_mouse = all_sims_df.states_visited_mouse.apply(deserialize_str)
    all_sims_df.n_states_visited = all_sims_df.n_states_visited.apply(deserialize_str)

    assert all_sims_df.policy_class.nunique() > 1, \
        f'251228 Author: all_sims_df should contain both fixed policy and RL sims but found: {all_sims_df.policy_class.nunique()} distinct policy_classes'

    # ---- choose grouping key (safer if exp_moment is not globally unique) ----
    # print('251221 Author: temporary hack: assuming fixed policy sims are those with non nan n_rewards_obtained_mouse')

    # # KEY_COLS = ["animal_ID","exp_moment"] if "animal_ID" in all_sims_df.columns else ["exp_moment"]

    # fixed_policy_sims_df = all_sims_df.dropna(subset=["exp_moment","n_rewards_obtained_mouse"]).copy()

    # rwd_nunique_by_exp = (fixed_policy_sims_df.groupby("exp_moment", sort=False)["n_rewards_obtained_mouse"]
    #                       .nunique(dropna=True)
    #                       .rename("n_unique_reward_counts"))

    # violating_exp_moments = rwd_nunique_by_exp.index[rwd_nunique_by_exp > 1]

    # violations_counts = (fixed_policy_sims_df.loc[fixed_policy_sims_df["exp_moment"].isin(violating_exp_moments)]
    #                      .groupby(["exp_moment","n_rewards_obtained_mouse"], sort=False).size().rename("n_rows")
    #                      .reset_index().sort_values(["exp_moment","n_rows"], ascending=[True,False]))

    # assert (rwd_nunique_by_exp <= 1).all(), f"Violating exp_moments: {list(violating_exp_moments)[:20]}"

    # ## ---- build a mapping KEY_COLS -> unique reward count (only where it's truly unique) ----
    # non_nan = all_sims_df.dropna(subset=KEY_COLS + [RWD_COL]).copy()

    # # nunique_by_key = non_nan.groupby(KEY_COLS, sort=False)[RWD_COL].nunique(dropna=True)
    # nunique_by_key = non_nan.groupby(KEY_COLS, sort=False)[RWD_COL].nunique()

    # unique_keys = nunique_by_key.index[nunique_by_key == 1]

    # # value is well-defined for unique_keys (all rows in group share the same value)
    # first_val_by_key = non_nan.groupby(KEY_COLS, sort=False)[RWD_COL].first()
    # valid_map = first_val_by_key.loc[unique_keys].rename("fill_val").reset_index()

    # # ---- fill NaNs using the mapping ----
    # mask_nan_before = all_sims_df[RWD_COL].isna()
    # n_nan_before = int(mask_nan_before.sum())

    # nan_key_df = all_sims_df.loc[mask_nan_before, KEY_COLS].copy()
    # nan_key_df["_orig_index"] = nan_key_df.index
    # nan_key_df = nan_key_df.merge(valid_map, on=KEY_COLS, how="left")

    # fill_vals = nan_key_df.set_index("_orig_index")["fill_val"]
    # all_sims_df.loc[fill_vals.index, RWD_COL] = fill_vals
    # assert initial_n_animals == all_sims_df.animal_ID.nunique(), f'Author: animals are getting dropped somehow (site 1)'

    # # ---- remaining failures (still NaN after attempted fill) ----
    # mask_nan_after = all_sims_df[RWD_COL].isna()
    # n_nan_after = int(mask_nan_after.sum())
    # n_filled = n_nan_before - n_nan_after

    # mouse_rwd_nan_fill_fail_df = all_sims_df.loc[mask_nan_after].copy()

    # print(f"Author: {RWD_COL} NaNs before fill: {n_nan_before}")
    # print(f"Author: {RWD_COL} filled via {KEY_COLS} map: {n_filled}")
    # print(f"Author: {RWD_COL} remaining NaNs after fill: {n_nan_after}")
    # print(f"Author: saved remaining failures to mouse_rwd_nan_fill_fail_df (n={len(mouse_rwd_nan_fill_fail_df)})")

    # # ---- drop remaining failures from all_sims_df ----
    # all_sims_df = all_sims_df.loc[~mask_nan_after].copy()
    # print(f"Author: all_sims_df rows after dropping remaining failures: {len(all_sims_df)}")

    assert initial_n_animals == all_sims_df.animal_ID.nunique(), f'Author: animals are getting dropped somehow (site 2)'
    if initial_n_animals != all_sims_df.animal_ID.nunique():
        print(f"Author WARNING: initial n animals: {initial_n_animals} vs final {all_sims_df.animal_ID.nunique()}")
        print(f"remaining: {list(all_sims_df.animal_ID.unique())}")

    # violations_counts
    return KEY_COLS, all_sims_df


@app.cell
def _(all_sims_df):
    all_sims_df.animal_ID.unique()
    return


@app.cell
def _(all_sims_df):
    all_sims_df.columns
    return


@app.cell
def _():
    # # ---- Diagnostic analysis of n_rewards_obtained_mouse fill failures ----

    # print(f"\n{'='*80}")
    # print(f"DIAGNOSTIC: Analyzing {len(mouse_rwd_nan_fill_fail_df)} failed fill cases")
    # print(f"{'='*80}\n")

    # # Basic distribution of failures
    # print("1. FAILURE DISTRIBUTION BY KEY COLUMNS:")
    # print("-" * 40)
    # if "animal_ID" in mouse_rwd_nan_fill_fail_df.columns:
    #     print(f"Unique animals with failures: {mouse_rwd_nan_fill_fail_df.animal_ID.nunique()}")
    #     print(f"Failures per animal:\n{mouse_rwd_nan_fill_fail_df.animal_ID.value_counts().head(10)}\n")

    # print(f"Failures by policy_class:\n{mouse_rwd_nan_fill_fail_df.policy_class.value_counts()}\n")

    # if "exp_moment" in mouse_rwd_nan_fill_fail_df.columns:
    #     print(f"Unique exp_moments with failures: {mouse_rwd_nan_fill_fail_df.exp_moment.nunique()}")

    # # Check what data exists for the failed keys in the full dataset
    # print("\n2. CHECKING FULL DATASET FOR FAILED KEYS:")
    # print("-" * 40)

    # FAILED_KEYS_DF = mouse_rwd_nan_fill_fail_df[KEY_COLS].drop_duplicates()
    # print(f"Unique key combinations that failed: {len(FAILED_KEYS_DF)}")

    # # Merge back to see ALL rows (not just NaN) for these keys
    # FULL_RECORDS_FOR_FAILED_KEYS = all_sims_df.merge(
    #     FAILED_KEYS_DF.assign(_failed_key=True),
    #     on=KEY_COLS,
    #     how='inner'
    # )

    # print(f"Total rows in full dataset matching failed keys: {len(FULL_RECORDS_FOR_FAILED_KEYS)}")
    # print(f"Of those, how many have non-NaN {RWD_COL}:")
    # NON_NAN_COUNT = FULL_RECORDS_FOR_FAILED_KEYS[RWD_COL].notna().sum()
    # print(f"  Non-NaN: {NON_NAN_COUNT}")
    # print(f"  NaN: {FULL_RECORDS_FOR_FAILED_KEYS[RWD_COL].isna().sum()}")

    # # If there ARE non-NaN values, why didn't they fill?
    # if NON_NAN_COUNT > 0:
    #     print("\n3. WHY DIDN'T THESE FILL? (Non-uniqueness check)")
    #     print("-" * 40)
    #     NON_NAN_SUBSET = FULL_RECORDS_FOR_FAILED_KEYS[
    #         FULL_RECORDS_FOR_FAILED_KEYS[RWD_COL].notna()
    #     ].copy()

    #     NUNIQUE_BY_FAILED_KEY = NON_NAN_SUBSET.groupby(KEY_COLS, sort=False)[RWD_COL].nunique()
    #     print(f"Keys with multiple distinct reward values:\n{NUNIQUE_BY_FAILED_KEY[NUNIQUE_BY_FAILED_KEY > 1]}\n")

    #     # Show example of non-unique case
    #     if (NUNIQUE_BY_FAILED_KEY > 1).any():
    #         EXAMPLE_NON_UNIQUE_KEY = NUNIQUE_BY_FAILED_KEY[NUNIQUE_BY_FAILED_KEY > 1].index[0]
    #         print(f"EXAMPLE: Multiple values for key {EXAMPLE_NON_UNIQUE_KEY}:")
    #         EXAMPLE_ROWS = NON_NAN_SUBSET.set_index(KEY_COLS).loc[[EXAMPLE_NON_UNIQUE_KEY]]
    #         print(EXAMPLE_ROWS[[RWD_COL, 'policy_class']].drop_duplicates())

    # # Sample of actual failure cases
    # print("\n4. SAMPLE OF FAILURE CASES:")
    # print("-" * 40)
    # SAMPLE_FAILURES = mouse_rwd_nan_fill_fail_df.head(5)[KEY_COLS + [RWD_COL, 'policy_class', 'rewarded_action_inds_mouse']]
    # print(SAMPLE_FAILURES.to_string(index=False))

    # # Check if failures are concentrated in specific policy classes or patterns
    # print("\n5. PATTERN ANALYSIS:")
    # print("-" * 40)
    # print("Cross-tab of policy_class vs presence of rewarded_action_inds_mouse:")
    # RAIMOUSE_PATTERN = mouse_rwd_nan_fill_fail_df.groupby('policy_class').apply(
    #     lambda g: pd.Series({'n_rows': len(g), 'pct_rewarded_action_inds_nan': (g.rewarded_action_inds_mouse.isna().sum() / len(g) * 100)}))

    # print(RAIMOUSE_PATTERN)

    # print(f"\n{'='*80}\n")
    return


@app.cell
def _():
    # n_filled, n_nan_after
    return


@app.cell
def _():
    # mask_nan_after
    return


@app.cell
def _(all_sims_df):
    all_sims_df.animal_ID.unique()
    return


@app.cell
def _(all_sims_df):
    all_sims_df.groupby(['policy_class', 'agent_model']).size()
    return


@app.cell
def _():
    # mouse_rwd_nan_fill_fail_df.exp_moment.value_counts(), mouse_rwd_nan_fill_fail_df
    return


@app.cell
def _(all_sims_df):
    all_sims_df.exp_moment.nunique()
    return


@app.cell
def _(all_sims_df):
    fixed_policy_sims_df = all_sims_df.dropna(subset=["exp_moment","n_rewards_obtained_mouse"]).copy()
    _z = fixed_policy_sims_df.groupby(['exp_moment','n_rewards_obtained_mouse']).size()\
        .sort_index(level='exp_moment',ascending=False)

    assert len(_z) == all_sims_df.exp_moment.nunique(), 'Author: some exp_moment has multiple n_rewards_obtained_mouse values'
    return


@app.cell
def _():
    return


@app.cell
def _(all_sims_df):

    all_sims_df.agent_model.value_counts()
    return


@app.cell
def _(all_sims_df):
    all_sims_df.full_sim_desc.value_counts()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # calculate stats
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    (GPT) sanity check stats

    - 260109 Author: is this still relevant/helpful?
    - 251216 GPT says it passed relative to the printed outputs
    """)
    return


@app.cell
def _(run_case_rank_biserial_n_wstat_sanity_checks):
    ## Toy cases
    run_case_rank_biserial_n_wstat_sanity_checks("all positive", [1,2,3,4,5])
    run_case_rank_biserial_n_wstat_sanity_checks("all negative", [-1,-2,-3,-4,-5])
    run_case_rank_biserial_n_wstat_sanity_checks("mixed", [1,2,-1,-2,3])
    run_case_rank_biserial_n_wstat_sanity_checks("with zeros", [1,0,2,0,-3,4], zero_tol=0.0)
    run_case_rank_biserial_n_wstat_sanity_checks("ties in |diff|", [2,-2,2,-2,1,-1])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## select the dataframe to run stats on
    """)
    return


@app.cell
def _(np, pd):
    # 251225 Author: new gpt code to fill missing mouse columns by matching rows by shared exp_moment-level

    # --- CHANGE 1: fix cols default arg (string -> 1-tuple) ---
    def fill_from_exp_moment(df: pd.DataFrame,
                                 cols=("n_rewards_obtained_mouse",), # CHANGED: added trailing comma
                                 key="exp_moment") -> pd.DataFrame:

    # def fill_from_exp_moment(df: pd.DataFrame,
    #                          cols=("n_rewards_obtained_mouse"),
    #                          key="exp_moment") -> pd.DataFrame:
        """
        For each col in `cols`, fill NaNs in rows by copying the (unique) non-null value
        found within the same `key` group (e.g., exp_moment).

        Safety checks:
          - If a group has >1 distinct non-null value for a column, raise.
          - If a group has no non-null value for a column, raise (can't infer fill).
        """
        out = df.copy()

        def _unique_nonnull_value(s: pd.Series):
            vals = pd.unique(s.dropna())
            if len(vals) == 1:
                return vals[0]
            if len(vals) == 0:
                return np.nan
            raise AssertionError(
                f"Non-unique values within {key}='{s.name}' for column '{s._colname}': {vals}"
            )

        for col in cols:
            if col not in out.columns:
                continue

            # compute the per-exp_moment value to broadcast back to rows
            g = out.groupby(key, dropna=False)[col]

            # attach colname for better error messages inside _unique_nonnull_value
            def _wrap_unique(s):
                s._colname = col  # type: ignore[attr-defined]
                return _unique_nonnull_value(s)

            per_group_val = g.apply(_wrap_unique)  # index: exp_moment -> value

            # if any group has no non-null value, we cannot fill
            missing_groups = per_group_val[per_group_val.isna()].index
            if len(missing_groups) > 0:
                raise AssertionError(
                    f"Cannot fill '{col}': no non-null value exists for these {key} groups: "
                    f"{list(missing_groups)[:10]}{'...' if len(missing_groups) > 10 else ''}"
                )

            # broadcast mapping back to each row and fill NaNs
            out[col] = out[col].fillna(out[key].map(per_group_val))

        return out
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## assign n_rewards_obtained_agent and n_rewards_obtained_mouse columns to rows with nans based on the exp_moment column
    """)
    return


@app.cell
def _():
    return


@app.cell
def _(all_sims_df):
    all_sims_df.groupby(['full_sim_desc', 'policy_class']).size()
    return


@app.cell
def _(all_sims_df):
    all_sims_df
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## STATS USER PARAMS n checks
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    260108 Author: removing some of this trying to get it to make plots for tomorrow. Recheck later!
    """)
    return


@app.cell
def _(
    KEY_COLS,
    all_sims_df,
    get_moment_strings,
    is_numeric_dtype,
    np,
    pd,
    split_early_late,
):
    DF_FOR_STATS = all_sims_df.copy() ## Author: just in case you want to filter the plots separately from above

    ## 260108 Author: hacking back in exp moment which was dropped for some reason...
    # DF_FOR_STATS.csv_data_path
    DF_FOR_STATS['exp_moment'] = DF_FOR_STATS.csv_data_path.apply(get_moment_strings)
    DF_FOR_STATS['moment'] = DF_FOR_STATS.csv_data_path.apply(get_moment_strings)
    DF_FOR_STATS['exp_moment']

    STATS_TEST = 'nonparametric'  # must be "parametric" (paired t tests) or "nonparametric" (Wilcoxon)
    TRAINING_STAGE_COL = 'training_stage'
    _VALUE_COLS=["reward_rate_mouse","reward_rate_agent"] # this are the columns that we actually care about

    ## assert no NaN remain
    assert DF_FOR_STATS["n_rewards_obtained_agent"].isnull().sum() == 0, "NaN in n_rewards_obtained_agent"
    assert DF_FOR_STATS["n_rewards_obtained_mouse"].isnull().sum() == 0, "NaN in n_rewards_obtained_mouse"
    assert DF_FOR_STATS["n_states_visited"].isnull().sum() == 0, "NaN in n_actions_agent"
    DF_FOR_STATS[DF_FOR_STATS.n_rewards_obtained_mouse.isnull()].full_sim_desc.unique()


    assert STATS_TEST in {"parametric","nonparametric"}, f"Author: set STATS_TEST to 'parametric' or 'nonparametric'; got {STATS_TEST}"

    # DF_FOR_STATS = per_exp_result_df.copy()
    # DF_FOR_STATS = RL_results_df.copy() ## just RL; 251217 Author: untested

    # KEY_COLS = ['animal_ID','exp_moment']


    # ## copy mouse data from fixed policy rows to RL rows based on exp_moment column
    # DF_FOR_STATS = fill_from_exp_moment(DF_FOR_STATS, cols=("n_rewards_obtained_mouse"), key="exp_moment")
    # ## sanity checks
    # assert DF_FOR_STATS["n_rewards_obtained_agent"].isnull().sum() == 0, "NaN in n_rewards_obtained_agent"
    # assert DF_FOR_STATS["n_rewards_obtained_mouse"].isnull().sum() == 0, "NaN in n_rewards_obtained_mouse"
    # assert DF_FOR_STATS["n_states_visited"].isnull().sum() == 0, "NaN in n_actions_agent"

    # --- CHANGE 2+3: fix call site tuple + force numeric before reward-rate division ---
    # DF_FOR_STATS = fill_from_exp_moment(DF_FOR_STATS, cols=("n_rewards_obtained_mouse",), key="exp_moment") # CHANGED: added trailing comma # 260108 Author: removing some of this trying to get it to make plots for tomorrow. Recheck later!

    ## sanity checks
    assert DF_FOR_STATS["n_rewards_obtained_agent"].isnull().sum() == 0, "NaN in n_rewards_obtained_agent"
    assert DF_FOR_STATS["n_rewards_obtained_mouse"].isnull().sum() == 0, "NaN in n_rewards_obtained_mouse"
    assert DF_FOR_STATS["n_states_visited"].isnull().sum() == 0, "NaN in n_actions_agent"

    ## CHANGED: force numeric dtypes (object/object division can yield object dtype reward_rate_*)
    _num_base_cols=["n_rewards_obtained_agent","n_rewards_obtained_mouse","n_states_visited"]
    for _c in _num_base_cols:
        DF_FOR_STATS[_c]=pd.to_numeric(DF_FOR_STATS[_c],errors="raise")

    ## CHANGED: compute as floats to guarantee numeric dtype
    DF_FOR_STATS["reward_rate_agent"]=DF_FOR_STATS["n_rewards_obtained_agent"].astype(float)/DF_FOR_STATS["n_states_visited"].astype(float)
    DF_FOR_STATS["reward_rate_mouse"]=DF_FOR_STATS["n_rewards_obtained_mouse"].astype(float)/DF_FOR_STATS["n_states_visited"].astype(float)


    #### add the reward rate columns

    ## 260106 GPT new: ensure numeric dtypes (object->object division yields object dtype reward_rate)
    _num_base_cols=["n_rewards_obtained_agent","n_rewards_obtained_mouse","n_states_visited"]
    for _c in _num_base_cols:
        if _c in DF_FOR_STATS.columns:
            DF_FOR_STATS[_c]=pd.to_numeric(DF_FOR_STATS[_c],errors="raise")

    DF_FOR_STATS["reward_rate_agent"]=DF_FOR_STATS["n_rewards_obtained_agent"].astype(float)/DF_FOR_STATS["n_states_visited"].astype(float)
    DF_FOR_STATS["reward_rate_mouse"]=DF_FOR_STATS["n_rewards_obtained_mouse"].astype(float)/DF_FOR_STATS["n_states_visited"].astype(float)

    # DF_FOR_STATS["reward_rate_agent"] = DF_FOR_STATS["n_rewards_obtained_agent"] / DF_FOR_STATS["n_states_visited"]
    # DF_FOR_STATS["reward_rate_mouse"] = DF_FOR_STATS["n_rewards_obtained_mouse"] / DF_FOR_STATS["n_states_visited"]

    bad_mask = (~np.isfinite(DF_FOR_STATS["reward_rate_agent"].to_numpy(dtype=float)) | ~np.isfinite(DF_FOR_STATS["reward_rate_mouse"].to_numpy(dtype=float)))
    bad_rows_df = DF_FOR_STATS.loc[bad_mask].copy()

    """ add the training stage column to DF_FOR_STATS """
    rows=[] ## build mapping table: one row per (animal_ID, exp_moment) that you KEEP
    for _animal_ID, _animal_df in DF_FOR_STATS.groupby('animal_ID', dropna=False):
        _animal_df = _animal_df.sort_values('exp_moment', inplace=False)
        session_moments =_animal_df['exp_moment'].drop_duplicates().tolist()
        assert len(session_moments) > 1, f"Author: need >=3 sessions to split early/late, got {len(session_moments)} for animal_ID {_animal_ID}"
    
        early_moments, late_moments = split_early_late(session_moments)
    
        rows.extend([{'animal_ID': _animal_ID, 'exp_moment': m, TRAINING_STAGE_COL: 'early'} for m in early_moments])
        rows.extend([{'animal_ID': _animal_ID, 'exp_moment': m, TRAINING_STAGE_COL: 'late'}  for m in late_moments])
    stage_df=pd.DataFrame(rows)
    ## merge back; rows for the dropped “middle” session (odd case) will get NaN
    DF_FOR_STATS=DF_FOR_STATS.merge(stage_df, on=KEY_COLS, how='left')
    ## optional: drop the middle-session rows entirely
    DF_FOR_STATS=DF_FOR_STATS[DF_FOR_STATS[TRAINING_STAGE_COL].notna()].copy()
    ## optional sanity checks
    assert DF_FOR_STATS[TRAINING_STAGE_COL].isin(['early','late']).all()


    # assert len(DF_FOR_STATS.query(f"{TRAINING_STAGE_COL} == 'early'")) == len(DF_FOR_STATS.query(f"{TRAINING_STAGE_COL} == 'late'")), "Author: early/late counts not equal?"

    # uncomment to sanity check / debug the splitting
    for _animal_ID, _animal_df in DF_FOR_STATS.groupby('animal_ID'):
        # _animal_df = _animal_df.sort_values('exp_moment')
        _animal_df = _animal_df.sort_values('exp_moment', inplace=False)
        assert _animal_df.exp_moment.is_monotonic_increasing, "Author: exp_moment not sorted ascending?"
        session_moments = _animal_df.exp_moment.drop_duplicates().to_list()
        print(f"animal_ID {_animal_ID}: n sessions = {_animal_df.exp_moment.nunique()}")
        early_moments, late_moments = split_early_late(session_moments)
        early_moments_str = '\n'.join(early_moments)
        late_moments_str = '\n'.join(late_moments)
        print(f"early moments: \n{early_moments_str}; \n\nlate moments:\n {late_moments_str}")    
        print('--------------'*10)

    ## validate the input df before actually trying it below in earnest
    assert 'DF_FOR_STATS' in globals(), "DF_FOR_STATS not defined in this marimo state"
    for _col in _VALUE_COLS:
        assert _col in DF_FOR_STATS.columns, f"missing {_col}"
        assert is_numeric_dtype(DF_FOR_STATS[_col]), f"{_col} must be numeric dtype; got {DF_FOR_STATS[_col].dtype}"
        x = DF_FOR_STATS[_col].to_numpy()
        assert x.dtype!=object, f"{_col} is object dtype; mean() may fail"
        assert np.all(np.isfinite(x.astype(float))), f"{_col} contains NaN/Inf"

    ## quick smoke test: this should not throw
    _ = DF_FOR_STATS.groupby(["animal_ID","exp_moment","full_sim_desc"], dropna=False)[_VALUE_COLS].mean()
    print("OK: reward_rate columns are numeric+finite; groupby.mean() succeeded")
    DF_FOR_STATS
    return DF_FOR_STATS, STATS_TEST, TRAINING_STAGE_COL, bad_rows_df


@app.cell
def _(mo):
    mo.md(r"""
    - 260205 Author: seems like this is just an artifact of different sims being ran; still checking...
    - 260204 Author: this needs to be resolved!
    """)
    return


@app.cell
def _():
    return


@app.cell
def _(DF_FOR_STATS, TRAINING_STAGE_COL):
    assert len(DF_FOR_STATS.query(f"{TRAINING_STAGE_COL} == 'early'")) == len(DF_FOR_STATS.query(f"{TRAINING_STAGE_COL} == 'late'")), "Author: early/late counts not equal?"
    return


@app.cell
def _(DF_FOR_STATS, TRAINING_STAGE_COL):
    n_early = DF_FOR_STATS.query(f"{TRAINING_STAGE_COL} == 'early'").groupby(['animal_ID', 'exp_moment']).ngroups
    n_late  = DF_FOR_STATS.query(f"{TRAINING_STAGE_COL} == 'late'").groupby(['animal_ID', 'exp_moment']).ngroups
    assert n_early == n_late, "Author: early/late session counts not equal?"
    return


@app.cell
def _(DF_FOR_STATS, TRAINING_STAGE_COL):
    for _aid, _adf in DF_FOR_STATS.groupby('animal_ID'):
        ne = len(_adf.query(f"{TRAINING_STAGE_COL} == 'early'"))
        nl = len(_adf.query(f"{TRAINING_STAGE_COL} == 'late'"))
        # assert ne == nl, f"animal {_aid}: early={ne}, late={nl}"
        if ne != nl:
            print(f"Warning: animal {_aid} has unequal early/late counts: early={ne}, late={nl}")
    return


@app.cell
def _(DF_FOR_STATS, TRAINING_STAGE_COL):
    ## 1. session-level balance (what split_early_late guarantees)
    for _aid, _adf in DF_FOR_STATS.groupby('animal_ID'):
        early_sessions = _adf.query(f"{TRAINING_STAGE_COL} == 'early'")['exp_moment'].nunique()
        late_sessions  = _adf.query(f"{TRAINING_STAGE_COL} == 'late'")['exp_moment'].nunique()
        assert early_sessions == late_sessions, \
            f"animal {_aid}: early has {early_sessions} sessions, late has {late_sessions}"

    ## 2. rows-per-session consistency (what sim completeness guarantees)
    ROWS_PER_SESSION = (
        DF_FOR_STATS
        .groupby(['animal_ID', 'exp_moment', TRAINING_STAGE_COL])
        .size()
        .reset_index(name='n_rows')
    )
    EXPECTED_ROWS = ROWS_PER_SESSION['n_rows'].mode().iloc[0]
    BAD_SESSIONS = ROWS_PER_SESSION.query("n_rows != @EXPECTED_ROWS")
    assert BAD_SESSIONS.empty, (
        f"Author: {len(BAD_SESSIONS)} sessions have unexpected row counts "
        f"(expected {EXPECTED_ROWS}):\n{BAD_SESSIONS.to_string()}"
    )
    return


@app.cell
def _(DF_FOR_STATS, TRAINING_STAGE_COL):
    (DF_FOR_STATS
     .query("animal_ID in ['a031', 'a033']")
     .groupby(['animal_ID', TRAINING_STAGE_COL, 'full_sim_desc'])
     .size()
     .unstack('full_sim_desc', fill_value=0)
    )
    return


@app.cell
def _():
    return


@app.cell
def _(DF_FOR_STATS, TRAINING_STAGE_COL):
    # len(DF_FOR_STATS.query(f"{TRAINING_STAGE_COL} == 'early'")) == len(DF_FOR_STATS.query(f"{TRAINING_STAGE_COL} == 'late'"))
    len(DF_FOR_STATS.query(f"{TRAINING_STAGE_COL} == 'early'"))
    return


@app.cell
def _(DF_FOR_STATS, TRAINING_STAGE_COL):
    len(DF_FOR_STATS.query(f"{TRAINING_STAGE_COL} == 'late'"))
    return


@app.cell
def _(bad_rows_df):
    bad_rows_df
    return


@app.cell
def _(DF_FOR_STATS):
    DF_FOR_STATS.full_sim_desc.value_counts()
    return


@app.cell
def _(DF_FOR_STATS):
    DF_FOR_STATS.columns
    return


@app.cell
def _(DF_FOR_STATS):
    """ Long-form df for per-mouse model whiskers / downstream plotting.

    Average across repeat_group_idx within (animal_ID, exp_moment, full_sim_desc),
    while keeping reward_rate_mouse aligned with the same session.
    """
    _req = ["animal_ID","exp_moment","full_sim_desc","reward_rate_mouse","reward_rate_agent"]
    _missing = [c for c in _req if c not in DF_FOR_STATS.columns]
    assert not _missing, f"avg_model_perf_df missing required cols: {_missing}"

    _group_cols = ["animal_ID","exp_moment","full_sim_desc"]
    _val_cols = ["reward_rate_mouse","reward_rate_agent"]
    avg_model_perf_df = DF_FOR_STATS.groupby(_group_cols, as_index=False, dropna=True)[_val_cols].mean()

    ## Optional: keep training_stage if it exists and is constant within session
    if "training_stage" in DF_FOR_STATS.columns:
        _ts = (DF_FOR_STATS.groupby(_group_cols, as_index=False, dropna=True)["training_stage"].first())
        avg_model_perf_df = avg_model_perf_df.merge(_ts, on=_group_cols, how="left", validate="one_to_one")

    assert len(avg_model_perf_df) > 0, "avg_model_perf_df ended up empty"
    return (avg_model_perf_df,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## all mice vs all sims (all episodes)

    251217 Author interim conclusion based on most recent 4 animals:
    - equal weighting for sessions -> highly significant
    - equal weighting for animals -> not significant
    """)
    return


@app.cell
def _(DF_FOR_STATS, STATS_TEST, avg_over_MR, get_paired_stats):
    INDEX_MAPPING = {
        'm': 'animal_ID',
        'e': 'exp_moment',
        's': 'full_sim_desc',
        'r': 'repeat_group_idx',}
    INDEX_MAPPING.update({v: k for k,v in INDEX_MAPPING.items()}) # make mapping bi-directional
    # STAT_COLS=["reward_rate_mouse","reward_rate_agent"]

    all_mice_vs_all_sims_stats_variants = {}

    ## notation: M = mouse performance; S = sim performance; lowercase indexing follows INDEX_MAPPING
    Mme_vs_Sme = avg_over_MR(DF_FOR_STATS, ['r', 's'], INDEX_MAPPING) 
    # assert Mme_vs_Sme.exp_moment.nunique() == DF_FOR_STATS.exp_moment.nunique(), \
    #     "Author: unexpected number of unique exp_moment after averaging over repeats and sim types"
    # assert Mme_vs_Sme.animal_ID.nunique() == DF_FOR_STATS.animal_ID.nunique(), \
    #     "Author: unexpected number of unique animal_ID after averaging over repeats and sim types"
    Mme_vs_Sme_expected_n_rows = DF_FOR_STATS.exp_moment.nunique()
    assert len(Mme_vs_Sme) == Mme_vs_Sme_expected_n_rows, \
        f"Author: unexpected number of rows after averaging over repeats and sim types: {len(Mme_vs_Sme)} vs {Mme_vs_Sme_expected_n_rows}"

    all_mice_vs_all_sims_stats_variants['Mme_vs_Sme-parametric'] = get_paired_stats(Mme_vs_Sme.reward_rate_mouse, 
                                                                                    Mme_vs_Sme.reward_rate_agent, 
                                                                                    test=STATS_TEST)

    all_mice_vs_all_sims_stats_variants['Mme_vs_Sme-nonparametric'] = get_paired_stats(Mme_vs_Sme.reward_rate_mouse, 
                                                                                       Mme_vs_Sme.reward_rate_agent, 
                                                                                       test=STATS_TEST)
    all_mice_vs_all_sims_stats_variants, Mme_vs_Sme
    return INDEX_MAPPING, Mme_vs_Sme, all_mice_vs_all_sims_stats_variants


@app.cell
def _(mo):
    mo.md(r"""
    251217 Author: almost significant; maybe revist with more animals
    """)
    return


@app.cell
def _(DF_FOR_STATS, INDEX_MAPPING, STATS_TEST, avg_over_MR, get_paired_stats):
    Mm_vs_Sm = avg_over_MR(DF_FOR_STATS, ['r', 's', 'e'], INDEX_MAPPING) 
    Mm_vs_Sm, get_paired_stats(Mm_vs_Sm['reward_rate_mouse'], Mm_vs_Sm['reward_rate_agent'], test=STATS_TEST)
    # DF_FOR_STATS.groupby(INDEX_MAPPING['m'])[STAT_COLS].mean() # Author: 
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Author: interesting; removing the bad performer doesn't bias the stats to favor the mice; presumably due to the decrease in N
    """)
    return


@app.cell
def _(DF_FOR_STATS, INDEX_MAPPING, STATS_TEST, avg_over_MR, get_paired_stats):
    # ANIMALS_EXCLUDED = ['a033']
    ANIMALS_EXCLUDED = []

    Mm_vs_Sm_generous = avg_over_MR(DF_FOR_STATS.query(f"animal_ID not in {ANIMALS_EXCLUDED}"), ['r', 's', 'e'], INDEX_MAPPING) 
    Mm_vs_Sm_generous, get_paired_stats(Mm_vs_Sm_generous['reward_rate_mouse'], Mm_vs_Sm_generous['reward_rate_agent'], test=STATS_TEST), 
    # DF_FOR_STATS.groupby(INDEX_MAPPING['m'])[STAT_COLS].mean() # Author: 
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Approaches that reuse mouse data: shelving/discarding for now...

    Author3: unfair weird due to reusing animal data
    """)
    return


@app.cell
def _():
    # Mmer_vs_Smer = avg_over_MR(DF_FOR_STATS, ['s'], INDEX_MAPPING)
    # M_vs_Ss = avg_over_MR(DF_FOR_STATS, ['r', 'm', 'e'], INDEX_MAPPING)
    # Mm_vs_Sms = avg_over_MR(DF_FOR_STATS, ['r', 'e'], INDEX_MAPPING)

    ## quick displays
    # Mmer_vs_Smer, get_paired_stats(Mmer_vs_Smer.reward_rate_mouse, \
    #     Mmer_vs_Smer.reward_rate_agent, test=STATS_TEST), get_paired_stats(Mmer_vs_Smer.reward_rate_mouse, Mmer_vs_Smer.reward_rate_agent, test=STATS_TEST)

    # get_paired_stats(M_vs_Ss.reward_rate_mouse, M_vs_Ss.reward_rate_agent, test=STATS_TEST), M_vs_Ss, DF_FOR_STATS.reward_rate_mouse.mean()

    # get_paired_stats(Mm_vs_Sms.reward_rate_mouse, Mm_vs_Sms.reward_rate_agent, test=STATS_TEST), Mm_vs_Sms
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## all mice vs each sim (all episodes)
    """)
    return


@app.cell
def _(
    DF_FOR_STATS,
    INDEX_MAPPING,
    STATS_TEST,
    avg_over_MR,
    get_paired_stats,
    pd,
):
    Mm_vs_Sms = avg_over_MR(DF_FOR_STATS, ['e', 'r'], INDEX_MAPPING)

    Mm_vs_Sms_expected_n_rows = DF_FOR_STATS.animal_ID.nunique() * DF_FOR_STATS.full_sim_desc.nunique()

    # assert len(Mm_vs_Sms) == Mm_vs_Sms_expected_n_rows, \
    #     f"Author: unexpected number of rows in Mm_vs_Sms: got {len(Mm_vs_Sms)} vs expected {Mm_vs_Sms_expected_n_rows}"

    _rows=[]
    for sim_str, sim_df in Mm_vs_Sms.groupby('full_sim_desc', dropna=True)[['reward_rate_mouse','reward_rate_agent']]:
        _stats = get_paired_stats(sim_df['reward_rate_mouse'].to_numpy(), sim_df['reward_rate_agent'].to_numpy(), test=STATS_TEST)
        print(f"{sim_str} -- mean diff (mouse - sim): {_stats['mean_diff']} -- p val: {_stats['p_val']}")
        _rows.append({'sim': sim_str, **_stats})

    stats_per_sim_df = pd.DataFrame(_rows)
    # Mm_vs_Sms, stats_per_sim_df,
    return (sim_str,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## each mouse vs all sims (all episodes)
    """)
    return


@app.cell
def _(Mme_vs_Sme, STATS_TEST, get_paired_stats, pd, sim_str):
    # Mme_vs_Sme  # assumed to be defined above

    _rows=[]
    for animal_ID, mouse_df in Mme_vs_Sme.groupby('animal_ID', dropna=True)[['reward_rate_mouse','reward_rate_agent']]:
        _stats = get_paired_stats(mouse_df['reward_rate_mouse'].to_numpy(), mouse_df['reward_rate_agent'].to_numpy(), test=STATS_TEST)
        print(f"{sim_str} -- mean diff (mouse - sim): {_stats['mean_diff']} -- p val: {_stats['p_val']}")
        _rows.append({'animal_ID': animal_ID, **_stats})

    stats_per_mouse_df=pd.DataFrame(_rows)
    Mme_vs_Sme, stats_per_mouse_df,
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## each mouse vs each sim (all episodes)
    """)
    return


@app.cell
def _(
    DF_FOR_STATS,
    INDEX_MAPPING,
    STATS_TEST,
    avg_over_MR,
    get_paired_stats,
    pd,
    sim_str,
):
    Mme_vs_Smes = avg_over_MR(DF_FOR_STATS, ['r'], INDEX_MAPPING)
    # Mme_vs_Smes

    _rows=[]
    for (_animal_ID, _full_sim_desc), mouse_x_sim_df in Mme_vs_Smes.groupby(['animal_ID', 'full_sim_desc'], dropna=True)[['reward_rate_mouse','reward_rate_agent']]:
        _stats = get_paired_stats(mouse_x_sim_df['reward_rate_mouse'].to_numpy(), mouse_x_sim_df['reward_rate_agent'].to_numpy(), test=STATS_TEST)
        print(f"{sim_str} -- mean diff (mouse - sim): {_stats['mean_diff']} -- p val: {_stats['p_val']}")
        _rows.append({'animal_ID': _animal_ID, 'full_sim_desc': _full_sim_desc, **_stats})

    stats_per_mouse_x_sim_df = pd.DataFrame(_rows)
    Mme_vs_Smes, stats_per_mouse_x_sim_df
    return Mme_vs_Smes, stats_per_mouse_x_sim_df


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## each mouse: early vs late

    - 260112 Author: fixed a couple days ago (solution unclear maybe single line filter above removed)?
    - 260108 Author: broken via something upstream
    """)
    return


@app.cell
def _(
    DF_FOR_STATS,
    STATS_TEST,
    get_paired_stats,
    move_df_col_to_leftmost,
    np,
    pd,
):
    mouse_training_stage_stat_rows=[]
    for _animal_ID, animal_df in DF_FOR_STATS.groupby('animal_ID', dropna=False):
        print(f"animal_ID {_animal_ID}: n sessions (raw unique exp_moment) = {animal_df.exp_moment.nunique()}")

        ## assert reward_rate_mouse is constant within each session (across different sims)
        chk = animal_df.groupby(['exp_moment'], dropna=True)['reward_rate_mouse'].nunique()
        assert (chk==1).all(), f"Author: reward_rate_mouse not constant within session:\n{chk[chk!=1]}"

        ## collapse to 1 row per session
        _mouse_per_session = animal_df.sort_values('exp_moment').groupby(['exp_moment','training_stage'], dropna=True)['reward_rate_mouse'].first().reset_index()

        rwd_rate_early=_mouse_per_session.query("training_stage=='early'").sort_values('exp_moment')['reward_rate_mouse'].to_numpy()
        rwd_rate_late =_mouse_per_session.query("training_stage=='late'").sort_values('exp_moment')['reward_rate_mouse'].to_numpy()

        assert np.isfinite(rwd_rate_early).all() and np.isfinite(rwd_rate_late).all()
        assert len(rwd_rate_early)==len(rwd_rate_late), f"Author: unequal number of sessions early vs late ({len(rwd_rate_early)} vs {len(rwd_rate_late)})"

        mouse_training_stage_stat_rows.append({
            'animal_ID': _animal_ID,
            'n_sessions_labeled': _mouse_per_session.exp_moment.nunique(),  # counts only early/late (middle dropped)
            'n_early': len(rwd_rate_early),
            'n_late': len(rwd_rate_late),
            **get_paired_stats(rwd_rate_late, rwd_rate_early, names=('late', 'early'), test=STATS_TEST, alternative='two-sided'),})

    mouse_training_stage_stat_df = pd.DataFrame(mouse_training_stage_stat_rows)
    mouse_training_stage_stat_df = move_df_col_to_leftmost(mouse_training_stage_stat_df, 'p_val')
    mouse_training_stage_stat_df = move_df_col_to_leftmost(mouse_training_stage_stat_df, 'mean_diff')
    mouse_training_stage_stat_df
    return (chk,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## all mice: early vs late
    """)
    return


@app.cell
def _(DF_FOR_STATS, STATS_TEST, chk, get_paired_stats):
    df_labeled=DF_FOR_STATS[DF_FOR_STATS.training_stage.isin(['early','late'])].copy()

    ## 1) ensure mouse reward is constant within session (across sims)
    _chk = df_labeled.groupby(['animal_ID','exp_moment'])['reward_rate_mouse'].nunique()
    assert (_chk == 1).all(), f"Author: reward_rate_mouse not constant within session:\n{chk[chk!=1]}"

    ## 2) collapse to one row per (mouse, session)
    mouse_per_session = df_labeled.sort_values(['animal_ID','exp_moment']).groupby(['animal_ID','exp_moment','training_stage'], dropna=True)['reward_rate_mouse'].first().reset_index()

    ## 3) per-mouse early/late means (one pair per mouse)
    mouse_stage_means = mouse_per_session.groupby(['animal_ID','training_stage'])['reward_rate_mouse'].mean().unstack('training_stage')
    assert set(['early','late']).issubset(mouse_stage_means.columns), f"Author: missing early/late cols: {mouse_stage_means.columns}"
    mouse_stage_means = mouse_stage_means.dropna(subset=['early','late'])

    rwd_early=mouse_stage_means['early'].to_numpy()
    rwd_late =mouse_stage_means['late'].to_numpy()

    ## positive diff = late - early
    all_mice_stats=get_paired_stats(rwd_late, rwd_early, names=('late', 'early'), test=STATS_TEST, alternative='two-sided')
    all_mice_stats
    return


@app.cell
def _(mo):
    mo.md(r"""
    # 260106 GPT new: audit
    """)
    return


@app.cell
def _(DF_FOR_STATS, MODEL_FORMATTING_DF, all_sims_df, np):
    def audit_pipeline_before_plot(all_sims_df, DF_FOR_STATS, MODEL_FORMATTING_DF, model_col="full_sim_desc", group_col="animal_ID"):
        print("=== all_sims_df ===")
        print("shape:", all_sims_df.shape)
        print("policy_class counts:\n", all_sims_df["policy_class"].value_counts(dropna=False))
        need_cols=["animal_ID","exp_moment","policy_class","full_sim_desc","n_rewards_obtained_agent","n_states_visited"]
        missing=[c for c in need_cols if c not in all_sims_df.columns]
        assert not missing, f"missing required cols in all_sims_df: {missing}"

        print("\n=== DF_FOR_STATS ===")
        print("shape:", DF_FOR_STATS.shape)
        for c in ["n_rewards_obtained_agent","n_rewards_obtained_mouse","n_states_visited","reward_rate_agent","reward_rate_mouse"]:
            assert c in DF_FOR_STATS.columns, f"missing {c} in DF_FOR_STATS"
            bad=~np.isfinite(DF_FOR_STATS[c].to_numpy(dtype=float))
            if bad.any():
                print(f"non-finite in {c}: n_bad={int(bad.sum())}")
                print(DF_FOR_STATS.loc[bad,["animal_ID","exp_moment","policy_class","full_sim_desc",c]].head(30))

        chk=DF_FOR_STATS.groupby(["animal_ID","exp_moment"], dropna=False)["reward_rate_mouse"].nunique(dropna=False)
        bad=chk[chk!=1]
        if len(bad):
            print("\nWARNING: reward_rate_mouse not constant within session (breaks paired assumptions):")
            print(bad.head(30))

        print("\n=== MODEL_FORMATTING_DF ===")
        assert MODEL_FORMATTING_DF.index.name=="model", f"MODEL_FORMATTING_DF.index.name={MODEL_FORMATTING_DF.index.name}"
        assert MODEL_FORMATTING_DF.index.is_unique, "MODEL_FORMATTING_DF index is not unique"
        for c in ["xtick_label","legend_group","color"]:
            assert c in MODEL_FORMATTING_DF.columns, f"missing required formatting col: {c}"
            bad=MODEL_FORMATTING_DF[c].isna()
            if bad.any():
                print(f"NaN in MODEL_FORMATTING_DF[{c}] for (showing up to 50):")
                print(MODEL_FORMATTING_DF.index[bad].tolist()[:50])

        models=sorted([str(m) for m in DF_FOR_STATS[model_col].dropna().unique().tolist()])
        missing_fmt=[m for m in models if m not in MODEL_FORMATTING_DF.index]
        if missing_fmt:
            print("\nWARNING: models missing from MODEL_FORMATTING_DF (showing up to 50):")
            print(missing_fmt[:50])

        if "model_order" in MODEL_FORMATTING_DF.columns:
            ord_ser=MODEL_FORMATTING_DF.loc[[m for m in models if m in MODEL_FORMATTING_DF.index],"model_order"]
            bad=~np.isfinite(ord_ser.astype(float))
            if bad.any():
                print("\nWARNING: non-finite model_order for (showing up to 50):")
                print(ord_ser.index[bad].tolist()[:50])

    audit_pipeline_before_plot(all_sims_df, DF_FOR_STATS, MODEL_FORMATTING_DF)
    print('Author: no news (asserts) is good news?')
    return


@app.cell
def _(DF_FOR_STATS):
    DF_FOR_STATS.columns
    return


@app.cell
def _(mo):
    mo.md(r"""
    # PLOT: mice vs sims for each session

    - 251216 Author: adapting from earlier notebook version labeled: v1.2 nicer formatting
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Default plotting params
    """)
    return


@app.cell
def _():
    AXIS_LABEL_FONT_SIZE = 16
    TITLE_FONT_SIZE = 20
    MOUSE_POINT_MARKER_SIZE = 9
    MOUSE_POINT_ALPHA = 0.95
    LINE_WIDTH_MOUSE = 1.7
    LINE_WIDTH_MODELS = 2.3
    LINE_WIDTH_ALL_MODELS = 3
    LEGEND_FONTSIZE = 13
    TICK_LABEL_FONT_SIZE = 13
    return (
        AXIS_LABEL_FONT_SIZE,
        LEGEND_FONTSIZE,
        LINE_WIDTH_ALL_MODELS,
        LINE_WIDTH_MODELS,
        LINE_WIDTH_MOUSE,
        MOUSE_POINT_ALPHA,
        MOUSE_POINT_MARKER_SIZE,
        TICK_LABEL_FONT_SIZE,
        TITLE_FONT_SIZE,
    )


@app.cell
def _(
    cm,
    deserialize_str,
    get_now_str,
    model_name_2_legend_str,
    np,
    pd,
    plt,
    sanity_check_all_agents_confidence_interval_inputs,
    sanity_check_per_model_confidence_interval_inputs,
    sanity_check_session_subset_df,
    ss,
):
    def plot_mice_vs_sims_per_session_GPT_dev(selected_animal_IDs, per_exp_result_df, normalize_by_n_actions_in_session=True, run_sanity_checks=True, MODEL_FORMATTING_DF=None, **kwargs):
        kwa = kwargs # shorten the name

        use_model_formatting = MODEL_FORMATTING_DF is not None
        if use_model_formatting:
            assert isinstance(MODEL_FORMATTING_DF, pd.DataFrame), "Author: MODEL_FORMATTING_DF must be a DataFrame"
            assert MODEL_FORMATTING_DF.index.name == "model", f"Author: MODEL_FORMATTING_DF.index.name must be 'model'; got {MODEL_FORMATTING_DF.index.name}"
            assert MODEL_FORMATTING_DF.index.is_unique, "Author: MODEL_FORMATTING_DF index must be unique"
            required_fmt_cols = {"color"}
            missing_fmt_cols = required_fmt_cols.difference(MODEL_FORMATTING_DF.columns)
            assert not missing_fmt_cols, f"Author: MODEL_FORMATTING_DF missing required columns: {sorted(missing_fmt_cols)}"

            label_col = "legend_label" if "legend_label" in MODEL_FORMATTING_DF.columns else "xtick_label"
            assert label_col in MODEL_FORMATTING_DF.columns, "Author: MODEL_FORMATTING_DF must contain either 'legend_label' or 'xtick_label'"

            def _order_models(model_list):
                model_list = list(model_list)
                if "model_order" in MODEL_FORMATTING_DF.columns:
                    ord_ser = pd.to_numeric(MODEL_FORMATTING_DF.loc[model_list, "model_order"], errors="raise")
                    assert np.isfinite(ord_ser.to_numpy()).all(), "Author: MODEL_FORMATTING_DF['model_order'] must be finite for plotted models"
                    return [m for m,_ in sorted(zip(model_list, ord_ser.to_numpy()), key=lambda z: z[1])]
                return sorted(model_list)
        else:
            label_col = None

        def _as_list(x):
            return deserialize_str(x) if isinstance(x, str) else x

        def _n_actions_from_states(states_visited):
            states_visited = _as_list(states_visited)
            n_states = len(states_visited)
            assert n_states >= 2, f"Author: need at least 2 states to define actions; got {n_states}"
            return n_states - 1

        def _metric(n_rewards, n_actions):
            if normalize_by_n_actions_in_session:
                assert n_actions > 0, f"Author: n_actions must be > 0; got {n_actions}"
                return float(n_rewards) / float(n_actions)
            return float(n_rewards)

        if normalize_by_n_actions_in_session:
            y_label = "n rewards / n actions"
            mouse_legend_label = "mouse reward rate"
        else:
            y_label = "n rewards"
            mouse_legend_label = "mouse rewards"

        for animal_ID in selected_animal_IDs:
            print(f'{get_now_str(hms=True)}: plotting/processing animal_ID {animal_ID}...')

            mice_subset = per_exp_result_df.query('animal_ID == @animal_ID').copy()

            plt.figure(figsize=kwa.get('figsize', (17, 6)))
            cyan_count = 0

            present_models = list(pd.unique(mice_subset['full_sim_desc']))
            if use_model_formatting:
                missing_models = [m for m in present_models if m not in MODEL_FORMATTING_DF.index]
                assert not missing_models, f"Author: models missing from MODEL_FORMATTING_DF: {missing_models}"
                model_names = _order_models(present_models)
            else:
                model_names = sorted(present_models)
                cmap = cm.get_cmap('turbo', len(model_names) if len(model_names) > 0 else 1)
                model_color_map = {name: cmap(j) for j, name in enumerate(model_names)}

            plt.plot([], [], '-', color='c',label=f'{mouse_legend_label}\n   (exceeds 95% CI of all yoked means)', linewidth=kwa['line_width_models'])
            plt.plot([], [], '-', color='g', label=f'{mouse_legend_label}\n   (exceeds yoked mean)', linewidth=kwa['line_width_models'])
            plt.plot([], [], '-', color='r', label=f'{mouse_legend_label}\n   (under yoked mean)', linewidth=kwa['line_width_models'])
            plt.plot([], [], color='black', label='yoked mean (95% CI)', linewidth=kwa['line_width_all_models'])

            sessions = list(mice_subset['exp_moment'].unique())
            for i, (exp_moment, session_subset_df) in enumerate(mice_subset.groupby('exp_moment')):

                rewarded_action_inds_mouse = _as_list(session_subset_df['rewarded_action_inds_mouse'].iloc[0])
                states_visited_mouse = _as_list(session_subset_df['states_visited_mouse'].iloc[0])

                n_rew_mouse = len(rewarded_action_inds_mouse)
                n_actions_mouse = _n_actions_from_states(states_visited_mouse)
                mouse_val = _metric(n_rew_mouse, n_actions_mouse)

                if run_sanity_checks:
                    sanity_check_session_subset_df(session_subset_df, exp_moment)

                    ## 260110 GPT debug suggestion
                    if "n_rewards_obtained_mouse" in session_subset_df.columns:
                        assert int(session_subset_df["n_rewards_obtained_mouse"].iloc[0])==n_rew_mouse
                    if "n_states_visited" in session_subset_df.columns:
                        n_states_scalar=int(session_subset_df["n_states_visited"].iloc[0])
                        assert n_states_scalar in {len(states_visited_mouse),n_actions_mouse,n_actions_mouse+1},(
                            f"n_states_visited={n_states_scalar}, len(states_visited_mouse)={len(states_visited_mouse)}, "
                            f"n_actions_mouse={n_actions_mouse}")
                    if "reward_rate_mouse" in session_subset_df.columns:
                        rr=float(session_subset_df["reward_rate_mouse"].iloc[0])
                        assert np.isfinite(rr)

                session_model_names = list(pd.unique(session_subset_df['full_sim_desc']))
                if use_model_formatting:
                    session_model_names = _order_models(session_model_names)
                else:
                    session_model_names = sorted(session_model_names)
                n_models_this_session = len(session_model_names)
                assert n_models_this_session > 0, f'Author: no models found for {exp_moment}'

                d = 1.0 / (n_models_this_session + 2)
                session_x_positions = i + d * np.arange(1, n_models_this_session + 1)

                all_agents_reward = []
                for model_idx, model_name in enumerate(session_model_names):
                    model_subset = session_subset_df[session_subset_df['full_sim_desc'] == model_name]

                    per_run_vals = []
                    for _, row in model_subset.iterrows():

                        rew_inds_agent = _as_list(row['rewarded_action_inds_agent'])
                        st_agent = _as_list(row['states_visited_agent'])

                        n_rew = len(rew_inds_agent)
                        n_actions_agent = _n_actions_from_states(st_agent)

                        assert n_actions_agent == n_actions_mouse, (
                            "Author: n actions agent != mouse; likely a deserialization/yoking issue "
                            f"(mouse={n_actions_mouse}, agent={n_actions_agent})")

                        per_run_vals.append(_metric(n_rew, n_actions_agent))

                    per_run_arr = np.asarray(per_run_vals, dtype=float)
                    sem_val = ss.sem(per_run_arr)

                    if run_sanity_checks:
                        sanity_check_per_model_confidence_interval_inputs(per_run_arr, sem_val, model_name, exp_moment)

                    model_mean = float(np.mean(per_run_arr))
                    all_agents_reward.append(model_mean)

                    ci = ss.t.interval(0.95, df=per_run_arr.size - 1, loc=model_mean, scale=sem_val)

                    x_offset = session_x_positions[model_idx]
                    if i == 0:
                        label_str = MODEL_FORMATTING_DF.loc[model_name, label_col] if use_model_formatting else model_name_2_legend_str(model_name)
                    else:
                        label_str = None

                    plt.plot([x_offset]*2, ci,
                             color=(MODEL_FORMATTING_DF.loc[model_name, "color"] if use_model_formatting else model_color_map.get(model_name, "0.5")),
                             label=label_str,
                             linewidth=kwa['line_width_models'])

                all_agents_reward_arr = np.asarray(all_agents_reward, dtype=float)
                sem_all = ss.sem(all_agents_reward_arr)
                if run_sanity_checks:
                    sanity_check_all_agents_confidence_interval_inputs(all_agents_reward_arr, sem_all, n_models_this_session, exp_moment)

                agents_reward_95ci = ss.t.interval(0.95, df=all_agents_reward_arr.size - 1, loc=float(np.mean(all_agents_reward_arr)), scale=sem_all)

                plt.plot([i]*2, agents_reward_95ci, color='black', linewidth=kwa['line_width_all_models'])

                mouse_line_color = ('c' if mouse_val > agents_reward_95ci[-1] else ('g' if mouse_val > np.nanmean(all_agents_reward_arr) else 'r'))

                if mouse_line_color == 'c':
                    cyan_count += 1

                x_end = float(np.max(session_x_positions))
                plt.plot([i, x_end], [mouse_val, mouse_val], '--', color=mouse_line_color, linewidth=kwa['line_width_mouse'], alpha=kwa['mouse_point_alpha'])

            plt.tick_params(axis='both', labelsize=kwa['tick_label_font_size'])
            plt.xlabel('Experiment session', fontsize=kwa['axis_label_font_size'])
            plt.ylabel(y_label, fontsize=kwa['axis_label_font_size'])

            n_sessions = len(sessions)

            title_prefix = kwa.get('title_prefix', '')

            plt.title(f"{title_prefix}mouse {animal_ID}: sessions with mouse > 95% CI of all sim. yoked means = {cyan_count} / {n_sessions}", fontsize=kwa['title_font_size'])

            plt.gcf().subplots_adjust(bottom=0.32)
            plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.22), fontsize=kwa['legend_fontsize'], ncols=kwa['n_legend_cols'], frameon=True)

            plt.savefig(f"./figures/c{get_now_str(hms=True)}_{animal_ID}_figS3_reward_rates_by_session-v1.2allModels.png", bbox_inches='tight', dpi=400)
            plt.savefig(f"./figures/c{get_now_str(hms=True)}_{animal_ID}_figS3_reward_rates_by_session-v1.2allModels.svg", bbox_inches='tight', dpi=400)
            plt.savefig(f"./figures/c{get_now_str(hms=True)}_{animal_ID}_figS3_reward_rates_by_session-v1.2allModels.pdf", bbox_inches='tight', dpi=400)
            plt.show()
    return (plot_mice_vs_sims_per_session_GPT_dev,)


@app.cell
def _(
    cm,
    deserialize_str,
    get_now_str,
    model_name_2_legend_str,
    np,
    pd,
    plt,
    sanity_check_all_agents_confidence_interval_inputs,
    sanity_check_per_model_confidence_interval_inputs,
    sanity_check_session_subset_df,
    ss,
):


    def plot_mice_vs_sims_per_session_GPT_dev_v2(selected_animal_IDs, per_exp_result_df, normalize_by_n_actions_in_session=True, run_sanity_checks=True, MODEL_FORMATTING_DF=None, **kwargs):

        def _print_skip(msg):
            print(f"{get_now_str(hms=True)}: Author: skip | {msg}")

        kwa=kwargs

        def _order_models(model_list):
            model_list=list(model_list)
            if "model_order" in MODEL_FORMATTING_DF.columns:
                ord_ser=pd.to_numeric(MODEL_FORMATTING_DF.loc[model_list,"model_order"],errors="raise")
                assert np.isfinite(ord_ser.to_numpy()).all(),"Author: MODEL_FORMATTING_DF['model_order'] must be finite for plotted models"
                return [m for m,_ in sorted(zip(model_list,ord_ser.to_numpy()),key=lambda z:z[1])]
            return sorted(model_list)

        use_model_formatting=MODEL_FORMATTING_DF is not None
        if use_model_formatting:

            assert isinstance(MODEL_FORMATTING_DF,pd.DataFrame),"Author: MODEL_FORMATTING_DF must be a DataFrame"
            assert MODEL_FORMATTING_DF.index.name=="model",f"Author: MODEL_FORMATTING_DF.index.name must be 'model'; got {MODEL_FORMATTING_DF.index.name}"
            assert MODEL_FORMATTING_DF.index.is_unique,"Author: MODEL_FORMATTING_DF index must be unique"
            required_fmt_cols={"color"}
            missing_fmt_cols=required_fmt_cols.difference(MODEL_FORMATTING_DF.columns)
            assert not missing_fmt_cols,f"Author: MODEL_FORMATTING_DF missing required columns: {sorted(missing_fmt_cols)}"

            label_col="legend_label" if "legend_label" in MODEL_FORMATTING_DF.columns else "xtick_label"
            assert label_col in MODEL_FORMATTING_DF.columns,"Author: MODEL_FORMATTING_DF must contain either 'legend_label' or 'xtick_label'"

            MODEL_FORMATTING_DF=MODEL_FORMATTING_DF.copy()
            MODEL_FORMATTING_DF.index=MODEL_FORMATTING_DF.index.astype(str)

        else:
            label_col=None

        def _as_list(x):
            return deserialize_str(x) if isinstance(x,str) else x

        def _n_actions_from_states(states_visited):
            states_visited=_as_list(states_visited)
            n_states=len(states_visited)
            assert n_states>=2,f"Author: need at least 2 states to define actions; got {n_states}"
            return n_states-1

        def _metric(n_rewards,n_actions):
            if normalize_by_n_actions_in_session:
                assert n_actions>0,f"Author: n_actions must be > 0; got {n_actions}"
                return float(n_rewards)/float(n_actions)
            return float(n_rewards)

        if normalize_by_n_actions_in_session:
            y_label="n rewards / n actions"
            mouse_legend_label="mouse reward rate"
        else:
            y_label="n rewards"
            mouse_legend_label="mouse rewards"

        figsize=kwa.get('figsize',(17,6))
        ax=kwa.get("ax",None)               # optional: draw into a provided subplot
        add_labels=kwa.get("add_labels",True) # optional: whether to attach labels (for shared legend workflows)
        draw_legend=kwa.get("draw_legend",True)
        do_save=kwa.get("do_save",True)
        do_show=kwa.get("do_show",True)

        for animal_ID in selected_animal_IDs:
            print(f'{get_now_str(hms=True)}: plotting/processing animal_ID {animal_ID}...')

            mice_subset=per_exp_result_df.query('animal_ID == @animal_ID').copy()

            mice_subset["full_sim_desc"]=mice_subset["full_sim_desc"].astype(str)

            if ax is None:
                plt.figure(figsize=figsize)
            else:
                plt.sca(ax)
                ax.cla()

            cyan_count=0
            n_sessions_skipped=0
            n_sessions_used=0

            present_models=list(pd.unique(mice_subset['full_sim_desc']))
            if use_model_formatting:
                missing_models=[m for m in present_models if m not in MODEL_FORMATTING_DF.index]
                assert not missing_models,f"Author: models missing from MODEL_FORMATTING_DF: {missing_models}"
                model_names=_order_models(present_models)
            else:
                model_names=sorted(present_models)
                cmap=cm.get_cmap('turbo',len(model_names) if len(model_names)>0 else 1)
                model_color_map={name:cmap(j) for j,name in enumerate(model_names)}

            # legend stubs (create handles/labels only if requested)
            if add_labels:
                plt.plot([],[],'-',color='c',label=f'{mouse_legend_label}\n   (exceeds 95% CI of all yoked means)',linewidth=kwa['line_width_models'])
                plt.plot([],[],'-',color='g',label=f'{mouse_legend_label}\n   (exceeds yoked mean)',linewidth=kwa['line_width_models'])
                plt.plot([],[],'-',color='r',label=f'{mouse_legend_label}\n   (under yoked mean)',linewidth=kwa['line_width_models'])
                plt.plot([],[],color='black',label='yoked mean (95% CI)',linewidth=kwa['line_width_all_models'])
            else:
                plt.plot([],[],'-',color='c',linewidth=kwa['line_width_models'])
                plt.plot([],[],'-',color='g',linewidth=kwa['line_width_models'])
                plt.plot([],[],'-',color='r',linewidth=kwa['line_width_models'])
                plt.plot([],[],color='black',linewidth=kwa['line_width_all_models'])

            sessions=list(mice_subset['exp_moment'].unique())
            for i,(exp_moment,session_subset_df) in enumerate(mice_subset.groupby('exp_moment')):

                rewarded_action_inds_mouse=_as_list(session_subset_df['rewarded_action_inds_mouse'].iloc[0])
                states_visited_mouse=_as_list(session_subset_df['states_visited_mouse'].iloc[0])

                n_rew_mouse=len(rewarded_action_inds_mouse)
                n_actions_mouse=_n_actions_from_states(states_visited_mouse)
                mouse_val=_metric(n_rew_mouse,n_actions_mouse)

                if run_sanity_checks:
                    sanity_check_session_subset_df(session_subset_df,exp_moment)

                    if "n_rewards_obtained_mouse" in session_subset_df.columns:
                        assert int(session_subset_df["n_rewards_obtained_mouse"].iloc[0])==n_rew_mouse
                    if "n_states_visited" in session_subset_df.columns:
                        n_states_scalar=int(session_subset_df["n_states_visited"].iloc[0])
                        assert n_states_scalar in {len(states_visited_mouse),n_actions_mouse,n_actions_mouse+1},(
                            f"n_states_visited={n_states_scalar}, len(states_visited_mouse)={len(states_visited_mouse)}, "
                            f"n_actions_mouse={n_actions_mouse}")
                    if "reward_rate_mouse" in session_subset_df.columns:
                        rr=float(session_subset_df["reward_rate_mouse"].iloc[0])
                        assert np.isfinite(rr)

                session_model_names=list(pd.unique(session_subset_df['full_sim_desc']))
                session_model_names=_order_models(session_model_names) if use_model_formatting else sorted(session_model_names)
                assert len(session_model_names)>0,f"Author: no models found for {exp_moment}"

                # --- keep ONLY this block for filtering + per-model CI + yoked CI ---

                n_runs_by_model=session_subset_df.groupby('full_sim_desc',dropna=False).size()
                kept_models=[]
                dropped_models=[]
                for m in session_model_names:
                    n_runs=int(n_runs_by_model.get(m,0))
                    if n_runs>=2:
                        kept_models.append(m)
                    else:
                        dropped_models.append((m,n_runs))

                if dropped_models:
                    _print_skip(
                        f"animal_ID={animal_ID} exp_moment={exp_moment} dropping models with <2 runs: {dropped_models}"
                    )

                if len(kept_models)==0:
                    _print_skip(
                        f"animal_ID={animal_ID} exp_moment={exp_moment} skipping session: no models remain after filtering (<2 runs each); "
                        f"original_models={len(session_model_names)}"
                    )
                    n_sessions_skipped+=1
                    continue

                session_model_names=kept_models
                n_models_this_session=len(session_model_names)

                d=1.0/(n_models_this_session+2)
                session_x_positions=i+d*np.arange(1,n_models_this_session+1)

                all_agents_reward=[]
                only_model_ci=None
                only_model_mean=None

                for model_idx,model_name in enumerate(session_model_names):
                    model_subset=session_subset_df[session_subset_df['full_sim_desc']==model_name]

                    per_run_vals=[]
                    for _,row in model_subset.iterrows():
                        rew_inds_agent=_as_list(row['rewarded_action_inds_agent'])
                        st_agent=_as_list(row['states_visited_agent'])

                        n_rew=len(rew_inds_agent)
                        n_actions_agent=_n_actions_from_states(st_agent)

                        assert n_actions_agent==n_actions_mouse,(
                            "Author: n actions agent != mouse; likely a deserialization/yoking issue "
                            f"(mouse={n_actions_mouse}, agent={n_actions_agent})")

                        per_run_vals.append(_metric(n_rew,n_actions_agent))

                    per_run_arr=np.asarray(per_run_vals,dtype=float)
                    if per_run_arr.size<2 or (not np.all(np.isfinite(per_run_arr))):
                        _print_skip(
                            f"animal_ID={animal_ID} exp_moment={exp_moment} model={model_name}: "
                            f"skipping model due to invalid CI inputs (n={per_run_arr.size}, finite={np.all(np.isfinite(per_run_arr))})"
                        )
                        continue

                    sem_val=ss.sem(per_run_arr)
                    if not np.isfinite(sem_val):
                        _print_skip(
                            f"animal_ID={animal_ID} exp_moment={exp_moment} model={model_name}: "
                            f"skipping model due to non-finite SEM (sem={sem_val})"
                        )
                        continue

                    if run_sanity_checks:
                        sanity_check_per_model_confidence_interval_inputs(per_run_arr,sem_val,model_name,exp_moment)

                    model_mean=float(np.mean(per_run_arr))
                    all_agents_reward.append(model_mean)

                    ci=ss.t.interval(0.95,df=per_run_arr.size-1,loc=model_mean,scale=sem_val)
                    if (not np.isfinite(ci[0])) or (not np.isfinite(ci[1])):
                        _print_skip(
                            f"animal_ID={animal_ID} exp_moment={exp_moment} model={model_name}: "
                            f"skipping model due to non-finite CI ({ci})"
                        )
                        continue

                    if n_models_this_session==1:
                        only_model_ci=ci
                        only_model_mean=model_mean

                    x_offset=session_x_positions[model_idx]
                    if add_labels and i==0:
                        label_str=MODEL_FORMATTING_DF.loc[model_name,label_col] if use_model_formatting else model_name_2_legend_str(model_name)
                    else:
                        label_str=None

                    plt.plot([x_offset]*2,ci,
                        color=(MODEL_FORMATTING_DF.loc[model_name,"color"] if use_model_formatting else model_color_map.get(model_name,"0.5")),
                        label=label_str,
                        linewidth=kwa['line_width_models'])

                # --- yoked CI: across-model if >=2 models, else fallback to the single model's repeat-based CI ---
                if n_models_this_session==1:
                    if (only_model_ci is None) or (only_model_mean is None) or (not np.all(np.isfinite(only_model_ci))):
                        _print_skip(f"animal_ID={animal_ID} exp_moment={exp_moment} skipping session: single-model fallback CI missing/invalid")
                        n_sessions_skipped+=1
                        continue
                    _print_skip(f"animal_ID={animal_ID} exp_moment={exp_moment} note: only 1 model; using that model's repeat-based CI as yoked CI")
                    agents_reward_95ci=only_model_ci
                    all_agents_reward_arr=np.asarray([only_model_mean],dtype=float)
                else:
                    all_agents_reward_arr=np.asarray(all_agents_reward,dtype=float)
                    if all_agents_reward_arr.size<2:
                        _print_skip(
                            f"animal_ID={animal_ID} exp_moment={exp_moment} skipping session post-filter: "
                            f"need >=2 model means for yoked CI, got {all_agents_reward_arr.size}"
                        )
                        n_sessions_skipped+=1
                        continue

                    if not np.all(np.isfinite(all_agents_reward_arr)):
                        _print_skip(f"animal_ID={animal_ID} exp_moment={exp_moment} skipping session: non-finite model means {all_agents_reward_arr}")
                        n_sessions_skipped+=1
                        continue

                    sem_all=ss.sem(all_agents_reward_arr)
                    if not np.isfinite(sem_all):
                        _print_skip(f"animal_ID={animal_ID} exp_moment={exp_moment} skipping session: non-finite SEM across models (sem_all={sem_all})")
                        n_sessions_skipped+=1
                        continue

                    if run_sanity_checks:
                        sanity_check_all_agents_confidence_interval_inputs(all_agents_reward_arr,sem_all,all_agents_reward_arr.size,exp_moment)

                    agents_reward_95ci=ss.t.interval(0.95,df=all_agents_reward_arr.size-1,
                        loc=float(np.mean(all_agents_reward_arr)),scale=sem_all)

                    if (not np.isfinite(agents_reward_95ci[0])) or (not np.isfinite(agents_reward_95ci[1])):
                        _print_skip(f"animal_ID={animal_ID} exp_moment={exp_moment} skipping session: non-finite yoked CI ({agents_reward_95ci})")
                        n_sessions_skipped+=1
                        continue

                plt.plot([i]*2,agents_reward_95ci,color='black',linewidth=kwa['line_width_all_models'])

                mouse_line_color=('c' if mouse_val>agents_reward_95ci[-1]
                    else ('g' if mouse_val>np.nanmean(all_agents_reward_arr) else 'r'))
                if mouse_line_color=='c':
                    cyan_count+=1

                x_end=float(np.max(session_x_positions))
                plt.plot([i,x_end],[mouse_val,mouse_val],'--',
                    color=mouse_line_color,linewidth=kwa['line_width_mouse'],alpha=kwa['mouse_point_alpha'])
                n_sessions_used+=1


                # # --- NEW: drop models with <2 runs (no t CI), with informative print ---
                # n_runs_by_model=session_subset_df.groupby('full_sim_desc',dropna=False).size()
                # kept_models=[]
                # dropped_models=[]
                # for m in session_model_names:
                #     n_runs=int(n_runs_by_model.get(m,0))
                #     if n_runs>=2:
                #         kept_models.append(m)
                #     else:
                #         dropped_models.append((m,n_runs))

                # if dropped_models:
                #     _print_skip(
                #         f"animal_ID={animal_ID} exp_moment={exp_moment} dropping models with <2 runs: "
                #         f"{[(m,n) for m,n in dropped_models]}"
                #     )

                # # --- REPLACE THE "kept_models < 2 => skip session" BLOCK WITH THIS ---

                # # keep models with >=2 runs (repeat-based CI possible)
                # n_runs_by_model=session_subset_df.groupby('full_sim_desc',dropna=False).size()
                # kept_models=[]
                # dropped_models=[]
                # for m in session_model_names:
                #     n_runs=int(n_runs_by_model.get(m,0))
                #     if n_runs>=2:
                #         kept_models.append(m)
                #     else:
                #         dropped_models.append((m,n_runs))

                # if dropped_models:
                #     _print_skip(
                #         f"animal_ID={animal_ID} exp_moment={exp_moment} dropping models with <2 runs: "
                #         f"{[(m,n) for m,n in dropped_models]}"
                #     )

                # # NOTE: do NOT require >=2 models; allow 1-model sessions (fallback to that model's repeat-CI)
                # if len(kept_models)==0:
                #     _print_skip(
                #         f"animal_ID={animal_ID} exp_moment={exp_moment} skipping session: no models remain after filtering (<2 runs each); "
                #         f"original_models={len(session_model_names)}"
                #     )
                #     n_sessions_skipped+=1
                #     continue

                # # --- NEW: if fewer than 2 models remain, skip the whole session (no yoked CI) ---
                # # if len(kept_models)<2:
                # #     _print_skip(
                # #         f"animal_ID={animal_ID} exp_moment={exp_moment} skipping session: need >=2 models after filtering, "
                # #         f"kept={len(kept_models)}; original_models={len(session_model_names)}"
                # #     )
                # #     n_sessions_skipped+=1
                # #     continue



                # session_model_names=kept_models
                # n_models_this_session=len(session_model_names)

                # d=1.0/(n_models_this_session+2)
                # session_x_positions=i+d*np.arange(1,n_models_this_session+1)

                # all_agents_reward=[]
                # only_model_ci=None
                # only_model_mean=None

                # for model_idx,model_name in enumerate(session_model_names):
                #     model_subset=session_subset_df[session_subset_df['full_sim_desc']==model_name]

                #     per_run_vals=[]
                #     for _,row in model_subset.iterrows():
                #         rew_inds_agent=_as_list(row['rewarded_action_inds_agent'])
                #         st_agent=_as_list(row['states_visited_agent'])

                #         n_rew=len(rew_inds_agent)
                #         n_actions_agent=_n_actions_from_states(st_agent)

                #         assert n_actions_agent==n_actions_mouse,(
                #             "Author: n actions agent != mouse; likely a deserialization/yoking issue "
                #             f"(mouse={n_actions_mouse}, agent={n_actions_agent})")

                #         per_run_vals.append(_metric(n_rew,n_actions_agent))

                #     per_run_arr=np.asarray(per_run_vals,dtype=float)

                #     if per_run_arr.size<2 or (not np.all(np.isfinite(per_run_arr))):
                #         _print_skip(
                #             f"animal_ID={animal_ID} exp_moment={exp_moment} model={model_name}: "
                #             f"skipping model due to invalid CI inputs (n={per_run_arr.size}, finite={np.all(np.isfinite(per_run_arr))})"
                #         )
                #         continue

                #     sem_val=ss.sem(per_run_arr)
                #     if (not np.isfinite(sem_val)):
                #         _print_skip(
                #             f"animal_ID={animal_ID} exp_moment={exp_moment} model={model_name}: "
                #             f"skipping model due to non-finite SEM (sem={sem_val})"
                #         )
                #         continue

                #     if run_sanity_checks:
                #         sanity_check_per_model_confidence_interval_inputs(per_run_arr,sem_val,model_name,exp_moment)

                #     model_mean=float(np.mean(per_run_arr))
                #     all_agents_reward.append(model_mean)

                #     ci=ss.t.interval(0.95,df=per_run_arr.size-1,loc=model_mean,scale=sem_val)
                #     if (not np.isfinite(ci[0])) or (not np.isfinite(ci[1])):
                #         _print_skip(
                #             f"animal_ID={animal_ID} exp_moment={exp_moment} model={model_name}: "
                #             f"skipping model due to non-finite CI ({ci})"
                #         )
                #         continue

                #     if n_models_this_session==1:
                #         only_model_ci=ci
                #         only_model_mean=model_mean

                #     x_offset=session_x_positions[model_idx]
                #     if add_labels and i==0:
                #         label_str=MODEL_FORMATTING_DF.loc[model_name,label_col] if use_model_formatting else model_name_2_legend_str(model_name)
                #     else:
                #         label_str=None

                #     plt.plot([x_offset]*2,ci,
                #         color=(MODEL_FORMATTING_DF.loc[model_name,"color"] if use_model_formatting else model_color_map.get(model_name,"0.5")),
                #         label=label_str,
                #         linewidth=kwa['line_width_models'])

                # session_model_names=kept_models
                # n_models_this_session=len(session_model_names)

                # d=1.0/(n_models_this_session+2)
                # session_x_positions=i+d*np.arange(1,n_models_this_session+1)

                # all_agents_reward=[]
                # for model_idx,model_name in enumerate(session_model_names):
                #     model_subset=session_subset_df[session_subset_df['full_sim_desc']==model_name]

                #     per_run_vals=[]
                #     for _,row in model_subset.iterrows():
                #         rew_inds_agent=_as_list(row['rewarded_action_inds_agent'])
                #         st_agent=_as_list(row['states_visited_agent'])

                #         n_rew=len(rew_inds_agent)
                #         n_actions_agent=_n_actions_from_states(st_agent)

                #         assert n_actions_agent==n_actions_mouse,(
                #             "Author: n actions agent != mouse; likely a deserialization/yoking issue "
                #             f"(mouse={n_actions_mouse}, agent={n_actions_agent})")

                #         per_run_vals.append(_metric(n_rew,n_actions_agent))

                #     per_run_arr=np.asarray(per_run_vals,dtype=float)

                #     # extra guard: even after row-count filtering, keep semantics safe
                #     if per_run_arr.size<2 or (not np.all(np.isfinite(per_run_arr))):
                #         _print_skip(
                #             f"animal_ID={animal_ID} exp_moment={exp_moment} model={model_name}: "
                #             f"skipping model due to invalid CI inputs (n={per_run_arr.size}, finite={np.all(np.isfinite(per_run_arr))})"
                #         )
                #         continue

                #     sem_val=ss.sem(per_run_arr)
                #     if (not np.isfinite(sem_val)):
                #         _print_skip(
                #             f"animal_ID={animal_ID} exp_moment={exp_moment} model={model_name}: "
                #             f"skipping model due to non-finite SEM (sem={sem_val})"
                #         )
                #         continue

                #     if run_sanity_checks:
                #         sanity_check_per_model_confidence_interval_inputs(per_run_arr,sem_val,model_name,exp_moment)

                #     model_mean=float(np.mean(per_run_arr))
                #     all_agents_reward.append(model_mean)

                #     ci=ss.t.interval(0.95,df=per_run_arr.size-1,loc=model_mean,scale=sem_val)
                #     if (not np.isfinite(ci[0])) or (not np.isfinite(ci[1])):
                #         _print_skip(
                #             f"animal_ID={animal_ID} exp_moment={exp_moment} model={model_name}: "
                #             f"skipping model due to non-finite CI ({ci})"
                #         )
                #         continue

                #     x_offset=session_x_positions[model_idx]
                #     if add_labels and i==0:
                #         label_str=MODEL_FORMATTING_DF.loc[model_name,label_col] if use_model_formatting else model_name_2_legend_str(model_name)
                #     else:
                #         label_str=None

                #     plt.plot([x_offset]*2,ci,
                #         color=(MODEL_FORMATTING_DF.loc[model_name,"color"] if use_model_formatting else model_color_map.get(model_name,"0.5")),
                #         label=label_str,
                #         linewidth=kwa['line_width_models'])

                # # if we lost too many models to semantic guards, bail out safely
                # all_agents_reward_arr=np.asarray(all_agents_reward,dtype=float)
                # if all_agents_reward_arr.size<2:
                #     _print_skip(
                #         f"animal_ID={animal_ID} exp_moment={exp_moment} skipping session post-filter: "
                #         f"need >=2 model means for yoked CI, got {all_agents_reward_arr.size}"
                #     )
                #     n_sessions_skipped+=1
                #     continue

                # if (not np.all(np.isfinite(all_agents_reward_arr))):
                #     _print_skip(
                #         f"animal_ID={animal_ID} exp_moment={exp_moment} skipping session: non-finite model means {all_agents_reward_arr}"
                #     )
                #     n_sessions_skipped+=1
                #     continue

                # sem_all=ss.sem(all_agents_reward_arr)
                # if (not np.isfinite(sem_all)):
                #     _print_skip(
                #         f"animal_ID={animal_ID} exp_moment={exp_moment} skipping session: non-finite SEM across models (sem_all={sem_all})"
                #     )
                #     n_sessions_skipped+=1
                #     continue

                # if run_sanity_checks:
                #     sanity_check_all_agents_confidence_interval_inputs(all_agents_reward_arr,sem_all,all_agents_reward_arr.size,exp_moment)

                # agents_reward_95ci=ss.t.interval(0.95,df=all_agents_reward_arr.size-1,loc=float(np.mean(all_agents_reward_arr)),scale=sem_all)
                # if (not np.isfinite(agents_reward_95ci[0])) or (not np.isfinite(agents_reward_95ci[1])):
                #     _print_skip(
                #         f"animal_ID={animal_ID} exp_moment={exp_moment} skipping session: non-finite yoked CI ({agents_reward_95ci})"
                #     )
                #     n_sessions_skipped+=1
                #     continue

                # plt.plot([i]*2,agents_reward_95ci,color='black',linewidth=kwa['line_width_all_models'])

                # mouse_line_color=('c' if mouse_val>agents_reward_95ci[-1] else ('g' if mouse_val>np.nanmean(all_agents_reward_arr) else 'r'))
                # if mouse_line_color=='c':
                #     cyan_count+=1

                # x_end=float(np.max(session_x_positions))
                # plt.plot([i,x_end],[mouse_val,mouse_val],'--',color=mouse_line_color,linewidth=kwa['line_width_mouse'],alpha=kwa['mouse_point_alpha'])

                # n_sessions_used+=1

            plt.tick_params(axis='both',labelsize=kwa['tick_label_font_size'])
            plt.xlabel('Experiment session',fontsize=kwa['axis_label_font_size'])
            plt.ylabel(y_label,fontsize=kwa['axis_label_font_size'])

            title_prefix=kwa.get('title_prefix','')
            denom=n_sessions_used
            if denom==0:
                title=f"{title_prefix}mouse {animal_ID}: no sessions plotted (skipped={n_sessions_skipped}/{len(sessions)})"
            else:
                title=f"{title_prefix}mouse {animal_ID}: sessions with mouse > 95% CI of all sim. yoked means = {cyan_count} / {denom} (skipped={n_sessions_skipped})"
            plt.title(title,fontsize=kwa['title_font_size'])

            if draw_legend:
                plt.gcf().subplots_adjust(bottom=0.32)
                plt.legend(loc='upper center',bbox_to_anchor=(0.5,-0.22),fontsize=kwa['legend_fontsize'],ncols=kwa['n_legend_cols'],frameon=True)

            if do_save:
                plt.savefig(f"./figures/c{get_now_str(hms=True)}_{animal_ID}_figS3_reward_rates_by_session-v1.2allModels.png",bbox_inches='tight',dpi=400)
                plt.savefig(f"./figures/c{get_now_str(hms=True)}_{animal_ID}_figS3_reward_rates_by_session-v1.2allModels.svg",bbox_inches='tight',dpi=400)
                plt.savefig(f"./figures/c{get_now_str(hms=True)}_{animal_ID}_figS3_reward_rates_by_session-v1.2allModels.pdf",bbox_inches='tight',dpi=400)

            if do_show:
                plt.show()
    return (plot_mice_vs_sims_per_session_GPT_dev_v2,)


@app.cell
def _(mo):
    mo.md(r"""
    separated by animal
    """)
    return


@app.cell
def _(
    AXIS_LABEL_FONT_SIZE,
    DF_FOR_STATS,
    LEGEND_FONTSIZE,
    LINE_WIDTH_ALL_MODELS,
    LINE_WIDTH_MODELS,
    LINE_WIDTH_MOUSE,
    MODEL_FORMATTING_DF,
    MOUSE_POINT_ALPHA,
    MOUSE_POINT_MARKER_SIZE,
    TICK_LABEL_FONT_SIZE,
    TITLE_FONT_SIZE,
    get_now_str,
    plot_mice_vs_sims_per_session_GPT_dev_v2,
    plt,
):
    def _():
        _RERUN_N_PLOT_MICE_VS_SIMS_PER_SESSION=True
        _SELECTED_ANIMAL_IDS_FIG_4=DF_FOR_STATS.animal_ID.unique().tolist()

        _N_LEGEND_COLS=4
        _RUN_SANITY_CHECKS=True

        if _RERUN_N_PLOT_MICE_VS_SIMS_PER_SESSION:

            _s=DF_FOR_STATS.groupby(['action_type'])['obs_type'].nunique(dropna=False)
            assert len(_s)>0,"no groups found for action_type"
            assert _s.eq(1).all(),f"obs_type not unique within some action_type:\n{_s[_s.ne(1)].sort_values(ascending=False)}"
            assert DF_FOR_STATS.action_type.isna().sum()==0,"Some action_type values are NaN"
            assert DF_FOR_STATS.obs_type.isna().sum()==0,"Some obs_type values are NaN"
            assert DF_FOR_STATS.policy_class.isna().sum()==0,"Some policy_class values are NaN"

            rep_groups=list(DF_FOR_STATS.groupby(['action_type']))
            n_reps=len(rep_groups)
            assert n_reps>0,"Author: no action_type groups found"

            for _animal_ID in _SELECTED_ANIMAL_IDS_FIG_4:

                fig,axs=plt.subplots(n_reps,1,figsize=(25,6*n_reps),sharex=True,sharey=True)
                if n_reps==1:
                    axs=[axs]

                for rep_idx,(rep_str,rep_df) in enumerate(rep_groups):
                    if isinstance(rep_str,tuple): rep_str=rep_str[0]
                    ax=axs[rep_idx]

                    _kwargs={
                        'axis_label_font_size':AXIS_LABEL_FONT_SIZE,
                        'title_font_size':TITLE_FONT_SIZE,
                        'mouse_point_marker_size':MOUSE_POINT_MARKER_SIZE,
                        'mouse_point_alpha':MOUSE_POINT_ALPHA,
                        'line_width_mouse':LINE_WIDTH_MOUSE,
                        'line_width_models':LINE_WIDTH_MODELS,
                        'line_width_all_models':LINE_WIDTH_ALL_MODELS,
                        'legend_fontsize':LEGEND_FONTSIZE,
                        'tick_label_font_size':TICK_LABEL_FONT_SIZE,
                        'n_legend_cols':_N_LEGEND_COLS,
                        'title_prefix':f'{rep_str}: ',
                        'figsize':(25,6),  # unused when ax provided, but kept for symmetry

                        # subplot controls
                        'ax':ax,
                        'add_labels':(rep_idx==0),  # only the first subplot carries labels for shared legend
                        'draw_legend':False,        # draw one legend for the whole figure
                        'do_save':False,            # save once per animal figure (below)
                        'do_show':False,
                    }

                    plot_mice_vs_sims_per_session_GPT_dev_v2(
                        [_animal_ID],rep_df,
                        MODEL_FORMATTING_DF=MODEL_FORMATTING_DF,
                        run_sanity_checks=_RUN_SANITY_CHECKS,
                        **_kwargs
                    )

                # shared legend from first axis
                handles,labels=axs[0].get_legend_handles_labels()
                fig.subplots_adjust(bottom=0.32)
                fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(0.5,-0.05),
                           fontsize=LEGEND_FONTSIZE,ncols=_N_LEGEND_COLS,frameon=True)

                fig.suptitle(f"mouse {_animal_ID}",y=1.02,fontsize=TITLE_FONT_SIZE)

                fig.savefig(f"./figures/c{get_now_str(hms=True)}_{_animal_ID}_figS3_reward_rates_by_session-repsSubplots.png",bbox_inches='tight',dpi=400)
                fig.savefig(f"./figures/c{get_now_str(hms=True)}_{_animal_ID}_figS3_reward_rates_by_session-repsSubplots.svg",bbox_inches='tight',dpi=400)
                fig.savefig(f"./figures/c{get_now_str(hms=True)}_{_animal_ID}_figS3_reward_rates_by_session-repsSubplots.pdf",bbox_inches='tight',dpi=400)
        return plt.show()


    _()
    return


@app.cell
def _(mo):
    mo.md(r"""
    separate by representation
    """)
    return


@app.cell
def _(
    DF_FOR_STATS,
    LEGEND_FONTSIZE,
    LINE_WIDTH_ALL_MODELS,
    LINE_WIDTH_MODELS,
    LINE_WIDTH_MOUSE,
    MODEL_FORMATTING_DF,
    MOUSE_POINT_ALPHA,
    MOUSE_POINT_MARKER_SIZE,
    RERUN_N_PLOT_MICE_VS_SIMS_PER_SESSION,
    TICK_LABEL_FONT_SIZE,
    TITLE_FONT_SIZE,
    get_now_str,
    np,
    plot_mice_vs_sims_per_session_GPT_dev_v2,
    plt,
):
    def _():
        # RERUN_N_PLOT_MICE_VS_SIMS_PER_SESSION=True
        _SELECTED_ANIMAL_IDS_FIG_4=DF_FOR_STATS.animal_ID.unique().tolist()

        _N_LEGEND_COLS=4
        _RUN_SANITY_CHECKS=True
        _SUBPLOT_TITLE_FONTSIZE = 10
        _AXIS_LABEL_FONT_SIZE = 8

        if RERUN_N_PLOT_MICE_VS_SIMS_PER_SESSION:

            _s=DF_FOR_STATS.groupby(['action_type'])['obs_type'].nunique(dropna=False)
            assert len(_s)>0,"no groups found for action_type"
            assert _s.eq(1).all(),f"obs_type not unique within some action_type:\n{_s[_s.ne(1)].sort_values(ascending=False)}"
            assert DF_FOR_STATS.action_type.isna().sum()==0,"Some action_type values are NaN"
            assert DF_FOR_STATS.obs_type.isna().sum()==0,"Some obs_type values are NaN"
            assert DF_FOR_STATS.policy_class.isna().sum()==0,"Some policy_class values are NaN"

            n_animals=len(_SELECTED_ANIMAL_IDS_FIG_4)
            assert n_animals>0,"Author: no animals selected"

            ncols=min(3,n_animals)
            nrows=int(np.ceil(n_animals/ncols))

            for rep_str,rep_df in DF_FOR_STATS.groupby(['action_type']):
                if isinstance(rep_str,tuple): rep_str=rep_str[0]

                _fig, _axs = plt.subplots(nrows,ncols,figsize=(25,6*nrows),squeeze=False)
                _axs=_axs.ravel()

                for animal_idx, animal_ID in enumerate(_SELECTED_ANIMAL_IDS_FIG_4):
                    ax=_axs[animal_idx]

                    _kwargs={
                        'axis_label_font_size': _AXIS_LABEL_FONT_SIZE,
                        'title_font_size':_SUBPLOT_TITLE_FONTSIZE,
                        'mouse_point_marker_size':MOUSE_POINT_MARKER_SIZE,
                        'mouse_point_alpha':MOUSE_POINT_ALPHA,
                        'line_width_mouse':LINE_WIDTH_MOUSE,
                        'line_width_models':LINE_WIDTH_MODELS,
                        'line_width_all_models':LINE_WIDTH_ALL_MODELS,
                        'legend_fontsize':LEGEND_FONTSIZE,
                        'tick_label_font_size':TICK_LABEL_FONT_SIZE,
                        'n_legend_cols':_N_LEGEND_COLS,

                        # subplot routing + suppress per-call legend/save/show
                        'ax':ax,
                        'add_labels':(animal_idx==0),  # collect labels once for shared legend
                        'draw_legend':False,
                        'do_save':False,
                        'do_show':False,

                        # optional: keep per-axis titles clean (rep title goes on the figure)
                        'title_prefix':''}

                    plot_mice_vs_sims_per_session_GPT_dev_v2([animal_ID],rep_df, MODEL_FORMATTING_DF=MODEL_FORMATTING_DF, run_sanity_checks=_RUN_SANITY_CHECKS, **_kwargs)

                # hide any unused axes
                for j in range(n_animals,len(_axs)):
                    _axs[j].set_visible(False)

                _fig.suptitle(f"{rep_str}",y=1.02,fontsize=TITLE_FONT_SIZE)

                # shared legend (from first subplot which had add_labels=True)
                handles,labels=_axs[0].get_legend_handles_labels()
                _fig.subplots_adjust(bottom=0.32)
                _fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(0.5,-0.05),
                           fontsize=LEGEND_FONTSIZE,ncols=_N_LEGEND_COLS,frameon=True)

                rep_str_safe=str(rep_str).replace(" ","_").replace("/","_").replace(";","_")
                _fig.savefig(f"./figures/c{get_now_str(hms=True)}_figS3_reward_rates_by_session_{rep_str_safe}_allAnimals.png",bbox_inches='tight',dpi=400)
                _fig.savefig(f"./figures/c{get_now_str(hms=True)}_figS3_reward_rates_by_session_{rep_str_safe}_allAnimals.svg",bbox_inches='tight',dpi=400)
                _fig.savefig(f"./figures/c{get_now_str(hms=True)}_figS3_reward_rates_by_session_{rep_str_safe}_allAnimals.pdf",bbox_inches='tight',dpi=400)
        return plt.show()


    _()
    return


@app.cell
def _(
    AXIS_LABEL_FONT_SIZE,
    DF_FOR_STATS,
    LEGEND_FONTSIZE,
    LINE_WIDTH_ALL_MODELS,
    LINE_WIDTH_MODELS,
    LINE_WIDTH_MOUSE,
    MODEL_FORMATTING_DF,
    MOUSE_POINT_ALPHA,
    MOUSE_POINT_MARKER_SIZE,
    TICK_LABEL_FONT_SIZE,
    TITLE_FONT_SIZE,
    plot_mice_vs_sims_per_session_GPT_dev,
):
    def _():
        RERUN_N_PLOT_MICE_VS_SIMS_PER_SESSION = True

        ## only used if RERUN_N_PLOT_MICE_VS_SIMS_PER_SESSION = True
        # _SELECTED_ANIMAL_IDS_FIG_4 = ['a031', 'a033', 'a030', 'a029']  
        # SELECTED_ANIMAL_IDS_FIG_4 =  per_exp_result_df.animal_ID.unique()
        # SELECTED_ANIMAL_IDS_FIG_4 = ['a030', 'a029', 'a003', 'a188', 'a033', 'a031']  
        # SELECTED_ANIMAL_IDS_FIG_4 = ['a030']  
        # SELECTED_ANIMAL_IDS_FIG_4 = ['a031', 'a033']
        _SELECTED_ANIMAL_IDS_FIG_4 = DF_FOR_STATS.animal_ID.unique().tolist()

        _N_LEGEND_COLS = 4
        _RUN_SANITY_CHECKS = True

        if RERUN_N_PLOT_MICE_VS_SIMS_PER_SESSION:

            ## Author: sanity check data
            # _s=DF_FOR_STATS.groupby(['policy_class','action_type'])['obs_type'].nunique(dropna=False)
            _s=DF_FOR_STATS.groupby(['action_type'])['obs_type'].nunique(dropna=False)
            assert len(_s)>0,"no groups found for (policy_class,action_type)"
            assert _s.eq(1).all(),f"obs_type not unique within some (policy_class,action_type):\n{_s[_s.ne(1)].sort_values(ascending=False)}"
            assert DF_FOR_STATS.action_type.isna().sum() == 0, "Some action_type values are NaN"
            assert DF_FOR_STATS.obs_type.isna().sum() == 0, "Some action_type values are NaN"
            assert DF_FOR_STATS.policy_class.isna().sum() == 0, "Some action_type values are NaN"

            # for rep_str, rep_df in DF_FOR_STATS.groupby(['policy_class','action_type']):
            for rep_str, rep_df in DF_FOR_STATS.groupby(['action_type']):

                if isinstance(rep_str, tuple): 
                    rep_str = rep_str[0]

                _kwargs = {'axis_label_font_size': AXIS_LABEL_FONT_SIZE,
                           'title_font_size': TITLE_FONT_SIZE,
                           'mouse_point_marker_size': MOUSE_POINT_MARKER_SIZE,
                           'mouse_point_alpha': MOUSE_POINT_ALPHA,
                           'line_width_mouse': LINE_WIDTH_MOUSE,
                           'line_width_models': LINE_WIDTH_MODELS,
                           'line_width_all_models': LINE_WIDTH_ALL_MODELS,
                           'legend_fontsize': LEGEND_FONTSIZE,
                           'tick_label_font_size': TICK_LABEL_FONT_SIZE,
                           'n_legend_cols': _N_LEGEND_COLS,
                           'title_prefix': f'{rep_str}: ',
                           'figsize': (25, 6)}

                # plot_mice_vs_sims_per_session(_SELECTED_ANIMAL_IDS_FIG_4, per_exp_result_df, run_sanity_checks=_RUN_SANITY_CHECKS, **_kwargs)
                # plot_mice_vs_sims_per_session(_SELECTED_ANIMAL_IDS_FIG_4, DF_FOR_STATS, run_sanity_checks=_RUN_SANITY_CHECKS, **_kwargs)
        return plot_mice_vs_sims_per_session_GPT_dev(_SELECTED_ANIMAL_IDS_FIG_4, rep_df, MODEL_FORMATTING_DF=MODEL_FORMATTING_DF, run_sanity_checks=_RUN_SANITY_CHECKS, **_kwargs)


        # _s=DF_FOR_STATS.groupby(['policy_class','action_type'])['obs_type'].nunique(dropna=False)
        # assert len(_s)>0,"no groups found for (policy_class,action_type)"
        # assert _s.eq(1).all(),f"obs_type not unique within some (policy_class,action_type):\n{_s[_s.ne(1)].sort_values(ascending=False)}"
        # assert DF_FOR_STATS.action_type.isna().sum() == 0, "Some action_type values are NaN"
        # assert DF_FOR_STATS.obs_type.isna().sum() == 0, "Some action_type values are NaN"
        # assert DF_FOR_STATS.policy_class.isna().sum() == 0, "Some action_type values are NaN"
        # DF_FOR_STATS.groupby(['policy_class', 'action_type'])['obs_type'].nunique() #.reset_index(name='count').sort_values('count', ascending=False)'])


    _()
    return


@app.cell
def _(mo):
    mo.md(r"""
    260109 Author TODO: large discrepancy between neurIPS plot and new ones. Find out why!
    """)
    return


@app.cell
def _(
    AXIS_LABEL_FONT_SIZE,
    DF_FOR_STATS,
    LEGEND_FONTSIZE,
    LINE_WIDTH_ALL_MODELS,
    LINE_WIDTH_MODELS,
    LINE_WIDTH_MOUSE,
    MODEL_FORMATTING_DF,
    MOUSE_POINT_ALPHA,
    MOUSE_POINT_MARKER_SIZE,
    TICK_LABEL_FONT_SIZE,
    TITLE_FONT_SIZE,
    plot_mice_vs_sims_per_session_GPT_dev,
):
    def _():
        RERUN_N_PLOT_MICE_VS_SIMS_PER_SESSION = True

        ## only used if RERUN_N_PLOT_MICE_VS_SIMS_PER_SESSION = True
        # _SELECTED_ANIMAL_IDS_FIG_4 = ['a031', 'a033', 'a030', 'a029']  
        # SELECTED_ANIMAL_IDS_FIG_4 =  per_exp_result_df.animal_ID.unique()
        # SELECTED_ANIMAL_IDS_FIG_4 = ['a030', 'a029', 'a003', 'a188', 'a033', 'a031']  
        # SELECTED_ANIMAL_IDS_FIG_4 = ['a030']  
        # SELECTED_ANIMAL_IDS_FIG_4 = ['a031', 'a033']
        _SELECTED_ANIMAL_IDS_FIG_4 = DF_FOR_STATS.animal_ID.unique().tolist()

        _N_LEGEND_COLS = 4
        _RUN_SANITY_CHECKS = True

        if RERUN_N_PLOT_MICE_VS_SIMS_PER_SESSION:

            _kwargs = {'axis_label_font_size': AXIS_LABEL_FONT_SIZE,
                       'title_font_size': TITLE_FONT_SIZE,
                       'mouse_point_marker_size': MOUSE_POINT_MARKER_SIZE,
                       'mouse_point_alpha': MOUSE_POINT_ALPHA,
                       'line_width_mouse': LINE_WIDTH_MOUSE,
                       'line_width_models': LINE_WIDTH_MODELS,
                       'line_width_all_models': LINE_WIDTH_ALL_MODELS,
                       'legend_fontsize': LEGEND_FONTSIZE,
                       'tick_label_font_size': TICK_LABEL_FONT_SIZE,
                       'n_legend_cols': _N_LEGEND_COLS,
                       'figsize': (25, 6)}

            # plot_mice_vs_sims_per_session(_SELECTED_ANIMAL_IDS_FIG_4, per_exp_result_df, run_sanity_checks=_RUN_SANITY_CHECKS, **_kwargs)
            # plot_mice_vs_sims_per_session(_SELECTED_ANIMAL_IDS_FIG_4, DF_FOR_STATS, run_sanity_checks=_RUN_SANITY_CHECKS, **_kwargs)
        return plot_mice_vs_sims_per_session_GPT_dev(_SELECTED_ANIMAL_IDS_FIG_4, DF_FOR_STATS, MODEL_FORMATTING_DF=MODEL_FORMATTING_DF, run_sanity_checks=_RUN_SANITY_CHECKS, **_kwargs)


    _()
    return


@app.cell
def _(
    AXIS_LABEL_FONT_SIZE,
    DF_FOR_STATS,
    LEGEND_FONTSIZE,
    LINE_WIDTH_ALL_MODELS,
    LINE_WIDTH_MODELS,
    LINE_WIDTH_MOUSE,
    MODEL_FORMATTING_DF,
    MOUSE_POINT_ALPHA,
    MOUSE_POINT_MARKER_SIZE,
    TICK_LABEL_FONT_SIZE,
    TITLE_FONT_SIZE,
    plot_mice_vs_sims_per_session_GPT_dev,
):
    def _():
        RERUN_N_PLOT_MICE_VS_SIMS_PER_SESSION = True
        _NORMALIZE_BY_N_ACTIONS_IN_SESSION = False

        ## only used if RERUN_N_PLOT_MICE_VS_SIMS_PER_SESSION = True
        # _SELECTED_ANIMAL_IDS_FIG_4 = ['a031', 'a033', 'a030', 'a029']  
        # SELECTED_ANIMAL_IDS_FIG_4 =  per_exp_result_df.animal_ID.unique()
        # SELECTED_ANIMAL_IDS_FIG_4 = ['a030', 'a029', 'a003', 'a188', 'a033', 'a031']  
        # SELECTED_ANIMAL_IDS_FIG_4 = ['a030']  
        # SELECTED_ANIMAL_IDS_FIG_4 = ['a031', 'a033']
        _SELECTED_ANIMAL_IDS_FIG_4 = DF_FOR_STATS.animal_ID.unique().tolist()

        _N_LEGEND_COLS = 4
        _RUN_SANITY_CHECKS = True

        if RERUN_N_PLOT_MICE_VS_SIMS_PER_SESSION:

            _kwargs = {'axis_label_font_size': AXIS_LABEL_FONT_SIZE,
                       'title_font_size': TITLE_FONT_SIZE,
                       'mouse_point_marker_size': MOUSE_POINT_MARKER_SIZE,
                       'mouse_point_alpha': MOUSE_POINT_ALPHA,
                       'line_width_mouse': LINE_WIDTH_MOUSE,
                       'line_width_models': LINE_WIDTH_MODELS,
                       'line_width_all_models': LINE_WIDTH_ALL_MODELS,
                       'legend_fontsize': LEGEND_FONTSIZE,
                       'tick_label_font_size': TICK_LABEL_FONT_SIZE,
                       'n_legend_cols': _N_LEGEND_COLS,
                       'figsize': (25, 6)}

            # plot_mice_vs_sims_per_session(_SELECTED_ANIMAL_IDS_FIG_4, per_exp_result_df, run_sanity_checks=_RUN_SANITY_CHECKS, **_kwargs)
            # plot_mice_vs_sims_per_session(_SELECTED_ANIMAL_IDS_FIG_4, DF_FOR_STATS, run_sanity_checks=_RUN_SANITY_CHECKS, **_kwargs)
        return plot_mice_vs_sims_per_session_GPT_dev(_SELECTED_ANIMAL_IDS_FIG_4, DF_FOR_STATS, 
                                                     MODEL_FORMATTING_DF=MODEL_FORMATTING_DF,
                                                     normalize_by_n_actions_in_session = _NORMALIZE_BY_N_ACTIONS_IN_SESSION,
                                                     run_sanity_checks=_RUN_SANITY_CHECKS, **_kwargs)


    _()
    return


@app.cell
def _(mo):
    mo.md(r"""
 
    """)
    return


@app.cell
def _(DF_FOR_STATS):
    assert DF_FOR_STATS.action_type.isna().sum() == 0, "Some action_type values are NaN"
    DF_FOR_STATS.action_type.value_counts(dropna=False)
    return


@app.cell
def _(DF_FOR_STATS):
    assert DF_FOR_STATS.obs_type.isna().sum() == 0, "Some action_type values are NaN"
    DF_FOR_STATS.obs_type.value_counts(dropna=False)
    return


@app.cell
def _(DF_FOR_STATS):
    assert DF_FOR_STATS.policy_class.isna().sum() == 0, "Some action_type values are NaN"
    DF_FOR_STATS.policy_class.value_counts(dropna=False)
    return


@app.cell
def _(DF_FOR_STATS):
    _s=DF_FOR_STATS.groupby(['policy_class','action_type'])['obs_type'].nunique(dropna=False)
    assert len(_s)>0,"no groups found for (policy_class,action_type)"
    assert _s.eq(1).all(),f"obs_type not unique within some (policy_class,action_type):\n{_s[_s.ne(1)].sort_values(ascending=False)}"
    assert DF_FOR_STATS.action_type.isna().sum() == 0, "Some action_type values are NaN"
    assert DF_FOR_STATS.obs_type.isna().sum() == 0, "Some action_type values are NaN"
    assert DF_FOR_STATS.policy_class.isna().sum() == 0, "Some action_type values are NaN"
    DF_FOR_STATS.groupby(['policy_class', 'action_type'])['obs_type'].nunique() #.reset_index(name='count').sort_values('count', ascending=False)'])
    return


@app.cell
def _(mo):
    mo.md(r"""
    260110 Author todo: add missing policy category info as early in the pipeline as possible; recreate here?
    """)
    return


@app.cell
def _(DF_FOR_STATS):
    DF_FOR_STATS.policy_category.value_counts(dropna=False)
    return


@app.cell
def _(DF_FOR_STATS):
    DF_FOR_STATS
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    - 260110 Author: this uses the legacy complicated formatting; rewritten above to use MODEL_FORMATTING_DICT
    - 251216 Author: 15 mins on compute-cluster even w/o sanity checks
    """)
    return


@app.cell
def _():
    # RERUN_N_PLOT_MICE_VS_SIMS_PER_SESSION = True

    # ## only used if RERUN_N_PLOT_MICE_VS_SIMS_PER_SESSION = True
    # # _SELECTED_ANIMAL_IDS_FIG_4 = ['a031', 'a033', 'a030', 'a029']  
    # # SELECTED_ANIMAL_IDS_FIG_4 =  per_exp_result_df.animal_ID.unique()
    # # SELECTED_ANIMAL_IDS_FIG_4 = ['a030', 'a029', 'a003', 'a188', 'a033', 'a031']  
    # # SELECTED_ANIMAL_IDS_FIG_4 = ['a030']  
    # # SELECTED_ANIMAL_IDS_FIG_4 = ['a031', 'a033']
    # _SELECTED_ANIMAL_IDS_FIG_4 = DF_FOR_STATS.animal_ID.unique().tolist()

    # _N_LEGEND_COLS = 4
    # _RUN_SANITY_CHECKS = True

    # if RERUN_N_PLOT_MICE_VS_SIMS_PER_SESSION:

    #     _kwargs = {'axis_label_font_size': AXIS_LABEL_FONT_SIZE,
    #                'title_font_size': TITLE_FONT_SIZE,
    #                'mouse_point_marker_size': MOUSE_POINT_MARKER_SIZE,
    #                'mouse_point_alpha': MOUSE_POINT_ALPHA,
    #                'line_width_mouse': LINE_WIDTH_MOUSE,
    #                'line_width_models': LINE_WIDTH_MODELS,
    #                'line_width_all_models': LINE_WIDTH_ALL_MODELS,
    #                'legend_fontsize': LEGEND_FONTSIZE,
    #                'tick_label_font_size': TICK_LABEL_FONT_SIZE,
    #                'n_legend_cols': _N_LEGEND_COLS}

    #     # plot_mice_vs_sims_per_session(_SELECTED_ANIMAL_IDS_FIG_4, per_exp_result_df, run_sanity_checks=_RUN_SANITY_CHECKS, **_kwargs)
    #     plot_mice_vs_sims_per_session(_SELECTED_ANIMAL_IDS_FIG_4, DF_FOR_STATS, run_sanity_checks=_RUN_SANITY_CHECKS, **_kwargs)
    return


@app.cell
def _():
    # DF_FOR_STATS[DF_FOR_STATS.rewarded_action_inds_mouse.isnull()]
    # DF_FOR_STATS.rewarded_action_inds_mouse.isnull().sum()
    # DF_FOR_STATS.rewarded_action_inds_mouse.astype(str).nunique()
    # len(DF_FOR_STATS)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    unnormalized version (ie just total number of rewards)
    """)
    return


@app.cell
def _(
    AXIS_LABEL_FONT_SIZE,
    DF_FOR_STATS,
    LEGEND_FONTSIZE,
    LINE_WIDTH_ALL_MODELS,
    LINE_WIDTH_MODELS,
    LINE_WIDTH_MOUSE,
    MODEL_FORMATTING_DF,
    MOUSE_POINT_ALPHA,
    MOUSE_POINT_MARKER_SIZE,
    RERUN_N_PLOT_MICE_VS_SIMS_PER_SESSION,
    TICK_LABEL_FONT_SIZE,
    TITLE_FONT_SIZE,
    plot_mice_vs_sims_per_session_GPT_dev_v2,
):
    ## only used if RERUN_N_PLOT_MICE_VS_SIMS_PER_SESSION = True

    # SELECTED_ANIMAL_IDS_FIG_4 =  per_exp_result_df.animal_ID.unique()
    # SELECTED_ANIMAL_IDS_FIG_4 = ['a030', 'a029', 'a003', 'a188', 'a033', 'a031']  
    # SELECTED_ANIMAL_IDS_FIG_4 = ['a030']  
    # SELECTED_ANIMAL_IDS_FIG_4 = ['a031', 'a033']  
    _SELECTED_ANIMAL_IDS_FIG_4 = ['a031', 'a033', 'a030', 'a029']  
    _N_LEGEND_COLS = 4
    _RUN_SANITY_CHECKS = True
    NORMALIZE_BY_N_ACTIONS_IN_SESSION = True

    if RERUN_N_PLOT_MICE_VS_SIMS_PER_SESSION:

        _kwargs = {
            'axis_label_font_size': AXIS_LABEL_FONT_SIZE,
            'title_font_size': TITLE_FONT_SIZE,
            'mouse_point_marker_size': MOUSE_POINT_MARKER_SIZE,
            'mouse_point_alpha': MOUSE_POINT_ALPHA,
            'line_width_mouse': LINE_WIDTH_MOUSE,
            'line_width_models': LINE_WIDTH_MODELS,
            'line_width_all_models': LINE_WIDTH_ALL_MODELS,
            'legend_fontsize': LEGEND_FONTSIZE,
            'tick_label_font_size': TICK_LABEL_FONT_SIZE,
            'n_legend_cols': _N_LEGEND_COLS}

        # plot_mice_vs_sims_per_session(_SELECTED_ANIMAL_IDS_FIG_4, per_exp_result_df, 
        #                               run_sanity_checks=_RUN_SANITY_CHECKS, normalize_by_n_actions_in_session=NORMALIZE_BY_N_ACTIONS_IN_SESSION, **_kwargs)

        # plot_mice_vs_sims_per_session_GPT_dev_v2(_SELECTED_ANIMAL_IDS_FIG_4, DF_FOR_STATS,
                                      # run_sanity_checks=_RUN_SANITY_CHECKS, normalize_by_n_actions_in_session=NORMALIZE_BY_N_ACTIONS_IN_SESSION, **_kwargs)

        plot_mice_vs_sims_per_session_GPT_dev_v2(_SELECTED_ANIMAL_IDS_FIG_4, DF_FOR_STATS, MODEL_FORMATTING_DF=MODEL_FORMATTING_DF, run_sanity_checks=_RUN_SANITY_CHECKS, normalize_by_n_actions_in_session=NORMALIZE_BY_N_ACTIONS_IN_SESSION, **_kwargs)
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


@app.cell
def _():
    # def _():
    #         RERUN = True
    #         if not RERUN:
    #             return

    #         ## ---- user parameters ----
    #         FIGSIZE = (14, 6)
    #         MARKER_SIZE = MOUSE_POINT_MARKER_SIZE
    #         ALPHA = MOUSE_POINT_ALPHA
    #         X_COL = "n_states_visited"  # proxy for n_actions (n_actions = n_states - 1)
    #         USE_N_ACTIONS = True        # if True, subtract 1 from n_states_visited for x-axis

    #         ## ---- derive session-level mouse df (one row per animal_ID × exp_moment) ----
    #         MOUSE_COLS = ["animal_ID", "exp_moment", "reward_rate_mouse", X_COL]
    #         assert all(c in DF_FOR_STATS.columns for c in MOUSE_COLS), \
    #             f"missing cols: {[c for c in MOUSE_COLS if c not in DF_FOR_STATS.columns]}"

    #         mouse_df = (
    #             DF_FOR_STATS[MOUSE_COLS]
    #             .drop_duplicates(subset=["animal_ID", "exp_moment"])
    #             .copy()
    #         )
    #         mouse_df["n_actions"] = mouse_df[X_COL].astype(float) - (1.0 if USE_N_ACTIONS else 0.0)

    #         ## ---- derive session-level agent df (one row per animal_ID × exp_moment × model) ----
    #         AGENT_COLS = ["animal_ID", "exp_moment", "full_sim_desc", "reward_rate_agent", X_COL]
    #         agent_df = (
    #             DF_FOR_STATS[AGENT_COLS]
    #             .groupby(["animal_ID", "exp_moment", "full_sim_desc"], dropna=False)
    #             .agg(reward_rate_agent=("reward_rate_agent", "mean"),
    #                  **{X_COL: (X_COL, "first")})
    #             .reset_index()
    #         )
    #         agent_df["n_actions"] = agent_df[X_COL].astype(float) - (1.0 if USE_N_ACTIONS else 0.0)

    #         ## ---- resolve model formatting ----
    #         USE_MODEL_FMT = MODEL_FORMATTING_DF is not None
    #         if USE_MODEL_FMT:
    #             LABEL_COL = "legend_label" if "legend_label" in MODEL_FORMATTING_DF.columns else "xtick_label"
    #             model_names = [m for m in agent_df["full_sim_desc"].unique() if m in MODEL_FORMATTING_DF.index]
    #             if "model_order" in MODEL_FORMATTING_DF.columns:
    #                 ord_s = MODEL_FORMATTING_DF.loc[model_names, "model_order"].astype(float).sort_values()
    #                 model_names = ord_s.index.tolist()
    #             model2color = MODEL_FORMATTING_DF.loc[model_names, "color"].to_dict()
    #             model2label = MODEL_FORMATTING_DF.loc[model_names, LABEL_COL].to_dict()
    #         else:
    #             model_names = sorted(agent_df["full_sim_desc"].unique())
    #             cmap = cm.get_cmap("turbo", max(len(model_names), 1))
    #             model2color = {m: cmap(i) for i, m in enumerate(model_names)}
    #             model2label = {m: m for m in model_names}

    #         ## ---- animal color map ----
    #         animal_ids = sorted(mouse_df["animal_ID"].unique())
    #         ANIMAL_CMAP = cm.get_cmap("tab10", max(len(animal_ids), 1))
    #         ANIMAL2COLOR = {a: ANIMAL_CMAP(i) for i, a in enumerate(animal_ids)}

    #         ## ---- legend layout ----
    #         N_LEGEND_COLS_LEFT = 5
    #         N_LEGEND_COLS_RIGHT = 4

    #         ## ---- plot ----
    #         fig, (AX_LEFT, AX_RIGHT) = plt.subplots(1, 2, figsize=FIGSIZE, sharey=True)

    #         # --- LEFT: one point per session, colored by animal ---
    #         for animal_id in animal_ids:
    #             sub = mouse_df[mouse_df["animal_ID"] == animal_id]
    #             AX_LEFT.scatter(
    #                 sub["n_actions"], sub["reward_rate_mouse"],
    #                 s=MARKER_SIZE, alpha=ALPHA,
    #                 color=ANIMAL2COLOR[animal_id], label=animal_id,
    #                 edgecolors="none",
    #             )
    #         AX_LEFT.set_xlabel("n actions", fontsize=AXIS_LABEL_FONT_SIZE)
    #         AX_LEFT.set_ylabel("reward rate (n rewards / n actions)", fontsize=AXIS_LABEL_FONT_SIZE)
    #         AX_LEFT.set_title("mouse sessions by animal", fontsize=TITLE_FONT_SIZE)
    #         AX_LEFT.tick_params(labelsize=TICK_LABEL_FONT_SIZE)
    #         AX_LEFT.set_ylim(bottom=0)  # let matplotlib auto-scale the top
    #         AX_LEFT.legend(
    #             fontsize=LEGEND_FONTSIZE, title="animal_ID", markerscale=1.5,
    #             loc="upper center", bbox_to_anchor=(0.5, -0.18),
    #             ncol=N_LEGEND_COLS_LEFT, frameon=True,
    #         )

    #         # --- RIGHT: one point per (session × model), colored by model ---
    #         for model_name in model_names:
    #             sub = agent_df[agent_df["full_sim_desc"] == model_name]
    #             AX_RIGHT.scatter(
    #                 sub["n_actions"], sub["reward_rate_agent"],
    #                 s=MARKER_SIZE, alpha=ALPHA,
    #                 color=model2color[model_name], label=model2label[model_name],
    #                 edgecolors="none",
    #             )
    #         AX_RIGHT.set_xlabel("n actions", fontsize=AXIS_LABEL_FONT_SIZE)
    #         AX_RIGHT.set_title("agent sessions by model", fontsize=TITLE_FONT_SIZE)
    #         AX_RIGHT.tick_params(labelsize=TICK_LABEL_FONT_SIZE)
    #         AX_RIGHT.legend(
    #             fontsize=LEGEND_FONTSIZE, title="model", markerscale=1.5,
    #             loc="upper center", bbox_to_anchor=(0.5, -0.18),
    #             ncol=N_LEGEND_COLS_RIGHT, frameon=True,
    #         )

    #         fig.subplots_adjust(bottom=0.35)
    #         fig.savefig(f"figures/{get_now_str()}_actions_vs_rwd_rate.png", bbox_inches="tight", pad_inches=0.02, dpi=200)
    #         fig.savefig(f"figures/{get_now_str()}_actions_vs_rwd_rate.svg", bbox_inches="tight", pad_inches=0.02)
    #         fig.savefig(f"figures/{get_now_str()}_actions_vs_rwd_rate.pdf", bbox_inches="tight", pad_inches=0.02)
    #         plt.show()

    # _()  # keep marimo happy


    return


@app.cell
def _(
    AXIS_LABEL_FONT_SIZE,
    DF_FOR_STATS,
    LEGEND_FONTSIZE,
    MODEL_FORMATTING_DF,
    MOUSE_POINT_ALPHA,
    MOUSE_POINT_MARKER_SIZE,
    TICK_LABEL_FONT_SIZE,
    TITLE_FONT_SIZE,
    cm,
    get_now_str,
    plt,
):
    def _():
        RERUN = True
        if not RERUN:
            return 

        ## ---- user parameters ----
        FIGSIZE = (14, 6)
        MARKER_SIZE = MOUSE_POINT_MARKER_SIZE
        ALPHA = MOUSE_POINT_ALPHA
        X_COL = "n_states_visited"  # proxy for n_actions (n_actions = n_states - 1)
        USE_N_ACTIONS = True        # if True, subtract 1 from n_states_visited for x-axis
        N_LEGEND_COLS_LEFT = 5
        N_LEGEND_COLS_RIGHT = 4

        ## ---- derive session-level mouse df (one row per animal_ID x exp_moment) ----
        mouse_cols = ["animal_ID", "exp_moment", "reward_rate_mouse", X_COL]
        assert all(c in DF_FOR_STATS.columns for c in mouse_cols), \
            f"missing cols: {[c for c in mouse_cols if c not in DF_FOR_STATS.columns]}"

        mouse_df = (
            DF_FOR_STATS[mouse_cols]
            .drop_duplicates(subset=["animal_ID", "exp_moment"])
            .copy()
        )
        mouse_df["n_actions"] = mouse_df[X_COL].astype(float) - (1.0 if USE_N_ACTIONS else 0.0)

        ## ---- derive session-level agent df (one row per animal_ID x exp_moment x model) ----
        agent_cols = ["animal_ID", "exp_moment", "full_sim_desc", "reward_rate_agent", X_COL]
        agent_df = (
            DF_FOR_STATS[agent_cols]
            .groupby(["animal_ID", "exp_moment", "full_sim_desc"], dropna=False)
            .agg(reward_rate_agent=("reward_rate_agent", "mean"),
                 **{X_COL: (X_COL, "first")})
            .reset_index()
        )
        agent_df["n_actions"] = agent_df[X_COL].astype(float) - (1.0 if USE_N_ACTIONS else 0.0)

        ## ---- resolve model formatting ----
        use_model_fmt = MODEL_FORMATTING_DF is not None
        if use_model_fmt:
            label_col = "legend_label" if "legend_label" in MODEL_FORMATTING_DF.columns else "xtick_label"
            model_names = [m for m in agent_df["full_sim_desc"].unique() if m in MODEL_FORMATTING_DF.index]
            if "model_order" in MODEL_FORMATTING_DF.columns:
                ord_s = MODEL_FORMATTING_DF.loc[model_names, "model_order"].astype(float).sort_values()
                model_names = ord_s.index.tolist()
            model2color = MODEL_FORMATTING_DF.loc[model_names, "color"].to_dict()
            model2label = MODEL_FORMATTING_DF.loc[model_names, label_col].to_dict()
        else:
            model_names = sorted(agent_df["full_sim_desc"].unique())
            _cmap = cm.get_cmap("turbo", max(len(model_names), 1))
            model2color = {m: _cmap(i) for i, m in enumerate(model_names)}
            model2label = {m: m for m in model_names}

        ## ---- animal color map ----
        animal_ids = sorted(mouse_df["animal_ID"].unique())
        animal_cmap = cm.get_cmap("tab10", max(len(animal_ids), 1))
        animal2color = {a: animal_cmap(i) for i, a in enumerate(animal_ids)}

        ## ---- plot ----
        fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=FIGSIZE, sharey=True)

        # --- LEFT: one point per session, colored by animal ---
        for animal_id in animal_ids:
            sub = mouse_df[mouse_df["animal_ID"] == animal_id]
            ax_left.scatter(
                sub["n_actions"], sub["reward_rate_mouse"],
                s=MARKER_SIZE, alpha=ALPHA,
                color=animal2color[animal_id], label=animal_id,
                edgecolors="none",
            )
        ax_left.set_xlabel("n actions", fontsize=AXIS_LABEL_FONT_SIZE)
        ax_left.set_ylabel("reward rate (n rewards / n actions)", fontsize=AXIS_LABEL_FONT_SIZE)
        ax_left.set_title("mouse sessions by animal", fontsize=TITLE_FONT_SIZE)
        ax_left.tick_params(labelsize=TICK_LABEL_FONT_SIZE)
        ax_left.set_ylim(bottom=0)
        ax_left.legend(
            fontsize=LEGEND_FONTSIZE, title="animal_ID", markerscale=1.5,
            loc="upper center", bbox_to_anchor=(0.5, -0.18),
            ncol=N_LEGEND_COLS_LEFT, frameon=True,
        )

        # --- RIGHT: one point per (session x model), colored by model ---
        for model_name in model_names:
            sub = agent_df[agent_df["full_sim_desc"] == model_name]
            ax_right.scatter(
                sub["n_actions"], sub["reward_rate_agent"],
                s=MARKER_SIZE, alpha=ALPHA,
                color=model2color[model_name], label=model2label[model_name],
                edgecolors="none",
            )
        ax_right.set_xlabel("n actions", fontsize=AXIS_LABEL_FONT_SIZE)
        ax_right.set_title("agent sessions by model", fontsize=TITLE_FONT_SIZE)
        ax_right.tick_params(labelsize=TICK_LABEL_FONT_SIZE)
        ax_right.legend(
            fontsize=LEGEND_FONTSIZE, title="model", markerscale=1.5,
            loc="upper center", bbox_to_anchor=(0.5, -0.18),
            ncol=N_LEGEND_COLS_RIGHT, frameon=True,
        )

        fig.subplots_adjust(bottom=0.35)
        fig.savefig(f"figures/{get_now_str()}_actions_vs_rwd_rate.png", bbox_inches="tight", pad_inches=0.02, dpi=200)
        fig.savefig(f"figures/{get_now_str()}_actions_vs_rwd_rate.svg", bbox_inches="tight", pad_inches=0.02)
        fig.savefig(f"figures/{get_now_str()}_actions_vs_rwd_rate.pdf", bbox_inches="tight", pad_inches=0.02)
        plt.show()

    _()
    return


@app.cell
def _():
    return


@app.cell
def _(
    AXIS_LABEL_FONT_SIZE,
    DF_FOR_STATS,
    LEGEND_FONTSIZE,
    MODEL_FORMATTING_DF,
    MOUSE_POINT_ALPHA,
    MOUSE_POINT_MARKER_SIZE,
    TICK_LABEL_FONT_SIZE,
    TITLE_FONT_SIZE,
    cm,
    get_now_str,
    os,
    plt,
):
    def _():
        RERUN = True
        if not RERUN:
            return

        ## ---- user parameters ----
        FIGSIZE = (14, 6)
        MARKER_SIZE = MOUSE_POINT_MARKER_SIZE
        ALPHA = MOUSE_POINT_ALPHA
        X_COL = "n_states_visited"  # proxy for n_actions (n_actions = n_states - 1)
        USE_N_ACTIONS = True        # if True, subtract 1 from n_states_visited for x-axis
        N_LEGEND_COLS_LEFT = 5
        N_LEGEND_COLS_RIGHT = 4
    
        X_LIM = None                # e.g. (0, 500) or None for auto
        # X_LIM = (0, 650)                # e.g. (0, 500) or None for auto
    
        # X_LOG_SCALE = False         # True for log scale on x-axis
        X_LOG_SCALE = True         # True for log scale on x-axis

        ## ---- derive session-level mouse df (one row per animal_ID x exp_moment) ----
        mouse_cols = ["animal_ID", "exp_moment", "reward_rate_mouse", X_COL]
        assert all(c in DF_FOR_STATS.columns for c in mouse_cols), \
            f"missing cols: {[c for c in mouse_cols if c not in DF_FOR_STATS.columns]}"

        mouse_df = (
            DF_FOR_STATS[mouse_cols]
            .drop_duplicates(subset=["animal_ID", "exp_moment"])
            .copy()
        )
        mouse_df["n_actions"] = mouse_df[X_COL].astype(float) - (1.0 if USE_N_ACTIONS else 0.0)

        ## ---- derive session-level agent df (one row per animal_ID x exp_moment x model) ----
        agent_cols = ["animal_ID", "exp_moment", "full_sim_desc", "reward_rate_agent", X_COL]
        agent_df = (
            DF_FOR_STATS[agent_cols]
            .groupby(["animal_ID", "exp_moment", "full_sim_desc"], dropna=False)
            .agg(reward_rate_agent=("reward_rate_agent", "mean"),
                 **{X_COL: (X_COL, "first")})
            .reset_index()
        )
        agent_df["n_actions"] = agent_df[X_COL].astype(float) - (1.0 if USE_N_ACTIONS else 0.0)

        ## ---- resolve model formatting ----
        use_model_fmt = MODEL_FORMATTING_DF is not None
        if use_model_fmt:
            label_col = "legend_label" if "legend_label" in MODEL_FORMATTING_DF.columns else "xtick_label"
            model_names = [m for m in agent_df["full_sim_desc"].unique() if m in MODEL_FORMATTING_DF.index]
            if "model_order" in MODEL_FORMATTING_DF.columns:
                ord_s = MODEL_FORMATTING_DF.loc[model_names, "model_order"].astype(float).sort_values()
                model_names = ord_s.index.tolist()
            model2color = MODEL_FORMATTING_DF.loc[model_names, "color"].to_dict()
            model2label = MODEL_FORMATTING_DF.loc[model_names, label_col].to_dict()
        else:
            model_names = sorted(agent_df["full_sim_desc"].unique())
            _cmap = cm.get_cmap("turbo", max(len(model_names), 1))
            model2color = {m: _cmap(i) for i, m in enumerate(model_names)}
            model2label = {m: m for m in model_names}

        ## ---- animal color map ----
        animal_ids = sorted(mouse_df["animal_ID"].unique())
        animal_cmap = cm.get_cmap("tab10", max(len(animal_ids), 1))
        animal2color = {a: animal_cmap(i) for i, a in enumerate(animal_ids)}

        ## ---- plot ----
        fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=FIGSIZE, sharey=True)

        # --- LEFT: one point per session, colored by animal ---
        for animal_id in animal_ids:
            sub = mouse_df[mouse_df["animal_ID"] == animal_id]
            ax_left.scatter(
                sub["n_actions"], sub["reward_rate_mouse"],
                s=MARKER_SIZE, alpha=ALPHA,
                color=animal2color[animal_id], label=animal_id,
                edgecolors="none",
            )
        ax_left.set_xlabel("n actions", fontsize=AXIS_LABEL_FONT_SIZE)
        ax_left.set_ylabel("reward rate (n rewards / n actions)", fontsize=AXIS_LABEL_FONT_SIZE)
        ax_left.set_title("mouse sessions by animal", fontsize=TITLE_FONT_SIZE)
        ax_left.tick_params(labelsize=TICK_LABEL_FONT_SIZE)
        ax_left.set_ylim(bottom=0)
        if X_LIM is not None:
            ax_left.set_xlim(X_LIM)
        if X_LOG_SCALE:
            ax_left.set_xscale("log")
        ax_left.legend(
            fontsize=LEGEND_FONTSIZE, title="animal_ID", markerscale=1.5,
            loc="upper center", bbox_to_anchor=(0.5, -0.18),
            ncol=N_LEGEND_COLS_LEFT, frameon=True,
        )

        # --- RIGHT: one point per (session x model), colored by model ---
        for model_name in model_names:
            sub = agent_df[agent_df["full_sim_desc"] == model_name]
            ax_right.scatter(
                sub["n_actions"], sub["reward_rate_agent"],
                s=MARKER_SIZE, alpha=ALPHA,
                color=model2color[model_name], label=model2label[model_name],
                edgecolors="none",
            )
        ax_right.set_xlabel("n actions", fontsize=AXIS_LABEL_FONT_SIZE)
        ax_right.set_title("agent sessions by model", fontsize=TITLE_FONT_SIZE)
        ax_right.tick_params(labelsize=TICK_LABEL_FONT_SIZE)
    
        if X_LIM is not None:
            ax_right.set_xlim(X_LIM)
        if X_LOG_SCALE:
            ax_right.set_xscale("log")
        ax_right.legend(
            fontsize=LEGEND_FONTSIZE, title="model", markerscale=1.5,
            loc="upper center", bbox_to_anchor=(0.5, -0.18),
            ncol=N_LEGEND_COLS_RIGHT, frameon=True,
        )

        log_str = '' # else condition
        if X_LOG_SCALE:
            log_str = '-log'
        
        x_lim_str = '' # else condition
        if X_LIM is not None:
            x_lim_str = f'-xlim{X_LIM[0]}-{X_LIM[1]}'
            x_lim_str = x_lim_str.replace('.', 'p')
        

        fig.subplots_adjust(bottom=0.35)
        os.makedirs("figures", exist_ok=True)
        fig.savefig(f"figures/{get_now_str()}_actions_vs_rwd_rate{log_str}{x_lim_str}.png", bbox_inches="tight", pad_inches=0.02, dpi=200)
        fig.savefig(f"figures/{get_now_str()}_actions_vs_rwd_rate{log_str}{x_lim_str}.svg", bbox_inches="tight", pad_inches=0.02)
        fig.savefig(f"figures/{get_now_str()}_actions_vs_rwd_rate{log_str}{x_lim_str}.pdf", bbox_inches="tight", pad_inches=0.02)
        plt.show()

    _()
    return


@app.cell
def _():
    return


@app.cell
def _(
    AXIS_LABEL_FONT_SIZE,
    DF_FOR_STATS,
    LEGEND_FONTSIZE,
    MODEL_FORMATTING_DF,
    MOUSE_POINT_ALPHA,
    MOUSE_POINT_MARKER_SIZE,
    TICK_LABEL_FONT_SIZE,
    TITLE_FONT_SIZE,
    cm,
    get_now_str,
    np,
    os,
    plt,
    ss,
):
    def _():
        RERUN = True
        if not RERUN:
            return

        ## ---- user parameters ----
        FIGSIZE = (14, 6)
        MARKER_SIZE = MOUSE_POINT_MARKER_SIZE
        ALPHA = MOUSE_POINT_ALPHA
        LINE_ALPHA = 0.7
        SHADING_ALPHA = 0.2
        LINE_WIDTH = 1.5
        X_COL = "n_states_visited"
        USE_N_ACTIONS = True
        N_LEGEND_COLS_LEFT = 5
        N_LEGEND_COLS_RIGHT = 4
        X_LIM = None
        X_LOG_SCALE = True
        SHADING_TYPE = "sem"        # "sem", "std", or "ci95"
        SORT_BY = "n_actions"       # "n_actions" or "exp_moment"

        ## ---- derive session-level mouse df (one row per animal_ID x exp_moment) ----
        mouse_cols = ["animal_ID", "exp_moment", "reward_rate_mouse", X_COL]
        assert all(c in DF_FOR_STATS.columns for c in mouse_cols), \
            f"missing cols: {[c for c in mouse_cols if c not in DF_FOR_STATS.columns]}"

        mouse_df = (
            DF_FOR_STATS[mouse_cols]
            .drop_duplicates(subset=["animal_ID", "exp_moment"])
            .copy()
        )
        mouse_df["n_actions"] = mouse_df[X_COL].astype(float) - (1.0 if USE_N_ACTIONS else 0.0)

        ## ---- derive session-level agent df WITH per-session spread ----
        agent_cols = ["animal_ID", "exp_moment", "full_sim_desc", "reward_rate_agent", X_COL]
        agent_agg = (
            DF_FOR_STATS[agent_cols]
            .groupby(["animal_ID", "exp_moment", "full_sim_desc"], dropna=False)
            .agg(
                rr_mean=("reward_rate_agent", "mean"),
                rr_std=("reward_rate_agent", "std"),
                rr_sem=("reward_rate_agent", lambda x: ss.sem(x) if len(x) > 1 else 0.0),
                rr_count=("reward_rate_agent", "count"),
                **{X_COL: (X_COL, "first")},
            )
            .reset_index()
        )
        agent_agg["n_actions"] = agent_agg[X_COL].astype(float) - (1.0 if USE_N_ACTIONS else 0.0)

        # compute shading half-width
        if SHADING_TYPE == "sem":
            agent_agg["rr_half"] = agent_agg["rr_sem"]
        elif SHADING_TYPE == "std":
            agent_agg["rr_half"] = agent_agg["rr_std"]
        elif SHADING_TYPE == "ci95":
            agent_agg["rr_half"] = agent_agg["rr_sem"] * 1.96
        else:
            raise ValueError(f"SHADING_TYPE must be 'sem', 'std', or 'ci95'; got '{SHADING_TYPE}'")
        agent_agg["rr_half"] = agent_agg["rr_half"].fillna(0.0)

        ## ---- resolve model formatting ----
        use_model_fmt = MODEL_FORMATTING_DF is not None
        if use_model_fmt:
            label_col = "legend_label" if "legend_label" in MODEL_FORMATTING_DF.columns else "xtick_label"
            model_names = [m for m in agent_agg["full_sim_desc"].unique() if m in MODEL_FORMATTING_DF.index]
            if "model_order" in MODEL_FORMATTING_DF.columns:
                ord_s = MODEL_FORMATTING_DF.loc[model_names, "model_order"].astype(float).sort_values()
                model_names = ord_s.index.tolist()
            model2color = MODEL_FORMATTING_DF.loc[model_names, "color"].to_dict()
            model2label = MODEL_FORMATTING_DF.loc[model_names, label_col].to_dict()
        else:
            model_names = sorted(agent_agg["full_sim_desc"].unique())
            _cmap = cm.get_cmap("turbo", max(len(model_names), 1))
            model2color = {m: _cmap(i) for i, m in enumerate(model_names)}
            model2label = {m: m for m in model_names}

        ## ---- animal color map ----
        animal_ids = sorted(mouse_df["animal_ID"].unique())
        animal_cmap = cm.get_cmap("tab10", max(len(animal_ids), 1))
        animal2color = {a: animal_cmap(i) for i, a in enumerate(animal_ids)}

        ## ---- plot ----
        fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=FIGSIZE, sharey=True)

        # --- LEFT: connected line per animal, sorted by SORT_BY ---
        sort_col = SORT_BY if SORT_BY in mouse_df.columns else "n_actions"
        for animal_id in animal_ids:
            sub = mouse_df[mouse_df["animal_ID"] == animal_id].sort_values(sort_col)
            ax_left.plot(
                sub["n_actions"].values, sub["reward_rate_mouse"].values,
                "-o", color=animal2color[animal_id], label=animal_id,
                markersize=np.sqrt(MARKER_SIZE), alpha=LINE_ALPHA,
                linewidth=LINE_WIDTH, markeredgecolor="none",
            )
        ax_left.set_xlabel("n actions", fontsize=AXIS_LABEL_FONT_SIZE)
        ax_left.set_ylabel("reward rate (n rewards / n actions)", fontsize=AXIS_LABEL_FONT_SIZE)
        ax_left.set_title("mouse sessions by animal", fontsize=TITLE_FONT_SIZE)
        ax_left.tick_params(labelsize=TICK_LABEL_FONT_SIZE)
        ax_left.set_ylim(bottom=0)
        if X_LIM is not None:
            ax_left.set_xlim(X_LIM)
        if X_LOG_SCALE:
            ax_left.set_xscale("log")
        ax_left.legend(
            fontsize=LEGEND_FONTSIZE, title="animal_ID", markerscale=1.5,
            loc="upper center", bbox_to_anchor=(0.5, -0.18),
            ncol=N_LEGEND_COLS_LEFT, frameon=True,
        )

        # --- RIGHT: mean line + shading per model ---
        for model_name in model_names:
            sub = agent_agg[agent_agg["full_sim_desc"] == model_name].sort_values("n_actions")
            x = sub["n_actions"].values
            y = sub["rr_mean"].values
            h = sub["rr_half"].values
            c = model2color[model_name]
            ax_right.plot(x, y, "-", color=c, label=model2label[model_name],
                          linewidth=LINE_WIDTH, alpha=LINE_ALPHA)
            ax_right.fill_between(x, y - h, y + h, color=c, alpha=SHADING_ALPHA)
        ax_right.set_xlabel("n actions", fontsize=AXIS_LABEL_FONT_SIZE)
        ax_right.set_title(f"agent sessions by model (shading = {SHADING_TYPE})", fontsize=TITLE_FONT_SIZE)
        ax_right.tick_params(labelsize=TICK_LABEL_FONT_SIZE)
        if X_LIM is not None:
            ax_right.set_xlim(X_LIM)
        if X_LOG_SCALE:
            ax_right.set_xscale("log")
        ax_right.legend(
            fontsize=LEGEND_FONTSIZE, title="model", markerscale=1.5,
            loc="upper center", bbox_to_anchor=(0.5, -0.18),
            ncol=N_LEGEND_COLS_RIGHT, frameon=True,
        )

        log_str = '-log' if X_LOG_SCALE else ''
        x_lim_str = ''
        if X_LIM is not None:
            x_lim_str = f'-xlim{X_LIM[0]}-{X_LIM[1]}'.replace('.', 'p')

        fig.subplots_adjust(bottom=0.35)
        os.makedirs("figures", exist_ok=True)
        fig.savefig(f"figures/{get_now_str()}_actions_vs_rwd_rate_lines{log_str}{x_lim_str}.png", bbox_inches="tight", pad_inches=0.02, dpi=200)
        fig.savefig(f"figures/{get_now_str()}_actions_vs_rwd_rate_lines{log_str}{x_lim_str}.svg", bbox_inches="tight", pad_inches=0.02)
        fig.savefig(f"figures/{get_now_str()}_actions_vs_rwd_rate_lines{log_str}{x_lim_str}.pdf", bbox_inches="tight", pad_inches=0.02)
        plt.show()

    _()
    return


@app.cell
def _():
    return


@app.cell
def _(
    AXIS_LABEL_FONT_SIZE,
    DF_FOR_STATS,
    LEGEND_FONTSIZE,
    MODEL_FORMATTING_DF,
    MOUSE_POINT_ALPHA,
    MOUSE_POINT_MARKER_SIZE,
    TICK_LABEL_FONT_SIZE,
    TITLE_FONT_SIZE,
    cm,
    get_now_str,
    np,
    os,
    plt,
    ss,
):
    def _():
        RERUN = True
        if not RERUN:
            return

        ## ---- user parameters ----
        PANEL_WIDTH = 5
        PANEL_HEIGHT = 5
        MARKER_SIZE = MOUSE_POINT_MARKER_SIZE
        ALPHA = MOUSE_POINT_ALPHA
        LINE_ALPHA = 0.7
        SHADING_ALPHA = 0.2
        LINE_WIDTH_MOUSE = 2.0
        LINE_WIDTH_MODELS = 1.5
        MOUSE_COLOR = "black"
        X_COL = "n_states_visited"
        USE_N_ACTIONS = True
        N_LEGEND_COLS = 3
        X_LIM = None
        X_LOG_SCALE = True
        SHADING_TYPE = "sem"        # "sem", "std", or "ci95"
        SORT_BY = "n_actions"       # "n_actions" or "exp_moment"
        SHARE_Y = True

        ## ---- derive session-level mouse df ----
        mouse_cols = ["animal_ID", "exp_moment", "reward_rate_mouse", X_COL]
        assert all(c in DF_FOR_STATS.columns for c in mouse_cols), \
            f"missing cols: {[c for c in mouse_cols if c not in DF_FOR_STATS.columns]}"

        mouse_df = (
            DF_FOR_STATS[mouse_cols]
            .drop_duplicates(subset=["animal_ID", "exp_moment"])
            .copy()
        )
        mouse_df["n_actions"] = mouse_df[X_COL].astype(float) - (1.0 if USE_N_ACTIONS else 0.0)

        ## ---- derive session-level agent df with spread ----
        agent_cols = ["animal_ID", "exp_moment", "full_sim_desc", "reward_rate_agent", X_COL]
        agent_agg = (
            DF_FOR_STATS[agent_cols]
            .groupby(["animal_ID", "exp_moment", "full_sim_desc"], dropna=False)
            .agg(
                rr_mean=("reward_rate_agent", "mean"),
                rr_std=("reward_rate_agent", "std"),
                rr_sem=("reward_rate_agent", lambda x: ss.sem(x) if len(x) > 1 else 0.0),
                rr_count=("reward_rate_agent", "count"),
                **{X_COL: (X_COL, "first")},
            )
            .reset_index()
        )
        agent_agg["n_actions"] = agent_agg[X_COL].astype(float) - (1.0 if USE_N_ACTIONS else 0.0)

        if SHADING_TYPE == "sem":
            agent_agg["rr_half"] = agent_agg["rr_sem"]
        elif SHADING_TYPE == "std":
            agent_agg["rr_half"] = agent_agg["rr_std"]
        elif SHADING_TYPE == "ci95":
            agent_agg["rr_half"] = agent_agg["rr_sem"] * 1.96
        else:
            raise ValueError(f"SHADING_TYPE must be 'sem', 'std', or 'ci95'; got '{SHADING_TYPE}'")
        agent_agg["rr_half"] = agent_agg["rr_half"].fillna(0.0)

        ## ---- resolve model formatting ----
        use_model_fmt = MODEL_FORMATTING_DF is not None
        if use_model_fmt:
            label_col = "legend_label" if "legend_label" in MODEL_FORMATTING_DF.columns else "xtick_label"
            model_names = [m for m in agent_agg["full_sim_desc"].unique() if m in MODEL_FORMATTING_DF.index]
            if "model_order" in MODEL_FORMATTING_DF.columns:
                ord_s = MODEL_FORMATTING_DF.loc[model_names, "model_order"].astype(float).sort_values()
                model_names = ord_s.index.tolist()
            model2color = MODEL_FORMATTING_DF.loc[model_names, "color"].to_dict()
            model2label = MODEL_FORMATTING_DF.loc[model_names, label_col].to_dict()
        else:
            model_names = sorted(agent_agg["full_sim_desc"].unique())
            _cmap = cm.get_cmap("turbo", max(len(model_names), 1))
            model2color = {m: _cmap(i) for i, m in enumerate(model_names)}
            model2label = {m: m for m in model_names}

        ## ---- layout ----
        animal_ids = sorted(mouse_df["animal_ID"].unique())
        n_animals = len(animal_ids)
        fig, axes = plt.subplots(
            1, n_animals,
            figsize=(PANEL_WIDTH * n_animals, PANEL_HEIGHT),
            sharey=SHARE_Y, sharex=False,
        )
        if n_animals == 1:
            axes = [axes]

        sort_col = SORT_BY if SORT_BY in mouse_df.columns else "n_actions"

        for idx, (animal_id, ax) in enumerate(zip(animal_ids, axes)):
            # --- mouse line ---
            m_sub = mouse_df[mouse_df["animal_ID"] == animal_id].sort_values(sort_col)
            ax.plot(
                m_sub["n_actions"].values, m_sub["reward_rate_mouse"].values,
                "-o", color=MOUSE_COLOR, label="mouse",
                markersize=np.sqrt(MARKER_SIZE), alpha=LINE_ALPHA,
                linewidth=LINE_WIDTH_MOUSE, markeredgecolor="none", zorder=10,
            )

            # --- model lines + shading ---
            for model_name in model_names:
                a_sub = agent_agg[
                    (agent_agg["full_sim_desc"] == model_name)
                    & (agent_agg["animal_ID"] == animal_id)
                ].sort_values("n_actions")
                if a_sub.empty:
                    continue
                x = a_sub["n_actions"].values
                y = a_sub["rr_mean"].values
                h = a_sub["rr_half"].values
                c = model2color[model_name]
                ax.plot(x, y, "-", color=c, label=model2label[model_name],
                        linewidth=LINE_WIDTH_MODELS, alpha=LINE_ALPHA)
                ax.fill_between(x, y - h, y + h, color=c, alpha=SHADING_ALPHA)

            ax.set_title(animal_id, fontsize=TITLE_FONT_SIZE)
            ax.set_xlabel("n actions", fontsize=AXIS_LABEL_FONT_SIZE)
            ax.tick_params(labelsize=TICK_LABEL_FONT_SIZE)
            ax.set_ylim(bottom=0)
            if X_LIM is not None:
                ax.set_xlim(X_LIM)
            if X_LOG_SCALE:
                ax.set_xscale("log")
            if idx == 0:
                ax.set_ylabel("reward rate (n rewards / n actions)", fontsize=AXIS_LABEL_FONT_SIZE)

        # --- shared legend from first axis ---
        handles, labels = axes[0].get_legend_handles_labels()
        # deduplicate while preserving order
        seen = set()
        unique_handles, unique_labels = [], []
        for h, l in zip(handles, labels):
            if l not in seen:
                seen.add(l)
                unique_handles.append(h)
                unique_labels.append(l)
        fig.legend(
            unique_handles, unique_labels,
            fontsize=LEGEND_FONTSIZE, markerscale=1.5,
            loc="upper center", bbox_to_anchor=(0.5, -0.02),
            ncol=N_LEGEND_COLS, frameon=True,
        )

        log_str = '-log' if X_LOG_SCALE else ''
        x_lim_str = ''
        if X_LIM is not None:
            x_lim_str = f'-xlim{X_LIM[0]}-{X_LIM[1]}'.replace('.', 'p')

        fig.subplots_adjust(bottom=0.25, wspace=0.15)
        os.makedirs("figures", exist_ok=True)
        fig.savefig(f"figures/{get_now_str()}_actions_vs_rwd_rate_per_mouse{log_str}{x_lim_str}.png", bbox_inches="tight", pad_inches=0.02, dpi=200)
        fig.savefig(f"figures/{get_now_str()}_actions_vs_rwd_rate_per_mouse{log_str}{x_lim_str}.svg", bbox_inches="tight", pad_inches=0.02)
        fig.savefig(f"figures/{get_now_str()}_actions_vs_rwd_rate_per_mouse{log_str}{x_lim_str}.pdf", bbox_inches="tight", pad_inches=0.02)
        plt.show()

    _()
    return


@app.cell
def _():
    return


@app.cell
def _(
    AXIS_LABEL_FONT_SIZE,
    DF_FOR_STATS,
    LEGEND_FONTSIZE,
    MODEL_FORMATTING_DF,
    MOUSE_POINT_ALPHA,
    MOUSE_POINT_MARKER_SIZE,
    TICK_LABEL_FONT_SIZE,
    TITLE_FONT_SIZE,
    cm,
    get_now_str,
    np,
    os,
    plt,
    ss,
):
    def _():
        RERUN = True
        if RERUN:

            ## ---- user parameters ----
            PANEL_WIDTH = 7
            PANEL_HEIGHT = 4
            MARKER_SIZE = MOUSE_POINT_MARKER_SIZE
            ALPHA = MOUSE_POINT_ALPHA
            LINE_ALPHA = 0.7
            SHADING_ALPHA = 0.2
            LINE_WIDTH = 1.5
            X_COL = "n_states_visited"
            USE_N_ACTIONS = True
            N_LEGEND_COLS_LEFT = 5
            N_LEGEND_COLS_RIGHT = 4
            X_LIM = None
            X_LOG_SCALE = True
            SHADING_TYPE = "sem"        # "sem", "std", or "ci95"
            SORT_BY = "n_actions"       # "n_actions" or "exp_moment"
            SHARE_X = True
            SHARE_Y = True
    
            ## ---- derive session-level mouse df ----
            mouse_cols = ["animal_ID", "exp_moment", "reward_rate_mouse", X_COL]
            assert all(c in DF_FOR_STATS.columns for c in mouse_cols), \
                f"missing cols: {[c for c in mouse_cols if c not in DF_FOR_STATS.columns]}"
    
            mouse_df = (
                DF_FOR_STATS[mouse_cols]
                .drop_duplicates(subset=["animal_ID", "exp_moment"])
                .copy()
            )
            mouse_df["n_actions"] = mouse_df[X_COL].astype(float) - (1.0 if USE_N_ACTIONS else 0.0)
    
            ## ---- derive session-level agent df with spread ----
            agent_cols = ["animal_ID", "exp_moment", "full_sim_desc", "reward_rate_agent", X_COL]
            agent_agg = (
                DF_FOR_STATS[agent_cols]
                .groupby(["animal_ID", "exp_moment", "full_sim_desc"], dropna=False)
                .agg(
                    rr_mean=("reward_rate_agent", "mean"),
                    rr_std=("reward_rate_agent", "std"),
                    rr_sem=("reward_rate_agent", lambda x: ss.sem(x) if len(x) > 1 else 0.0),
                    rr_count=("reward_rate_agent", "count"),
                    **{X_COL: (X_COL, "first")},
                )
                .reset_index()
            )
            agent_agg["n_actions"] = agent_agg[X_COL].astype(float) - (1.0 if USE_N_ACTIONS else 0.0)
    
            if SHADING_TYPE == "sem":
                agent_agg["rr_half"] = agent_agg["rr_sem"]
            elif SHADING_TYPE == "std":
                agent_agg["rr_half"] = agent_agg["rr_std"]
            elif SHADING_TYPE == "ci95":
                agent_agg["rr_half"] = agent_agg["rr_sem"] * 1.96
            else:
                raise ValueError(f"SHADING_TYPE must be 'sem', 'std', or 'ci95'; got '{SHADING_TYPE}'")
            agent_agg["rr_half"] = agent_agg["rr_half"].fillna(0.0)
    
            ## ---- resolve model formatting ----
            use_model_fmt = MODEL_FORMATTING_DF is not None
            if use_model_fmt:
                label_col = "legend_label" if "legend_label" in MODEL_FORMATTING_DF.columns else "xtick_label"
                model_names = [m for m in agent_agg["full_sim_desc"].unique() if m in MODEL_FORMATTING_DF.index]
                if "model_order" in MODEL_FORMATTING_DF.columns:
                    ord_s = MODEL_FORMATTING_DF.loc[model_names, "model_order"].astype(float).sort_values()
                    model_names = ord_s.index.tolist()
                model2color = MODEL_FORMATTING_DF.loc[model_names, "color"].to_dict()
                model2label = MODEL_FORMATTING_DF.loc[model_names, label_col].to_dict()
            else:
                model_names = sorted(agent_agg["full_sim_desc"].unique())
                _cmap = cm.get_cmap("turbo", max(len(model_names), 1))
                model2color = {m: _cmap(i) for i, m in enumerate(model_names)}
                model2label = {m: m for m in model_names}
    
            ## ---- animal color map ----
            animal_ids = sorted(mouse_df["animal_ID"].unique())
            n_animals = len(animal_ids)
            animal_cmap = cm.get_cmap("tab10", max(n_animals, 1))
            animal2color = {a: animal_cmap(i) for i, a in enumerate(animal_ids)}
    
            sort_col = SORT_BY if SORT_BY in mouse_df.columns else "n_actions"
    
            ## ---- plot: n_animals rows x 2 cols ----
            fig, axes = plt.subplots(
                n_animals, 2,
                figsize=(PANEL_WIDTH * 2, PANEL_HEIGHT * n_animals),
                sharex=SHARE_X, sharey=SHARE_Y,
                squeeze=False,
            )
    
            for row_idx, animal_id in enumerate(animal_ids):
                ax_left = axes[row_idx, 0]
                ax_right = axes[row_idx, 1]
    
                # --- LEFT: mouse connected line ---
                m_sub = mouse_df[mouse_df["animal_ID"] == animal_id].sort_values(sort_col)
                ax_left.plot(
                    m_sub["n_actions"].values, m_sub["reward_rate_mouse"].values,
                    "-o", color=animal2color[animal_id],
                    label=animal_id if row_idx == 0 else None,
                    markersize=np.sqrt(MARKER_SIZE), alpha=LINE_ALPHA,
                    linewidth=LINE_WIDTH, markeredgecolor="none",
                )
                ax_left.set_ylabel(animal_id, fontsize=AXIS_LABEL_FONT_SIZE, fontweight="bold")
                ax_left.tick_params(labelsize=TICK_LABEL_FONT_SIZE)
                ax_left.set_ylim(bottom=0)
                if X_LIM is not None:
                    ax_left.set_xlim(X_LIM)
                if X_LOG_SCALE:
                    ax_left.set_xscale("log")
                if row_idx == 0:
                    ax_left.set_title("mouse sessions", fontsize=TITLE_FONT_SIZE)
    
                # --- RIGHT: model mean + shading ---
                for model_name in model_names:
                    a_sub = agent_agg[
                        (agent_agg["full_sim_desc"] == model_name)
                        & (agent_agg["animal_ID"] == animal_id)
                    ].sort_values("n_actions")
                    if a_sub.empty:
                        continue
                    x = a_sub["n_actions"].values
                    y = a_sub["rr_mean"].values
                    h = a_sub["rr_half"].values
                    c = model2color[model_name]
                    ax_right.plot(
                        x, y, "-", color=c,
                        label=model2label[model_name] if row_idx == 0 else None,
                        linewidth=LINE_WIDTH, alpha=LINE_ALPHA,
                    )
                    ax_right.fill_between(x, y - h, y + h, color=c, alpha=SHADING_ALPHA)
    
                ax_right.tick_params(labelsize=TICK_LABEL_FONT_SIZE)
                if X_LIM is not None:
                    ax_right.set_xlim(X_LIM)
                if X_LOG_SCALE:
                    ax_right.set_xscale("log")
                if row_idx == 0:
                    ax_right.set_title(f"agent sessions (shading = {SHADING_TYPE})", fontsize=TITLE_FONT_SIZE)
    
            # x-axis labels only on bottom row
            axes[-1, 0].set_xlabel("n actions", fontsize=AXIS_LABEL_FONT_SIZE)
            axes[-1, 1].set_xlabel("n actions", fontsize=AXIS_LABEL_FONT_SIZE)
    
            # --- shared legend below figure ---
            all_handles, all_labels = [], []
            for ax_row in axes:
                for ax in ax_row:
                    h, l = ax.get_legend_handles_labels()
                    all_handles.extend(h)
                    all_labels.extend(l)
            seen = set()
            unique_handles, unique_labels = [], []
            for h, l in zip(all_handles, all_labels):
                if l not in seen:
                    seen.add(l)
                    unique_handles.append(h)
                    unique_labels.append(l)
            n_legend_cols = max(N_LEGEND_COLS_LEFT, N_LEGEND_COLS_RIGHT)
            fig.legend(
                unique_handles, unique_labels,
                fontsize=LEGEND_FONTSIZE, markerscale=1.5,
                loc="upper center", bbox_to_anchor=(0.5, -0.01),
                ncol=n_legend_cols, frameon=True,
            )
    
            log_str = '-log' if X_LOG_SCALE else ''
            x_lim_str = ''
            if X_LIM is not None:
                x_lim_str = f'-xlim{X_LIM[0]}-{X_LIM[1]}'.replace('.', 'p')
    
            fig.subplots_adjust(hspace=0.25, wspace=0.1)
            os.makedirs("figures", exist_ok=True)
            fig.savefig(f"figures/{get_now_str()}_actions_vs_rwd_rate_per_mouse{log_str}{x_lim_str}.png", bbox_inches="tight", pad_inches=0.02, dpi=200)
            fig.savefig(f"figures/{get_now_str()}_actions_vs_rwd_rate_per_mouse{log_str}{x_lim_str}.svg", bbox_inches="tight", pad_inches=0.02)
            fig.savefig(f"figures/{get_now_str()}_actions_vs_rwd_rate_per_mouse{log_str}{x_lim_str}.pdf", bbox_inches="tight", pad_inches=0.02)
            plt.show()

    _()
    return


@app.cell
def _():
    return


@app.cell
def _(
    AXIS_LABEL_FONT_SIZE,
    DF_FOR_STATS,
    LEGEND_FONTSIZE,
    MODEL_FORMATTING_DF,
    MOUSE_POINT_ALPHA,
    MOUSE_POINT_MARKER_SIZE,
    TICK_LABEL_FONT_SIZE,
    TITLE_FONT_SIZE,
    cm,
    get_now_str,
    np,
    os,
    plt,
    ss,
):
    def _():
        RERUN = True
        if RERUN:

            ## ---- user parameters ----
            PANEL_WIDTH = 7
            PANEL_HEIGHT = 4
            MARKER_SIZE = MOUSE_POINT_MARKER_SIZE
            ALPHA = MOUSE_POINT_ALPHA
            LINE_ALPHA = 0.7
            SHADING_ALPHA = 0.2
            LINE_WIDTH = 1.5
            X_COL = "n_states_visited"
            USE_N_ACTIONS = True
            N_LEGEND_COLS_LEFT = 5
            N_LEGEND_COLS_RIGHT = 4
            X_LIM = None
            X_LOG_SCALE = True
            SHADING_TYPE = "sem"        # "sem", "std", or "ci95"
            SORT_BY = "n_actions"       # "n_actions" or "exp_moment"
            SHARE_X = True
            SHARE_Y = True
    
            ## ---- derive session-level mouse df ----
            mouse_cols = ["animal_ID", "exp_moment", "reward_rate_mouse", X_COL]
            assert all(c in DF_FOR_STATS.columns for c in mouse_cols), \
                f"missing cols: {[c for c in mouse_cols if c not in DF_FOR_STATS.columns]}"
    
            mouse_df = (
                DF_FOR_STATS[mouse_cols]
                .drop_duplicates(subset=["animal_ID", "exp_moment"])
                .copy()
            )
            mouse_df["n_actions"] = mouse_df[X_COL].astype(float) - (1.0 if USE_N_ACTIONS else 0.0)
    
            ## ---- map full_sim_desc -> legend_group for pooling ----
            use_model_fmt = MODEL_FORMATTING_DF is not None
            if use_model_fmt:
                label_col = "legend_label" if "legend_label" in MODEL_FORMATTING_DF.columns else "xtick_label"
                group_col_name = "legend_group" if "legend_group" in MODEL_FORMATTING_DF.columns else None
    
                # build lookup: full_sim_desc -> group name, color, label
                known_models = [m for m in DF_FOR_STATS["full_sim_desc"].unique() if m in MODEL_FORMATTING_DF.index]
                desc2group = {}
                group2color = {}
                group2label = {}
                for m in known_models:
                    grp = MODEL_FORMATTING_DF.loc[m, group_col_name] if group_col_name else m
                    desc2group[m] = grp
                    group2color[grp] = MODEL_FORMATTING_DF.loc[m, "color"]
                    group2label[grp] = MODEL_FORMATTING_DF.loc[m, label_col]
    
                # determine group order
                if "legend_order" in MODEL_FORMATTING_DF.columns:
                    grp_order_map = {}
                    for m in known_models:
                        grp = desc2group[m]
                        val = float(MODEL_FORMATTING_DF.loc[m, "legend_order"])
                        if grp not in grp_order_map or val < grp_order_map[grp]:
                            grp_order_map[grp] = val
                    group_names = sorted(grp_order_map.keys(), key=lambda g: grp_order_map[g])
                else:
                    # preserve first-appearance order
                    seen = set()
                    group_names = []
                    for m in known_models:
                        grp = desc2group[m]
                        if grp not in seen:
                            seen.add(grp)
                            group_names.append(grp)
            else:
                # no formatting df: each full_sim_desc is its own group
                all_descs = sorted(DF_FOR_STATS["full_sim_desc"].unique())
                desc2group = {m: m for m in all_descs}
                _cmap = cm.get_cmap("turbo", max(len(all_descs), 1))
                group2color = {m: _cmap(i) for i, m in enumerate(all_descs)}
                group2label = {m: m for m in all_descs}
                group_names = all_descs
    
            ## ---- build pooled agent df: one flat average per (animal, session, group) ----
            # attach group label to every raw row, then aggregate ALL runs within that group
            agent_raw = DF_FOR_STATS[["animal_ID", "exp_moment", "full_sim_desc", "reward_rate_agent", X_COL]].copy()
            agent_raw["legend_group"] = agent_raw["full_sim_desc"].map(desc2group)
            agent_raw = agent_raw.dropna(subset=["legend_group"])  # drop unknown models
    
            agent_agg = (
                agent_raw
                .groupby(["animal_ID", "exp_moment", "legend_group"], dropna=False)
                .agg(
                    rr_mean=("reward_rate_agent", "mean"),
                    rr_std=("reward_rate_agent", "std"),
                    rr_sem=("reward_rate_agent", lambda x: ss.sem(x) if len(x) > 1 else 0.0),
                    rr_count=("reward_rate_agent", "count"),
                    **{X_COL: (X_COL, "first")},
                )
                .reset_index()
            )
            agent_agg["n_actions"] = agent_agg[X_COL].astype(float) - (1.0 if USE_N_ACTIONS else 0.0)
    
            if SHADING_TYPE == "sem":
                agent_agg["rr_half"] = agent_agg["rr_sem"]
            elif SHADING_TYPE == "std":
                agent_agg["rr_half"] = agent_agg["rr_std"]
            elif SHADING_TYPE == "ci95":
                agent_agg["rr_half"] = agent_agg["rr_sem"] * 1.96
            else:
                raise ValueError(f"SHADING_TYPE must be 'sem', 'std', or 'ci95'; got '{SHADING_TYPE}'")
            agent_agg["rr_half"] = agent_agg["rr_half"].fillna(0.0)
    
            ## ---- animal color map ----
            animal_ids = sorted(mouse_df["animal_ID"].unique())
            n_animals = len(animal_ids)
            animal_cmap = cm.get_cmap("tab10", max(n_animals, 1))
            animal2color = {a: animal_cmap(i) for i, a in enumerate(animal_ids)}
    
            sort_col = SORT_BY if SORT_BY in mouse_df.columns else "n_actions"
    
            ## ---- plot: n_animals rows x 2 cols ----
            fig, axes = plt.subplots(
                n_animals, 2,
                figsize=(PANEL_WIDTH * 2, PANEL_HEIGHT * n_animals),
                sharex=SHARE_X, sharey=SHARE_Y,
                squeeze=False,
            )
    
            for row_idx, animal_id in enumerate(animal_ids):
                ax_left = axes[row_idx, 0]
                ax_right = axes[row_idx, 1]
    
                # --- LEFT: mouse connected line ---
                m_sub = mouse_df[mouse_df["animal_ID"] == animal_id].sort_values(sort_col)
                ax_left.plot(
                    m_sub["n_actions"].values, m_sub["reward_rate_mouse"].values,
                    "-o", color=animal2color[animal_id],
                    label=animal_id if row_idx == 0 else None,
                    markersize=np.sqrt(MARKER_SIZE), alpha=LINE_ALPHA,
                    linewidth=LINE_WIDTH, markeredgecolor="none",
                )
                ax_left.set_ylabel(animal_id, fontsize=AXIS_LABEL_FONT_SIZE, fontweight="bold")
                ax_left.tick_params(labelsize=TICK_LABEL_FONT_SIZE)
                ax_left.set_ylim(bottom=0)
                if X_LIM is not None:
                    ax_left.set_xlim(X_LIM)
                if X_LOG_SCALE:
                    ax_left.set_xscale("log")
                if row_idx == 0:
                    ax_left.set_title("mouse sessions", fontsize=TITLE_FONT_SIZE)
    
                # --- RIGHT: pooled model group mean + shading ---
                for grp_name in group_names:
                    a_sub = agent_agg[
                        (agent_agg["legend_group"] == grp_name)
                        & (agent_agg["animal_ID"] == animal_id)
                    ].sort_values("n_actions")
                    if a_sub.empty:
                        continue
                    x = a_sub["n_actions"].values
                    y = a_sub["rr_mean"].values
                    h = a_sub["rr_half"].values
                    c = group2color[grp_name]
                    ax_right.plot(
                        x, y, "-", color=c,
                        label=group2label[grp_name] if row_idx == 0 else None,
                        linewidth=LINE_WIDTH, alpha=LINE_ALPHA,
                    )
                    ax_right.fill_between(x, y - h, y + h, color=c, alpha=SHADING_ALPHA)
    
                ax_right.tick_params(labelsize=TICK_LABEL_FONT_SIZE)
                if X_LIM is not None:
                    ax_right.set_xlim(X_LIM)
                if X_LOG_SCALE:
                    ax_right.set_xscale("log")
                if row_idx == 0:
                    ax_right.set_title(f"agent sessions — pooled by model group ({SHADING_TYPE})", fontsize=TITLE_FONT_SIZE)
    
            # x-axis labels only on bottom row
            axes[-1, 0].set_xlabel("n actions", fontsize=AXIS_LABEL_FONT_SIZE)
            axes[-1, 1].set_xlabel("n actions", fontsize=AXIS_LABEL_FONT_SIZE)
    
            # --- shared legend below figure ---
            all_handles, all_labels = [], []
            for ax_row in axes:
                for ax in ax_row:
                    h, l = ax.get_legend_handles_labels()
                    all_handles.extend(h)
                    all_labels.extend(l)
            seen = set()
            unique_handles, unique_labels = [], []
            for h, l in zip(all_handles, all_labels):
                if l not in seen:
                    seen.add(l)
                    unique_handles.append(h)
                    unique_labels.append(l)
            n_legend_cols = max(N_LEGEND_COLS_LEFT, N_LEGEND_COLS_RIGHT)
            fig.legend(
                unique_handles, unique_labels,
                fontsize=LEGEND_FONTSIZE, markerscale=1.5,
                loc="upper center", bbox_to_anchor=(0.5, -0.01),
                ncol=n_legend_cols, frameon=True,
            )
    
            log_str = '-log' if X_LOG_SCALE else ''
            x_lim_str = ''
            if X_LIM is not None:
                x_lim_str = f'-xlim{X_LIM[0]}-{X_LIM[1]}'.replace('.', 'p')
    
            fig.subplots_adjust(hspace=0.25, wspace=0.1)
            os.makedirs("figures", exist_ok=True)
            fig.savefig(f"figures/{get_now_str()}_actions_vs_rwd_rate_per_mouse_pooled{log_str}{x_lim_str}.png", bbox_inches="tight", pad_inches=0.02, dpi=200)
            fig.savefig(f"figures/{get_now_str()}_actions_vs_rwd_rate_per_mouse_pooled{log_str}{x_lim_str}.svg", bbox_inches="tight", pad_inches=0.02)
            fig.savefig(f"figures/{get_now_str()}_actions_vs_rwd_rate_per_mouse_pooled{log_str}{x_lim_str}.pdf", bbox_inches="tight", pad_inches=0.02)
            plt.show()

    _()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # PLOT: violin and connected lines b/w mouse n sims

    - 251218 Author: are these comparing mice vs sims or early vs late?
    - 251218 GTP: proposed updates and fixes
    """)
    return


@app.cell
def _(DF_FOR_STATS, STATS_TEST, get_paired_stats, mean_and_ci, np, plt):
    PLOT_VIOLINS_N_LINES = True

    VIOLIN_ALPHA = 1.0
    POINT_ALPHA = 1.0

    # --- build a session-level paired df WITH training_stage, then compare weightings + plot consistently ---
    # Expected in scope: DF_FOR_STATS, get_paired_stats, np, pd, plt, ss
    if PLOT_VIOLINS_N_LINES:
        assert 'DF_FOR_STATS' in globals()
        assert 'get_paired_stats' in globals()
        for _name in ['np','pd','plt','ss']:
            assert _name in globals(), f"expected {_name} in scope"

        _required = ['animal_ID','exp_moment','reward_rate_mouse','reward_rate_agent']
        _missing = [c for c in _required if c not in DF_FOR_STATS.columns]
        assert not _missing, f"Missing required cols in DF_FOR_STATS: {_missing}"

        # training_stage may or may not be present depending on where you paste this cell.
        # If absent, we *do not* re-implement your early/late labeling here; we just build the session-level df without stage.
        _has_stage = ('training_stage' in DF_FOR_STATS.columns)

        _key = ['animal_ID','exp_moment']

        # 0) sanity: mouse reward should be constant within (mouse, session) across sims/repeats
        _mouse_const = DF_FOR_STATS.groupby(_key, dropna=False)['reward_rate_mouse'].nunique(dropna=False)
        assert (_mouse_const == 1).all(), f"reward_rate_mouse not constant within session for some keys:\n{_mouse_const[_mouse_const!=1].head(20)}"

        # 1) session-level aggregation: avg over sims+repeats -> one row per (mouse, session)
        mme_vs_sme = (DF_FOR_STATS
                     .groupby(_key, as_index=False, dropna=False)[['reward_rate_mouse','reward_rate_agent']]
                     .mean())

        # 2) attach training_stage if available (and enforce constancy within session)
        if _has_stage:
            _stage_const = DF_FOR_STATS.groupby(_key, dropna=False)['training_stage'].nunique(dropna=False)
            assert (_stage_const == 1).all(), f"training_stage not constant within session for some keys:\n{_stage_const[_stage_const!=1].head(20)}"
            _stage_map = (DF_FOR_STATS.groupby(_key, as_index=False, dropna=False)['training_stage'].first())
            mme_vs_sme = mme_vs_sme.merge(_stage_map, on=_key, how='left', validate='one_to_one')

        # 3) derive paired difference column (this is the estimand for paired inference)
        mme_vs_sme['diff_mouse_minus_sim'] = mme_vs_sme['reward_rate_mouse'] - mme_vs_sme['reward_rate_agent']

        print("\nMR: mme_vs_sme session-level df")
        print("  n rows (mouse×session):", len(mme_vs_sme))
        print("  n mice:", mme_vs_sme['animal_ID'].nunique(dropna=False))
        print("  n sessions:", mme_vs_sme['exp_moment'].nunique(dropna=False))
        print("  columns:", list(mme_vs_sme.columns))

        ## A) Compare three inferential choices on the SAME session-level rows

        ## A1) Naive session-weighted paired test (treats sessions as independent units)
        stats_session_naive = get_paired_stats(mme_vs_sme['reward_rate_mouse'].to_numpy(), mme_vs_sme['reward_rate_agent'].to_numpy(), test=STATS_TEST, alternative='two-sided')

        ## A2) Mouse-weighted paired test (each mouse contributes one mean; "across mice" estimand)
        _per_mouse = mme_vs_sme.groupby('animal_ID', as_index=False, dropna=False)[['reward_rate_mouse','reward_rate_agent']].mean()
        stats_mouse_weighted = get_paired_stats(_per_mouse['reward_rate_mouse'].to_numpy(), _per_mouse['reward_rate_agent'].to_numpy(), test=STATS_TEST, alternative='two-sided')

        print("\nMR: SAME DATA, different weighting assumptions")
        print("  session-weighted naive paired t (units = sessions):")
        print("   ", {k:stats_session_naive[k] for k in ['n','mean_diff','p_val','effect_name','effect_size'] if k in stats_session_naive})
        print("  mouse-weighted paired t (units = mice):")
        print("   ", {k:stats_mouse_weighted[k] for k in ['n','mean_diff','p_val','effect_name','effect_size'] if k in stats_mouse_weighted})

        ## A3) Cluster-aware session-level model (cluster-robust SE by mouse)
        ##     This directly addresses the "sessions aren't independent within mouse" issue.
        _cluster_out = None
        try:
            import statsmodels.formula.api as _smf
        except Exception as _e:
            print("\nMR warning: statsmodels not available; skipping cluster-robust OLS + MixedLM.")
            print(" ", _e)
        else:
            _sess_df = mme_vs_sme.copy()
            _sess_df['intercept'] = 1.0  # for explicitness/debug

            # cluster-robust OLS on diff (no stage here; this is just mouse-vs-sim average diff)
            _ols = _smf.ols('diff_mouse_minus_sim ~ 1', data=_sess_df).fit(
                cov_type='cluster', cov_kwds={'groups': _sess_df['animal_ID']}
            )
            _b = float(_ols.params.get('Intercept', np.nan))
            _p = float(_ols.pvalues.get('Intercept', np.nan))
            _ci = _ols.conf_int().loc['Intercept'].to_numpy() if 'Intercept' in _ols.params.index else [np.nan, np.nan]
            print("\nMR: cluster-robust OLS on diff (units = sessions, SE clustered by mouse)")
            print(f"  mean(diff_mouse_minus_sim)={_b:.6g}, p={_p:.6g}, 95% CI=({_ci[0]:.6g},{_ci[1]:.6g})")
            _cluster_out = {'b':_b,'p':_p,'ci':_ci}

            ## If training_stage exists, estimate late-early at the session level with clustered SE
            if _has_stage:
                _stage_df = _sess_df[_sess_df['training_stage'].isin(['early','late'])].copy()
                assert len(_stage_df) > 0, "no early/late rows after filtering"
                _stage_df['stage_late'] = (_stage_df['training_stage'] == 'late').astype(int)

                _ols_stage = _smf.ols('diff_mouse_minus_sim ~ stage_late', data=_stage_df).fit(
                    cov_type='cluster', cov_kwds={'groups': _stage_df['animal_ID']}
                )
                _b = float(_ols_stage.params.get('stage_late', np.nan))
                _p = float(_ols_stage.pvalues.get('stage_late', np.nan))
                _ci = _ols_stage.conf_int().loc['stage_late'].to_numpy() if 'stage_late' in _ols_stage.params.index else [np.nan, np.nan]
                print("\nMR: cluster-robust stage effect on diff (late - early), session-level with SE clustered by mouse")
                print(f"  b(late-early)={_b:.6g}, p={_p:.6g}, 95% CI=({_ci[0]:.6g},{_ci[1]:.6g})")

                ## Optional mixed model (random intercept) for the same stage effect
                try:
                    _lme = _smf.mixedlm('diff_mouse_minus_sim ~ stage_late', data=_stage_df, groups=_stage_df['animal_ID']).fit(reml=False)
                    _b = float(_lme.params.get('stage_late', np.nan))
                    _p = float(_lme.pvalues.get('stage_late', np.nan))
                    _ci = _lme.conf_int().loc['stage_late'].to_numpy() if 'stage_late' in _lme.params.index else [np.nan, np.nan]
                    print("\nMR: MixedLM random-intercept stage effect on diff (late - early)")
                    print(f"  b(late-early)={_b:.6g}, p={_p:.6g}, 95% CI=({_ci[0]:.6g},{_ci[1]:.6g})")
                except Exception as _e:
                    print("\nMR note: MixedLM random-intercept failed (often fine for small n_mice); skipping.")
                    print(" ", _e)

        ## B) Plots that match the stats (all based on mme_vs_sme session-level rows)
        ##    This is the “miscommunication fix”: plot the SAME rows you test.

        def _p_txt(stats_dict):
            p = stats_dict.get('p_val_raw', stats_dict.get('p_val', np.nan))
            try:
                p = float(p)
            except Exception:
                return "p=NA"
            if not np.isfinite(p): return "p=NA"
            if p == 0.0: return f"p<{np.finfo(float).tiny:.1e}"
            return f"p={p:.1e}"

        _mv = mme_vs_sme['reward_rate_mouse'].to_numpy(dtype=float)
        _av = mme_vs_sme['reward_rate_agent'].to_numpy(dtype=float)
        _diff = (_mv - _av)

        # Plot 1: two marginal violins (DESCRIPTIVE ONLY) + explicit paired-diff CI in the annotation
        fig1, ax1 = plt.subplots(figsize=(5.5, 4.5))
        parts = ax1.violinplot([_mv, _av], positions=[0,1], showmeans=False, showextrema=False, widths=0.8)
        for pc in parts['bodies']:
            pc.set_alpha(VIOLIN_ALPHA)

        # m_mouse, ci_mouse = _mean_and_ci(_mv)
        # m_agent, ci_agent = _mean_and_ci(_av)
        # m_diff, ci_diff = _mean_and_ci(_diff)
        m_mouse, ci_mouse = mean_and_ci(_mv)
        m_agent, ci_agent = mean_and_ci(_av)
        m_diff, ci_diff = mean_and_ci(_diff)

        ax1.plot([0],[m_mouse], marker='o')
        ax1.plot([1],[m_agent], marker='o')
        ax1.vlines([0],[ci_mouse[0]],[ci_mouse[1]], linewidth=2)
        ax1.vlines([1],[ci_agent[0]],[ci_agent[1]], linewidth=2)

        ax1.set_xticks([0,1])
        ax1.set_xticklabels(["mouse","agent"])
        ax1.set_ylabel("reward rate")
        ax1.set_title("marginal distributions (descriptive)")

        # IMPORTANT: annotate with paired effect CI (the actual inferential target)
        txt = f"{_p_txt(stats_session_naive)}; Δ(mouse−agent)={m_diff:.3g} [{ci_diff[0]:.3g},{ci_diff[1]:.3g}]"
        y_top = max(ci_mouse[1], ci_agent[1])
        y_span = float(max(_mv.max(), _av.max()) - min(_mv.min(), _av.min()))
        y_span = y_span if y_span > 0 else 1.0
        y = y_top + 0.06*y_span
        h = 0.03*y_span
        ax1.plot([0,0,1,1],[y,y+h,y+h,y], linewidth=1.5)
        ax1.text(0.5, y+h, txt, ha='center', va='bottom')
        plt.tight_layout()

        # Plot 2: paired differences violin (this matches the paired test visually)
        fig2, ax2 = plt.subplots(figsize=(5.5, 4.5))
        parts = ax2.violinplot([_diff], positions=[0], showmeans=False, showextrema=False, widths=0.8)
        for pc in parts['bodies']:
            pc.set_alpha(VIOLIN_ALPHA)
        ax2.axhline(0, linestyle='--')
        ax2.plot([0],[m_diff], marker='o')
        ax2.vlines([0],[ci_diff[0]],[ci_diff[1]], linewidth=2)
        ax2.set_xticks([0])
        ax2.set_xticklabels(["mouse − agent"])
        ax2.set_ylabel("paired difference")
        ax2.set_title("paired differences (inferential target)")
        ax2.text(0, ci_diff[1], _p_txt(stats_session_naive), ha='center', va='bottom')
        plt.tight_layout()

        # Plot 3: show the pairing explicitly (each point-pair is one mouse×session row)
        fig3, ax3 = plt.subplots(figsize=(5.5, 4.5))
        _x0 = np.zeros_like(_mv)
        _x1 = np.ones_like(_av)
        for i in range(_mv.size):
            ax3.plot([0,1], [_mv[i], _av[i]], alpha=POINT_ALPHA)
        ax3.scatter(_x0, _mv, s=12)
        ax3.scatter(_x1, _av, s=12)
        ax3.set_xticks([0,1])
        ax3.set_xticklabels(["mouse","agent"])
        ax3.set_ylabel("reward rate")
        ax3.set_title("paired data shown explicitly (one line per session)")
        plt.tight_layout()

        plt.show()

        # Expose the session-level df for reuse (this is the "input I expect")
        mme_vs_sme
    return PLOT_VIOLINS_N_LINES, POINT_ALPHA, VIOLIN_ALPHA


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    same as above or not?
    """)
    return


@app.cell
def _(
    DF_FOR_STATS,
    PLOT_VIOLINS_N_LINES,
    POINT_ALPHA,
    STATS_TEST,
    VIOLIN_ALPHA,
    get_paired_stats,
    np,
    plt,
    ss,
):
    def _():

        if PLOT_VIOLINS_N_LINES:
            # --- build a session-level paired df WITH training_stage, then compare weightings + plot consistently ---
            # Expected in scope: DF_FOR_STATS, get_paired_stats, np, pd, plt, ss

            assert 'DF_FOR_STATS' in globals()
            assert 'get_paired_stats' in globals()
            for _name in ['np','pd','plt','ss']:
                assert _name in globals(), f"expected {_name} in scope"

            _required = ['animal_ID','exp_moment','reward_rate_mouse','reward_rate_agent']
            _missing = [c for c in _required if c not in DF_FOR_STATS.columns]
            assert not _missing, f"Missing required cols in DF_FOR_STATS: {_missing}"

            # training_stage may or may not be present depending on where you paste this cell.
            # If absent, we *do not* re-implement your early/late labeling here; we just build the session-level df without stage.
            _has_stage = ('training_stage' in DF_FOR_STATS.columns)

            _key = ['animal_ID','exp_moment']

            # 0) sanity: mouse reward should be constant within (mouse, session) across sims/repeats
            _mouse_const = DF_FOR_STATS.groupby(_key, dropna=False)['reward_rate_mouse'].nunique(dropna=False)
            assert (_mouse_const == 1).all(), f"reward_rate_mouse not constant within session for some keys:\n{_mouse_const[_mouse_const!=1].head(20)}"

            # 1) session-level aggregation: avg over sims+repeats -> one row per (mouse, session)
            mme_vs_sme = (DF_FOR_STATS
                         .groupby(_key, as_index=False, dropna=False)[['reward_rate_mouse','reward_rate_agent']]
                         .mean())

            # 2) attach training_stage if available (and enforce constancy within session)
            if _has_stage:
                _stage_const = DF_FOR_STATS.groupby(_key, dropna=False)['training_stage'].nunique(dropna=False)
                assert (_stage_const == 1).all(), f"training_stage not constant within session for some keys:\n{_stage_const[_stage_const!=1].head(20)}"
                _stage_map = (DF_FOR_STATS.groupby(_key, as_index=False, dropna=False)['training_stage'].first())
                mme_vs_sme = mme_vs_sme.merge(_stage_map, on=_key, how='left', validate='one_to_one')

            # 3) derive paired difference column (this is the estimand for paired inference)
            mme_vs_sme['diff_mouse_minus_sim'] = mme_vs_sme['reward_rate_mouse'] - mme_vs_sme['reward_rate_agent']

            print("\nMR: mme_vs_sme session-level df")
            print("  n rows (mouse×session):", len(mme_vs_sme))
            print("  n mice:", mme_vs_sme['animal_ID'].nunique(dropna=False))
            print("  n sessions:", mme_vs_sme['exp_moment'].nunique(dropna=False))
            print("  columns:", list(mme_vs_sme.columns))


            """ A) Compare three inferential choices on the SAME session-level rows """
            ## A1) Naive session-weighted paired test (treats sessions as independent units)
            stats_session_naive = get_paired_stats(mme_vs_sme['reward_rate_mouse'].to_numpy(), mme_vs_sme['reward_rate_agent'].to_numpy(),
                                                   test=STATS_TEST, alternative='two-sided')

            ## A2) Mouse-weighted paired test (each mouse contributes one mean; "across mice" estimand)
            _per_mouse = (mme_vs_sme.groupby('animal_ID', as_index=False, dropna=False)[['reward_rate_mouse','reward_rate_agent']].mean())

            stats_mouse_weighted = get_paired_stats(_per_mouse['reward_rate_mouse'].to_numpy(), _per_mouse['reward_rate_agent'].to_numpy(),
                                                    test=STATS_TEST, alternative='two-sided')

            print("\nMR: SAME DATA, different weighting assumptions")
            print("  session-weighted naive paired t (units = sessions):")
            print("   ", {k:stats_session_naive[k] for k in ['n','mean_diff','p_val','effect_name','effect_size'] if k in stats_session_naive})
            print("  mouse-weighted paired t (units = mice):")
            print("   ", {k:stats_mouse_weighted[k] for k in ['n','mean_diff','p_val','effect_name','effect_size'] if k in stats_mouse_weighted})

            # A3) Cluster-aware session-level model (cluster-robust SE by mouse)
            #     This directly addresses the "sessions aren't independent within mouse" issue.
            _cluster_out = None
            try:
                import statsmodels.formula.api as _smf
            except Exception as _e:
                print("\nMR warning: statsmodels not available; skipping cluster-robust OLS + MixedLM.")
                print(" ", _e)
            else:
                _sess_df = mme_vs_sme.copy()
                _sess_df['intercept'] = 1.0  # for explicitness/debug

                # cluster-robust OLS on diff (no stage here; this is just mouse-vs-sim average diff)
                _ols = _smf.ols('diff_mouse_minus_sim ~ 1', data=_sess_df).fit(
                    cov_type='cluster', cov_kwds={'groups': _sess_df['animal_ID']}
                )
                _b = float(_ols.params.get('Intercept', np.nan))
                _p = float(_ols.pvalues.get('Intercept', np.nan))
                _ci = _ols.conf_int().loc['Intercept'].to_numpy() if 'Intercept' in _ols.params.index else [np.nan, np.nan]
                print("\nMR: cluster-robust OLS on diff (units = sessions, SE clustered by mouse)")
                print(f"  mean(diff_mouse_minus_sim)={_b:.6g}, p={_p:.6g}, 95% CI=({_ci[0]:.6g},{_ci[1]:.6g})")
                _cluster_out = {'b':_b,'p':_p,'ci':_ci}

                # If training_stage exists, estimate late-early at the session level with clustered SE
                if _has_stage:
                    _stage_df = _sess_df[_sess_df['training_stage'].isin(['early','late'])].copy()
                    assert len(_stage_df) > 0, "no early/late rows after filtering"
                    _stage_df['stage_late'] = (_stage_df['training_stage'] == 'late').astype(int)

                    _ols_stage = _smf.ols('diff_mouse_minus_sim ~ stage_late', data=_stage_df).fit(
                        cov_type='cluster', cov_kwds={'groups': _stage_df['animal_ID']}
                    )
                    _b = float(_ols_stage.params.get('stage_late', np.nan))
                    _p = float(_ols_stage.pvalues.get('stage_late', np.nan))
                    _ci = _ols_stage.conf_int().loc['stage_late'].to_numpy() if 'stage_late' in _ols_stage.params.index else [np.nan, np.nan]
                    print("\nMR: cluster-robust stage effect on diff (late - early), session-level with SE clustered by mouse")
                    print(f"  b(late-early)={_b:.6g}, p={_p:.6g}, 95% CI=({_ci[0]:.6g},{_ci[1]:.6g})")

                    # Optional mixed model (random intercept) for the same stage effect
                    try:
                        _lme = _smf.mixedlm('diff_mouse_minus_sim ~ stage_late', data=_stage_df, groups=_stage_df['animal_ID']).fit(reml=False)
                        _b = float(_lme.params.get('stage_late', np.nan))
                        _p = float(_lme.pvalues.get('stage_late', np.nan))
                        _ci = _lme.conf_int().loc['stage_late'].to_numpy() if 'stage_late' in _lme.params.index else [np.nan, np.nan]
                        print("\nMR: MixedLM random-intercept stage effect on diff (late - early)")
                        print(f"  b(late-early)={_b:.6g}, p={_p:.6g}, 95% CI=({_ci[0]:.6g},{_ci[1]:.6g})")
                    except Exception as _e:
                        print("\nMR note: MixedLM random-intercept failed (often fine for small n_mice); skipping.")
                        print(" ", _e)


            """ B) Plots that match the stats (all based on mme_vs_sme session-level rows) """
            def _mean_and_ci(x, conf=0.95):
                x = np.asarray(x, dtype=float).ravel()
                assert x.size > 1 and np.all(np.isfinite(x))
                m = float(x.mean())
                se = float(x.std(ddof=1) / np.sqrt(x.size))
                tcrit = float(ss.t.ppf(0.5 + conf/2.0, x.size - 1))
                half = tcrit * se
                return m, (m-half, m+half)

            def _p_txt(stats_dict):
                p = stats_dict.get('p_val_raw', stats_dict.get('p_val', np.nan))
                try:
                    p = float(p)
                except Exception:
                    return "p=NA"
                if not np.isfinite(p): return "p=NA"
                if p == 0.0: return f"p<{np.finfo(float).tiny:.1e}"
                return f"p={p:.1e}"

            _mv = mme_vs_sme['reward_rate_mouse'].to_numpy(dtype=float)
            _av = mme_vs_sme['reward_rate_agent'].to_numpy(dtype=float)
            _diff = (_mv - _av)

            # Plot 1: two marginal violins (DESCRIPTIVE ONLY) + explicit paired-diff CI in the annotation
            fig1, ax1 = plt.subplots(figsize=(5.5, 4.5))
            parts = ax1.violinplot([_mv, _av], positions=[0,1], showmeans=False, showextrema=False, widths=0.8)
            for pc in parts['bodies']:
                pc.set_alpha(VIOLIN_ALPHA)

            m_mouse, ci_mouse = _mean_and_ci(_mv)
            m_agent, ci_agent = _mean_and_ci(_av)
            m_diff, ci_diff = _mean_and_ci(_diff)

            ax1.plot([0],[m_mouse], marker='o')
            ax1.plot([1],[m_agent], marker='o')
            ax1.vlines([0],[ci_mouse[0]],[ci_mouse[1]], linewidth=2)
            ax1.vlines([1],[ci_agent[0]],[ci_agent[1]], linewidth=2)

            ax1.set_xticks([0,1])
            ax1.set_xticklabels(["mouse","agent"])
            ax1.set_ylabel("reward rate")
            ax1.set_title("marginal distributions (descriptive)")

            # IMPORTANT: annotate with paired effect CI (the actual inferential target)
            txt = f"{_p_txt(stats_session_naive)}; Δ(mouse−agent)={m_diff:.3g} [{ci_diff[0]:.3g},{ci_diff[1]:.3g}]"
            y_top = max(ci_mouse[1], ci_agent[1])
            y_span = float(max(_mv.max(), _av.max()) - min(_mv.min(), _av.min()))
            y_span = y_span if y_span > 0 else 1.0
            y = y_top + 0.06*y_span
            h = 0.03*y_span
            ax1.plot([0,0,1,1],[y,y+h,y+h,y], linewidth=1.5)
            ax1.text(0.5, y+h, txt, ha='center', va='bottom')
            plt.tight_layout()

            # Plot 2: paired differences violin (this matches the paired test visually)
            fig2, ax2 = plt.subplots(figsize=(5.5, 4.5))
            parts = ax2.violinplot([_diff], positions=[0], showmeans=False, showextrema=False, widths=0.8)
            for pc in parts['bodies']:
                pc.set_alpha(VIOLIN_ALPHA)
            ax2.axhline(0, linestyle='--')
            ax2.plot([0],[m_diff], marker='o')
            ax2.vlines([0],[ci_diff[0]],[ci_diff[1]], linewidth=2)
            ax2.set_xticks([0])
            ax2.set_xticklabels(["mouse − agent"])
            ax2.set_ylabel("paired difference")
            ax2.set_title("paired differences (inferential target)")
            ax2.text(0, ci_diff[1], _p_txt(stats_session_naive), ha='center', va='bottom')
            plt.tight_layout()

            # Plot 3: show the pairing explicitly (each point-pair is one mouse×session row)
            fig3, ax3 = plt.subplots(figsize=(5.5, 4.5))
            _x0 = np.zeros_like(_mv)
            _x1 = np.ones_like(_av)
            for i in range(_mv.size):
                ax3.plot([0,1], [_mv[i], _av[i]], alpha=POINT_ALPHA)
            ax3.scatter(_x0, _mv, s=12)
            ax3.scatter(_x1, _av, s=12)
            ax3.set_xticks([0,1])
            ax3.set_xticklabels(["mouse","agent"])
            ax3.set_ylabel("reward rate")
            ax3.set_title("paired data shown explicitly (one line per session)")
            plt.tight_layout()
            return plt.show()

            # Expose the session-level df for reuse (this is the "input I expect")
            # return mme_vs_sme

    _()
    return


@app.cell
def _():
    return


@app.cell
def _():
    BRACKET_COLOR   = "black"
    COLOR_MOUSE     = "black"
    COLOR_AGENT     = "0.6"    # gray
    Y_OFFSET        = 0.018     # relative offset above tallest CI cap
    H_SCALE         = 0.70     # shorten verticals ~30%
    TEXT_PAD        = 0.002     # extra bump above bracket
    YMAX_FIXED      = 0.5      # hard cap

    # --- Font size user parameters ---
    TITLE_FONTSIZE  = 16
    XLABEL_FONTSIZE = 14
    YLABEL_FONTSIZE = 15
    XTICK_FONTSIZE  = 12
    XTICK_FONTSIZE_LEFT = 14
    YTICK_FONTSIZE  = 12
    # LEGEND_FONTSIZE = 12
    ANNOT_FONTSIZE  = 12
    return (
        ANNOT_FONTSIZE,
        COLOR_MOUSE,
        TITLE_FONTSIZE,
        YLABEL_FONTSIZE,
        YMAX_FIXED,
        YTICK_FONTSIZE,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    PLOT: all mice vs all sims (2 bars)
    """)
    return


@app.cell
def _(
    Mme_vs_Sme,
    all_mice_vs_all_sims_stats_variants,
    combined_barplots_brackets_stars,
    get_now_str,
    os,
    plt,
):
    PLOT_2_BAR_SUMMARY = True

    if PLOT_2_BAR_SUMMARY:
        _stats_all = all_mice_vs_all_sims_stats_variants['Mme_vs_Sme-parametric']

        _fig, ax_all = combined_barplots_brackets_stars(Mme_vs_Sme, _stats_all, mouse_col="reward_rate_mouse", agent_col="reward_rate_agent", ylabel="reward rate (n rewards / n actions)")

        os.makedirs("figures", exist_ok=True)
        _fig.savefig(f"figures/{get_now_str()}_figS1_reward_rate_mice_vs_agents-leftOnly.png", dpi=500, bbox_inches="tight", pad_inches=0.02)
        _fig.savefig(f"figures/{get_now_str()}_figS1_reward_rate_mice_vs_agents-leftOnly.pdf", bbox_inches="tight", pad_inches=0.02)  # dpi ignored for vector-only
        _fig.savefig(f"figures/{get_now_str()}_figS1_reward_rate_mice_vs_agents-leftOnly.svg", bbox_inches="tight", pad_inches=0.02)
        plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # PLOT: all mice vs all baseline models (left panel); individual mice vs yoked sims (right panel)
    """)
    return


@app.cell
def _(
    Mme_vs_Sme,
    all_mice_vs_all_sims_stats_variants,
    combined_barplots_brackets_stars_2panel,
    get_now_str,
    os,
    plt,
):
    PLOT_ALL_VS_ALL_SIMS = True

    if PLOT_ALL_VS_ALL_SIMS:
        _stats_all = all_mice_vs_all_sims_stats_variants['Mme_vs_Sme-parametric']

        _fig, (_ax_all, _ax_by) = combined_barplots_brackets_stars_2panel(
            # avg_model_perf_per_session_df, stats_all, # liberal way
            Mme_vs_Sme, _stats_all, # conservative way
            mouse_col="reward_rate_mouse",
            agent_col="reward_rate_agent",
            group_col="animal_ID",
            ylabel="reward rate (n rewards / n actions)")

        os.makedirs("figures", exist_ok=True)

        _fig.savefig(f"figures/{get_now_str()}_fig1_reward_rate_mice_vs_agents.png",
                    dpi=500, bbox_inches="tight", pad_inches=0.02)
        _fig.savefig(f"figures/{get_now_str()}_fig1_reward_rate_mice_vs_agents.pdf",
                    bbox_inches="tight", pad_inches=0.02)  # dpi ignored for vector-only
        _fig.savefig(f"figures/{get_now_str()}_fig1_reward_rate_mice_vs_agents.svg",
                    bbox_inches="tight", pad_inches=0.02)

        plt.show()
    return


@app.cell
def _(Mme_vs_Smes):
    Mme_vs_Smes
    return


@app.cell
def _(mo):
    mo.md(r"""
    # PLOT: mouse vs sim variants
    """)
    return


@app.cell
def _(stats_per_mouse_x_sim_df):
    stats_per_mouse_x_sim_df
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## mouse vs RL agents
    """)
    return


@app.cell
def _(stats_per_mouse_x_sim_df):
    stats_per_mouse_x_sim_df.full_sim_desc.value_counts(sort=False)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    251228 Author TODO:
        - wrong policy_class for RL models
        - missing animals in DF_FOR_STATS
    """)
    return


@app.cell
def _(DF_FOR_STATS):
    DF_FOR_STATS.animal_ID.value_counts()
    return


@app.cell
def _(DF_FOR_STATS):
    # DF_FOR_STATS.groupby(['animal_ID', 'policy_class', 'agent_model', 'obs_type', 'action_type', 'full_sim_desc']).size()
    DF_FOR_STATS.groupby(['full_sim_desc', 'policy_class', 'agent_model', 'obs_type', 'action_type']).size()
    return


@app.cell
def _(DF_FOR_STATS):
    DF_FOR_STATS.policy_class.value_counts()
    return


@app.cell
def _(DF_FOR_STATS):
    DF_FOR_STATS.agent_model.value_counts()
    return


@app.cell
def _(stats_per_mouse_x_sim_df):
    stats_per_mouse_x_sim_df
    return


@app.cell
def _():
    "xtick_label","legend_group","color","model_order","legend_order","legend_label"
    return


@app.cell
def _():
    # stats_per_mouse_x_sim_df.full_sim_desc
    return


@app.cell
def _(stats_per_mouse_x_sim_df):
    stats_per_mouse_x_sim_df.index
    return


@app.cell
def _(mo):
    mo.md(r"""
    251225 Author: rather complicated plot... trying to figure out how to customize to add the RL agents..
    - key_to_group
    - GROUP_ORDER: passed to plot func
    - COLOR_MAP: generated from {m: GROUP_PALETTE.get(key_to_group(m), DEFAULT_COLOR) for m in ORDERED_MODELS}
    - model2color: generated from base_colors and COLOR_MAP
    - GROUP_PALETTE: actual color maps
    """)
    return


@app.cell
def _(stats_per_mouse_x_sim_df):
    stats_per_mouse_x_sim_df
    return


@app.cell
def _(pd):
    def key_to_group(k: str) -> str:
        # group == policy (first token), e.g. 'action biased', 'random', ...
        return k.split(';', 1)[0].replace('_', ' ').strip()

    def ensure_column_df(df, model_col="full_sim_desc"):
        if isinstance(df.index, pd.MultiIndex) or (df.index.name == model_col):
            df = df.reset_index()
        return df

    def rename_models(stats_by_mouse_model, model_col="full_sim_desc"): # 251225 Author: note that these strings get changed again by tidy_axis_label() later
        # model_keys = get_model_strings(stats_by_mouse_model, model_col="full_sim_desc")
        # model_keys = [str(m) for m in df.reset_index()[model_col].unique().tolist()]
        model_keys = [str(m) for m in stats_by_mouse_model[model_col].unique().tolist()] # 251225 Author: trying to simplify

        displayed_model_names = []
        for m in model_keys:
            print(f"raw string in full_sim_desc: {m}")

            # m_out = 'test - uncomment and resume debugging if all models affected'

            if 'nodes_adj' in m:
                m_out = 'RL: nodes'
            if 'obs_allo_latent' in m:
                m_out = 'RL: room allocentric'
            if 'obs_allo_real' in m:
                m_out = 'RL: virtual allocentric'
            else:

                ## hack it into a more readable form
                m_parse = m.replace("avoidReversal-False", '').replace(";", " ").replace("seqLength-", "len ").replace('avoidReversal-True', '(avoid back)')
                m_parse = m_parse.replace("  ", " ") # remove double spaces
                if 'action_biased' not in m_parse or 'observation_action_biased' in m_parse:
                    m_parse = m_parse.replace('len 1', '')

                if 'latent_state_biased' in m_parse:
                    m_out = 'Policy: latent state biased'
                else:
                    policy = m_parse.split(' ')[0].replace('_', ' ').strip()
                    rep = m_parse.split(' ')[1].replace('_', ' ').strip()
                    extra = m_parse.split(' ')[2:] if len(m_parse.split(' ')) > 2 else []
                    extra_str = ' '.join(extra) if extra else ''
                    if extra_str != '':
                        m_out = f"Policy: {policy} {extra_str.strip()} - Rep.: {rep}"
                    else:
                        m_out = f"Policy: {policy} - Rep.: {rep}"

            print(f"reformatted string via rename_models(): {m_out}"); print()
            displayed_model_names.append(m_out.strip())

        return displayed_model_names, model_keys
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### color assignments
    """)
    return


@app.cell
def _():
    GROUP_PALETTE = {
        "random":                    "#377eb8",  # bold blue
        "action biased":             "#e41a1c",  # bold red
        "observation action biased": "#4daf4a",  # bold green
        "latent state biased":       "#984ea3",  # bold purple
        "rwd dist oracle": 'gray',
        "A2C":                      "#ff7f00",  # bold orange
        "PPO":                       "olive",  
        "DQN":                       "orange"}

    # _base = plt.rcParams['axes.prop_cycle'].by_key()['color']
    # GROUP_PALETTE = {'random': _base[0], 'action biased': _base[1], 'observation action biased': _base[2], 'latent state biased': _base[3],}

    DEFAULT_COLOR = "black"  # bold orange fallback (if needed)
    MODEL_COLOR_GROUPS_N_PLOT_ORDER = ['random', 'action biased', 'observation action biased', 'latent state biased', 'rwd dist oracle', 'A2C', 'PPO', 'DQN']
    return DEFAULT_COLOR, GROUP_PALETTE, MODEL_COLOR_GROUPS_N_PLOT_ORDER


@app.cell
def _(
    ANNOT_FONTSIZE,
    COLOR_MOUSE,
    LEGEND_FONTSIZE,
    Line2D,
    Patch,
    TITLE_FONTSIZE,
    YLABEL_FONTSIZE,
    YMAX_FIXED,
    YTICK_FONTSIZE,
    mean_and_ci,
    np,
    pd,
    plt,
    prepare_table,
    significance2stars,
):
    # 2512xx Author: MODEL_FORMATTING_DF drives ALL per-model formatting (labels/colors/grouping/order)
    def plot_mouse_vs_each_model_all_sessions(stats_by_mouse_model, model_col="full_sim_desc", group_col="animal_ID",
        MODEL_FORMATTING_DF=None, PVAL_COL="p_val", MOUSE_FILTER=None, FIGSIZE=(11, 6), YLIM=(0.0, YMAX_FIXED),
        TITLE="Individual mouse vs simulations",
        y_label="reward rate",
        numerator_str="n rewards",
        denominator_str="n actions",
        SHOW_BRACKETS=False,
        TOP_K_BRACKETS=None,
        SHOW_WHISKERS=True,
        DF_LONG=None,
        MOUSE_COL="reward_rate_mouse",
        AGENT_COL="reward_rate_agent",
        WHISKER_CONF=0.95,
        WHISKER_CAPSIZE=4,
        GROUP_ORDER=None,
        LEGEND_FRAME=True,
        MOUSE_WHISKER_DEDUP_COL="exp_moment",
    ):
        """ Refactor goals:
              - ALL model-specific formatting is controlled by MODEL_FORMATTING_DF (no LABEL_MAP/COLOR_MAP wrappers)
              - y_label is composed explicitly: f"{y_label} ({numerator_str} / {denominator_str})"
              - DF_LONG usage clarified:
                  * If SHOW_WHISKERS=True, DF_LONG must provide per-sample values to compute CI whiskers.
                  * Required DF_LONG columns: {group_col, model_col, MOUSE_COL, AGENT_COL}
                  * Optional de-dup column for mouse whiskers: MOUSE_WHISKER_DEDUP_COL

            Convenience fallback: if DF_LONG is None but stats_by_mouse_model itself is already long-format
            (contains required columns), we will use it as DF_LONG. """

        """ validate model formatting df """
        assert MODEL_FORMATTING_DF is not None, "MODEL_FORMATTING_DF is required (explicit per-model formatting)"
        assert isinstance(MODEL_FORMATTING_DF, pd.DataFrame), "MODEL_FORMATTING_DF must be a DataFrame"
        assert MODEL_FORMATTING_DF.index.name == "model", "MODEL_FORMATTING_DF.index.name must be 'model'"
        missing = {"xtick_label", "legend_group", "color"}.difference(MODEL_FORMATTING_DF.columns)
        assert not missing, f"MODEL_FORMATTING_DF missing required columns: {sorted(missing)}"
        ## Enforce explicitness: no NaNs for required per-model formatting fields
        for c in ["xtick_label", "legend_group", "color"]:
            bad = MODEL_FORMATTING_DF[c].isna()
            assert not bad.any(), f"MODEL_FORMATTING_DF has NaN in required column '{c}' for models: {MODEL_FORMATTING_DF.index[bad].tolist()}"


        df = prepare_table(stats_by_mouse_model, model_col=model_col, group_col=group_col)

        # Normalize keys to strings for robust matching
        df[model_col] = df[model_col].astype(str)
        df[group_col] = df[group_col].astype(str)

        # ensure we’re actually plotting the intended p’s (raw or FDR)
        assert PVAL_COL in df.columns, f"{PVAL_COL} not in stats table"
        assert np.isfinite(df[PVAL_COL]).all(), f"NaNs in {PVAL_COL}"

        if MOUSE_FILTER is not None:
            keep = [str(x) for x in (MOUSE_FILTER if isinstance(MOUSE_FILTER, (list, tuple, set, np.ndarray)) else [MOUSE_FILTER])]
            df = df[df[group_col].astype(str).isin(keep)]
            assert len(df) > 0, f"No rows left after filtering to {MOUSE_FILTER!r}"

        present_models = list(dict.fromkeys(df[model_col].tolist()))
        present_models_set = set(present_models)

        ## formatting df index normalized to str for matching
        fmt = MODEL_FORMATTING_DF.copy()
        fmt.index = fmt.index.astype(str)

        ## enforce explicit coverage for all models present in data
        missing_fmt = [m for m in present_models if m not in fmt.index]
        assert not missing_fmt, ("MODEL_FORMATTING_DF is missing rows for models present in df:\n"
                                 f"{missing_fmt}\n\n"
                                 "Add these models as index entries (index name must be 'model') and provide at least:\n"
                                 "  xtick_label, legend_group, color")

        # apply include filter (default True if absent)
        if "include" in fmt.columns:
            include_models = fmt.index[fmt["include"].fillna(True).astype(bool)]
        else:
            include_models = fmt.index

        models = [m for m in present_models if (m in include_models)]

        # order by explicit model_order if provided, else keep present order
        if "model_order" in fmt.columns:
            ord_ser = fmt.loc[models, "model_order"]
            assert np.isfinite(ord_ser.astype(float)).all(), (
                "MODEL_FORMATTING_DF['model_order'] must be finite for plotted models. "
                f"Bad models: {ord_ser[~np.isfinite(ord_ser.astype(float, errors='ignore'))].index.tolist()}"
            )
            models = list(ord_ser.astype(float).sort_values(kind="mergesort").index)

        # groups (mice)
        groups_present = [str(g) for g in df[group_col].unique()]
        if GROUP_ORDER:
            groups = [str(g) for g in GROUP_ORDER if str(g) in groups_present]
        else:
            groups = [str(g) for g in sorted(groups_present, key=lambda x: str(x))]

        # fig2 assumes a single mouse group
        assert len(groups) == 1, f"Author: expected single mouse group for fig2 plot but found {len(groups)}: {groups}"

        # map formatting
        model2color = fmt.loc[models, "color"].to_dict()
        model2xtick = fmt.loc[models, "xtick_label"].to_dict()
        model2leggrp = fmt.loc[models, "legend_group"].to_dict()

        # legend_label handling (optional)
        if "legend_label" in fmt.columns:
            model2leglbl = fmt.loc[models, "legend_label"].to_dict()
        else:
            model2leglbl = {m: model2leggrp[m] for m in models}

        # ----------------- means -----------------
        mouse_means = []
        agent_means = {m: [] for m in models}
        for gid in groups:
            sub = df[df[group_col] == gid]
            mouse_means.append(float(sub["mouse_mean"].iloc[0]))
            for m in models:
                row = sub[sub[model_col] == m]
                agent_means[m].append(np.nan if row.empty else float(row["model_mean"].iloc[0]))

        mouse_means = np.array(mouse_means, float)
        for m in models:
            agent_means[m] = np.array(agent_means[m], float)

        ## ----------------- whiskers (CI) -----------------
        have_whiskers = False
        if SHOW_WHISKERS:
            # GPT TODO: compute DF_LONG internally if possible -> best-effort fallback:
            if DF_LONG is None and isinstance(stats_by_mouse_model, pd.DataFrame):
                need_long = {group_col, model_col, MOUSE_COL, AGENT_COL}
                if need_long.issubset(set(stats_by_mouse_model.columns)):
                    DF_LONG = stats_by_mouse_model

            if DF_LONG is not None:
                need = {group_col, model_col, MOUSE_COL, AGENT_COL}
                dfl = DF_LONG
                if (isinstance(dfl.index, pd.MultiIndex) or (getattr(dfl.index, "name", None) in (group_col, model_col))):
                    dfl = dfl.reset_index()

                assert need.issubset(dfl.columns), ("DF_LONG missing required columns for whiskers.\n"
                                                    f"Need: {sorted(need)}\nHave: {sorted(dfl.columns.tolist())}")

                mouse_ci_lo, mouse_ci_hi = [], []
                for gid in groups:
                    g_long = dfl[dfl[group_col].astype(str) == str(gid)]
                    if (MOUSE_WHISKER_DEDUP_COL is not None) and (MOUSE_WHISKER_DEDUP_COL in g_long.columns):
                        vals = (g_long[[MOUSE_WHISKER_DEDUP_COL, MOUSE_COL]].dropna()
                                .drop_duplicates(subset=[MOUSE_WHISKER_DEDUP_COL]))[MOUSE_COL].values
                    else:
                        vals = g_long[MOUSE_COL].dropna().values
                    _, (lo, hi) = mean_and_ci(vals, conf=WHISKER_CONF)
                    mouse_ci_lo.append(lo); mouse_ci_hi.append(hi)

                mouse_ci_lo = np.array(mouse_ci_lo, float)
                mouse_ci_hi = np.array(mouse_ci_hi, float)

                agent_ci_lo = {m: [] for m in models}
                agent_ci_hi = {m: [] for m in models}
                for gid in groups:
                    g_mouse = dfl[dfl[group_col].astype(str) == str(gid)]
                    for m in models:
                        gm = g_mouse[g_mouse[model_col].astype(str) == str(m)]
                        assert not gm.empty, f"Author: No data for group {gid}, model {m} in DF_LONG"
                        vals = gm[AGENT_COL].values
                        _, (lo, hi) = mean_and_ci(vals, conf=WHISKER_CONF)
                        agent_ci_lo[m].append(lo); agent_ci_hi[m].append(hi)

                for m in models:
                    agent_ci_lo[m] = np.array(agent_ci_lo[m], float)
                    agent_ci_hi[m] = np.array(agent_ci_hi[m], float)

                have_whiskers = True

        ## ----------------- plot -----------------
        if SHOW_BRACKETS:
            raise NotImplementedError("SHOW_BRACKETS/TOP_K_BRACKETS not implemented in this refactor (explicitly add if needed).")

        fig, ax = plt.subplots(figsize=FIGSIZE)
        x = np.arange(len(groups))
        w = 0.1

        ## mouse bar
        mouse_label = f"mouse: {groups[0]}"  # fig2 is single mouse
        mouse_centers = x - w * (len(models) / 2)

        if have_whiskers:
            yerr_mouse = np.vstack([np.clip(mouse_means - mouse_ci_lo, 0, np.inf), 
                                    np.clip(mouse_ci_hi - mouse_means, 0, np.inf)])

            ax.bar(mouse_centers, mouse_means, w, color=COLOR_MOUSE, label=mouse_label, 
                   yerr=yerr_mouse, capsize=WHISKER_CAPSIZE, ecolor=COLOR_MOUSE)
        else:
            ax.bar(mouse_centers, mouse_means, w, color=COLOR_MOUSE, label=mouse_label)

        # agent bars + sig annotations
        agent_centers = []
        sig_codes_present = set()

        for j, m in enumerate(models):
            offs = x - w * (len(models) / 2) + (j + 1) * w
            if have_whiskers:
                yerr = np.vstack([np.clip(agent_means[m] - agent_ci_lo[m], 0, np.inf),
                                  np.clip(agent_ci_hi[m] - agent_means[m], 0, np.inf)])

                ax.bar(offs, agent_means[m], w, color=model2color[m], label="_nolegend_",
                    yerr=yerr, capsize=WHISKER_CAPSIZE, ecolor=model2color[m])
            else:
                ax.bar(offs, agent_means[m], w, color=model2color[m], label="_nolegend_")

            agent_centers.append(offs)

            for i, gid in enumerate(groups):
                row = df.loc[(df[group_col] == gid) & (df[model_col] == m), PVAL_COL]
                assert len(row) == 1, f"Expected 1 row for ({gid}, {m}), found {len(row)}"
                p = float(row.iloc[0])
                sig = significance2stars(p)
                sig_codes_present.add(sig)

                y_top = agent_means[m][i]
                if have_whiskers and np.isfinite(agent_means[m][i]):
                    hi = agent_ci_hi[m][i]
                    if np.isfinite(hi):
                        y_top = max(y_top, hi)

                ax.text(offs[i], y_top + 0.012 * (YLIM[1] - YLIM[0]), sig, 
                        ha="center", va="bottom", fontsize=ANNOT_FONTSIZE)

        # x-axis ticks (fig2)
        centers = [mouse_centers[0]] + [arr[0] for arr in agent_centers]
        labels  = [mouse_label] + [model2xtick[m] for m in models]
        ax.set_xticks(centers)
        ax.set_xticklabels(labels, rotation=-30, ha="left", fontsize=XTICK_FONTSIZE_MOUSE_VS_SIMS)

        # trim whitespace at both ends
        all_centers = np.concatenate([mouse_centers] + agent_centers, axis=0)
        ax.set_xlim(np.nanmin(all_centers) - 0.55 * w, np.nanmax(all_centers) + 0.55 * w)

        # title / y label (GPT TODO)
        ax.set_title(TITLE.replace("mouse", f"mouse {groups[0]}"), fontsize=TITLE_FONTSIZE)
        ax.set_ylim(*YLIM)

        y_label_full = f"{y_label} ({numerator_str} / {denominator_str})"
        ax.set_ylabel(y_label_full, fontsize=YLABEL_FONTSIZE)
        ax.tick_params(axis="y", labelsize=YTICK_FONTSIZE)

        # ----------------- legend: mouse + collapsed legend groups + notation -----------------
        if ax.get_legend() is not None:
            ax.get_legend().remove()

        # build legend groups from MODEL_FORMATTING_DF (explicit grouping)
        # enforce: within a legend_group, all colors identical and legend_label identical
        grp_rows = []
        for m in models:
            grp_rows.append({"model": m, "legend_group": model2leggrp[m], "legend_label": model2leglbl[m], "color": model2color[m],
                "legend_order": float(fmt.loc[m, "legend_order"]) if "legend_order" in fmt.columns and pd.notna(fmt.loc[m, "legend_order"]) else np.nan})
        grp_df = pd.DataFrame(grp_rows)

        for g, sub in grp_df.groupby("legend_group", sort=False):
            colors = list(pd.unique(sub["color"]))
            labels_u = list(pd.unique(sub["legend_label"]))
            assert len(colors) == 1, f"Legend group '{g}' has multiple colors (must be explicit + consistent): {colors}"
            assert len(labels_u) == 1, f"Legend group '{g}' has multiple legend_label values (must be consistent): {labels_u}"

        # determine legend group order
        if GROUP_ORDER:
            legend_groups = [g for g in GROUP_ORDER if g in grp_df["legend_group"].unique()]
        elif "legend_order" in grp_df.columns and grp_df["legend_order"].notna().any():
            legend_groups = (grp_df.groupby("legend_group", sort=False)["legend_order"].min()
                .sort_values(kind="mergesort").index.tolist())
        else:
            # preserve first appearance
            seen = set()
            legend_groups = []
            for g in grp_df["legend_group"].tolist():
                if g not in seen:
                    seen.add(g)
                    legend_groups.append(g)

        handles = [Patch(facecolor=COLOR_MOUSE, edgecolor="none", label=mouse_label)]
        labels_legend = [mouse_label]

        for g in legend_groups:
            row0 = grp_df[grp_df["legend_group"] == g].iloc[0]
            handles.append(Patch(facecolor=row0["color"], edgecolor="none", label=row0["legend_label"]))
            labels_legend.append(row0["legend_label"])

        parts = []
        if "***" in sig_codes_present:
            parts.append("all mouse vs simulation comparisons Bonferroni-corrected for multiple comparisons \n*** p<0.001 (two-sided paired t-test)")
        if "*" in sig_codes_present:
            parts.append("* p<0.05")
        if "ns" in sig_codes_present:
            parts.append("ns p≥0.05")
        notation_txt = "; ".join(parts) if parts else "ns p≥0.05"

        handles.append(Line2D([], [], color="none", label=notation_txt))
        labels_legend.append(notation_txt)

        # ax.legend(handles, labels_legend, fontsize=LEGEND_FONTSIZE, frameon=LEGEND_FRAME, bbox_to_anchor=(1.05, 1), loc='lower right')
        ax.legend(handles, labels_legend, fontsize=LEGEND_FONTSIZE, frameon=LEGEND_FRAME, bbox_to_anchor=(1.05, 1))

        return fig, ax

    XTICK_FONTSIZE_MOUSE_VS_SIMS = 10
    return (plot_mouse_vs_each_model_all_sessions,)


@app.cell
def _(stats_per_mouse_x_sim_df):
    stats_per_mouse_x_sim_df.animal_ID.unique()
    return


@app.cell
def _(
    DEFAULT_COLOR,
    DF_FOR_STATS,
    GROUP_PALETTE,
    MODEL_COLOR_GROUPS_N_PLOT_ORDER,
    avg_model_perf_df,
    bonferroni,
    get_now_str,
    np,
    os,
    pd,
    plot_mouse_vs_each_model_all_sessions,
    plt,
    stats_per_mouse_x_sim_df,
):
    PLOT_EACH_MOUSE_VS_EACH_SIM = True

    if PLOT_EACH_MOUSE_VS_EACH_SIM: 

        stats_by_mouse_model = stats_per_mouse_x_sim_df.copy()  # ensure defined before parsing model keys

        ################

        # one-time normalization: force MultiIndex with named levels
        assert "animal_ID" in stats_by_mouse_model.columns and "full_sim_desc" in stats_by_mouse_model.columns,(
            "Author: cannot normalize stats_by_mouse_model: missing animal_ID/full_sim_desc columns. "
            f"cols={list(stats_by_mouse_model.columns)}")

        stats_by_mouse_model = stats_by_mouse_model.set_index(["animal_ID","full_sim_desc"],drop=True).sort_index()
        stats_by_mouse_model.index.set_names(["animal_ID","full_sim_desc"],inplace=True)

        # enforce canonical column names expected by utils_latMaz_stats_n_plotting.prepare_table
        rename_map={
            "mean_mouse":"mouse_mean",
            "mean_mice":"mouse_mean",
            "mouse_avg":"mouse_mean",
            "avg_mouse":"mouse_mean",

            "mean_sim":"model_mean",
            "mean_sims":"model_mean",
            "mean_model":"model_mean",
            "mean_agent":"model_mean",
            "agent_mean":"model_mean",
            "sim_mean":"model_mean",
            "avg_model":"model_mean",
            "avg_agent":"model_mean",
        }

        stats_by_mouse_model=stats_by_mouse_model.rename(columns=rename_map)

        # fail fast if renaming created ambiguous duplicates
        dups=stats_by_mouse_model.columns[stats_by_mouse_model.columns.duplicated()].tolist()
        assert not dups,f"Author: duplicate columns after rename -> {dups}"

        req=["mouse_mean","model_mean"]
        miss=[c for c in req if c not in stats_by_mouse_model.columns]
        assert not miss,f"Author: stats_by_mouse_model missing required columns for prepare_table: {miss}. cols={list(stats_by_mouse_model.columns)}"

        # ensure numeric (prevents downstream surprises)
        stats_by_mouse_model["mouse_mean"]=pd.to_numeric(stats_by_mouse_model["mouse_mean"],errors="raise")
        stats_by_mouse_model["model_mean"]=pd.to_numeric(stats_by_mouse_model["model_mean"],errors="raise")

        #################

        # Build ONE table that contains both “registry” fields + plot formatting fields.
        # Index = full_sim_desc (the model key used in stats_by_mouse_model).

        MODEL_COL = model_col = "full_sim_desc"

        idx = stats_by_mouse_model.index
        assert isinstance(idx,pd.MultiIndex),("Author: expected stats_by_mouse_model.index to be a MultiIndex. "
                                              f"Got type={type(idx)}, name={getattr(idx,'name',None)}, nlevels={getattr(idx,'nlevels',None)}. "
                                              "Fix upstream: set a MultiIndex that includes full_sim_desc.")

        assert (idx.names is not None) and (MODEL_COL in idx.names),("Author: expected stats_by_mouse_model.index.names to include 'full_sim_desc'. "
                                                                     f"Got index.names={idx.names}. "
                                                                     "Fix upstream: name the MultiIndex levels and include full_sim_desc.")

        # models actually present in plot input
        models=pd.Index(stats_by_mouse_model.index.get_level_values(model_col)).unique()
        # models=pd.Index(stats_by_mouse_model.index.get_level_values("full_sim_desc")).dropna().unique()

        models=[m for m in models.tolist() if pd.notna(m)]

        need=[model_col,"agent_model","policy_class","action_type","obs_type"]
        missing=[c for c in need if c not in DF_FOR_STATS.columns]
        assert not missing,f"Author: DF_FOR_STATS missing cols needed for MODEL_FORMATTING_DF: {missing}"

        fmt=(DF_FOR_STATS.loc[DF_FOR_STATS[model_col].isin(models),need].drop_duplicates().rename(columns={model_col:"model"}).copy())

        def _norm_key(x):
            if pd.isna(x): return ""
            return " ".join(str(x).lower().replace("_"," ").replace("-"," ").split())

        # prefer agent_model; fallback to token0 of full_sim_desc
        def _infer_base_model(row):
            am=row.get("agent_model",pd.NA)
            if pd.notna(am) and str(am) not in ("<NA>","nan","None",""):
                return str(am).strip()
            m=str(row.get("model",""))
            tok=m.split(";",1)[0].strip().replace("_"," ")
            return tok if tok else "unknown_model"

        fmt["base_model"]=fmt.apply(_infer_base_model,axis=1)
        fmt["base_model"]=(fmt["base_model"].astype("string")
                           .replace({"<NA>":pd.NA,"nan":pd.NA,"None":pd.NA,"":pd.NA})
                           .fillna("unknown_model"))

        # map normalized -> canonical palette key (handles underscores/spaces/case)
        palette_norm2key={_norm_key(k):k for k in GROUP_PALETTE.keys()}
        fmt["palette_key"]=fmt["base_model"].map(lambda s: palette_norm2key.get(_norm_key(s),str(s)))

        # parse seqLength-* from full_sim_desc (optional, but useful for ordering)
        def _parse_seq_len(m):
            if not isinstance(m,str): return np.nan
            for part in m.split(";"):
                if part.startswith("seqLength-"):
                    v=part.split("-",1)[1]
                    try: return int(v)
                    except Exception: return v
            return np.nan

        fmt["seq_len"]=fmt["model"].map(_parse_seq_len)

        # color
        fmt["color"]=fmt["palette_key"].map(GROUP_PALETTE)
        missing_color=sorted(fmt.loc[fmt["color"].isna(),"palette_key"].dropna().unique().tolist())
        if missing_color:
            print("Author: palette_key(s) missing from GROUP_PALETTE -> using DEFAULT_COLOR:",missing_color)
        fmt["color"]=fmt["color"].fillna(DEFAULT_COLOR)

        # labels (edit here if you want different labeling rules)
        action_str=fmt["action_type"].astype("string").fillna("")
        fmt["xtick_label"]=fmt["palette_key"]
        has_action=action_str.ne("")
        if has_action.any():
            fmt.loc[has_action,"xtick_label"]=fmt.loc[has_action].apply(lambda r:f"{r['palette_key']} ({r['action_type']})",axis=1)
        if fmt["seq_len"].notna().any():
            fmt["xtick_label"]=fmt["xtick_label"]+fmt["seq_len"].map(lambda x:f" L{x}" if pd.notna(x) else "")

        fmt["legend_group"]=fmt["palette_key"]
        fmt["legend_label"]=fmt["palette_key"]

        # ordering (group order from your list; then remaining palette keys; then unknowns)
        order=[]
        for k in MODEL_COLOR_GROUPS_N_PLOT_ORDER:
            kk=palette_norm2key.get(_norm_key(k),k)
            if kk not in order: order.append(kk)
        for k in GROUP_PALETTE.keys():
            if k not in order: order.append(k)

        order_map={k:i for i,k in enumerate(order)}
        fmt["base_order"]=fmt["palette_key"].map(order_map)

        unk_mask=fmt["base_order"].isna()
        if unk_mask.any():
            unknown=sorted(fmt.loc[unk_mask,"palette_key"].dropna().unique().tolist())
            start=len(order_map)
            unk_map={m:start+i for i,m in enumerate(unknown)}
            fmt.loc[unk_mask,"base_order"]=fmt.loc[unk_mask,"palette_key"].map(unk_map)
            print("Author: unknown palette_key(s) not in ordering list; appended to end:",unknown)

        fmt["base_order"]=pd.to_numeric(fmt["base_order"],errors="raise")

        # within-group order: use seq_len (if present) then model string for stability
        seq_sort=fmt["seq_len"].copy()
        seq_sort=seq_sort.where(pd.to_numeric(seq_sort,errors="coerce").notna(),np.inf)
        fmt["_seq_sort"]=pd.to_numeric(seq_sort,errors="coerce")

        fmt=fmt.sort_values(["base_order","palette_key","_seq_sort","model"],kind="mergesort").copy()
        fmt["within_order"]=fmt.groupby("palette_key",sort=False).cumcount()

        mult=int(fmt.groupby("palette_key",sort=False).size().max())+1
        fmt["legend_order"]=fmt["base_order"].astype(int)
        fmt["model_order"]=(fmt["base_order"]*mult+fmt["within_order"]).astype(int)

        # final: everything lives here
        MODEL_FORMATTING_DF=(fmt.drop(columns=["_seq_sort"])
            .set_index("model"))
        MODEL_FORMATTING_DF.index.name="model"

        # hard contract (plot expects these)
        required_cols=["xtick_label","legend_group","legend_label","color","model_order","legend_order"]
        for c in required_cols:
            assert c in MODEL_FORMATTING_DF.columns,f"Author: MODEL_FORMATTING_DF missing {c}"
            bad=MODEL_FORMATTING_DF[c].isna()
            assert not bad.any(),f"Author: MODEL_FORMATTING_DF has NaN in {c} for: {MODEL_FORMATTING_DF.index[bad].tolist()}"

        # optional: ensure full coverage
        missing_models=[m for m in models if m not in MODEL_FORMATTING_DF.index]
        assert not missing_models,f"Author: models missing from MODEL_FORMATTING_DF: {missing_models}"

        idx=stats_by_mouse_model.index
        assert isinstance(idx,pd.MultiIndex),f"Author: expected MultiIndex; got {type(idx)}"
        assert idx.names==["animal_ID",MODEL_COL],f"Author: expected index.names ['animal_ID','{MODEL_COL}']; got {idx.names}"
        assert MODEL_COL not in stats_by_mouse_model.columns,f"Author: expected '{MODEL_COL}' to be an index level, not a column"
        assert not idx.duplicated().any(),"Author: duplicate (animal_ID, full_sim_desc) rows"

        # Plot contract: MODEL_FORMATTING_DF must fully cover plotted models and have required fields
        present_models=pd.Index(idx.get_level_values(MODEL_COL)).dropna().unique().tolist()

        missing_models=[m for m in present_models if m not in MODEL_FORMATTING_DF.index]
        assert not missing_models,f"Author: models missing from MODEL_FORMATTING_DF: {missing_models}"

        required_fmt_cols=["xtick_label","legend_group","legend_label","color","model_order","legend_order"]
        missing_fmt_cols=[c for c in required_fmt_cols if c not in MODEL_FORMATTING_DF.columns]
        assert not missing_fmt_cols,f"Author: MODEL_FORMATTING_DF missing columns: {missing_fmt_cols}"

        for c in required_fmt_cols:
            bad=MODEL_FORMATTING_DF.loc[present_models,c].isna()
            assert not bad.any(),f"Author: MODEL_FORMATTING_DF has NaN in '{c}' for: {MODEL_FORMATTING_DF.loc[present_models].index[bad].tolist()}"

        # OPTIONAL: numeric sanity for p-values used in the plot annotations
        assert "p_val" in stats_by_mouse_model.columns,f"Author: expected p_val column; cols={list(stats_by_mouse_model.columns)}"
        p=pd.to_numeric(stats_by_mouse_model["p_val"],errors="raise")
        assert np.isfinite(p.to_numpy()).all(),"Author: NaN/Inf in p_val"

        """ 4) Bonferroni correction within mouse (across models)  """ 
        ## use raw paired t-test p's
        stats_by_mouse_model = stats_by_mouse_model.assign(p_use=stats_by_mouse_model["p_val"])
        ## check the upstream data format
        assert isinstance(stats_by_mouse_model.index, pd.MultiIndex) and 'animal_ID' in stats_by_mouse_model.index.names, \
            f"Author: need MultiIndex w/ level 'animal_ID'; got {type(stats_by_mouse_model.index).__name__} names={getattr(stats_by_mouse_model.index,'names',None)}"
        ## apply per mouse and keep index alignment
        stats_by_mouse_model['p_val_bonf'] = stats_by_mouse_model.groupby(level='animal_ID')['p_use'].transform(
            lambda s: bonferroni(s.to_numpy(dtype=float, copy=False)))

        """ finally generate the plots """
        os.makedirs("figures", exist_ok=True)

        for _animal_ID in stats_by_mouse_model.index.get_level_values("animal_ID").unique():

            _fig2, _ax2 = plot_mouse_vs_each_model_all_sessions(stats_by_mouse_model, MOUSE_FILTER=_animal_ID,
                PVAL_COL="p_val_bonf",     # ⟵ use Bonferroni-adjusted p-values or p_val_adj (p_val_bonf or p_val_adj)
                MODEL_FORMATTING_DF = MODEL_FORMATTING_DF,
                SHOW_WHISKERS=True, DF_LONG=avg_model_perf_df, MOUSE_COL="reward_rate_mouse", AGENT_COL="reward_rate_agent")

            _fig2.savefig(f"figures/{get_now_str()}_figS2_reward_rate_{_animal_ID}_vs_agents-bonf.png", dpi=500, bbox_inches="tight", pad_inches=0.02)
            _fig2.savefig(f"figures/{get_now_str()}_figS2_reward_rate_{_animal_ID}_vs_agents-bonf.pdf", bbox_inches="tight", pad_inches=0.02)  # dpi ignored for vector-only
            _fig2.savefig(f"figures/{get_now_str()}_figS2_reward_rate_{_animal_ID}_vs_agents-bonf.svg", bbox_inches="tight", pad_inches=0.02)

            plt.show()
    return (MODEL_FORMATTING_DF,)


@app.cell
def _():
    # stats_by_mouse_model
    return


@app.cell
def _():
    """ already moved to utils """

    ## ----------------- helpers -----------------
    # def significance2stars(p):
    #     # if not np.isfinite(p): return "ns"
    #     assert np.isfinite(p), "sig_code: non-finite p-value"
    #     if p < 1e-3: return "***"
    #     if p < 1e-2: return "**"
    #     if p < 0.05: return "*"
    #     return "ns"
    return


@app.cell
def _():
    """ Keep Author/Author3 discussion for a reference for a bit longer. See older versions of the code before 251219 for a more complete discussion (but probably not worth it) """

    # """ neurIPS: each mouse vs all (yoked/matched) sims """ 
    # #### option 1 (Author3: don't do it): original naive approach --> yields tiny p-values due to large n
    # # stats_by_mouse = df_for_stats.groupby("animal_ID").apply(lambda g: pd.Series(get_paired_stats(g["reward_rate_mouse"], g["reward_rate_agent"], test=STATS_TEST)), include_groups=False)

    # #### option 2 (Author3: don't do it): average over all sim repeats then average over each experiment (for each mouse), then compare each mice to its sims
    # # stats_by_mouse = avg_mice_vs_sims_all_sessions.groupby(['animal_ID']).apply(lambda g: pd.Series(get_paired_stats(g["reward_rate_mouse"], g["reward_rate_agent"], test=STATS_TEST)), include_groups=False)

    # # #### option 3 (Author3): further averaging 
    # avg_mice_session_vs_all_sims = avg_sim_perf_per_session.groupby(['animal_ID', 'exp_moment'], as_index=False)[['reward_rate_mouse', 'reward_rate_agent']].mean() # already avg'd over repeats
    # assert avg_mice_session_vs_all_sims.equals(avg_over_sims_per_session)
    # stats_by_mouse = avg_mice_session_vs_all_sims.groupby(['animal_ID']).apply(lambda g: pd.Series(get_paired_stats(g["reward_rate_mouse"], g["reward_rate_agent"], test=STATS_TEST)), include_groups=False)

    # """ each mouse vs each sim type (full_sim_desc) over ALL sessions """ 
    # #### option 1: avg over both repeats and sessions? Author3: suspect cuz not taking the mean over sessions first (or the repeats)
    # # stats_by_mouse_sim = df_for_stats.groupby(["animal_ID", "full_sim_desc"]).apply(lambda g: pd.Series(get_paired_stats(g["reward_rate_mouse"], g["reward_rate_agent"], test=STATS_TEST)), include_groups=False)

    # #### option 2: average repeats -> average sessions -> stats
    # ## mouse: avg over sessions
    # ## sims: avg repeats -> avg sessions --> stats
    # ## sims: avg sessions (multiple repeats per session) --> stats # Author3: dimensionality needs to match, so no

    # """ neurIPS: each mouse vs all sims PER session """
    # ## Author: presumably wrong
    # # stats_by_mouse_exp = df_for_stats.groupby(["animal_ID", "exp_moment"]).apply(lambda g: pd.Series(get_paired_stats(g["reward_rate_mouse"], g["reward_rate_agent"], test=STATS_TEST)), include_groups=False)
    # ## option 1: avg sim repeats --> n sims vs mouse (1) (per day): Author3: 1-sample t test:
    # ## option 2: n sims x m repeats vs mouse (1) (per day): 1-sample t test: Author3: hierarchical; complicated
    # ## Author3: sim repeats vs mouse (1) (per day per sim type): 1-sample t test 
    return


@app.cell
def _():
    """ 260112 Author: todo have LLM compare these old alternative stats versions to above and then merge/discard"""

    # """ Benjamini–Hochberg (non-negative, monotone) q-values. Asserts no NaN/Inf in the input vector. """
    # def fdr_bh_strict(pvals):
    #     p = np.asarray(pvals, dtype=float)
    #     assert np.isfinite(p).all(), "fdr_bh_strict: NaN/Inf in p-values"
    #     m = p.size
    #     order = np.argsort(p, kind="mergesort")        # sort ascending
    #     p_sorted = p[order]
    #     ranks = np.arange(1, m + 1, dtype=float)       # 1..m
    #     q_sorted = p_sorted * m / ranks
    #     q_sorted = np.minimum.accumulate(q_sorted[::-1])[::-1]   # monotone
    #     assert np.isfinite(q_sorted).all(), "fdr_bh_strict: non-finite q-values"
    #     assert ((0.0 <= q_sorted) & (q_sorted <= 1.0)).all(), "fdr_bh_strict: q-values out of [0,1]"
    #     # q = np.empty_like(p); q[order] = np.clip(q_sorted, 0, 1)
    #     q = np.empty_like(p); 
    #     q[order] = q_sorted
    #     return q

    # def fdr_bh(pvals, alpha=0.05):
    #     """Return array of BH-adjusted p-values (q-values)."""
    #     p = np.asarray(pvals, float)
    #     m = np.sum(np.isfinite(p))
    #     assert np.all(np.isfinite(p)), "fdr_bh_strict: NaN/Inf in p-values"
    #     order = np.argsort(p, kind="mergesort")
    #     ranks = np.empty_like(order); ranks[order] = np.arange(1, len(p)+1)
    #     p_sorted = p[order]
    #     q_sorted = np.minimum.accumulate((p_sorted * m / ranks[::-1])[::-1])
    #     q = np.empty_like(p); q[order] = q_sorted
    #     return np.clip(q, 0, 1)

    # # Choose the p source (prefer raw when available)
    # p_use = np.where(
    #     stats_by_mouse_model.get("p_val_raw").notna(),
    #     stats_by_mouse_model["p_val_raw"].to_numpy(),
    #     stats_by_mouse_model["p_val"].to_numpy()
    # )
    # assert np.isfinite(p_use).all(), "NaN/Inf in p_use before FDR"
    # stats_by_mouse_model = stats_by_mouse_model.assign(p_use=p_use)

    # # FDR per mouse family WITH INDEX ALIGNMENT
    # stats_by_mouse_model["p_val_adj"] = (
    #     stats_by_mouse_model
    #       .groupby("animal_ID", group_keys=False)["p_use"]
    #       .apply(lambda s: pd.Series(fdr_bh_strict(s.values), index=s.index))
    # )
    return


@app.cell
def _():
    """ 251225 Author: unclear what the status of this commented code is... check along with above carefully """

    ## 251218 Author: GPT says this code from the neurIPS era notebook is misleading / wrong

    # def violin_two_groups_with_sig(df, mouse_col, agent_col, stats_all,
    #                                conf=0.95, ylabel="reward rate", title="mouse vs agent"):
    #     # data
    #     mv = np.asarray(df[mouse_col].values, dtype=float)
    #     av = np.asarray(df[agent_col].values, dtype=float)
    #     assert mv.shape == av.shape, "paired vectors must have same length"
    #     assert np.all(np.isfinite(mv)) and np.all(np.isfinite(av)), "NaN/Inf present"

    #     # means + CIs
    #     m_mouse, ci_mouse = mean_and_ci(mv, conf=conf)
    #     m_agent, ci_agent = mean_and_ci(av, conf=conf)

    #     # figure
    #     fig = plt.figure()
    #     ax = plt.gca()

    #     parts = ax.violinplot([mv, av], positions=[0, 1], showmeans=False, showextrema=False, widths=0.8)

    #     # optional: lighten fill and draw outlines
    #     for pc in parts['bodies']:
    #         pc.set_alpha(0.6)

    #     # overlay means and 95% CI as thin lines/points
    #     ax.plot([0], [m_mouse], marker='o')
    #     ax.plot([1], [m_agent], marker='o')
    #     ax.vlines([0], [ci_mouse[0]], [ci_mouse[1]], linewidth=2)
    #     ax.vlines([1], [ci_agent[0]], [ci_agent[1]], linewidth=2)

    #     # axes, labels
    #     ax.set_xticks([0, 1])
    #     ax.set_xticklabels(["mouse", "agent"])
    #     ax.set_ylabel(ylabel)
    #     ax.set_title(title)

    #     # significance bracket
    #     ptxt = p_display_from_stats(stats_all)
    #     eff_name = stats_all.get("effect_name", "")
    #     eff_val = stats_all.get("effect_size", np.nan)
    #     eff_txt = f", {eff_name}={eff_val:.3f}" if np.isfinite(eff_val) else ""
    #     n = stats_all.get("n", mv.size)
    #     alt = stats_all.get("alternative", "two-sided")

    #     y_top = max(ci_mouse[1], ci_agent[1])
    #     y = y_top + (abs(y_top) + 1e-12) * 0.05
    #     h = (abs(y_top) + 1e-12) * 0.03
    #     draw_sig_bracket(ax, 0, 1, y, h, f"{ptxt} (n={n}, {alt}){eff_txt}")

    #     ax.margins(y=0.15)
    #     plt.tight_layout()
    #     return fig, ax

    # def violin_paired_differences(df, mouse_col, agent_col,
    #                               conf=0.95, ylabel="agent − mouse", title="paired differences"):
    #     mv = np.asarray(df[mouse_col].values, dtype=float)
    #     av = np.asarray(df[agent_col].values, dtype=float)
    #     assert mv.shape == av.shape, "paired vectors must have same length"
    #     assert np.all(np.isfinite(mv)) and np.all(np.isfinite(av)), "NaN/Inf present"
    #     diffs = av - mv

    #     m_diff, ci_diff = mean_and_ci(diffs, conf=conf)

    #     fig = plt.figure()
    #     ax = plt.gca()
    #     parts = ax.violinplot([diffs], positions=[0], showmeans=False, showextrema=False, widths=0.8)
    #     for pc in parts['bodies']:
    #         pc.set_alpha(0.6)

    #     ax.plot([0], [m_diff], marker='o')
    #     ax.vlines([0], [ci_diff[0]], [ci_diff[1]], linewidth=2)

    #     ax.axhline(0, linestyle='--')
    #     ax.set_xticks([0])
    #     ax.set_xticklabels(["agent − mouse"])
    #     ax.set_ylabel(ylabel)
    #     ax.set_title(title)
    #     plt.tight_layout()
    #     return fig, ax

    # stats_all = all_mice_vs_all_sims_stats_variants['Mme_vs_Sme-parametric']

    # ## two-group violin with bracket (uses stats_all for the annotation)
    # fig1, ax1 = violin_two_groups_with_sig(DF_FOR_STATS, "reward_rate_mouse", "reward_rate_agent", stats_all,
    #                                        conf=0.95, ylabel="reward rate", title="mouse vs agent reward rate")

    # ## single violin of paired differences
    # fig2, ax2 = violin_paired_differences(DF_FOR_STATS, "reward_rate_mouse", "reward_rate_agent",
    #                                       conf=0.95, ylabel="agent − mouse", title="paired differences in reward rate")
    # plt.show()

    # _MODEL_FORMATTING_DF = pd.DataFrame(
    #     [
    #         # model (index)                         xtick_label                    legend_group               color       model_order legend_order legend_label
    #         ("Policy: fixed; Foo",                 "Foo",                         "fixed policy",            "#1f77b4",  0,         0,          "fixed policy"),
    #         ("Policy: MDP; Bar",                   "Bar",                         "MDP",                     "#ff7f0e",  1,         1,          "MDP"),
    #         ("Policy: POMDP; Baz",                 "Baz",                         "POMDP",                   "#2ca02c",  2,         2,          "POMDP"),
    #     ],
    #     columns=["model","xtick_label","legend_group","color","model_order","legend_order","legend_label"],
    # ).set_index("model")
    # _MODEL_FORMATTING_DF.index.name = "model"
    # _MODEL_FORMATTING_DF


    #########################

    # ## 251228 Author: note that n_rows is a sanity check... they shouldn't be crazy different unless a mixture of impartial results are being loaded
    # _MODEL_FORMATTING_DF = DF_FOR_STATS.groupby(['full_sim_desc', 'policy_class', 'agent_model', 'obs_type', 'action_type'], dropna=False).size().to_frame('n_rows').reset_index()
    # print('251228 Author: currently assuming that the action_type is sufficient to recover the full representation, ie no mismatches between obs and action types')
    # # _MODEL_FORMATTING_DF
    # # _MODEL_FORMATTING_DF['xtick_label'] = _MODEL_FORMATTING_DF.apply(lambda row: f"{row.agent_model} ({row.action_type})")
    # _MODEL_FORMATTING_DF['xtick_label'] = _MODEL_FORMATTING_DF.apply(lambda row: f"{row['agent_model']} ({row['action_type']})", axis=1)
    # _MODEL_FORMATTING_DF['legend_group'] = _MODEL_FORMATTING_DF.agent_model.astype(str)
    # # _MODEL_FORMATTING_DF['color'] = _MODEL_FORMATTING_DF.agent_model.apply(lambda row: MODEL2COLOR_DICT[row['agent_model']], axis=1)
    # # _MODEL_FORMATTING_DF['color'] = _MODEL_FORMATTING_DF.apply(lambda row: MODEL2COLOR_DICT[row['agent_model']], axis=1)

    # # legend_group: simplest (just copy or astype)
    # # _MODEL_FORMATTING_DF['legend_group'] = _MODEL_FORMATTING_DF['agent_model'].astype(str)

    # # color: map via dict (best)
    # _MODEL_FORMATTING_DF['color'] = _MODEL_FORMATTING_DF['agent_model'].map(MODEL2COLOR_DICT)
    # _MODEL_FORMATTING_DF['display_order'] = _MODEL_FORMATTING_DF['agent_model'].map(MODEL2ORDER_DICT)

    # MODEL2ORDER_DICT
    # MODEL_FORMATTING_DF = _MODEL_FORMATTING_DF
    # MODEL_FORMATTING_DF.index.name = "model"

    # # print(list(MODEL2COLOR_DICT.keys()))

    #########################

    #####################


    # import matplotlib.pyplot as plt
    # import matplotlib.colors as mcolors

    # # use agent_model when present; otherwise fall back to full_sim_desc for labeling + color hashing
    # agent_label=_MODEL_FORMATTING_DF["agent_model"].astype("string")
    # agent_label=agent_label.fillna(_MODEL_FORMATTING_DF["full_sim_desc"].astype("string"))

    # action_label=_MODEL_FORMATTING_DF["action_type"].astype("string").fillna("")
    # _MODEL_FORMATTING_DF["xtick_label"]=agent_label+" ("+action_label+")"
    # _MODEL_FORMATTING_DF["legend_group"]=agent_label

    # # try dictionary colors first
    # _MODEL_FORMATTING_DF["color"]=agent_label.map(MODEL2COLOR_DICT)

    # # fill any remaining missing colors with stable hashed colormap colors
    # missing_mask=_MODEL_FORMATTING_DF["color"].isna()
    # if missing_mask.any():
    #     missing_keys=sorted(agent_label[missing_mask].astype(str).unique().tolist())
    #     print("WARNING: missing MODEL2COLOR_DICT keys; assigning stable hashed colors:\n",missing_keys)

    #     cmap=plt.get_cmap("turbo")
    #     N=getattr(cmap,"N",256)

    #     def stable_hex(s):
    #         h=int(hashlib.md5(str(s).encode("utf-8")).hexdigest(),16)
    #         u=(h%N)/max(N-1,1)
    #         return mcolors.to_hex(cmap(u))

    #     _MODEL_FORMATTING_DF.loc[missing_mask,"color"]=agent_label[missing_mask].map(stable_hex)

    # # IMPORTANT: only add model_order if it's fully defined; otherwise omit it
    # model_order=_MODEL_FORMATTING_DF["agent_model"].map(MODEL2ORDER_DICT)
    # if model_order.notna().all():
    #     _MODEL_FORMATTING_DF["model_order"]= model_order

    ###############

    # # ---- labels ----
    # agent_model=_MODEL_FORMATTING_DF["agent_model"].astype("string")
    # full_sim_desc=_MODEL_FORMATTING_DF["full_sim_desc"].astype("string")

    # # (A) agent_model is NA
    # na_agent_mask=agent_model.isna()
    # if na_agent_mask.any():
    #     print("WARNING: agent_model is <NA> for these full_sim_desc rows (showing up to 50):")
    #     print(_MODEL_FORMATTING_DF.loc[na_agent_mask,["full_sim_desc","policy_class","obs_type","action_type"]].drop_duplicates().head(50))

    # # choose a label for ticks/legend even when agent_model is NA
    # agent_label=agent_model.fillna(full_sim_desc)
    # action_label=_MODEL_FORMATTING_DF["action_type"].astype("string").fillna("")
    # _MODEL_FORMATTING_DF["xtick_label"]=agent_label+" ("+action_label+")"
    # _MODEL_FORMATTING_DF["legend_group"]=agent_label

    # # ---- color ----
    # _MODEL_FORMATTING_DF["color"]=agent_label.map(MODEL2COLOR_DICT)

    # missing_color_mask=_MODEL_FORMATTING_DF["color"].isna()
    # if missing_color_mask.any():
    #     # (B) keys missing from MODEL2COLOR_DICT (exclude <NA> keys because those are handled above)
    #     missing_color_keys=sorted(agent_label[missing_color_mask].dropna().astype(str).unique().tolist())
    #     if len(missing_color_keys) > 0:
    #         print("WARNING: MODEL2COLOR_DICT missing keys (showing up to 100):")
    #         print(missing_color_keys[:100])

    #     # also show which full_sim_desc rows are affected (useful when agent_label==full_sim_desc fallback)
    #     print("WARNING: assigning default color='gray' for these models (showing up to 50):")
    #     print(_MODEL_FORMATTING_DF.loc[missing_color_mask,["full_sim_desc","agent_model","policy_class","obs_type","action_type"]].drop_duplicates().head(50))

    #     _MODEL_FORMATTING_DF.loc[missing_color_mask,"color"]="gray"

    # # ---- model_order (only if fully defined; otherwise omit) ----
    # model_order=_MODEL_FORMATTING_DF["agent_model"].map(MODEL2ORDER_DICT)
    # if model_order.notna().all():
    #     _MODEL_FORMATTING_DF["model_order"]=model_order
    # else:
    #     missing_order_keys=sorted(_MODEL_FORMATTING_DF.loc[model_order.isna(),"agent_model"].dropna().astype(str).unique().tolist())
    #     if len(missing_order_keys) > 0:
    #         print("WARNING: MODEL2ORDER_DICT missing keys (omitting model_order). Showing up to 100:")
    #         print(missing_order_keys[:100])


    ########### 

    # import hashlib

    # # ---- fill missing colors (so validator can never fail on 'color') ----
    # # import hashlib
    # # import matplotlib.pyplot as plt
    # # import matplotlib.colors as mcolors

    # _missing_mask = _MODEL_FORMATTING_DF["color"].isna()
    # if _missing_mask.any():
    #     _missing_agent_models = sorted(
    #         _MODEL_FORMATTING_DF.loc[_missing_mask, "agent_model"].astype(str).unique().tolist()
    #     )
    #     print(
    #         "WARNING: agent_model(s) missing from MODEL2COLOR_DICT; assigning stable hashed colors:\n"
    #         f"{_missing_agent_models}"
    #     )

    #     _cmap = plt.get_cmap("turbo")
    #     _N = getattr(_cmap, "N", 256)

    #     def _stable_hex(_s: str) -> str:
    #         _h = int(hashlib.md5(_s.encode("utf-8")).hexdigest(), 16)
    #         _u = (_h % _N) / max(_N - 1, 1)
    #         return mcolors.to_hex(_cmap(_u))

    #     _MODEL_FORMATTING_DF.loc[_missing_mask, "color"] = (
    #         _MODEL_FORMATTING_DF.loc[_missing_mask, "agent_model"].astype(str).map(_stable_hex)
    #     )

    # # ---- order (optional but useful because your plotter expects 'model_order', not 'display_order') ----
    # _MODEL_FORMATTING_DF["model_order"] = _MODEL_FORMATTING_DF["agent_model"].map(MODEL2ORDER_DICT)

    # # ---- CRITICAL: set index to match df[model_col] i.e., full_sim_desc ----
    # MODEL_FORMATTING_DF = _MODEL_FORMATTING_DF.set_index("full_sim_desc")
    # MODEL_FORMATTING_DF.index.name = "model"

    # # ---- labels ----
    # agent_model=_MODEL_FORMATTING_DF["agent_model"].astype("string")
    # full_sim_desc=_MODEL_FORMATTING_DF["full_sim_desc"].astype("string")

    # # choose a label for ticks/legend even when agent_model is NA
    # agent_label=agent_model.fillna(full_sim_desc)
    # action_label=_MODEL_FORMATTING_DF["action_type"].astype("string").fillna("")
    # _MODEL_FORMATTING_DF["xtick_label"]=agent_label+" ("+action_label+")"
    # _MODEL_FORMATTING_DF["legend_group"]=agent_label

    # # ---- color ----
    # _MODEL_FORMATTING_DF["color"]=agent_label.map(MODEL2COLOR_DICT)

    # missing_color_mask=_MODEL_FORMATTING_DF["color"].isna()
    # if missing_color_mask.any():
    #     missing_color_keys=sorted(agent_label[missing_color_mask].dropna().astype(str).unique().tolist())
    #     if len(missing_color_keys)>0:
    #         print("WARNING: MODEL2COLOR_DICT missing keys (showing up to 200):")
    #         print(missing_color_keys[:200])

    #     print("WARNING: assigning default color='gray' for these models (showing up to 50):")
    #     print(_MODEL_FORMATTING_DF.loc[missing_color_mask,["full_sim_desc","agent_model","policy_class","obs_type","action_type"]]
    #           .drop_duplicates().head(50))

    #     _MODEL_FORMATTING_DF.loc[missing_color_mask,"color"]="gray"

    # # ---- model_order (ONLY if fully defined; otherwise OMIT) ----
    # model_order=agent_model.map(MODEL2ORDER_DICT)

    # if model_order.notna().all():
    #     _MODEL_FORMATTING_DF["model_order"]=model_order.astype(float)
    # else:
    #     # IMPORTANT: ensure we do NOT leave a partial 'model_order' column around,
    #     # because the plotter will try to use it and assert finiteness.
    #     missing_order_keys=sorted(agent_model[model_order.isna()].dropna().astype(str).unique().tolist())
    #     if len(missing_order_keys)>0:
    #         print("WARNING: MODEL2ORDER_DICT missing keys (omitting model_order). Showing up to 200:")
    #         print(missing_order_keys[:200])

    #     if "model_order" in _MODEL_FORMATTING_DF.columns:
    #         _MODEL_FORMATTING_DF=_MODEL_FORMATTING_DF.drop(columns=["model_order"])

    # # ---- FINAL: rebuild formatting df AFTER all modifications ----
    # MODEL_FORMATTING_DF=_MODEL_FORMATTING_DF.set_index("full_sim_desc")
    # MODEL_FORMATTING_DF.index.name="model"

    # displayed_model_names, model_keys = rename_models(stats_by_mouse_model)

    # ## 1) map original keys -> your display labels
    # label_map = dict(zip(model_keys, displayed_model_names))

    # ## 2) derive each model's policy/group from the original key
    # ## 3) establish desired overall model order
    # _group_rank = {g: i for i, g in enumerate(MODEL_COLOR_GROUPS_N_PLOT_ORDER)}
    # _first_seen  = {k: i for i, k in enumerate(model_keys)}  # preserve first-appearance within group
    # ordered_models = sorted(model_keys, key=lambda k: (_group_rank.get(key_to_group(k), 999), _first_seen[k]))

    # ## 4) reorder the stats table so first-appearance == ordered_models (controls bar positions)
    # _sbmm = ensure_column_df(stats_by_mouse_model, model_col="full_sim_desc").copy()
    # _order_map = {m: i for i, m in enumerate(ordered_models)}
    # _sbmm["__model_order__"] = _sbmm["full_sim_desc"].map(_order_map)
    # _sbmm = _sbmm.sort_values(["__model_order__"], kind="mergesort").drop(columns="__model_order__")

    # ## 5) colors by group (you can tweak this palette anytime)
    # ## --- coerce stats_per_mouse_x_sim_df -> 1b_-style stats_by_mouse_model ---
    # # stats_by_mouse_model = stats_per_mouse_x_sim_df.copy()

    # ## tolerate older naming
    # if "sim" in stats_by_mouse_model.columns and "full_sim_desc" not in stats_by_mouse_model.columns:
    #     stats_by_mouse_model = stats_by_mouse_model.rename(columns={"sim":"full_sim_desc"})

    # # match 1b_ column names expected by prepare_table / plot_fig2_from_stats
    # stats_by_mouse_model = stats_by_mouse_model.rename(columns={"mean_mouse":"mouse_mean", "mean_sim":"model_mean", 
    #                                                             "mean_model":"model_mean", "mean_sims":"model_mean"})

    # required_cols = ["animal_ID","full_sim_desc","mouse_mean","model_mean","p_val"] # used for an assert below but
    # _missing = [c for c in required_cols if c not in stats_by_mouse_model.columns]
    # assert not _missing, (
    #     f"stats_by_mouse_model missing required columns after coercion: {_missing}\n"
    #     f"columns={list(stats_by_mouse_model.columns)}")

    # ## make the MultiIndex that your downstream Bonferroni + loop expects
    # stats_by_mouse_model = stats_by_mouse_model.set_index(["animal_ID","full_sim_desc"]).sort_index()
    # assert isinstance(stats_by_mouse_model.index, pd.MultiIndex)
    # assert stats_by_mouse_model.index.names == ["animal_ID","full_sim_desc"]
    # assert not stats_by_mouse_model.index.duplicated().any(), "duplicate (animal_ID, full_sim_desc) rows"
    # assert np.isfinite(stats_by_mouse_model["p_val"].to_numpy(dtype=float)).all(), "NaN/Inf in p_val"


    # _MODEL_FORMATTING_DF = pd.DataFrame(
    #     [
    #         # model (index)                         xtick_label                    legend_group               color       model_order legend_order legend_label
    #         ("Policy: fixed; Foo",                 "Foo",                         "fixed policy",            "#1f77b4",  0,         0,          "fixed policy"),
    #         ("Policy: MDP; Bar",                   "Bar",                         "MDP",                     "#ff7f0e",  1,         1,          "MDP"),
    #         ("Policy: POMDP; Baz",                 "Baz",                         "POMDP",                   "#2ca02c",  2,         2,          "POMDP"),
    #     ],
    #     columns=["model","xtick_label","legend_group","color","model_order","legend_order","legend_label"],
    # ).set_index("model")
    # _MODEL_FORMATTING_DF.index.name = "model"
    # _MODEL_FORMATTING_DF

    # _MODEL_FORMATTING_DF = pd.DataFrame(
    #     [
    #         # model (index)                         xtick_label                    legend_group               color       model_order legend_order legend_label
    #         ("Policy: fixed; Foo",                 "Foo",                         "fixed policy",            "#1f77b4",  0,         0,          "fixed policy"),
    #         ("Policy: MDP; Bar",                   "Bar",                         "MDP",                     "#ff7f0e",  1,         1,          "MDP"),
    #         ("Policy: POMDP; Baz",                 "Baz",                         "POMDP",                   "#2ca02c",  2,         2,          "POMDP"),
    #     ],
    #     columns=["model","xtick_label","legend_group","color","model_order","legend_order","legend_label"],
    # ).set_index("model")
    # _MODEL_FORMATTING_DF.index.name = "model"
    # _MODEL_FORMATTING_DF

    # model_col="full_sim_desc"
    # # Only format models that actually appear in the plot input
    # models=pd.Index(stats_by_mouse_model.index.get_level_values(model_col)).unique()
    # models=[m for m in models.tolist() if pd.notna(m)]

    # cols=[model_col,"agent_model","policy_class","obs_type","action_type"]
    # missing=[c for c in cols if c not in DF_FOR_STATS.columns]
    # assert not missing,f"Author: DF_FOR_STATS missing required cols for model formatting: {missing}"

    # base=(DF_FOR_STATS.loc[DF_FOR_STATS[model_col].isin(models),cols]
    #       .drop_duplicates()
    #       .rename(columns={model_col:"model"})
    #       .copy())

    # def _infer_base_model(row):
    #     am=row.get("agent_model",pd.NA)
    #     if pd.isna(am) or str(am) in ("<NA>","nan","None",""):
    #         m=str(row["model"])
    #         tok=m.split(";",1)[0].strip().replace("_"," ")
    #         return tok if tok else "unknown_model"
    #     return str(am).strip()

    # base["base_model"]=base.apply(_infer_base_model,axis=1)

    # # Optional: pull seqLength-* from full_sim_desc (mostly fixed-policy)
    # def _parse_seq_len(m):
    #     if not isinstance(m,str): return np.nan
    #     for part in m.split(";"):
    #         if part.startswith("seqLength-"):
    #             v=part.split("-",1)[1]
    #             try: return int(v)
    #             except Exception: return v
    #     return np.nan


    # seq=base["model"].map(_parse_seq_len)

    # # xtick_label: include action_type when available; include seq length when present
    # base["xtick_label"]=base["base_model"]
    # has_action=base["action_type"].notna()
    # if has_action.any():
    #     base.loc[has_action,"xtick_label"]=base.loc[has_action].apply(
    #         lambda r:f"{r['base_model']} ({r['action_type']})",axis=1
    #     )
    # if seq.notna().any():
    #     base["xtick_label"]=base["xtick_label"]+seq.map(lambda x:f" L{x}" if pd.notna(x) else "")

    # base["legend_group"]=base["base_model"]
    # base["legend_label"]=base["base_model"]

    # # Colors: fixed policy + RL. Unknowns default to gray (and are printed).
    # model2color={
    #     "random":"#377eb8",
    #     "action biased":"#e41a1c",
    #     "observation action biased":"#4daf4a",
    #     "latent state biased":"#984ea3",
    #     "rwd dist oracle": 'gray',
    #     "A2C":"#ff7f00",
    #     "PPO":"olive",
    #     "DQN":"orange",
    # }
    # base["color"]=base["base_model"].map(model2color)
    # missing_color=sorted(base.loc[base["color"].isna(),"base_model"].dropna().unique().tolist())
    # if missing_color:
    #     print("Author: base_model(s) missing from model2color -> using gray:",missing_color)
    # base["color"]=base["color"].fillna("gray")

    # # If DF_FOR_STATS has inconsistent metadata per model, warn and keep first deterministically.
    # dup=(base["model"].value_counts())
    # _bad=dup[dup>1]
    # if len(_bad)>0:
    #     print("Author: multiple metadata rows for same model; keeping first. Models:",_bad.index.tolist())
    #     base=(base.sort_values(["model","base_model","action_type","obs_type"],kind="mergesort")
    #              .drop_duplicates("model",keep="first")
    #              .copy())

    # ##########

    # # Orders: stable, fully-defined (required by your plot validator)
    # base_order_map={k:i for i,k in enumerate(model2color.keys())}
    # base["base_order"]=base["base_model"].map(base_order_map)

    # unknown_mask=base["base_order"].isna()
    # if unknown_mask.any():
    #     unknown=sorted(base.loc[unknown_mask,"base_model"].astype(str).unique().tolist())
    #     start=len(base_order_map)
    #     unk_map={m:start+i for i,m in enumerate(unknown)}
    #     base.loc[unknown_mask,"base_order"]=base.loc[unknown_mask,"base_model"].astype(str).map(unk_map)

    # base["base_order"]=pd.to_numeric(base["base_order"],errors="raise")
    # base=base.sort_values(["base_order","base_model","model"],kind="mergesort").copy()
    # base["within_order"]=base.groupby("base_model",sort=False).cumcount()

    # # float is fine; your downstream assert checks np.isfinite(...)
    # base["legend_order"]=base["base_order"].astype(float)
    # base["model_order"]=base["base_order"].astype(float)*1000.0+base["within_order"].astype(float)

    # _MODEL_FORMATTING_DF=(base[["model","xtick_label","legend_group","color","model_order","legend_order","legend_label"]].set_index("model"))
    # _MODEL_FORMATTING_DF.index.name="model"

    # # Sanity prints that directly explain missing formatting
    # missing_rows=[m for m in models if m not in _MODEL_FORMATTING_DF.index]
    # if missing_rows:
    #     print("Author: models present in stats_by_mouse_model but missing from _MODEL_FORMATTING_DF:",missing_rows)

    # # Hard asserts to match your plot validator expectations
    # req=["xtick_label","legend_group","color","model_order","legend_order","legend_label"]
    # for c in req:
    #     assert c in _MODEL_FORMATTING_DF.columns,f"Author: _MODEL_FORMATTING_DF missing required column: {c}"
    #     _bad=_MODEL_FORMATTING_DF[c].isna()
    #     assert not _bad.any(),f"Author: _MODEL_FORMATTING_DF has NaN in '{c}' for models: {_MODEL_FORMATTING_DF.index[_bad].tolist()}"

    # def _():
    #     # import numpy as np
    #     # import pandas as pd
    #     # import matplotlib.pyplot as plt

    #     model_col="full_sim_desc"

    #     # Only format models that actually appear in the plot input
    #     models=pd.Index(stats_by_mouse_model.index.get_level_values(model_col)).unique()
    #     models=[m for m in models.tolist() if pd.notna(m)]

    #     cols=[model_col,"agent_model","policy_class","obs_type","action_type"]
    #     missing=[c for c in cols if c not in DF_FOR_STATS.columns]
    #     assert not missing,f"Author: DF_FOR_STATS missing required cols for model formatting: {missing}"

    #     base=(DF_FOR_STATS.loc[DF_FOR_STATS[model_col].isin(models),cols]
    #           .drop_duplicates()
    #           .rename(columns={model_col:"model"})
    #           .copy())

    #     def _canonical_base_model(agent_model, model_str):
    #         """
    #         Returns one of:
    #           random / action biased / observation action biased / latent state biased / A2C / PPO / DQN
    #         or a normalized fallback string for unknowns.
    #         """
    #         s=None
    #         if pd.notna(agent_model) and str(agent_model) not in ("<NA>","nan","None"):
    #             s=str(agent_model)
    #         else:
    #             s=str(model_str) if pd.notna(model_str) else ""
    #             s=s.split(";",1)[0]  # first token of full_sim_desc

    #         s0=s.strip()
    #         sL=s0.lower().strip()
    #         sL=" ".join(sL.replace("_"," ").replace("-"," ").split())

    #         # RL (match by substring, not exact)
    #         if "ppo" in sL: return "PPO"
    #         if "dqn" in sL: return "DQN"
    #         if "a2c" in sL: return "A2C"

    #         # fixed-policy families (order matters: longer first)
    #         if "observation action biased" in sL or "obs action biased" in sL: return "observation action biased"
    #         if "latent state biased" in sL or "state biased" in sL: return "latent state biased"
    #         if "action biased" in sL: return "action biased"
    #         if "random" in sL: return "random"

    #         # unknown: keep a readable normalized base name
    #         return sL if sL else "unknown"

    #     base["base_model"]=base.apply(lambda r:_canonical_base_model(r.get("agent_model",np.nan), r.get("model","")),axis=1)
    #     assert base["base_model"].notna().all(),"Author: base_model unexpectedly has NaNs"

    #     # Optional: pull seqLength-* from full_sim_desc (mostly fixed-policy)
    #     def _parse_seq_len(m):
    #         if not isinstance(m,str): return np.nan
    #         for part in m.split(";"):
    #             if part.startswith("seqLength-"):
    #                 v=part.split("-",1)[1]
    #                 try: return int(v)
    #                 except Exception: return v
    #         return np.nan

    #     seq=base["model"].map(_parse_seq_len)

    #     # xtick_label: include action_type when available; include seq length when present
    #     base["xtick_label"]=base["base_model"]
    #     has_action=base["action_type"].notna()
    #     if has_action.any():
    #         base.loc[has_action,"xtick_label"]=base.loc[has_action].apply(
    #             lambda r:f"{r['base_model']} ({r['action_type']})",axis=1
    #         )
    #     if seq.notna().any():
    #         base["xtick_label"]=base["xtick_label"]+seq.map(lambda x:f" L{x}" if pd.notna(x) else "")

    #     base["legend_group"]=base["base_model"]
    #     base["legend_label"]=base["base_model"]

    #     # Explicit colors you care about (kept stable)
    #     known_model2color={
    #         "random":"#377eb8",
    #         "action biased":"#e41a1c",
    #         "observation action biased":"#4daf4a",
    #         "latent state biased":"#984ea3",
    #         "A2C":"#ff7f00",
    #         "PPO":"olive",
    #         "DQN":"orange",
    #     }

    #     base["color"]=base["base_model"].map(known_model2color)

    #     # Assign NON-gray colors to unknowns (stable by sorted name)
    #     unknown=sorted(base.loc[base["color"].isna(),"base_model"].dropna().unique().tolist())
    #     if unknown:
    #         print("Author: unknown base_model(s) not in known_model2color; assigning tab20 colors:",unknown)
    #         cmap=plt.get_cmap("tab20")
    #         unk2color={m:cmap(i % cmap.N) for i,m in enumerate(unknown)}
    #         base.loc[base["color"].isna(),"color"]=base.loc[base["color"].isna(),"base_model"].map(unk2color)

    #     assert base["color"].notna().all(),"Author: color assignment still has NaNs"

    #     # If DF_FOR_STATS has inconsistent metadata per model, warn and keep first deterministically.
    #     dup=base["model"].value_counts()
    #     bad=dup[dup>1]
    #     if len(bad)>0:
    #         print("Author: multiple metadata rows for same model; keeping first. Models:",bad.index.tolist())
    #         base=(base.sort_values(["model","base_model","action_type","obs_type"],kind="mergesort")
    #                  .drop_duplicates("model",keep="first")
    #                  .copy())

    #     # Fully-defined ordering (required by your plot validator)
    #     display_order=list(known_model2color.keys())+unknown
    #     order_map={m:i for i,m in enumerate(display_order)}
    #     base["base_order"]=base["base_model"].map(order_map)
    #     assert base["base_order"].notna().all(),"Author: base_order still has NaNs (should be impossible here)"

    #     base=base.sort_values(["base_order","base_model","model"],kind="mergesort").copy()
    #     base["within_order"]=base.groupby("base_model",sort=False).cumcount()

    #     base["legend_order"]=base["base_order"].astype(int)
    #     base["model_order"]=base["base_order"].astype(int)*1000+base["within_order"].astype(int)

    #     _MODEL_FORMATTING_DF=(base[["model","xtick_label","legend_group","color","model_order","legend_order","legend_label"]]
    #                           .set_index("model"))
    #     _MODEL_FORMATTING_DF.index.name="model"

    #     # Direct debug: show anything that would have gone gray previously
    #     debug=base.loc[base["base_model"].isin(unknown),["model","agent_model","base_model","color"]].copy()
    #     if len(debug)>0:
    #         print("Author: models using auto-colors (previously gray):")
    #     return print(debug.to_string(index=False))


    # _()

    # def build_model_registry_and_formatting(stats_by_mouse_model,DF_FOR_STATS,*,model_col="full_sim_desc",group_palette=None,model_groups_n_plot_order=None,default_color="gray"):
    #     import numpy as np
    #     import pandas as pd

    #     if group_palette is None: group_palette={}
    #     if model_groups_n_plot_order is None: model_groups_n_plot_order=[]

    #     def _norm(x):
    #         if pd.isna(x): return ""
    #         s=str(x).lower().replace("_"," ").replace("-"," ")
    #         return " ".join(s.split())

    #     palette_norm2key={_norm(k):k for k in group_palette.keys()}

    #     # --- which models are actually in the plot input? ---
    #     if isinstance(stats_by_mouse_model.index,pd.MultiIndex) and model_col in stats_by_mouse_model.index.names:
    #         models=pd.Index(stats_by_mouse_model.index.get_level_values(model_col)).unique()
    #     elif model_col in getattr(stats_by_mouse_model,"columns",[]):
    #         models=pd.Index(stats_by_mouse_model[model_col].unique())
    #     else:
    #         models=pd.Index(stats_by_mouse_model.reset_index()[model_col].unique())
    #     models=[m for m in models.tolist() if pd.notna(m)]

    #     need=[model_col,"agent_model","policy_class","obs_type","action_type"]
    #     missing=[c for c in need if c not in DF_FOR_STATS.columns]
    #     assert not missing,f"Author: DF_FOR_STATS missing cols needed for model formatting: {missing}"

    #     base=(DF_FOR_STATS.loc[DF_FOR_STATS[model_col].isin(models),need]
    #           .drop_duplicates()
    #           .rename(columns={model_col:"model"})
    #           .copy())

    #     # base_model: prefer agent_model; fallback to parsing full_sim_desc token-0
    #     def _infer_base_model(row):
    #         am=row.get("agent_model",None)
    #         if pd.isna(am) or str(am) in ("<NA>","nan","None"):
    #             tok=str(row["model"]).split(";",1)[0].strip()
    #             tok=tok.replace("_"," ")
    #             return tok
    #         return str(am)

    #     base["base_model"]=base.apply(_infer_base_model,axis=1)

    #     # optional: parse seqLength-* from full_sim_desc
    #     def _parse_seq_len(m):
    #         if not isinstance(m,str): return np.nan
    #         for part in m.split(";"):
    #             if part.startswith("seqLength-"):
    #                 v=part.split("-",1)[1]
    #                 try: return int(v)
    #                 except Exception: return v
    #         return np.nan

    #     base["seq_len"]=base["model"].map(_parse_seq_len)

    #     # map base_model -> palette_key using normalization (handles underscores/spaces)
    #     base["palette_key"]=base["base_model"].map(lambda x: palette_norm2key.get(_norm(x),str(x)))

    #     # colors
    #     base["color"]=base["palette_key"].map(group_palette)
    #     missing_color=sorted(base.loc[base["color"].isna(),"palette_key"].dropna().unique().tolist())
    #     if missing_color:
    #         print("Author: palette_key(s) missing from GROUP_PALETTE -> using default_color:",missing_color)
    #     base["color"]=base["color"].fillna(default_color)

    #     # xtick/legend labels (edit here if you want richer strings)
    #     base["xtick_label"]=base["palette_key"]
    #     has_action=base["action_type"].notna()
    #     if has_action.any():
    #         base.loc[has_action,"xtick_label"]=base.loc[has_action].apply(lambda r:f"{r['palette_key']} ({r['action_type']})",axis=1)
    #     if base["seq_len"].notna().any():
    #         base["xtick_label"]=base["xtick_label"]+base["seq_len"].map(lambda x:f" L{x}" if pd.notna(x) else "")

    #     base["legend_group"]=base["palette_key"]
    #     base["legend_label"]=base["palette_key"]

    #     # ordering: start with your explicit order list, then append remaining palette keys, then any truly unknown keys
    #     explicit=[k for k in model_groups_n_plot_order if _norm(k) in palette_norm2key]
    #     explicit=[palette_norm2key[_norm(k)] for k in explicit]

    #     palette_rest=[k for k in group_palette.keys() if k not in explicit]
    #     group_order=explicit+palette_rest

    #     order_map={k:i for i,k in enumerate(group_order)}
    #     base["base_order"]=base["palette_key"].map(order_map)

    #     # append unknown palette_keys after known ones (stable)
    #     if base["base_order"].isna().any():
    #         unknown=sorted(base.loc[base["base_order"].isna(),"palette_key"].dropna().unique().tolist())
    #         start=len(order_map)
    #         unk_map={m:start+i for i,m in enumerate(unknown)}
    #         base.loc[base["base_order"].isna(),"base_order"]=base.loc[base["base_order"].isna(),"palette_key"].map(unk_map)
    #         print("Author: unknown palette_key(s) not in group_order; appended at end:",unknown)

    #     base=base.sort_values(["base_order","palette_key","model"],kind="mergesort").copy()
    #     base["within_order"]=base.groupby("palette_key",sort=False).cumcount()

    #     # construct an integer model_order without magic numbers
    #     mult=int(base.groupby("palette_key",sort=False).size().max())+1
    #     model_order=base["base_order"]*mult+base["within_order"]

    #     # IMPORTANT: only add model_order if it's fully defined; otherwise omit it
    #     if pd.notna(model_order).all() and np.isfinite(model_order.astype(float)).all():
    #         base["model_order"]=model_order.astype(int)
    #     else:
    #         print("Author: model_order not fully defined; omitting model_order")
    #         if "model_order" in base.columns: base=base.drop(columns=["model_order"])

    #     base["legend_order"]=base["base_order"].astype(int)

    #     MODEL_REGISTRY_DF=base.copy()

    #     keep_cols=["xtick_label","legend_group","color","legend_label","legend_order","base_model","palette_key","policy_class","obs_type","action_type","seq_len"]
    #     if "model_order" in base.columns: keep_cols.insert(3,"model_order")

    #     MODEL_FORMATTING_DF=(base[["model"]+keep_cols].set_index("model"))
    #     MODEL_FORMATTING_DF.index.name="model"

    #     # hard checks matching the plotting expectations
    #     for c in ["xtick_label","legend_group","color","legend_label","legend_order"]:
    #         assert c in MODEL_FORMATTING_DF.columns,f"Author: MODEL_FORMATTING_DF missing {c}"
    #         bad=MODEL_FORMATTING_DF[c].isna()
    #         assert not bad.any(),f"Author: MODEL_FORMATTING_DF has NaN in {c} for: {MODEL_FORMATTING_DF.index[bad].tolist()}"

    #     return MODEL_REGISTRY_DF,MODEL_FORMATTING_DF


    # MODEL_REGISTRY_DF,MODEL_FORMATTING_DF=build_model_registry_and_formatting(
    #     stats_by_mouse_model,DF_FOR_STATS,
    #     group_palette=GROUP_PALETTE,
    #     model_groups_n_plot_order=MODEL_COLOR_GROUPS_N_PLOT_ORDER,
    #     default_color="gray",
    # )

    # MODEL_REGISTRY_DF.sort_values(["base_order","palette_key","within_order","model"]).reset_index(drop=True)
    # MODEL_FORMATTING_DF.sort_values(["legend_order"]+(["model_order"] if "model_order" in MODEL_FORMATTING_DF.columns else [])).head(50)


    # Infer a usable base model name for fixed-policy rows where agent_model is <NA>
    # def _infer_base_model(row):
    #     am=row.get("agent_model",None)
    #     if pd.isna(am) or str(am) in ("<NA>","nan","None"):
    #         m=str(row["model"])
    #         tok=m.split(";",1)[0].strip()
    #         tok=tok.replace("_"," ")

    ##########

    # Orders: stable, fully-defined (required by your plot validator)
    # base_order={k:i for i,k in enumerate(model2color.keys())}
    # base["base_order"]=base["base_model"].map(base_order)

    # Unknown base models get appended after known ones in alphabetical order
    # if base["base_order"].isna().any():
    #     unknown=sorted(base.loc[base["base_order"].isna(),"base_model"].unique().tolist())
    #     start=len(base_order)
    #     unk_map={m:start+i for i,m in enumerate(unknown)}
    #     base.loc[base["base_order"].isna(),"base_order"]=base.loc[base["base_order"].isna(),"base_model"].map(unk_map)

    # base=base.sort_values(["base_order","base_model","model"],kind="mergesort").copy()
    # base["within_order"]=base.groupby("base_model",sort=False).cumcount()

    # base["legend_order"]=base["base_order"].astype(int)
    # base["model_order"]=base["base_order"].astype(int)*1000+base["within_order"].astype(int)

    # # import re
    # # import numpy as np
    # # import pandas as pd
    # # import matplotlib.pyplot as plt
    # # from matplotlib.patches import Patch
    # # from matplotlib.lines import Line2D

    # MODEL_FORMATTING_DICT = {}
    # # GPT TODO: consolidate these into MODEL_FORMATTING_DICT: COLOR_MAP=None, LABEL_MAP=None, MODEL_ORDER=None, STRIP_GROUP_FROM_XLABELS=True
    # # GPT TODO: y_label= f"{y_label} ({numerator_str} / {demoninator_str})", # replace Y_LABEL 

    # # def plot_fig2_from_stats_MRts_by_mouse_model(
    # #     stats_by_mouse_model,
    # #     model_col="full_sim_desc",
    # #     group_col="animal_ID",
    # #     MODEL_FORMATTING_DICT=None,
    # #     PVAL_COL="p_val",
    # #     MOUSE_FILTER=None,
    # #     FIGSIZE=(11, 6),
    # #     YLIM=(0.0, None),  # if None, we'll try to use global YMAX_FIXED, else auto
    # #     TITLE="Individual mouse vs simulations",
    # #     YLABEL=None,  # can be set via MODEL_FORMATTING_DICT["YLABEL"]
    # #     SHOW_BRACKETS=False,
    # #     TOP_K_BRACKETS=None,
    # #     SHOW_WHISKERS=True,
    # #     DF_LONG=None,
    # #     MOUSE_COL="reward_rate_mouse",
    # #     AGENT_COL="reward_rate_agent",
    # #     WHISKER_CONF=0.95,
    # #     WHISKER_CAPSIZE=4,
    # #     GROUP_ORDER=None,          # legend group order
    # #     LEGEND_FRAME=True,
    # #     MOUSE_WHISKER_DEDUP_COL="exp_moment",
    # # ):
    # def plot_fig2_from_stats_MR(stats_by_mouse_model, model_col="full_sim_desc", group_col="animal_ID", MODEL_FORMATTING_DICT
    #     PVAL_COL="p_val", MOUSE_FILTER=None,
    #     FIGSIZE=(11,6), YLIM=(0.0, YMAX_FIXED),
    #     TITLE="Individual mouse vs simulations", 
    #     SHOW_BRACKETS=False, TOP_K_BRACKETS=None, SHOW_WHISKERS=True, DF_LONG=None, # GPT TODO: can DF_LONG be calculated internally or otherwise usage clarified?
    #     MOUSE_COL="reward_rate_mouse", AGENT_COL="reward_rate_agent",
    #     WHISKER_CONF=0.95, WHISKER_CAPSIZE=4,
    #     GROUP_ORDER=None, LEGEND_FRAME=True,
    #     MOUSE_WHISKER_DEDUP_COL="exp_moment"):

    #     ## GPT TODO: fill in here

    #     # expects these to exist somewhere in your codebase:
    #     # - prepare_table(stats_by_mouse_model, model_col=..., group_col=...)
    #     # - mean_and_ci(values, conf=...) -> (mean, (lo, hi))

    #     """Consolidated replacement for plot_fig2_from_stats(...):
    #       - formatting is pulled from MODEL_FORMATTING_DICT
    #       - x tick labels actually use LABEL_MAP / tidy logic (this is why rename_models() didn't show)"""

    #     fmt = MODEL_FORMATTING_DICT

    #     # ---- formatting pulls (consolidated) ----

    # # ======= visual constants (as before) =======
    # # BRACKET_COLOR   = "black"
    # # COLOR_MOUSE     = "black"
    # # Y_OFFSET        = 0.02
    # # H_SCALE         = 0.70
    # # TEXT_PAD        = 0.01
    # # YMAX_FIXED      = 0.5
    # # # YMAX_FIXED      = 0.9

    # # TITLE_FONTSIZE      = 16
    # # XLABEL_FONTSIZE     = 14
    # # YLABEL_FONTSIZE     = 15
    # # YTICK_FONTSIZE      = 12
    # # LEGEND_FONTSIZE     = 10
    # # ANNOT_FONTSIZE      = 12
    # XTICK_FONTSIZE_MOUSE_VS_SIMS = 10

    # def plot_fig2_from_stats(stats_by_mouse_model, model_col="full_sim_desc", group_col="animal_ID",
    #     COLOR_MAP=None, LABEL_MAP=None,
    #     PVAL_COL="p_val",
    #     MODEL_ORDER=None, MOUSE_FILTER=None,
    #     FIGSIZE=(11,6), YLIM=(0.0, YMAX_FIXED),
    #     TITLE="Individual mouse vs simulations", YLABEL="reward rate (n rewards / n actions)",
    #     SHOW_BRACKETS=False, TOP_K_BRACKETS=None, SHOW_WHISKERS=True, DF_LONG=None,
    #     MOUSE_COL="reward_rate_mouse", AGENT_COL="reward_rate_agent",
    #     WHISKER_CONF=0.95, WHISKER_CAPSIZE=4,
    #     GROUP_ORDER=None, LEGEND_FRAME=True,
    #     MOUSE_WHISKER_DEDUP_COL="exp_moment",
    #     STRIP_GROUP_FROM_XLABELS=True):

    #     ## ---------- helpers ----------
    #     def _name(m): 
    #         return LABEL_MAP.get(m, m) if LABEL_MAP else m

    #     def key_to_group(k: str) -> str: 
    #         return k.split(';', 1)[0].replace('_', ' ').strip()

    #     def p_from_row(row):
    #         p = row.get("p_val_raw", np.nan)
    #         if not np.isfinite(p): p = row.get("p_val", np.nan)
    #         return p

    #     def significance2stars(p):
    #         if not np.isfinite(p): return "ns"
    #         if p < 1e-3: return "***"
    #         if p < 0.05: return "*"
    #         return "ns"

    #     # small text utilities -------------------------------
    #     def _rm_empty_parens(t: str) -> str:
    #         t = re.sub(r'\(\s*\)', '', t)
    #         t = re.sub(r'\s*[-–—]\s*$', '', t).strip()
    #         t = re.sub(r'\s{2,}', ' ', t).strip()
    #         return t

    #     def _move_avoid_reversal_to_end(txt: str) -> str:
    #         if not txt: return txt
    #         t = re.sub(r'avoid\s*back\s*[:=-]?\s*(true|false)?', 'avoid back', txt, flags=re.I)
    #         m = re.search(r'avoid back', t, flags=re.I)
    #         if not m: return _rm_empty_parens(t)
    #         t_removed = (t[:m.start()] + t[m.end():]).strip()
    #         t_removed = re.sub(r'\s*[-–—,:;]\s*$', '', t_removed).strip()
    #         out = f"{t_removed} (avoid back)".strip()
    #         return _rm_empty_parens(out)

    #     def _move_len_to_end(txt: str) -> str:
    #         m = re.search(r'\blen\s*(\d+)\b', txt, flags=re.I)
    #         if not m: return _rm_empty_parens(txt)
    #         n = m.group(1)
    #         t = re.sub(r'\b(len\s*\d+)\b\s*[-–—:]?\s*', '', txt, flags=re.I).strip()
    #         t = re.sub(r'\s{2,}', ' ', t)
    #         t = (f"{t} — len {n}" if t else f"len {n}")
    #         return _rm_empty_parens(t)

    #     def _fix_rep_prefix(txt: str) -> str:
    #         """Ensure 'Rep.: latent node idx' wording and clean junk before 'Rep.:'."""
    #         # add 'Rep.:' if the token 'latent node idx' appears without it
    #         if re.search(r'(?i)\blatent\s*node\s*idx\b', txt) and not re.search(r'(?i)\bRep\.\s*:', txt):
    #             txt = re.sub(r'(?i)\blatent\s*node\s*idx\b', 'Rep.: latent node idx', txt)
    #         # remove any dash/colon junk immediately before 'Rep.:'
    #         txt = re.sub(r'\s*[-–—:]\s*(?=Rep\.\s*:)', ' ', txt)
    #         txt = re.sub(r'^\s*[-–—:]+\s*(Rep\.\s*:)', r'\1', txt)
    #         # collapse spaces
    #         txt = re.sub(r'\s{2,}', ' ', txt).strip()
    #         return txt

    #     def tidy_axis_label(mkey: str) -> str:
    #         label = _name(mkey)
    #         g = key_to_group(mkey)

    #         if STRIP_GROUP_FROM_XLABELS:
    #             for pref in (f"Policy: {g}", g, f"{g} -", f"{g}:", f"{g} —"):
    #                 if label.lower().startswith(pref.lower()):
    #                     label = label[len(pref):].lstrip(" -:—")
    #                     break
    #             if not label:
    #                 parts = [p.strip().replace('_',' ') for p in mkey.split(';')]
    #                 label = parts[1] if len(parts) > 1 else _name(mkey)

    #         if g.lower() == "latent state biased":
    #             label = re.sub(r'(?i)^policy:\s*', '', label).strip()

    #         label = _move_avoid_reversal_to_end(label)
    #         label = _move_len_to_end(label)
    #         # strip Rep.: completely
    #         label = re.sub(r'\bRep\.\s*:\s*', '', label)

    #         # NEW: remove any leading dash/colon/space left hanging
    #         label = re.sub(r'^\s*[-:—]+\s*', '', label)

    #         label = _rm_empty_parens(label)
    #         return label

    #     # ---------- prep ----------
    #     df = prepare_table(stats_by_mouse_model, model_col=model_col, group_col=group_col)

    #     ## ensure we’re actually plotting the intended p’s (raw or FDR)
    #     assert PVAL_COL in df.columns, f"{PVAL_COL} not in stats table"
    #     assert np.isfinite(df[PVAL_COL]).all(), f"NaNs in {PVAL_COL}"

    #     if MOUSE_FILTER is not None:
    #         keep = [str(x) for x in (MOUSE_FILTER if isinstance(MOUSE_FILTER,(list,tuple,set,np.ndarray)) else [MOUSE_FILTER])]
    #         df = df[df[group_col].astype(str).isin(keep)]
    #         assert len(df) > 0, f"No rows left after filtering to {MOUSE_FILTER!r}"

    #     ## 251224 Author: how does this work?
    #     present_models = list(dict.fromkeys(df[model_col].tolist()))
    #     models = [m for m in (MODEL_ORDER or present_models) if m in present_models]

    #     if len(models) != len(present_models):
    #         missing_models = list(set(models).difference(present_models))
    #         print(f"251225 Author warning: MODEL_ORDER has models not present in data; these will be skipped: {missing_models}.")

    #     base_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    #     if COLOR_MAP is None: COLOR_MAP = {}
    #     model2color = {m: COLOR_MAP.get(m, base_colors[i % len(base_colors)]) for i, m in enumerate(models)}
    #     groups = [str(g) for g in sorted(df[group_col].unique(), key=lambda x: str(x))]

    #     ## means
    #     mouse_means = []
    #     agent_means = {m: [] for m in models}
    #     for gid in groups:
    #         sub = df[df[group_col]==gid]
    #         mouse_means.append(float(sub["mouse_mean"].iloc[0]))
    #         for m in models:
    #             row = sub[sub[model_col]==m]
    #             agent_means[m].append(np.nan if row.empty else float(row["model_mean"].iloc[0]))
    #     mouse_means = np.array(mouse_means, float)
    #     for m in models:
    #         agent_means[m] = np.array(agent_means[m], float)

    #     ## whiskers (mouse CI with de-dup)
    #     have_whiskers = False
    #     if SHOW_WHISKERS and DF_LONG is not None:
    #         need = {group_col, model_col, MOUSE_COL, AGENT_COL}
    #         dfl = DF_LONG.reset_index() if (isinstance(DF_LONG.index, pd.MultiIndex) or DF_LONG.index.name in (group_col, model_col)) else DF_LONG
    #         if need.issubset(dfl.columns):
    #             mouse_ci_lo, mouse_ci_hi = [], []
    #             for gid in groups:
    #                 g_long = dfl[dfl[group_col]==gid]
    #                 if (MOUSE_WHISKER_DEDUP_COL is not None) and (MOUSE_WHISKER_DEDUP_COL in g_long.columns):
    #                     vals = (g_long[[MOUSE_WHISKER_DEDUP_COL, MOUSE_COL]].dropna()
    #                         .drop_duplicates(subset=[MOUSE_WHISKER_DEDUP_COL]))[MOUSE_COL].values
    #                 else:
    #                     vals = g_long[MOUSE_COL].dropna().values
    #                 _, (lo, hi) = mean_and_ci(vals, conf=WHISKER_CONF)
    #                 mouse_ci_lo.append(lo); mouse_ci_hi.append(hi)
    #             mouse_ci_lo = np.array(mouse_ci_lo, float); mouse_ci_hi = np.array(mouse_ci_hi, float)

    #             agent_ci_lo = {m: [] for m in models}; agent_ci_hi = {m: [] for m in models}
    #             for gid in groups:
    #                 g_mouse = dfl[dfl[group_col]==gid]
    #                 for m in models:
    #                     gm = g_mouse[g_mouse[model_col]==m]
    #                     assert not gm.empty, f"Author: No data for group {gid}, model {m} in DF_LONG"
    #                     # if gm.empty:
    #                     #     agent_ci_lo[m].append(np.nan); agent_ci_hi[m].append(np.nan)
    #                     # else:
    #                     vals = gm[AGENT_COL].values
    #                     _, (lo, hi) = mean_and_ci(vals, conf=WHISKER_CONF)
    #                     agent_ci_lo[m].append(lo); agent_ci_hi[m].append(hi)
    #             for m in models:
    #                 agent_ci_lo[m] = np.array(agent_ci_lo[m], float)
    #                 agent_ci_hi[m] = np.array(agent_ci_hi[m], float)
    #             have_whiskers = True

    #     ## ---------- plot ----------
    #     fig, ax = plt.subplots(figsize=FIGSIZE)
    #     x = np.arange(len(groups))
    #     w = 0.1

    #     ## mouse bar & label (legend and x tick must match)
    #     mouse_label = f"mouse: {groups[0]}" if len(groups)==1 else "mice"
    #     mouse_centers = x - w*(len(models)/2)
    #     if have_whiskers:
    #         yerr_mouse = np.vstack([
    #             np.clip(mouse_means - mouse_ci_lo, 0, np.inf),
    #             np.clip(mouse_ci_hi - mouse_means, 0, np.inf)])
    #         ax.bar(mouse_centers, mouse_means, w, color=COLOR_MOUSE, label=mouse_label, yerr=yerr_mouse, capsize=WHISKER_CAPSIZE, ecolor=COLOR_MOUSE)
    #     else:
    #         ax.bar(mouse_centers, mouse_means, w, color=COLOR_MOUSE, label=mouse_label)

    #     ## agent bars + annotations
    #     agent_centers = []
    #     sig_codes_present = set()
    #     for j, m in enumerate(models):
    #         offs = x - w*(len(models)/2) + (j+1)*w
    #         if have_whiskers:
    #             yerr = np.vstack([np.clip(agent_means[m] - agent_ci_lo[m], 0, np.inf), np.clip(agent_ci_hi[m] - agent_means[m], 0, np.inf)])
    #             ax.bar(offs, agent_means[m], w, color=model2color[m], label="_nolegend_", yerr=yerr, capsize=WHISKER_CAPSIZE, ecolor=model2color[m])
    #         else:
    #             ax.bar(offs, agent_means[m], w, color=model2color[m], label="_nolegend_")
    #         agent_centers.append(offs)

    #         for i, gid in enumerate(groups):

    #             ## new: read the selected p-value column directly
    #             row = df.loc[(df[group_col]==gid) & (df[model_col]==m), PVAL_COL]
    #             assert len(row) == 1, f"Expected 1 row for ({gid}, {m}), found {len(row)}"
    #             p = float(row.iloc[0])
    #             sig = sig_code(p)

    #             sig_codes_present.add(sig)
    #             y_top = agent_means[m][i]
    #             if have_whiskers and np.isfinite(agent_means[m][i]):
    #                 hi = agent_ci_hi[m][i]
    #                 if np.isfinite(hi): y_top = max(y_top, hi)
    #             ax.text(offs[i], y_top + 0.012*(YLIM[1]-YLIM[0]), sig, ha='center', va='bottom', fontsize=ANNOT_FONTSIZE)

    #     ## ---------- x-axis ticks ----------
    #     assert len(mouse_centers) == 1, f'Author: expected single mouse group for fig2 plot but found: {len(mouse_centers)}'
    #     # if len(groups) == 1:
    #     centers = [mouse_centers[0]] + [arr[0] for arr in agent_centers]
    #     labels  = [mouse_label] + models

    #     ax.set_xticks(centers)
    #     ax.set_xticklabels(labels, rotation=-30, ha='left', fontsize=XTICK_FONTSIZE_MOUSE_VS_SIMS)
    #     # ax.set_xlabel("model type", fontsize=XLABEL_FONTSIZE)
    #     # else:
    #     #     ax.set_xticks(x)
    #     #     ax.set_xticklabels(groups, rotation=-30, ha='left', fontsize=XTICK_FONTSIZE_MOUSE_VS_SIMS)
    #     #     ax.set_xlabel("mouse ID", fontsize=XLABEL_FONTSIZE)

    #     ## trim whitespace at both ends
    #     all_centers = np.concatenate([mouse_centers] + agent_centers, axis=0)
    #     ax.set_xlim(np.nanmin(all_centers) - 0.55*w, np.nanmax(all_centers) + 0.55*w)

    #     ## title / y
    #     ax.set_title(TITLE.replace('mouse', f'mouse {groups[0]}') if len(groups)==1 else TITLE, fontsize=TITLE_FONTSIZE)
    #     ax.set_ylim(*YLIM)
    #     ax.set_ylabel(YLABEL, fontsize=YLABEL_FONTSIZE)
    #     ax.tick_params(axis='y', labelsize=YTICK_FONTSIZE)

    #     ## ---------- Legend: mouse + one entry per color group + notation mapping ----------
    #     if ax.get_legend() is not None:
    #         ax.get_legend().remove()

    #     present_groups = [key_to_group(m) for m in models]
    #     # assert GROUP_ORDER is not None, "Author: GROUP_ORDER must be provided or code altered"
    #     if GROUP_ORDER:
    #         group_list = [g for g in GROUP_ORDER if g in present_groups]
    #     else:
    #         seen = set(); group_list = []
    #         for g in present_groups:
    #             if g not in seen:
    #                 seen.add(g); group_list.append(g)

    #     group2color = {}
    #     for m in models:
    #         g = key_to_group(m)
    #         if g not in group2color:
    #             group2color[g] = model2color[m]

    #     handles = [Patch(facecolor=COLOR_MOUSE, edgecolor='none', label=mouse_label)]
    #     labels  = [mouse_label]
    #     for g in group_list:
    #         handles.append(Patch(facecolor=group2color[g], edgecolor='none', label=g))
    #         labels.append(g)

    #     parts = []
    #     if '***' in sig_codes_present: 
    #         parts.append('all mouse vs simulation comparisons Bonferroni-corrected for multiple comparisons \n*** p<0.001 (two-sided paired t-test)')
    #     if '*'   in sig_codes_present: 
    #         parts.append('* p<0.05')
    #     if 'ns'  in sig_codes_present: 
    #         parts.append('ns p≥0.05')
    #     notation_txt = '; '.join(parts) if parts else 'ns p≥0.05'
    #     handles.append(Line2D([], [], color='none', label=notation_txt))
    #     labels.append(notation_txt)

    #     ax.legend(handles, labels, fontsize=LEGEND_FONTSIZE, frameon=LEGEND_FRAME)

    #     return fig, ax



    # # sanity
    # assert np.isfinite(stats_by_mouse_model["p_val_adj"]).all(), "NaN in p_val_adj after FDR"

    # def get_model_strings(df, model_col="full_sim_desc"):
    #     # returns a stable, first-appearance order list of model keys
    #     if isinstance(df.index, pd.MultiIndex) and model_col in df.index.names:
    #         print('took index approach to parsing model strings')
    #         models = df.index.get_level_values(model_col).unique().tolist()
    #     elif df.index.name == model_col:
    #         print('took index.name approach to parsing model strings')
    #         models = df.index.unique().tolist()
    #     else:
    #         print('took generic else approach to parsing model strings')
    #         models = df.reset_index()[model_col].unique().tolist()
    #     return [str(m) for m in models]

    # sub = stats_by_mouse_model.xs(animal_ID, level="animal_ID")
    # n_tests = sub["n"].max()
    # bonf = np.clip(sub["p_val"] * n_tests, 0, 1)
    # assert np.allclose(bonf, sub["p_val_bonf"], equal_nan=True), f"Bonferroni mismatch for {animal_ID}"

    # plot_fig2_from_stats
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
