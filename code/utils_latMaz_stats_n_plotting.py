# import os

# import pandas as pd
# # from google.colab import drive
# from glob import glob
# import re
# import collections
# from datetime import datetime
# import numpy as np
# from copy import deepcopy
# import imageio
# from matplotlib import pyplot as plt
# from PIL import Image
# import plotly.graph_objects as go
# # import plotly.io as pio
# import ast
# try: import RPi.GPIO as GPIO
# except Exception as e: print(e); print('RPi.GPIO not imported, which is probably ok if you are not running on a Raspberry Pi...')
# from time import sleep
# import os
# import math
# from time import time
# import time
# import shutil
# import hashlib
# from collections import defaultdict

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

sys.path.append('./')
print('240618 Author: I assume utils_latMaz is in the same directory as the notebook')

from utils_latMaz import get_most_recent_file, deserialize_str, get_moment_strings, load_maze, get_now_str, get_adj_states, \
    displacement_to_compass_heading, get_direction_from_radian, allo_radian_map_dict, get_ego_direction, get_verified_st_traj, move_df_col_to_leftmost


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
LEGEND_FONTSIZE = 12
ANNOT_FONTSIZE  = 12

def validate_paired_stats_inputs(vals_a, vals_b, *, allow_singleton_2d=False, name_a="mouse_vals", name_b="sim_vals"):
    """ Convert inputs to float arrays and enforce 1D pairing.

    Author: * in a function signature makes everything after it keyword-only.

    allow_singleton_2d=False:
        require ndim==1 exactly (reject (n,1), (1,n), and any nested list-of-lists)
    allow_singleton_2d=True:
        allow (n,1) or (1,n) by squeeze(), but still reject true 2D (n,m) with m>1
    """
    if allow_singleton_2d:
        array_a=np.asarray(vals_a, dtype=float).squeeze()
        array_b=np.asarray(vals_b, dtype=float).squeeze()
        assert array_a.ndim==1 and array_b.ndim==1, \
            f"expected 1D after squeeze; got {name_a} shape={array_a.shape}, {name_b} shape={array_b.shape}"
    else:
        array_a=np.asarray(vals_a, dtype=float)
        array_b=np.asarray(vals_b, dtype=float)
        assert array_a.ndim==1 and array_b.ndim==1, \
            f"expected 1D inputs; got {name_a} shape={array_a.shape}, {name_b} shape={array_b.shape}"

    assert array_a.shape==array_b.shape, f"shape mismatch: {name_a} {array_a.shape} vs {name_b} {array_b.shape}"
    assert array_a.size>1, f"need >=2 pairs, got {array_a.size}"
    assert np.all(np.isfinite(array_a)), f"{name_a} contain NaN/Inf"
    assert np.all(np.isfinite(array_b)), f"{name_b} contain NaN/Inf"

    return array_a, array_b


def get_paired_stats(mouse_vals, sim_vals, names=('mean_mouse', 'mean_sim'), delta_degrees_of_freedom=1, test='parametric', alternative='two-sided', verbose=False):
    """
    Strict paired comparison (agent vs mouse) with robust checks and p-flooring.
    test: 'parametric' -> paired t-test; 'nonparametric' -> Wilcoxon signed-rank
    alternative: 'two-sided' | 'greater' | 'less'
    Returns dict with means, test stat/p (raw + clipped), effect size, and counts.
    """
    mouse_vals, sim_vals = validate_paired_stats_inputs(mouse_vals, sim_vals, allow_singleton_2d=False, name_a="mouse", name_b="sim")

    n = mouse_vals.size
    assert n > 1, f"need at least 2 pairs, got {n}"
    warn_if_small_n_for_hedges(n, small_n_threshold=20, label=f"get_paired_stats::{test}") # --- hedges small-n warning (runs once per call) ---
    assert delta_degrees_of_freedom == 1, \
        f"Expected delta_degrees_of_freedom==1 for paired inference, got {delta_degrees_of_freedom}"

    diff = mouse_vals - sim_vals # Author: core paired differences

    mean_mouse = float(mouse_vals.mean())
    mean_sims = float(sim_vals.mean())
    mean_diff = float(diff.mean())
    sd_diff = float(diff.std(ddof=delta_degrees_of_freedom))
    assert sd_diff >= 0.0 and np.isfinite(sd_diff), f"sd_diff was not positive and finite: {sd_diff}"
    se_diff = (sd_diff/np.sqrt(n)) 

    if verbose:
        print(
            f"{names[0]}: {mean_mouse}, {names[1]}: {mean_sims}, mean_diff: {mean_diff}, sd_diff: {sd_diff}, se_diff: {se_diff}; n: {n}, test: {test}, alternative: {alternative}"
        )

    result = {
        names[0]: mean_mouse, names[1]: mean_sims,
        "mean_diff": mean_diff, "se_diff": se_diff, "sd_diff": sd_diff, 
        "n": int(n), "test": test, "alternative": alternative}

    ## p-floor to avoid exact zeros from underflow
    p_floor = np.nextafter(0.0, 1.0)  # smallest positive float

    ## --- tests ---
    if test == 'parametric':
        # degenerate: zero variance of differences
        if not (np.isfinite(sd_diff) and sd_diff > 0):
            stat_raw, p_raw = 0.0, 1.0
            effect_name, effect_size = "cohen_d_z", np.nan
        else:
            # t = ss.ttest_rel(sim_vals, mouse_vals, alternative=alternative)
            t = ss.ttest_rel(mouse_vals, sim_vals, alternative=alternative)
            stat_raw, p_raw = float(t.statistic), float(t.pvalue)
            effect_name = "cohen_d_z"
            effect_size = float(mean_diff/sd_diff)
        p_clipped = max(p_raw, p_floor)
        result.update({"stat": stat_raw, "p_val_raw": p_raw, "p_val": p_clipped,
            "neg_log10_p": -np.log10(p_clipped), "effect_name": effect_name, "effect_size": effect_size})

    elif test=='nonparametric':

        ## nonparametric: Wilcoxon signed-rank on paired diffs        
        assert not np.allclose(diff, 0.0, equal_nan=False), 'Author: all diffs are exactly zero; cannot perform Wilcoxon test' # all zero diffs -> no signal

        ## exclude exact zeros from ranking per Wilcoxon convention; 251215 Author: GPT is saying this is standard practice; but it's applied via zero_method='wilcox' anyway
        # d_eff = diff[nz]
        # w = ss.wilcoxon(d_eff, zero_method='wilcox', alternative=alternative, correction=False, mode='auto')
        w = ss.wilcoxon(diff, zero_method='wilcox', alternative=alternative, correction=False, mode='auto') # Let SciPy apply the standard Wilcoxon convention for zeros via zero_method='wilcox'
        w_stat, p_raw = float(w.statistic), float(w.pvalue)
        nz = ~np.isclose(diff, 0.0) ## Effective n for rank-biserial denominator: count nonzero diffs (use same zero tolerance policy)
        assert nz.dtype == bool and nz.shape == diff.shape, f"unexpected nz mask type or shape: {nz.dtype}, {nz.shape}"
        n_eff = int(nz.sum()) 
        assert n_eff > 0, "internal: expected nonzero diffs after filtering"        

        denom = n_eff*(n_eff+1)/2
        p_clipped = max(p_raw, p_floor)

        ## naive
        # r_rb = float(1 - 2*w_stat/denom) if denom > 0 else np.nan 
        ## using this to get the directionality of the effect which is otherwise lost
        # sign = np.sign(diff.sum())  
        # r_rb = float(sign*(1 - 2*w_stat/denom)) if denom > 0 else np.nan
        ## use helper we sanity-checked
        rb = rank_biserial_from_diff(diff)  

        result.update({"stat": w_stat, "p_val_raw": p_raw, "p_val": p_clipped, "neg_log10_p": -np.log10(p_clipped),
            "n_eff_nonzero": int(n_eff), "effect_name": "rank_biserial_r", "effect_size": float(rb["r_rb"])})

    else:
        raise ValueError(f"unknown test={test}; must be either 'parametric' or 'nonparametric'")

    return result


def assert_required_cols(df, required_cols):
    assert isinstance(df, pd.DataFrame), f"expected DataFrame, got {type(df)}"
    assert len(required_cols) == len(set(required_cols)), f"Duplicate entries in required_cols: {required_cols}" # assert no column duplicates 
    missing=[c for c in required_cols if c not in df.columns]
    assert not missing,f"Missing required cols: {missing}"

def make_hashable(x):
    if x is None: 
        return "__none__"
    # normalize numpy scalars
    if isinstance(x, np.generic):
        x = x.item()
    # normalize NaN
    if isinstance(x, float) and np.isnan(x):
        return "__nan__"
    if isinstance(x, list):
        return tuple(make_hashable(v) for v in x)
    if isinstance(x, tuple):
        return tuple(make_hashable(v) for v in x)
    if isinstance(x, set):
        items=[make_hashable(v) for v in x]
        return ("__set__", tuple(sorted(items, key=lambda v: repr(v))))
    if isinstance(x, dict):
        items=[(repr(k), make_hashable(v)) for k,v in x.items()]
        return ("__dict__", tuple(sorted(items, key=lambda kv: kv[0])))
    if isinstance(x, np.ndarray):
        flat=tuple(make_hashable(v) for v in x.ravel().tolist())
        return ("__ndarray__", x.shape, str(x.dtype), flat)
    return x

def warn_if_small_n_for_hedges(n, *, small_n_threshold=20, label=""):
    """ Prints a warning if n is 'small' enough that Hedges' correction may be useful.
            For paired d_z (diff-based), df = n-1 and the small-sample correction is:
                g_z = J * d_z,  with  J ≈ 1 - 3/(4*df - 1)
        Direction: for finite df, J < 1, so g_z shrinks d_z toward 0 (same sign). """
    n=int(n)
    assert n >= 2, f"need n>=2 for paired effect sizes, got {n}"
    df=n-1
    J=1 - 3/(4*df - 1)  # standard approximation
    assert 0 <= J <= 1, f"unexpected J={J} for df={df}"
    if n <= small_n_threshold:
        tag=f"[{label}] " if label else ""
        pct=(1-J)*100
        print(
            f"WARNING {tag}n={n} (df={df}) is small-ish; Hedges' correction might be appropriate.\n"
            f"  It would multiply d_z by J≈{J:.4f} (shrink |d_z| by ~{pct:.1f}%, keep the same sign).\n"
            f"  i.e., g_z ≈ {J:.4f} * d_z")
    return J
# --- Example usage with your stats output dict ---
# stats = summarize_stats(...)
# J = warn_if_small_n_for_hedges(stats["n"], small_n_threshold=20, label="each_mouse_vs_all_models")

def rank_biserial_from_diff(diff, *, zero_tol=0.0):
    """
    Signed rank-biserial correlation for Wilcoxon signed-rank:
        r_rb = (W_plus - W_minus) / (W_plus + W_minus)
    where W_plus/W_minus are sums of ranks of |diff| for diff>0 and diff<0,
    after dropping diffs with |diff| <= zero_tol.

    This avoids ambiguity about SciPy returning min(W+,W-) and preserves sign.
    """
    diff=np.asarray(diff,dtype=float).ravel()
    assert diff.ndim==1, f"expected 1D diff, got shape {diff.shape}"
    assert np.all(np.isfinite(diff)), "diff contains NaN/Inf"

    if zero_tol==0.0:
        nz=(diff!=0.0)
    else:
        nz=(np.abs(diff)>float(zero_tol))

    d=diff[nz]
    n_eff=int(d.size)
    assert n_eff>0, "all diffs are zero (after tolerance); effect undefined"

    ranks=ss.rankdata(np.abs(d), method="average")  # tie-safe
    w_plus=float(ranks[d>0].sum())
    w_minus=float(ranks[d<0].sum())
    denom=w_plus+w_minus
    assert denom>0, "internal: denom should be >0 if n_eff>0"

    r_rb=float((w_plus-w_minus)/denom)
    return {"r_rb":r_rb,"w_plus":w_plus,"w_minus":w_minus,"denom":denom,"n_eff":n_eff}

def rrb_from_scipy_wstat(diff, *, alternative="two-sided", zero_method="wilcox", zero_tol=0.0):
    """
    Reproduce the common magnitude formula from SciPy's wilcoxon statistic:
        mag = 1 - 2*w_stat/denom
    where denom = n_eff(n_eff+1)/2.
    Note: this is magnitude-only unless you attach a sign separately.
    """
    diff=np.asarray(diff,dtype=float).ravel()
    assert np.all(np.isfinite(diff))

    w=ss.wilcoxon(diff, zero_method=zero_method, alternative=alternative, correction=False, mode="auto")
    w_stat=float(w.statistic)
    p_val=float(w.pvalue)

    if zero_tol==0.0:
        n_eff=int(np.sum(diff!=0.0))
    else:
        n_eff=int(np.sum(np.abs(diff)>float(zero_tol)))
    assert n_eff>0, "all diffs are zero (after tolerance); Wilcoxon degenerate"

    denom=float(n_eff*(n_eff+1)/2)
    mag=float(1 - 2*w_stat/denom)
    return {"w_stat":w_stat,"p_val":p_val,"mag":mag,"denom":denom,"n_eff":n_eff}

"""Compatibility alias: Some plotting cells were written against a `summarize_stats()` helper. 
This wrapper simply forwards to `get_paired_stats()` so the rest of the notebook
can stay unchanged."""
def summarize_stats(mouse_vals, sim_vals, *, test="parametric", alternative="two-sided", names=("mean_mouse","mean_sim")):
    return get_paired_stats(mouse_vals, sim_vals, names=names, test=test, alternative=alternative)

def run_case_rank_biserial_n_wstat_sanity_checks(name, diff, *, alternative="two-sided", zero_tol=0.0):
    diff=np.asarray(diff,dtype=float).ravel()
    print(f"\n=== {name} ===")
    print("diff:", diff.tolist())
    print("mean(diff):", float(diff.mean()))

    rb=rank_biserial_from_diff(diff, zero_tol=zero_tol)
    sc=rrb_from_scipy_wstat(diff, alternative=alternative, zero_tol=zero_tol)

    print("rank-biserial (signed):", rb["r_rb"])
    print("  W+:", rb["w_plus"], "W-:", rb["w_minus"], "denom:", rb["denom"], "n_eff:", rb["n_eff"])
    print("SciPy wilcoxon:", "w_stat:", sc["w_stat"], "p_val:", sc["p_val"])
    print("derived magnitude 1 - 2*w/denom:", sc["mag"], " (magnitude-only)")

    # Sanity checks:
    # 1) signed r_rb sign should match the direction of the rank mass difference
    # 2) magnitude from scipy formula should match abs(r_rb) when SciPy uses min(W+,W-)
    #    (ties/zero handling can still make this approximate; we keep a loose check)
    assert np.sign(rb["r_rb"]) in (-1.0, 0.0, 1.0)
    assert rb["r_rb"] <= 1.0 + 1e-12 and rb["r_rb"] >= -1.0 - 1e-12

    # If there is a clear mean direction, r_rb should generally agree in sign
    if not np.isclose(diff.mean(), 0.0):
        assert np.sign(rb["r_rb"]) == np.sign(diff.mean()), \
            f"sign mismatch: sign(r_rb)={np.sign(rb['r_rb'])} vs sign(mean(diff))={np.sign(diff.mean())}"

    # magnitude consistency (loose)
    assert np.isclose(abs(rb["r_rb"]), sc["mag"], rtol=1e-8, atol=1e-8), \
        f"magnitude mismatch: abs(r_rb)={abs(rb['r_rb'])} vs mag={sc['mag']}"
    
def split_early_late(dates):
    # dates: ordered (ascending) sequence of comparable items
    dates=list(dates)
    n=len(dates)
    assert n>=2, f"need >=2 dates, got {n}"

    mid=n//2
    if n%2==1:
        print('Author: note odd number of dates, so dropping the middle session')
        # drop the middle
        dates_early=dates[:mid]
        dates_late=dates[mid+1:]
    else:
        dates_early=dates[:mid]
        dates_late=dates[mid:]

    assert len(dates_early)==len(dates_late)
    assert set(dates_early).isdisjoint(set(dates_late)), "Author: early/late date sets not disjoint?"
    return sorted(list(set(dates_early))), sorted(list(set(dates_late)))

def groupby_mean_MR(df, group_cols, value_cols, validate=True):
    ## convert to expected types
    if isinstance(group_cols, str): group_cols=[group_cols]
    if isinstance(group_cols, tuple): group_cols=list(group_cols)
    if isinstance(value_cols, str): value_cols=[value_cols]

    if validate:
        ## presence checks
        for c in group_cols+list(value_cols):
            if c in group_cols and c not in df.columns:
                print(f'Author: warning: {c} not in df.columns -> removing it from consideration')
                group_cols.remove(c)
                continue

            assert c in df.columns, f"missing col: {c}"
            assert df[c].notna().all(), f"group col {c} contains NaN/NaT; dropna=True would drop rows" ## if you're going to rely on dropna=True, enforce "no missing group keys"

        ## numeric checks for values being averaged
        for c in value_cols:
            assert is_numeric_dtype(df[c]), f"value col {c} must be numeric dtype; got {df[c].dtype}"
            x=df[c].to_numpy()
            assert x.dtype!=object, f"value col {c} is object dtype; mean() may fail"
            assert np.all(np.isfinite(x.astype(float))), f"value col {c} contains NaN/Inf"

    return df.groupby(group_cols, as_index=False, dropna=True)[list(value_cols)].mean() # dropna=True should be safe we check for NaN above

def avg_over_MR(df, inds_averaged, index_mapping_dict, *, value_cols=("reward_rate_mouse","reward_rate_agent"), index_ordering=("m","e","s","r")):
    """ convert explicit index averaged over into the implicit remaining indices for groupby
    index_mapping_dict should be set of indices that fully identifies the data point being considered""" 

    ## cast to expected types
    if isinstance(value_cols, str): 
        value_cols=(value_cols,)
    if isinstance(inds_averaged, str): 
        inds_averaged=[inds_averaged]
    else: 
        inds_averaged=list(inds_averaged)

    full_index_char_set=set(index_ordering)

    inds_averaged_chars = []
    for idx in inds_averaged:
        if len(idx) == 1:
            inds_averaged_chars.append(idx)
        else:
            assert idx in index_mapping_dict, f"Author: unknown index spec: {idx}"
            inds_averaged_chars.append(index_mapping_dict[idx])

    inds_averaged_chars_set=set(inds_averaged_chars)
    assert inds_averaged_chars_set.issubset(full_index_char_set), f"Author: unexpected inds: {sorted(inds_averaged_chars_set-full_index_char_set)}"
    remaining_ind_chars_set = full_index_char_set - inds_averaged_chars_set # pandas groupby operates over the EXCLUDED column...

    ## reorder remaining inds to match standard ordering
    remaining_ind_chars=[idx for idx in index_ordering if idx in remaining_ind_chars_set] # enforce standard ordering 
    remaining_ind_cols=[index_mapping_dict[idx] for idx in remaining_ind_chars]
    assert all([len(idx) > 1 for idx in remaining_ind_cols]), "Author: expected all remaining_ind_cols to be full string col names"

    ## compute the mean over the remaining indices
    if len(remaining_ind_cols)==0:
        print('Author warning: remaining columns empty; returning mean over entire df')
        return df[list(value_cols)].mean().to_frame().T # convert back from series to df
    else:
        return groupby_mean_MR(df, remaining_ind_cols, value_cols)

def sanity_check_session_subset_df(session_subset_df, exp_moment):
    assert session_subset_df.rewarded_action_inds_mouse.astype(str).nunique() == 1, (
        f"Author: expected exactly one unique rewarded_action_inds_mouse for exp_moment {exp_moment},"
        f" but got {session_subset_df.rewarded_action_inds_mouse.nunique()} unique values")
    assert session_subset_df.states_visited_mouse.astype(str).nunique() == 1, (
        f"Author: expected exactly one unique states_visited_mouse for exp_moment {exp_moment},"
        f" but got {session_subset_df.states_visited_mouse.nunique()} unique values")
    rewarded_action_inds_mouse = session_subset_df['rewarded_action_inds_mouse'].iloc[0]
    states_visited_mouse = session_subset_df['states_visited_mouse'].iloc[0]
    assert type(rewarded_action_inds_mouse) in [list, np.ndarray] and type(states_visited_mouse) in [list, np.ndarray], (
        f"Author: expected rewarded_action_inds_mouse and states_visited_mouse to be lists or arrays,"
        f" but got {type(rewarded_action_inds_mouse)} and {type(states_visited_mouse)}")
    assert len(states_visited_mouse) >= 2, f"Author: need at least 2 states for mouse; got {len(states_visited_mouse)}"
    assert session_subset_df['full_sim_desc'].nunique() != 0, 'Author: no models found for this session for some reason...'

def sanity_check_per_model_confidence_interval_inputs(reward_rate_arr, sem_val, model_name, exp_moment):
    ## Preconditions for t-based CI on model's mean
    assert reward_rate_arr.size >= 2, (
        f"Author: need at least 2 runs per model to define a t-based CI; "
        f"got {reward_rate_arr.size} for model {model_name} in session {exp_moment}")
    assert np.all(np.isfinite(reward_rate_arr)), (
        f"Author: non-finite values in reward_rate for model {model_name} in session {exp_moment}: "
        f"{reward_rate_arr}")
    assert np.isfinite(sem_val), (
        f"Author: SEM is non-finite for model {model_name} in session {exp_moment}: sem={sem_val}, "
        f"values={reward_rate_arr}")

def sanity_check_all_agents_confidence_interval_inputs(all_agents_reward_arr, sem_all, n_models_this_session, exp_moment):
        # Sanity check: we have one mean per model
        assert all_agents_reward_arr.size == n_models_this_session, (
            f"Author: expected one mean reward per model, but got "
            f"{all_agents_reward_arr.size} means for {n_models_this_session} models "
            f"in session {exp_moment}")
        # Preconditions for across-model t-based CI
        assert all_agents_reward_arr.size >= 2, (
            f"Author: need at least 2 models to define a yoked t-based CI; "
            f"got {all_agents_reward_arr.size} in session {exp_moment}")
        assert np.all(np.isfinite(all_agents_reward_arr)), (
            f"Author: non-finite model means in all_agents_reward for session {exp_moment}: "
            f"{all_agents_reward_arr}")
        assert np.isfinite(sem_all), (
            f"Author: SEM is non-finite for yoked means in session {exp_moment}: "
            f"sem={sem_all}, values={all_agents_reward_arr}")

def model_name_2_legend_str(model_name):
    """Short/clean legend label for a model name."""
    if 'observation_action_biased' in model_name:
        return model_name.split(';')[0].replace('_', ' ')
    elif 'action_biased' in model_name:
        return (model_name.replace(';', ' ')
                         .replace('avoidReversal-False', '')
                         .replace('seqLength-', 'seq. length ')
                         .replace('_', ' ')
                         .replace('  ', ' ')
                         .replace('action biased', 'action biased\n'))
    elif 'random' in model_name:
        return (model_name.replace(';', ' ')
                         .replace('avoidReversal-False', '')
                         .replace('avoidReversal-True', 'avoid reversal')
                         .replace('seqLength-1', '')
                         .replace('_', ' ')
                         .replace('  ', ' ')
                         .replace('random', 'random\n'))
    elif 'latent_state_biased' in model_name:
        return model_name.split(';')[0].replace('_', ' ')
    elif any(s in model_name for s in ['DQN', 'PPO', 'A2C']):
        return model_name.replace(';', '-')
    else:
        raise NotImplementedError(f"Author: model_name_2_legend_str not implemented for {model_name}, please add a case for it")
    
    
    
## 251216 Author: new GPT version to support optional y axis normalization by n actions; visually verified that the normalized version matches the prior version of this function.
def plot_mice_vs_sims_per_session(selected_animal_IDs, per_exp_result_df, normalize_by_n_actions_in_session=True, run_sanity_checks=True, **kwargs):
    kwa = kwargs # shorten the name

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

        plt.figure(figsize=(17, 6))
        cyan_count = 0  # count sessions where mouse > upper bound of mean-across-models 95% CI

        ## --- consistent color mapping across all models seen for this mouse ---
        model_names = sorted(mice_subset['full_sim_desc'].unique())
        cmap = cm.get_cmap('turbo', len(model_names) if len(model_names) > 0 else 1)
        MODEL_COLOR_MAP = {name: cmap(j) for j, name in enumerate(model_names)}

        ## --- legend stubs (mouse lines & all-models CI) ---
        plt.plot([], [], '-', color='c',
                 label=f'{mouse_legend_label}\n   (exceeds 95% CI of all yoked means)',
                 linewidth=kwa['line_width_models'])
        plt.plot([], [], '-', color='g',
                 label=f'{mouse_legend_label}\n   (exceeds yoked mean)',
                 linewidth=kwa['line_width_models'])
        plt.plot([], [], '-', color='r',
                 label=f'{mouse_legend_label}\n   (under yoked mean)',
                 linewidth=kwa['line_width_models'])
        plt.plot([], [], color='black', label='yoked mean (95% CI)', linewidth=kwa['line_width_all_models'])

        sessions = list(mice_subset['exp_moment'].unique())
        for i, (exp_moment, session_subset_df) in enumerate(mice_subset.groupby('exp_moment')):

            if run_sanity_checks:
                sanity_check_session_subset_df(session_subset_df, exp_moment)

            ## mouse performance (per-session)
            rewarded_action_inds_mouse = _as_list(session_subset_df['rewarded_action_inds_mouse'].iloc[0])
            states_visited_mouse = _as_list(session_subset_df['states_visited_mouse'].iloc[0])

            n_rew_mouse = len(rewarded_action_inds_mouse)
            n_actions_mouse = _n_actions_from_states(states_visited_mouse)
            mouse_val = _metric(n_rew_mouse, n_actions_mouse)

            ## models for this session
            session_model_names = sorted(session_subset_df['full_sim_desc'].unique())
            n_models_this_session = len(session_model_names)
            assert n_models_this_session > 0, f'Author: no models found for {exp_moment}'

            ## x-positions for model CIs within the unit interval assigned to this session
            d = 1.0 / (n_models_this_session + 2)  # base spacing
            session_x_positions = i + d * np.arange(1, n_models_this_session + 1)

            ## Draw each model’s CI (vertical line)
            all_agents_reward = []
            for model_idx, (model_name, model_subset) in enumerate(session_subset_df.groupby('full_sim_desc')):

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

                """ 95% CI for this model’s mean (in the chosen metric units) """
                per_run_arr = np.asarray(per_run_vals, dtype=float)
                sem_val = ss.sem(per_run_arr)
                if run_sanity_checks:
                    sanity_check_per_model_confidence_interval_inputs(per_run_arr, sem_val, model_name, exp_moment)

                model_mean = float(np.mean(per_run_arr))
                all_agents_reward.append(model_mean)

                ci = ss.t.interval(0.95, df=per_run_arr.size - 1, loc=model_mean, scale=sem_val)

                x_offset = session_x_positions[model_idx]
                label_str = model_name_2_legend_str(model_name) if i == 0 else None
                plt.plot([x_offset]*2, ci,
                         color=MODEL_COLOR_MAP.get(model_name, '0.5'),
                         label=label_str,
                         linewidth=kwa['line_width_models'])

            """ Black reference line: 95% CI of the mean across models for this session """
            all_agents_reward_arr = np.asarray(all_agents_reward, dtype=float)
            sem_all = ss.sem(all_agents_reward_arr)
            if run_sanity_checks:
                sanity_check_all_agents_confidence_interval_inputs(all_agents_reward_arr, sem_all, n_models_this_session, exp_moment)

            agents_reward_95ci = ss.t.interval(
                0.95,
                df=all_agents_reward_arr.size - 1,
                loc=float(np.mean(all_agents_reward_arr)),
                scale=sem_all)

            plt.plot([i]*2, agents_reward_95ci, color='black', linewidth=kwa['line_width_all_models'])

            ## decide mouse-line color using EXACT same rule as cyan definition
            mouse_line_color = (
                'c' if mouse_val > agents_reward_95ci[-1]
                else ('g' if mouse_val > np.nanmean(all_agents_reward_arr) else 'r'))

            if mouse_line_color == 'c':
                cyan_count += 1

            ## draw mouse horizontal line once (from session x=i to last model x)
            x_end = float(np.max(session_x_positions))
            plt.plot([i, x_end], [mouse_val, mouse_val], '--',
                     color=mouse_line_color,
                     linewidth=kwa['line_width_mouse'],
                     alpha=kwa['mouse_point_alpha'])

        ## axes & title
        plt.tick_params(axis='both', labelsize=kwa['tick_label_font_size'])
        plt.xlabel('Experiment session', fontsize=kwa['axis_label_font_size'])
        plt.ylabel(y_label, fontsize=kwa['axis_label_font_size'])

        n_sessions = len(sessions)
        plt.title(
            f"Mouse {animal_ID}: sessions with mouse > 95% CI of all sim. yoked means = {cyan_count} / {n_sessions}",
            fontsize=kwa['title_font_size'])

        ## legend underneath
        plt.gcf().subplots_adjust(bottom=0.32)
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.22),
                   fontsize=kwa['legend_fontsize'], ncols=kwa['n_legend_cols'], frameon=True)

        plt.savefig(f"./figures/c{get_now_str(hms=True)}_{animal_ID}_figS3_reward_rates_by_session-v1.2allModels.png", bbox_inches='tight', dpi=400)
        plt.savefig(f"./figures/c{get_now_str(hms=True)}_{animal_ID}_figS3_reward_rates_by_session-v1.2allModels.svg", bbox_inches='tight', dpi=400)
        plt.savefig(f"./figures/c{get_now_str(hms=True)}_{animal_ID}_figS3_reward_rates_by_session-v1.2allModels.pdf", bbox_inches='tight', dpi=400)
        plt.show()

# def mean_and_ci(x, conf=0.95):
#     x = np.asarray(x, float); 
#     if x.size < 2 or not np.all(np.isfinite(x)): return np.nan, (np.nan, np.nan)
#     m = x.mean(); s = x.std(ddof=1)
#     if s <= 0: return m, (m, m)
#     tcrit = ss.t.ppf(0.5 + conf/2.0, x.size-1)
#     half = tcrit * (s/np.sqrt(x.size))
#     return m, (m-half, m+half)

def bonferroni(pvals):
    p=np.asarray(pvals,dtype=float)
    # if p.size==0: return p
    assert p.size != 0, "bonferroni: empty p-values array"
    assert np.isfinite(p).all(),"NaN/Inf in p-values"
    assert ((0<=p)&(p<=1)).all(),"p-values outside [0,1]"
    m=p.size
    return np.clip(p*m,0,1)

def mean_and_ci(x, conf=0.95):
    x = np.asarray(x, dtype=float)
    assert np.all(np.isfinite(x)), "mean_and_ci: NaN/Inf in input"
    n = x.size
    assert n > 1, "mean_and_ci: need at least 2 observations"
    m = x.mean()
    s = x.std(ddof=1)
    se = s/np.sqrt(n) if s > 0 else 0.0
    tcrit = ss.t.ppf(0.5 + conf/2.0, n-1)
    half = tcrit * se
    return m, (m-half, m+half)

def draw_sig_bracket(ax, x1, x2, y, h, text, lw=1.5, color='black'):
    ax.plot([x1, x1, x2, x2], [y, y+h, y+h, y], linewidth=lw, color=color)
    ax.text((x1+x2)/2.0, y+h+0.002, text, ha='center', va='bottom', color=color, fontsize=12)

def p_display_from_stats(stats_all):
    p_raw = stats_all.get("p_val_raw", np.nan)
    if np.isnan(p_raw):
        p_raw = stats_all.get("p_val", np.nan)
    if np.isfinite(p_raw) and p_raw > 0:
        return f"p={p_raw:.1e}"
    if np.isfinite(p_raw) and p_raw == 0.0:
        return f"p<{np.finfo(float).tiny:.1e}"
    return "p=NA"

def p_from_result(res):
    p = res.get("p_val_raw", np.nan)
    if not np.isfinite(p): p = res.get("p_val", np.nan)
    return p

def stars_for_p(p):
    assert np.isfinite(p), f"stars_for_p: p is not finite: {p}"
    if p < 0.001: return "***"
    elif p < 0.01: return "**"
    elif p < 0.05: return "*"
    return ""

def stats_for_df(df, stats, mouse_col, agent_col):
    """Return stats computed on *df*. If a precomputed stats dict is passed
    but its n doesn't match len(df), recompute to avoid mismatches."""
    mv = df[mouse_col].to_numpy(dtype=float)
    av = df[agent_col].to_numpy(dtype=float)
    # strict paired, finite check (will raise if violated)
    assert mv.shape == av.shape and np.all(np.isfinite(mv)) and np.all(np.isfinite(av))
    if (stats is None) or (int(stats.get("n", -1)) != len(df)):
        print('Author warning: calling summarize_stats inside stats_for_df w/ parametric test...')
        return summarize_stats(mv, av, test="parametric")
    return stats

def combined_barplots_brackets_stars(df, stats_all,
                                     mouse_col="reward_rate_mouse",
                                     agent_col="reward_rate_agent",
                                     conf=0.95, ylabel="reward rate",
                                     y_max_fixed=0.6,
                                     color_mouse='black',
                                     color_agent="0.6",
                                     title_font_size = 16,
                                     x_tick_font_size_left=14,
                                     figsize=(7,6)):  # shrink width since only 1 panel

    ## NEW: guarantee stats are computed on exactly this df
    stats_all = stats_for_df(df, stats_all, mouse_col, agent_col)

    # >>> NEW: extract t, p, n actually used for THIS plot <<<
    t_val = float(stats_all["stat"])
    p_val = float(stats_all.get("p_val_raw", stats_all.get("p_val")))
    n_all = int(stats_all["n"])
    df_all = n_all - 1
    print(f"[plot-stats] two-sided paired t-test: t({df_all}) = {t_val:.2f}, p = {p_val:.2e}")

    mv = np.asarray(df[mouse_col].values, dtype=float)
    av = np.asarray(df[agent_col].values, dtype=float)

    # m_mouse, ci_mouse, _ = mean_and_ci(mv, conf=conf) # Author: outdated format?
    # m_agent, ci_agent, _ = mean_and_ci(av, conf=conf)
    m_mouse, ci_mouse = mean_and_ci(mv, conf=conf)
    m_agent, ci_agent = mean_and_ci(av, conf=conf)

    per_mouse_min = float(0)
    per_mouse_max = float(np.nanmax([ci_mouse[1], ci_agent[1]]))
    span = (per_mouse_max - per_mouse_min) if per_mouse_max > per_mouse_min else 1.0 # Author: do we need the conditional?
    shared_ylim = (per_mouse_min, y_max_fixed)

    fig, ax_all = plt.subplots(figsize=figsize)

    x_mouse, x_agent = 0, 1 # Author: bar positions
    err_mouse = np.array([[m_mouse - ci_mouse[0]], [ci_mouse[1] - m_mouse]])
    err_agent = np.array([[m_agent - ci_agent[0]], [ci_agent[1] - m_agent]])

    ax_all.bar([x_mouse], [m_mouse], yerr=err_mouse, capsize=4, color=color_mouse, ecolor=color_mouse) # color=COLOR_MOUSE, ecolor=COLOR_MOUSE, label="mice")
    ax_all.bar([x_agent], [m_agent], yerr=err_agent, capsize=4, color=color_agent, ecolor=color_agent) # color=COLOR_AGENT, ecolor=COLOR_AGENT, label="simulations")

    ax_all.set_xticks([x_mouse, x_agent])
    ax_all.set_xticklabels(["all mice","all simulations"], fontsize=x_tick_font_size_left)
    ax_all.set_ylabel(ylabel, fontsize=YLABEL_FONTSIZE)
    ax_all.set_title("All mice vs all baseline models", fontsize=title_font_size)
    ax_all.set_ylim(shared_ylim)
    ax_all.tick_params(axis='y', labelsize=YTICK_FONTSIZE)

    # significance annotation
    p_all = p_from_result(stats_all)
    print('p_all:', p_all)
    stars_all = stars_for_p(p_all)
    n_all = int(stats_all.get("n", len(df)))
    y_top = max(ci_mouse[1], ci_agent[1])
    y = y_top + Y_OFFSET*span
    h = H_SCALE * (0.03*span)
    text_all = f"{stars_all} n = {n_all}" if stars_all else f"n = {n_all}"
    draw_sig_bracket(ax_all, x_mouse, x_agent, y, h, text_all)

    ## significance legend handles (two-sided paired t-test; n = mouse × session pairs)
    # label='*** p<0.001 (two-sided paired t-test)'),
    # label='** p<0.01 (two-sided paired t-test)'),
    # label='* p<0.05 (two-sided paired t-test)')
    #     label='*** p<0.001 (two-sided paired t-test; n = mouse × session pairs)'),
    #     label='* p<0.05 (two-sided paired t-test; n = mouse × session pairs)')

    # star_handles = [Line2D([], [], color='none', label='** p<0.01 (two-sided paired t-test\n n = mouse × session pairs)'),]
    star_handles = [
        Line2D([], [], color='none', label='*** p<0.001 (two-sided paired t-test)\n n = mouse × session pairs'),
        # Line2D([], [], color='none', label='** p<0.01 (two-sided paired t-test)\n n = mouse × session pairs'),
        # Line2D([], [], color='none', label='* p<0.05 (two-sided paired t-test)\n n = mouse × session pairs'),
    ]

    handles, labels = ax_all.get_legend_handles_labels()
    ax_all.legend(handles + star_handles, labels + [h.get_label() for h in star_handles], fontsize=LEGEND_FONTSIZE)
    return fig, ax_all

def combined_barplots_brackets_stars_2panel(df, stats_all,
                                     mouse_col="reward_rate_mouse",
                                     agent_col="reward_rate_agent",
                                     group_col="animal_ID",
                                     conf=0.95, ylabel="reward rate",
                                     y_max_fixed=0.6,
                                     title_font_size = 16,
                                     x_label_font_size=14,
                                     y_label_font_size=14,
                                     x_tick_font_size_left=14,
                                     y_tick_font_size=12,
                                     figsize=(14,6)):

    # ===== precompute per-mouse summaries (RIGHT panel) =====
    groups = []
    mouse_means, mouse_lo, mouse_hi = [], [], []
    agent_means, agent_lo, agent_hi = [], [], []
    pvals, ns = [], []
    for gid, g in df.groupby(group_col):
        gm = np.asarray(g[mouse_col].values, dtype=float)
        ga = np.asarray(g[agent_col].values, dtype=float)

        # m_m, ci_m, _ = mean_and_ci(gm, conf=conf)
        # m_a, ci_a, _ = mean_and_ci(ga, conf=conf)
        m_m, ci_m = mean_and_ci(gm, conf=conf)
        m_a, ci_a = mean_and_ci(ga, conf=conf)

        # res = summarize_stats(gm, ga, test="parametric")
        res = get_paired_stats(gm, ga, test="parametric")

        pvals.append(p_from_result(res)); ns.append(int(res.get("n", len(g))))
        groups.append(str(gid))
        mouse_means.append(m_m); mouse_lo.append(ci_m[0]); mouse_hi.append(ci_m[1])
        agent_means.append(m_a); agent_lo.append(ci_a[0]); agent_hi.append(ci_a[1])

    groups = np.array(groups)
    mouse_means = np.array(mouse_means); agent_means = np.array(agent_means)
    mouse_err = np.vstack([mouse_means - np.array(mouse_lo), np.array(mouse_hi) - mouse_means])
    agent_err = np.vstack([agent_means - np.array(agent_lo), np.array(agent_hi) - agent_means])

    # --- FIX: flatten before nanmax (right & left share ylim) ---
    per_mouse_min = float(0)
    per_mouse_max = float(np.nanmax(np.r_[mouse_hi, agent_hi]))
    span = (per_mouse_max - per_mouse_min) if per_mouse_max > per_mouse_min else 1.0
    shared_ylim = (per_mouse_min, y_max_fixed)

    fig = plt.figure(figsize=figsize, constrained_layout=True)
    gs = gridspec.GridSpec(1, 2, width_ratios=[1,2], wspace=0.05)
    ax_all = fig.add_subplot(gs[0,0])
    ax_by  = fig.add_subplot(gs[0,1], sharey=ax_all)

    # ===== RIGHT panel: per-mouse paired comparisons =====
    x = np.arange(groups.size); w = 0.35
    ax_by.bar(x - w/2, mouse_means, w, yerr=mouse_err, capsize=4,
              label="mice", color=COLOR_MOUSE, ecolor=COLOR_MOUSE)
    ax_by.bar(x + w/2, agent_means, w, yerr=agent_err, capsize=4,
              label="simulations", color=COLOR_AGENT, ecolor=COLOR_AGENT)
    ax_by.set_xticks(x)
    ax_by.set_xticklabels(groups, rotation=45, ha='right', fontsize=XTICK_FONTSIZE)
    ax_by.set_title("individual mice vs yoked-simulations", fontsize=title_font_size)
    ax_by.set_ylim(shared_ylim)
    ax_by.tick_params(axis='y', labelleft=False, labelsize=y_tick_font_size)
    ax_by.set_xlabel("mouse ID", labelpad=10, fontsize=x_label_font_size)

    for i, (m_hi, a_hi, p, n) in enumerate(zip(mouse_hi, agent_hi, pvals, ns)):
        y_top = max(m_hi, a_hi)
        y = y_top + Y_OFFSET*span
        h = H_SCALE * (0.03*span)
        stars = stars_for_p(p)
        text = f"{stars} n={n}" if stars else f"n={n}"
        draw_sig_bracket(ax_by, i - w/2, i + w/2, y, h, text)

    # --- UPDATED RIGHT legend with graded tiers (two lines optional) ---
    star_handles = [
        Line2D([], [], color='none', label='*** p<0.001 (two-sided paired t-test)\n n = per-mouse sessions'),
        Line2D([], [], color='none', label='** p<0.01 (two-sided paired t-test)\n n = per-mouse sessions'),
        Line2D([], [], color='none', label='* p<0.05 (two-sided paired t-test)\n n = per-mouse sessions'),
    ]
    handles, labels = ax_by.get_legend_handles_labels()

    # ax_by.legend(handles + star_handles, labels + [h.get_label() for h in star_handles], fontsize=LEGEND_FONTSIZE) # 251218 Author: works but crowded and overlaid over other bars

    ## 251218 GPT: put legend outside, to the right
    ax_by.legend(
        handles + star_handles,
        labels + [h.get_label() for h in star_handles],
        fontsize=LEGEND_FONTSIZE,
        loc="upper left",
        bbox_to_anchor=(1.02,1.0),  # (x,y) in axes coords; x>1 pushes it outside
        borderaxespad=0.0,)
    # make room on the right (do this once per figure)
    ax_by.figure.subplots_adjust(right=0.78)

    # ===== LEFT panel: all mice vs all simulations (pooled over models, paired across sessions) =====
    # Recompute stats on THIS df to avoid mismatches with external stats_all
    stats_all_local = stats_for_df(df, stats_all, mouse_col, agent_col)

    # (Optionally print the exact test used for the left panel)
    t_val = float(stats_all_local["stat"])
    p_val = float(stats_all_local.get("p_val_raw", stats_all_local.get("p_val")))
    n_all = int(stats_all_local["n"]); df_all = n_all - 1
    print(f"[left-panel] two-sided paired t-test: t({df_all}) = {t_val:.2f}, p = {p_val:.2e}, n = {n_all}")

    mv = np.asarray(df[mouse_col].values, dtype=float)
    av = np.asarray(df[agent_col].values, dtype=float)
    # m_mouse, ci_mouse, _ = mean_and_ci(mv, conf=conf)
    # m_agent, ci_agent, _ = mean_and_ci(av, conf=conf)
    m_mouse, ci_mouse = mean_and_ci(mv, conf=conf)
    m_agent, ci_agent = mean_and_ci(av, conf=conf)

    x_mouse, x_agent = 0, 1
    err_mouse = np.array([[m_mouse - ci_mouse[0]], [ci_mouse[1] - m_mouse]])
    err_agent = np.array([[m_agent - ci_agent[0]], [ci_agent[1] - m_agent]])

    ax_all.bar([x_mouse], [m_mouse], yerr=err_mouse, capsize=4,
               color=COLOR_MOUSE, ecolor=COLOR_MOUSE, label="mice")
    ax_all.bar([x_agent], [m_agent], yerr=err_agent, capsize=4,
               color=COLOR_AGENT, ecolor=COLOR_AGENT, label="simulations")

    ax_all.set_xticks([x_mouse, x_agent])
    ax_all.set_xticklabels(["all mice","all simulations"], fontsize=x_tick_font_size_left)
    ax_all.set_ylabel(ylabel, fontsize=y_label_font_size)
    ax_all.set_title("all mice vs all baseline models", fontsize=TITLE_FONTSIZE)  # sentence case ok
    ax_all.set_ylim(shared_ylim)
    ax_all.tick_params(axis='y', labelsize=y_tick_font_size)

    # significance annotation (stars and n wording consistent with single-panel plot)
    stars_all = stars_for_p(p_val)
    y_top = max(ci_mouse[1], ci_agent[1])
    y = y_top + Y_OFFSET*span
    h = H_SCALE * (0.03*span)
    text_all = f"{stars_all} n = {n_all} (mouse × session pairs)" if stars_all else f"n = {n_all} (mouse × session pairs)"
    draw_sig_bracket(ax_all, x_mouse, x_agent, y, h, text_all)

    # Optional: add a small legend block mirroring the right-panel legend (to be symmetric)
    star_handles_left = [
        Line2D([], [], color='none',
               label='*** p<0.001 (two-sided paired t-test)\n n = mouse × session pairs'),
        Line2D([], [], color='none',
               label='** p<0.01 (two-sided paired t-test)\n n = mouse × session pairs'),
        Line2D([], [], color='none',
               label='* p<0.05 (two-sided paired t-test)\n n = mouse × session pairs'),
    ]
    hL, lL = ax_all.get_legend_handles_labels()
    # ax_all.legend(hL + star_handles_left, lL + [h.get_label() for h in star_handles_left], fontsize=LEGEND_FONTSIZE)

    return fig, (ax_all, ax_by)

def p_from_row(row):
    p = row.get("p_val_raw", np.nan)
    if not np.isfinite(p): p = row.get("p_val", np.nan)
    return p

def significance2stars(p):
    # if not np.isfinite(p): return "ns"
    assert np.isfinite(p), "sig_code: non-finite p-value"
    if p < 1e-3: return "***"
    if p < 1e-2: return "**"
    if p < 0.05: return "*"
    return "ns"

def prepare_table(stats_by_mouse_model, model_col="full_sim_desc", group_col="animal_ID"):
    df = stats_by_mouse_model
    if isinstance(df.index, pd.MultiIndex) or (df.index.name in (group_col, model_col)):
        df = df.reset_index()
    req = {group_col, model_col, "mouse_mean", "model_mean"}
    miss = [c for c in req if c not in df.columns]
    assert not miss, f"stats_by_mouse_model missing: {miss}"
    return df

######## legacy code keeping for a bit longer for peace of mind:

# def tidy_axis_label(mkey: str) -> str:
    #     """Final formatter for x-axis labels."""
    #     label = _name(mkey)
    #     g = key_to_group(mkey)

    #     # strip group prefix if present
    #     if STRIP_GROUP_FROM_XLABELS:
    #         for pref in (f"Policy: {g}", g, f"{g} -", f"{g}:", f"{g} —"):
    #             if label.lower().startswith(pref.lower()):
    #                 label = label[len(pref):].lstrip(" -:—")
    #                 break
    #         if not label:
    #             parts = [p.strip().replace('_',' ') for p in mkey.split(';')]
    #             label = parts[1] if len(parts) > 1 else _name(mkey)

    #     # latent-state-biased: remove leading "Policy:" but keep content
    #     if g.lower() == "latent state biased":
    #         label = re.sub(r'(?i)^policy:\s*', '', label).strip()

    #     # reorder tokens and clean
    #     label = _move_avoid_reversal_to_end(label)
    #     label = _move_len_to_end(label)
    #     label = _fix_rep_prefix(label)
    #     label = _rm_empty_parens(label)
    #     return label
    # def tidy_axis_label(mkey: str) -> str:
    #     """Final formatter for x-axis labels."""
    #     label = _name(mkey)
    #     g = key_to_group(mkey)

    #     if STRIP_GROUP_FROM_XLABELS:
    #         for pref in (f"Policy: {g}", g, f"{g} -", f"{g}:", f"{g} —"):
    #             if label.lower().startswith(pref.lower()):
    #                 label = label[len(pref):].lstrip(" -:—")
    #                 break
    #         if not label:
    #             parts = [p.strip().replace('_',' ') for p in mkey.split(';')]
    #             label = parts[1] if len(parts) > 1 else _name(mkey)

    #     if g.lower() == "latent state biased":
    #         label = re.sub(r'(?i)^policy:\s*', '', label).strip()

    #     # reorder tokens and clean
    #     label = _move_avoid_reversal_to_end(label)
    #     label = _move_len_to_end(label)
    #     # comment this out to stop enforcing "Rep.: ..." 
    #     # label = _fix_rep_prefix(label)

    #     # NEW: drop "Rep.:" entirely
    #     label = re.sub(r'\bRep\.\s*:\s*', '', label)

    #     label = _rm_empty_parens(label)
    #     return label

## 251216 Author: seemingly unused/legacy. Let's leave it a little longer though.
# def mean_ci(x, conf=0.95): # pooling the models as per GPT suggestion
#     x = np.asarray(x, dtype=float)

#     # strict inputs
#     assert np.all(np.isfinite(x)), "mean_ci: input contains NaN/Inf"
#     n = x.size
#     assert n > 1, "mean_ci: need at least 2 observations"

#     m = x.mean()

#     # estimated SE of the mean with unknown sigma
#     se = ss.sem(x, ddof=1)  # same as nan_policy='omit' after NaN assert
#     # assert se > 0, f"mean_ci: zero variance -> CI undefined (handle upstream) {se} {x}"
#     if se == 0: 
#         print(f'warning se == 0 in mean_ci, returning (mean, mean) as the CI')
#         return (m, m)

#     m = x.mean()
#     h = ss.t.ppf((1 + conf) / 2, n - 1) * se
#     return (m - h, m + h)

# def run_all_comparisons(tables, *, test='parametric', alternative='two-sided',
#                         also_run_pingouin=False, p_adjust='fdr_bh'):
#     """
#     Returns a single tidy results df with comparison_name and (optionally) pingouin columns.
#     p_adjust: None or one of statsmodels multipletests methods (e.g. 'fdr_bh', 'bonferroni')
#     """
#     rows=[]

#     ## 1) all mice vs all models (from mouse_model: one row per (animal,model))
#     print('running all_mice vs all_models')
#     df= tables["mouse_model"]
#     r=get_paired_stats(df["reward_rate_mouse"],df["reward_rate_agent"],test=test,alternative=alternative)
#     rows.append({"comparison_name":"all_mice_vs_all_models__mouse_model_unit",**r})

#     ## 2) each mouse vs all models (pooled across models per session, then per mouse paired test across sessions)
#     print('running each_mouse vs all_models')
#     df= tables["mouse_session_pooled"]
#     for a_id,g in df.groupby("animal_ID",dropna=False):
#         r=get_paired_stats(g["reward_rate_mouse"],g["reward_rate_agent"],test=test,alternative=alternative)
#         rows.append({"comparison_name":"each_mouse_vs_all_models__mouse_session_unit",
#                      "animal_ID":a_id,**r})

#     ## 3) each mouse vs each model type (from session_model: session-level paired diffs for each (mouse,model))
#     print('running each_mouse vs each_model')
#     df= tables["session_model"]
#     for (a_id,model),g in df.groupby(["animal_ID","full_sim_desc"],dropna=False):
#         r=get_paired_stats(g["reward_rate_mouse"],g["reward_rate_agent"],test=test,alternative=alternative)
#         rows.append({"comparison_name":"each_mouse_vs_each_model__session_unit",
#                      "animal_ID":a_id,"full_sim_desc":model,**r})

#     out=pd.DataFrame(rows)

#     ## p-adjust within each comparison family (common “multiple comparisons” policy)
#     if p_adjust is not None:
#         try:
#             from statsmodels.stats.multitest import multipletests
#             out["p_val_adj"]=np.nan
#             out["rejected_adj"]=False

#             for cname,idx in out.groupby("comparison_name").groups.items():
#                 p=out.loc[idx,"p_val"].to_numpy(dtype=float)
#                 rej,p_adj,_,_=multipletests(p,method=p_adjust)
#                 out.loc[idx,"p_val_adj"]=p_adj
#                 out.loc[idx,"rejected_adj"]=rej
#         except Exception as e:
#             raise RuntimeError(f"p-adjust requested but failed (statsmodels missing or error): {e}")

#     # Optional: Pingouin cross-check for the same comparisons
#     # if also_run_pingouin:
#     #     pg_stat=[]
#     #     pg_p=[]
#     #     for i,row in out.iterrows():
#     #         # reconstruct the sample vectors for the specific comparison
#     #         cname=row["comparison_name"]
#     #         if cname=="all_mice_vs_all_models__mouse_model_unit":
#     #             df=tables["mouse_model"]
#     #             mouse_vals=df["reward_rate_mouse"].to_numpy()
#     #             sim_vals=df["reward_rate_agent"].to_numpy()
#     #         elif cname=="each_mouse_vs_all_models__mouse_session_unit":
#     #             df=tables["mouse_session_pooled"]
#     #             g=df[df["animal_ID"]==row["animal_ID"]]
#     #             mouse_vals=g["reward_rate_mouse"].to_numpy()
#     #             sim_vals=g["reward_rate_agent"].to_numpy()
#     #         else:
#     #             df=tables["session_model"]
#     #             g=df[(df["animal_ID"]==row["animal_ID"])&(df["full_sim_desc"]==row["full_sim_desc"])]
#     #             mouse_vals=g["reward_rate_mouse"].to_numpy()
#     #             sim_vals=g["reward_rate_agent"].to_numpy()

#     #         pg=paired_stats_pingouin(mouse_vals,sim_vals,test=test,alternative=alternative)
#     #         pg_stat.append(pg.get("stat"))
#     #         pg_p.append(pg.get("p_val"))

#     #     out["pkg_stat"]=pg_stat
#     #     out["pkg_p_val"]=pg_p
#     #     out["abs_p_diff"]=np.abs(out["p_val"]-out["pkg_p_val"])

#     return out

# def paired_stats_pingouin(mouse_vals, sim_vals, test='parametric', alternative='two-sided'):
#     import pingouin as pg  # optional dependency

#     mouse_vals=np.asarray(mouse_vals,dtype=float).ravel()
#     sim_vals=np.asarray(sim_vals,dtype=float).ravel()
#     assert mouse_vals.shape==sim_vals.shape,f"shape mismatch {mouse_vals.shape} vs {sim_vals.shape}"
#     assert mouse_vals.size>1,f"need >=2 pairs, got {mouse_vals.size}"
#     assert np.all(np.isfinite(mouse_vals)),"mouse has NaN/Inf"
#     assert np.all(np.isfinite(sim_vals)),"agent has NaN/Inf"

#     if test=='parametric':
#         # pingouin expects x,y; paired=True gives paired t-test
#         res=pg.ttest(sim_vals,mouse_vals,paired=True,alternative=alternative)
#         # columns often include: 'T', 'p-val', 'CI95%', 'cohen-d' (check exact names in your version)
#         row=res.iloc[0].to_dict()
#         # normalize names a bit
#         return {"stat":float(row.get("T")),
#                 "p_val":float(row.get("p-val")),
#                 "effect_name":"cohen_d",
#                 "effect_size":float(row.get("cohen-d")) if row.get("cohen-d") is not None else np.nan}

#     if test=='nonparametric':
#         res=pg.wilcoxon(sim_vals,mouse_vals,alternative=alternative)
#         row=res.iloc[0].to_dict()
#         # pingouin often returns W-val + p-val + RBC or CLES depending on version
#         return {"stat":float(row.get("W-val")) if row.get("W-val") is not None else np.nan,
#                 "p_val":float(row.get("p-val")),
#                 "effect_name":"rbc",
#                 "effect_size":float(row.get("RBC")) if row.get("RBC") is not None else np.nan}

#     raise ValueError(f"unknown test={test}")