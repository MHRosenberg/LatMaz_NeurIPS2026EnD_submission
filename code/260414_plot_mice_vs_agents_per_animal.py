#!/usr/bin/env python3
"""
260414_plot_mice_vs_agents_per_animal.py

Per-animal bar charts: mouse RPA vs canonical agent models.
Supports two session sets (select via SESSION_SET at bottom of file):

  '60'  — canonical 60-session set (a031=24, a033=36). All 23 agents available.
           Run by default.

  '152' — expanded 152-session set (9 animals: a001, a002, a003, a029, a030,
           a031, a033, a185, a188). Requires:
             data_out/rl_sims/c*_expanded92_no_training_agents.csv
           No-training agents only (POMCP, FSC_bio, heuristics, cloning);
           RL training agents (DQN, DRQN, RecurrentSAC, etc.) are unavailable
           for the older 92 sessions.

Ported from Author's Marimo notebooks:
  code/1b_aa_260205b_plot_mice_vs_all_sims-mrm-DEV.py
  code/1a_aa_260204_latMaz_analyze_mice_vs_all_sims-mrm-STABLE.py

Produces:
  reports/figures/c{ts}_per_animal_{set}_overview.png
  reports/figures/c{ts}_per_animal_{set}_{animal}_agents.png
  reports/figures/c{ts}_per_session_{set}_{animal}.png
"""
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
from matplotlib import gridspec
from scipy import stats as ss

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR   = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
DATA_DIR     = PROJECT_ROOT / 'data_out' / 'rl_sims'
FIG_DIR      = PROJECT_ROOT / 'reports' / 'figures'
FIG_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(PROJECT_ROOT / 'code'))

from experiment_config import load_yoking_df, filter_sessions
from utils_latMaz import get_most_recent_file

TS = 'c' + datetime.now().strftime('%y%m%d-%H%M%S')

# ---------------------------------------------------------------------------
# Agent color families (matches cld benchmark report conventions)
# ---------------------------------------------------------------------------
COLOR_MOUSE   = 'black'
COLOR_MOUSE_BAR = '#1a1a1a'

AGENT_FAMILY_COLORS = {
    'POMCP':        '#2ca02c',   # green
    'POMCP_bio':    '#17becf',   # cyan
    'RecurrentSAC': '#9467bd',   # purple
    'DQN':          '#1f77b4',   # blue
    'DRQN':         '#aec7e8',   # light blue
    'PPO':          '#ff7f0e',   # orange
    'A2C':          '#ffbb78',   # light orange
    'TRPO':         '#d62728',   # red
    'QRDQN':        '#ff9896',   # light red
    'RecurrentPPO': '#e377c2',   # pink
    'FSC_bio':      '#bcbd22',   # yellow-green
    'VarMarkov':    '#8c564b',   # brown
    'OOI':          '#c49c94',   # light brown
    'NoBkFullFwd':  '#7f7f7f',   # mid-gray
    'FullFwd':      '#c7c7c7',   # light gray
    'Clone':        '#dbdb8d',   # pale yellow
    'Greedy_oracle':'#98df8a',   # light green
    'Random_forward':'#d9d9d9',  # very light gray
    'Random':       '#bdbdbd',   # gray
}

def _family_color(model):
    for key, col in AGENT_FAMILY_COLORS.items():
        if model.startswith(key):
            return col
    return '#aaaaaa'


# ---------------------------------------------------------------------------
# Statistics helpers (adapted from utils_latMaz_stats_n_plotting.py)
# ---------------------------------------------------------------------------

def mean_and_ci(x, conf=0.95):
    x = np.asarray(x, dtype=float)
    assert x.size > 1 and np.all(np.isfinite(x))
    m = x.mean()
    se = x.std(ddof=1) / np.sqrt(x.size)
    tcrit = ss.t.ppf(0.5 + conf / 2.0, x.size - 1)
    half = tcrit * se
    return m, (m - half, m + half)


def paired_ttest(a, b):
    """Two-sided paired t-test; returns (t, p)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    assert a.shape == b.shape and a.size > 1
    return ss.ttest_rel(a, b)


def stars_for_p(p, bonf_n=1):
    p_adj = min(p * bonf_n, 1.0)
    if p_adj < 0.001: return '***'
    if p_adj < 0.01:  return '**'
    if p_adj < 0.05:  return '*'
    return 'ns'


def draw_bracket(ax, x1, x2, y, h, text, lw=1.5, color='black', fontsize=11):
    ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=lw, color=color)
    ax.text((x1 + x2) / 2, y + h + 0.002, text,
            ha='center', va='bottom', color=color, fontsize=fontsize)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _get_session_set(mode='60'):
    """Return (session_set, label) for mode '60' or '152'."""
    yoke = load_yoking_df()
    if mode == '60':
        sessions = set(filter_sessions(yoke))
        return sessions, '60sess'
    elif mode == '152':
        fbr_path = str(DATA_DIR / 'c260331-182639_fbr_152sessions.csv')
        assert os.path.exists(fbr_path), f'FBR 152-session CSV not found: {fbr_path}'
        fbr = pd.read_csv(fbr_path)
        sessions = set(fbr['exp_moment'].unique())
        return sessions, '152sess'
    else:
        raise ValueError(f"Unknown mode '{mode}'. Use '60' or '152'.")


def load_mouse_rpa(sessions):
    """Return DataFrame: exp_moment, animal_ID, rpa_mouse for given session set."""
    yoke = load_yoking_df()
    sub = yoke[yoke['exp_moment'].isin(sessions)].copy()
    sub['n_actions'] = sub['n_states_visited'] - 1
    sub['rpa_mouse'] = sub['n_rewards'] / sub['n_actions']
    return sub[['exp_moment', 'animal_ID', 'rpa_mouse']].reset_index(drop=True)


def load_all_agents(sessions, mode='60'):
    """Return DataFrame: exp_moment, model, rpa_agent (mean over seeds per session).

    mode='60':  all 23 agents (canonical CSVs only)
    mode='152': no-training agents from both canonical CSVs and expanded92 CSV
    """
    records = []

    def _add(df, model_col='model', rpa_col='rpa', reward_col='total_reward',
             act_col='n_actions', model_override=None):
        if df is None or df.empty:
            return
        if 'rpa' not in df.columns:
            df = df.copy()
            df['rpa'] = df[reward_col] / df[act_col]
        if model_override:
            df = df.copy()
            df['model'] = model_override
        if model_col != 'model':
            df = df.rename(columns={model_col: 'model'})
        grouped = df.groupby(['exp_moment', 'model'])['rpa'].mean().reset_index()
        grouped.rename(columns={'rpa': 'rpa_agent'}, inplace=True)
        records.append(grouped)

    def _load(fname):
        p = DATA_DIR / fname
        if p.exists():
            return pd.read_csv(p)
        print(f'  WARNING: {fname} not found')
        return pd.DataFrame()

    def _filter(df):
        if df is None or df.empty or 'exp_moment' not in df.columns:
            return df if df is not None else pd.DataFrame()
        return df[df['exp_moment'].isin(sessions)].copy()

    # Always load canonical 60-session results (apply session filter for 60-mode)
    sess60 = set(filter_sessions(load_yoking_df()))

    def _filter_canonical(df):
        """Filter to canonical 60 sessions (not the full session set)."""
        if df is None or df.empty or 'exp_moment' not in df.columns:
            return df if df is not None else pd.DataFrame()
        return df[df['exp_moment'].isin(sessions & sess60)].copy()

    # HPO RL agents (canonical only; not run on expanded sessions)
    _add(_filter_canonical(_load('c260316-171035_hpo_tuning_FIXED.csv')))
    # Baseline heuristics (canonical)
    _add(_filter_canonical(_load('c260318-144507_baseline_heuristics.csv')))
    # Intermediate agents (canonical)
    _add(_filter_canonical(_load('c260318-144446_intermediate_agents.csv')))
    # FSC_bio corrected params (canonical)
    df_fsc = _filter_canonical(_load('c260331-200128_fsc_bio_correct_params.csv'))
    _add(df_fsc)
    # POMCP_bio (canonical)
    _add(_filter_canonical(_load('c260316-171035_pomcp_bio_FIXED.csv')))
    # RecurrentSAC standard (canonical only)
    df_rsac = _filter_canonical(_load('c260315-020734_recurrent_sac_d32_benchmark.csv'))
    if not df_rsac.empty:
        df_rsac = df_rsac.copy()
        df_rsac['variant'] = df_rsac['variant'].replace({'recurrent_sac_d32': 'RecurrentSAC'})
        _add(df_rsac, model_col='variant', rpa_col='rpa')
    # RecurrentSAC causal (canonical only)
    df_rsac_c = _filter_canonical(_load('c260327-172440_recurrent_sac_causal.csv'))
    if not df_rsac_c.empty:
        _add(df_rsac_c, model_override='RecurrentSAC_causal', rpa_col='rpa')
    # Cloning / heuristic agents (canonical)
    _add(_filter_canonical(_load('c260413-103634_cloning_agents.csv')))

    if mode == '152':
        # Load the expanded92 CSV (no-training agents on older 92 sessions)
        exp_pattern = str(DATA_DIR / '*_expanded92_no_training_agents.csv')
        exp_path = get_most_recent_file(exp_pattern)
        if exp_path:
            exp_df = pd.read_csv(exp_path)
            older_sessions = sessions - sess60
            _add(_filter(exp_df))
            print(f'  Loaded expanded92 CSV: {os.path.basename(exp_path)} '
                  f'({len(exp_df)} rows, covering {exp_df["exp_moment"].nunique()} sessions)')
        else:
            print('  WARNING: expanded92 CSV not found. Run '
                  '260414_run_expanded152_no_training_agents.py first.')

    if not records:
        raise RuntimeError('No agent CSVs loaded.')

    agents = pd.concat(records, ignore_index=True)
    # Drop FSC_bio from intermediate agents if the corrected-params file was loaded
    # (keep only the most recent FSC_bio result)
    if not df_fsc.empty:
        agents = agents.drop_duplicates(subset=['exp_moment', 'model'], keep='last')
    return agents


# ---------------------------------------------------------------------------
# Figure 1: 2-panel overview (left = all pooled, right = per-animal)
# ---------------------------------------------------------------------------

def fig_overview(mouse_df, agents_df, tag='60sess'):
    """
    Compare mouse vs ALL agents pooled (mean over models per session).

    Left panel: single bar pair — all 60 sessions combined.
    Right panel: per-animal bar pairs (a031, a033).
    """
    # Per-session mean over all agent models
    pool = agents_df.groupby('exp_moment')['rpa_agent'].mean().reset_index()
    df = mouse_df.merge(pool, on='exp_moment', how='inner')
    assert len(df) > 0, f'No sessions in merged df'

    CONF = 0.95
    Y_MAX = 0.55
    SPAN = Y_MAX

    fig = plt.figure(figsize=(12, 5), constrained_layout=True)
    gs = gridspec.GridSpec(1, 2, figure=fig, width_ratios=[1, 2], wspace=0.08)
    ax_all = fig.add_subplot(gs[0, 0])
    ax_by  = fig.add_subplot(gs[0, 1], sharey=ax_all)

    # ---- LEFT: all mice vs pooled agents ----
    mv = df['rpa_mouse'].values
    av = df['rpa_agent'].values
    m_m, ci_m = mean_and_ci(mv, CONF)
    m_a, ci_a = mean_and_ci(av, CONF)
    t, p = paired_ttest(mv, av)
    n = len(mv)

    err_m = np.array([[m_m - ci_m[0]], [ci_m[1] - m_m]])
    err_a = np.array([[m_a - ci_a[0]], [ci_a[1] - m_a]])
    ax_all.bar([0], [m_m], yerr=err_m, capsize=5, color=COLOR_MOUSE_BAR, ecolor=COLOR_MOUSE_BAR)
    ax_all.bar([1], [m_a], yerr=err_a, capsize=5, color='0.6', ecolor='0.6')
    ax_all.set_xticks([0, 1])
    ax_all.set_xticklabels(['all mice', 'all agents\n(pooled)'], fontsize=12)
    ax_all.set_ylabel('reward rate (n rewards / n actions)', fontsize=12)
    ax_all.set_title('all mice vs all agents', fontsize=13)
    ax_all.set_ylim(0, Y_MAX)
    # significance annotation
    y_top = max(ci_m[1], ci_a[1])
    draw_bracket(ax_all, 0, 1, y_top + 0.01 * SPAN, 0.015 * SPAN,
                 f'{stars_for_p(p)} n={n}', fontsize=11)
    print(f'[all pooled] t={t:.2f}, p={p:.2e}, n={n}')

    # ---- RIGHT: per-animal ----
    animals = sorted(df['animal_ID'].unique())
    x = np.arange(len(animals))
    w = 0.35
    mm_list, ma_list = [], []
    ci_mm_lo, ci_mm_hi = [], []
    ci_ma_lo, ci_ma_hi = [], []
    pvals, ns = [], []

    for aid in animals:
        sub = df[df['animal_ID'] == aid]
        gmv = sub['rpa_mouse'].values
        gav = sub['rpa_agent'].values
        m_m2, ci_m2 = mean_and_ci(gmv, CONF)
        m_a2, ci_a2 = mean_and_ci(gav, CONF)
        t2, p2 = paired_ttest(gmv, gav)
        mm_list.append(m_m2); ci_mm_lo.append(ci_m2[0]); ci_mm_hi.append(ci_m2[1])
        ma_list.append(m_a2); ci_ma_lo.append(ci_a2[0]); ci_ma_hi.append(ci_a2[1])
        pvals.append(float(p2)); ns.append(len(gmv))
        print(f'[animal {aid}] t={t2:.2f}, p={p2:.2e}, n={len(gmv)}')

    mm_arr = np.array(mm_list); ma_arr = np.array(ma_list)
    err_mm = np.vstack([mm_arr - np.array(ci_mm_lo), np.array(ci_mm_hi) - mm_arr])
    err_ma = np.vstack([ma_arr - np.array(ci_ma_lo), np.array(ci_ma_hi) - ma_arr])

    b_m = ax_by.bar(x - w/2, mm_arr, w, yerr=err_mm, capsize=5,
                    color=COLOR_MOUSE_BAR, ecolor=COLOR_MOUSE_BAR, label='mouse')
    b_a = ax_by.bar(x + w/2, ma_arr, w, yerr=err_ma, capsize=5,
                    color='0.6', ecolor='0.6', label='agents (pooled)')
    ax_by.set_xticks(x)
    ax_by.set_xticklabels([f'a{a}' for a in animals], fontsize=12)
    ax_by.set_xlabel('mouse ID', fontsize=12)
    ax_by.set_title('individual mice vs pooled agents', fontsize=13)
    ax_by.tick_params(axis='y', labelleft=False)
    ax_by.set_ylim(0, Y_MAX)

    for i, (mhi, ahi, p, n) in enumerate(zip(ci_mm_hi, ci_ma_hi, pvals, ns)):
        y_top = max(mhi, ahi)
        draw_bracket(ax_by, i - w/2, i + w/2, y_top + 0.01 * SPAN, 0.015 * SPAN,
                     f'{stars_for_p(p)} n={n}', fontsize=11)

    ax_by.legend(fontsize=11, loc='upper right')

    out = FIG_DIR / f'{TS}_per_animal_{tag}_overview.png'
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {out.name}')
    return out


# ---------------------------------------------------------------------------
# Figure 2: sorted bar chart — all agents vs one animal (or all combined)
# ---------------------------------------------------------------------------

DISPLAY_NAMES = {
    'POMCP':              'POMCP',
    'POMCP_bio':          'POMCP_bio',
    'RecurrentSAC_causal': 'RecSAC-causal',
    'RecurrentSAC':       'RecSAC',
    'FSC_bio':            'FSC_bio',
    'DQN':                'DQN',
    'DRQN_seq':           'DRQN-seq',
    'DRQN_rand':          'DRQN-rand',
    'QRDQN':              'QRDQN',
    'RecurrentPPO':       'RecurrentPPO',
    'PPO':                'PPO',
    'A2C':                'A2C',
    'TRPO':               'TRPO',
    'VarMarkov':          'VarMarkov',
    'OOI':                'OOI',
    'NoBkFullFwd':        'NoBk+FullFwd',
    'FullFwd':            'FullFwd',
    'Clone_ego':          'Clone-ego',
    'Clone_allo_real':    'Clone-allo-r',
    'Clone_allo_latent':  'Clone-allo-l',
    'Greedy_oracle':      'GreedyOracle',
    'Random_forward':     'Rnd+Fwd',
    'Random':             'Random',
}


def fig_animal_sorted_agents(mouse_df, agents_df, animal_id=None, tag='60sess'):
    """
    For a given animal (or all combined if animal_id is None), plot all agent
    models as bars sorted by mean RPA descending.  Mouse RPA shown as dashed
    horizontal reference line.  Per-model paired t-test vs mouse with Bonferroni.

    animal_id: '031', '033', or None for all combined.
    """
    if animal_id is not None:
        m_sub = mouse_df[mouse_df['animal_ID'] == animal_id].copy()
        a_sub = agents_df[agents_df['exp_moment'].isin(m_sub['exp_moment'])].copy()
        label = f'a{animal_id}'
        n_sess = len(m_sub)
    else:
        m_sub = mouse_df.copy()
        a_sub = agents_df.copy()
        label = 'all'
        n_sess = len(m_sub)

    mouse_rpa_per_sess = m_sub.set_index('exp_moment')['rpa_mouse']

    models = sorted(a_sub['model'].unique())
    n_models = len(models)
    BONF_N = n_models  # per-figure Bonferroni correction

    rows = []
    for model in models:
        mod_df = a_sub[a_sub['model'] == model].copy()
        # per-session mean over seeds
        sess_rpa = mod_df.groupby('exp_moment')['rpa_agent'].mean()
        # align with mouse sessions
        common = sess_rpa.index.intersection(mouse_rpa_per_sess.index)
        if len(common) < 2:
            print(f'  WARNING: {model} has <2 sessions in common, skipping')
            continue
        av = sess_rpa.loc[common].values
        mv = mouse_rpa_per_sess.loc[common].values
        mean_rpa = float(av.mean())
        sem_rpa  = float(av.std(ddof=1) / np.sqrt(len(av)))
        t, p = paired_ttest(mv, av)
        rows.append({
            'model': model,
            'mean_rpa': mean_rpa,
            'sem': sem_rpa,
            'n': len(common),
            't': float(t),
            'p': float(p),
            'stars': stars_for_p(float(p), bonf_n=BONF_N),
        })

    df_plot = pd.DataFrame(rows).sort_values('mean_rpa', ascending=False).reset_index(drop=True)

    mouse_mean = float(mouse_rpa_per_sess.mean())
    mouse_sem  = float(mouse_rpa_per_sess.std(ddof=1) / np.sqrt(len(mouse_rpa_per_sess)))

    # --- plot ---
    fig_h = max(5, 0.45 * len(df_plot))
    fig, ax = plt.subplots(figsize=(10, fig_h))

    colors = [_family_color(m) for m in df_plot['model']]
    y_pos  = np.arange(len(df_plot))

    ax.barh(y_pos, df_plot['mean_rpa'], xerr=df_plot['sem'],
            color=colors, ecolor='0.3', capsize=3, height=0.7, error_kw={'lw': 1.2})

    # mouse reference lines (mean ± SEM)
    ax.axvline(mouse_mean, color='black', lw=2.0, ls='--', label=f'mouse {label} (mean)')
    ax.axvspan(mouse_mean - mouse_sem, mouse_mean + mouse_sem,
               alpha=0.12, color='black', label='mouse ±SEM')

    # significance text to the right of each bar
    x_text = df_plot['mean_rpa'] + df_plot['sem'] + 0.003
    for i, row in df_plot.iterrows():
        s = row['stars']
        if s and s != 'ns':
            col = 'firebrick' if row['mean_rpa'] < mouse_mean else 'seagreen'
            ax.text(x_text.iloc[i], i, s, va='center', ha='left', fontsize=9, color=col, fontweight='bold')
        else:
            ax.text(x_text.iloc[i], i, 'ns', va='center', ha='left', fontsize=8, color='0.5')

    display_labels = [DISPLAY_NAMES.get(m, m) for m in df_plot['model']]
    ax.set_yticks(y_pos)
    ax.set_yticklabels(display_labels, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel('reward rate (RPA)', fontsize=12)
    ax.set_title(f'Individual mouse {label} vs simulations  (n_sess={n_sess}, Bonferroni n={BONF_N})',
                 fontsize=12)
    ax.legend(fontsize=10, loc='lower right')

    # legend: family color patches + mouse reference
    seen = {}
    for m, c in zip(df_plot['model'], colors):
        fam = next((k for k in AGENT_FAMILY_COLORS if m.startswith(k)), m)
        if fam not in seen:
            seen[fam] = c
    patches = [mpatches.Patch(color=c, label=fam) for fam, c in seen.items()]
    ref_handles = [
        mlines.Line2D([], [], color='black', ls='--', lw=2, label=f'mouse {label} mean'),
        mpatches.Patch(color='black', alpha=0.25, label='mouse ±SEM'),
    ]
    ax.legend(handles=patches + ref_handles, fontsize=8, loc='lower right',
              ncol=3, title='Agent family')

    fname = f'a{animal_id}' if animal_id else 'combined'
    out = FIG_DIR / f'{TS}_per_animal_{tag}_{fname}_agents.png'
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {out.name}')
    return out


# ---------------------------------------------------------------------------
# Figure 3: per-session timeseries for one animal
# ---------------------------------------------------------------------------

def fig_per_session_timeseries(mouse_df, agents_df, animal_id, tag='60sess'):
    """
    Per-session timeseries: mouse RPA and agent 95% CI band across sessions.
    Sessions sorted chronologically by exp_moment.
    One line per major agent family; mouse shown as bold black.
    """
    m_sub = mouse_df[mouse_df['animal_ID'] == animal_id].copy()
    a_sub = agents_df[agents_df['exp_moment'].isin(m_sub['exp_moment'])].copy()

    sessions = sorted(m_sub['exp_moment'].unique())
    sess_idx  = {s: i for i, s in enumerate(sessions)}
    xs = np.arange(len(sessions))

    # Mouse RPA per session
    mouse_rpa = m_sub.set_index('exp_moment')['rpa_mouse'].reindex(sessions)

    # Per-model per-session mean RPA
    pivot = a_sub.groupby(['exp_moment', 'model'])['rpa_agent'].mean().unstack('model')
    pivot = pivot.reindex(sessions)

    # 95% CI band across all agents (for each session)
    all_vals = pivot.values
    ci_lo = np.nanpercentile(all_vals, 2.5, axis=1)
    ci_hi = np.nanpercentile(all_vals, 97.5, axis=1)
    ci_mid = np.nanmean(all_vals, axis=1)

    fig, ax = plt.subplots(figsize=(14, 4))

    # Per-session: 95% CI of agent mean RPAs across models (within-session)
    # For each session compute mean ± 95% CI across models, then check if mouse exceeds CI hi
    ci_lo_sess = np.full(len(sessions), np.nan)
    ci_hi_sess = np.full(len(sessions), np.nan)
    for i, sess in enumerate(sessions):
        row = all_vals[i]
        finite = row[np.isfinite(row)]
        if len(finite) > 1:
            _, ci = mean_and_ci(finite, conf=0.95)
            ci_lo_sess[i] = ci[0]
            ci_hi_sess[i] = ci[1]

    ax.fill_between(xs, ci_lo_sess, ci_hi_sess, alpha=0.2, color='steelblue',
                    label='agents: 95% CI across models')
    ax.plot(xs, ci_mid, '--', color='steelblue', lw=1.2, label='agents: mean')

    # Mouse
    ax.plot(xs, mouse_rpa.values, 'o-', color='black', lw=2, ms=6,
            label=f'mouse a{animal_id}')

    # Mark sessions where mouse > 95% CI upper bound (within-session)
    exceeds = mouse_rpa.values > ci_hi_sess
    n_exceed = int(np.nansum(exceeds))
    ax.scatter(xs[exceeds], mouse_rpa.values[exceeds],
               color='red', zorder=5, s=60, label=f'mouse > 95% CI upper ({n_exceed}/{len(xs)})')

    ax.set_xticks(xs)
    ax.set_xticklabels(range(len(sessions)), fontsize=8)
    ax.set_xlabel('Experiment session (chronological)', fontsize=11)
    ax.set_ylabel('reward rate (RPA)', fontsize=11)
    ax.set_title(f'Mouse a{animal_id}: per-session RPA vs agent 95% CI (across models)  '
                 f'(above={n_exceed}/{len(xs)})', fontsize=12)
    ax.legend(fontsize=9, loc='upper left')
    ax.set_ylim(bottom=0)

    out = FIG_DIR / f'{TS}_per_session_{tag}_a{animal_id}.png'
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {out.name}')
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def run_all_figures(mode='60'):
    """Generate all figures for the given session mode ('60' or '152')."""
    sessions, set_label = _get_session_set(mode)
    print(f'\nSession set: {set_label} ({len(sessions)} sessions)')

    print('\n--- Loading mouse data ---')
    mouse_df = load_mouse_rpa(sessions)
    animals = sorted(mouse_df['animal_ID'].unique())
    print(f'  {len(mouse_df)} sessions, animals: {[f"a{a}" for a in animals]}')
    for a in animals:
        n = (mouse_df['animal_ID'] == a).sum()
        print(f'    a{a}: {n} sessions')

    print('\n--- Loading agent data ---')
    agents_df = load_all_agents(sessions, mode=mode)
    models = sorted(agents_df['model'].unique())
    print(f'  {len(agents_df)} rows, {len(models)} models: {models}')

    tag = set_label  # e.g. '60sess' or '152sess'

    print(f'\n--- Figure 1: overview 2-panel ({tag}) ---')
    fig_overview(mouse_df, agents_df, tag=tag)

    print(f'\n--- Figure 2: sorted agents per animal ({tag}) ---')
    for aid in animals:
        fig_animal_sorted_agents(mouse_df, agents_df, animal_id=aid, tag=tag)
    fig_animal_sorted_agents(mouse_df, agents_df, animal_id=None, tag=tag)

    print(f'\n--- Figure 3: per-session timeseries ({tag}) ---')
    for aid in animals:
        fig_per_session_timeseries(mouse_df, agents_df, animal_id=aid, tag=tag)

    print(f'\nDone. All figures saved to {FIG_DIR}')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['60', '152', 'both'], default='60',
                        help='Session set to use (default: 60)')
    args = parser.parse_args()

    print(f'Timestamp: {TS}')
    if args.mode == 'both':
        run_all_figures('60')
        run_all_figures('152')
    else:
        run_all_figures(args.mode)
