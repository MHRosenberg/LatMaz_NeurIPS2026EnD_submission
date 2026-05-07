#!/usr/bin/env python3
"""
260405_generate_paper_values.py

Generate paper/paper_values.tex — all \newcommand definitions for paper statistics.
Each statistic is computed from canonical CSVs (260316 master rerun, 60-session set).

Usage:
    conda activate latMaz_RL
    python code/260405_generate_paper_values.py

Outputs:
    paper/paper_values.tex              — LaTeX \newcommand definitions (overwritten)
    data_out/rl_sims/c{ts}_paper_values.json  — timestamped JSON of all values

Macro naming convention:
    CamelCase, letters only (no numbers, underscores, or symbols).
    Grouped by: Dataset / Mouse / Heuristics / Agents / StatTests / Pretrain / DoseResponse

SD convention:
    All SDs are computed as std of per-session mean RPAs (one value per session,
    averaged over seeds). This is the session-to-session variability.

Stat test sign convention:
    Mouse vs heuristic (rows 1-2): t = ttest_rel(mouse, heuristic)
        positive t → mouse > heuristic
    Agent vs mouse (rows 3+): t = ttest_rel(agent, mouse)
        positive t → agent > mouse
    RecSAC vs POMCP: t = ttest_rel(RecSAC, POMCP)
        negative t → POMCP > RecSAC
"""
import json
import sys
import os
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import networkx as nx
from scipy import stats

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
# Layout-agnostic project root: walk up looking for data_released/ or paper/.
# Works for code/<script>.py (depth 2),
# code/<script>.py (depth 1), and submission code/<script>.py (depth 1).
def _find_project_root(start: Path) -> Path:
    cur = start
    for _ in range(5):
        if (cur / 'data_released').is_dir() or (cur / 'paper').is_dir():
            return cur
        if cur.parent == cur:
            break
        cur = cur.parent
    return start.parent.parent  # legacy fallback (depth-2 dev layout)
PROJECT_ROOT = _find_project_root(SCRIPT_DIR)
# Reviewers reproduce paper values from data_released/results/ first; falls
# back to data_out/rl_sims/ for in-development re-runs (260505).
RELEASED_DATA_DIR = PROJECT_ROOT / 'data_released' / 'results'
DEV_DATA_DIR = PROJECT_ROOT / 'data_out' / 'rl_sims'
DATA_DIR = RELEASED_DATA_DIR if RELEASED_DATA_DIR.exists() else DEV_DATA_DIR
PAPER_DIR = PROJECT_ROOT / 'paper'

sys.path.insert(0, str(SCRIPT_DIR))
from experiment_config import load_yoking_df, filter_sessions
from utils_latMaz import load_maze

TS = 'c' + datetime.now().strftime('%y%m%d-%H%M%S')

# Canonical CSVs — all from 260316 master rerun (fully corrected rebait, 60 sessions)
FILES = {
    'hpo':         DATA_DIR / 'c260316-171035_hpo_tuning_FIXED.csv',
    'heuristics':  DATA_DIR / 'c260318-144507_baseline_heuristics.csv',
    'intermediate':DATA_DIR / 'c260318-144446_intermediate_agents.csv',  # FSC_bio, OOI, VarMarkov
    'recsac':      DATA_DIR / 'c260315-020734_recurrent_sac_d32_benchmark.csv',
    'recsac_causal':DATA_DIR / 'c260327-172440_recurrent_sac_causal.csv',
    'fsc_bio':     DATA_DIR / 'c260331-200128_fsc_bio_correct_params.csv',  # timeout=1, greediness=3 — paper params
    'pomcp_bio':   DATA_DIR / 'c260316-171035_pomcp_bio_FIXED.csv',
    'pretrain':    DATA_DIR / 'c260316-171035_pretrain_FIXED.csv',
    'target_pretrain': DATA_DIR / 'c260316-171035_target_session_pretrain_FIXED.csv',
    'cloning':     DATA_DIR / 'c260413-103634_cloning_agents.csv',  # FullFwd, NoBkFullFwd, Clone variants
}

PAPER_AGENT_MAP = {
    # paper_key: internal CSV/model name, paper-facing label, source CSV key
    'Random': {'model': 'Random', 'label': 'Random', 'source': 'heuristics'},
    'Fwd': {'model': 'Random_forward', 'label': 'Random_forward', 'source': 'heuristics'},
    'Greedy': {'model': 'Greedy_oracle', 'label': 'Greedy_oracle', 'source': 'heuristics'},
    'Pomcp': {'model': 'POMCP/fewer_sims', 'label': 'POMCP (c=20)', 'source': 'hpo'},
    'PomcpBio': {'model': 'POMCP_bio', 'label': 'Frontier-Plan', 'source': 'pomcp_bio'},
    'RecSac': {'model': 'RecurrentSAC_D32', 'label': 'RecurrentSAC D=32', 'source': 'recsac'},
    'RecSacCausal': {'model': 'RecurrentSAC_causal', 'label': 'RecurrentSAC (causal)', 'source': 'recsac_causal'},
    'Dqn': {'model': 'DQN', 'label': 'DQN', 'source': 'hpo'},
    'DqnPre': {'model': 'DQN/prior_sessions_same_maze', 'label': 'DQN pretrained', 'source': 'pretrain'},
    'DrqnSeq': {'model': 'DRQN_seq', 'label': 'DRQN_seq', 'source': 'hpo'},
    'DrqnRand': {'model': 'DRQN_rand', 'label': 'DRQN_rand', 'source': 'hpo'},
    'Qrdqn': {'model': 'QRDQN', 'label': 'QRDQN', 'source': 'hpo'},
    'Trpo': {'model': 'TRPO', 'label': 'TRPO', 'source': 'hpo'},
    'Ppo': {'model': 'PPO', 'label': 'PPO', 'source': 'hpo'},
    'Atwoc': {'model': 'A2C', 'label': 'A2C', 'source': 'hpo'},
    'RecPpo': {'model': 'RecurrentPPO', 'label': 'RecurrentPPO', 'source': 'hpo'},
    'FscBio': {'model': 'FSC_bio', 'label': 'FSC_bio', 'source': 'intermediate'},
    'FullFwd': {'model': 'FullFwd', 'label': 'FullFwd', 'source': 'cloning'},
    'NoBkFullFwd': {'model': 'NoBkFullFwd', 'label': 'NoBk+FullFwd', 'source': 'cloning'},
    'CloneEgo': {'model': 'Clone_ego', 'label': 'Clone (ego)', 'source': 'cloning'},
    'CloneAlloReal': {'model': 'Clone_allo_real', 'label': 'Clone (allo-real)', 'source': 'cloning'},
    'CloneAlloLatent': {'model': 'Clone_allo_latent', 'label': 'Clone (allo-latent)', 'source': 'cloning'},
}

N_BONF = 7  # number of comparisons for Bonferroni correction

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
def load_all():
    yoked = load_yoking_df()
    # Canonical-60 = exact HPO-tuning session set (a031, a033). Pinning here
    # so paper values are reproducible after the yoking df grows (260505;
    # filter_sessions(>251001, n>50) had drifted to 166 sessions).
    hpo_path = FILES['hpo']
    if hpo_path.exists():
        sessions_60 = set(pd.read_csv(hpo_path)['exp_moment'].unique())
    else:
        sessions_60 = set(filter_sessions(yoked))
    print(f'  60-session set: {len(sessions_60)} sessions')

    mouse = yoked[yoked['exp_moment'].isin(sessions_60)].copy()
    mouse['n_actions'] = mouse['n_states_visited'] - 1
    mouse['rpa'] = mouse['n_rewards'] / mouse['n_actions']

    dfs = {'mouse': mouse, 'sessions_60': sessions_60}
    missing_files = []
    for name, path in FILES.items():
        if not path.exists():
            missing_files.append((name, path))
            continue
        df = pd.read_csv(path)
        if 'exp_moment' in df.columns:
            df = df[df['exp_moment'].isin(sessions_60)]
        dfs[name] = df
        print(f'  Loaded {name}: {len(df)} rows from {path.name}')
    if missing_files:
        # 260507 v12: was `print('WARNING'); dfs[name] = pd.DataFrame()` per
        # file (silent empty-df fallback). Downstream stat code processed
        # the empty frame and produced nonsense values without crashing
        # loudly. Now: collect missing files, then fail-fast with a clear
        # multi-file error before any stat is computed.
        msg = ['ERROR: required CSV(s) not found in DATA_DIR:']
        for name, path in missing_files:
            msg.append(f'  - {name}: {path}')
        msg.append('')
        msg.append('All FILES entries above feed into headline numbers in')
        msg.append('paper_values.tex. Set DATA_DIR (env var) to the dir')
        msg.append('containing them, or ship the missing files.')
        raise FileNotFoundError('\n'.join(msg))
    return dfs


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def add_rpa(df, rew_col='total_reward', act_col='n_actions'):
    df = df.copy()
    if 'rpa' not in df.columns:
        df['rpa'] = df[rew_col] / df[act_col]
    return df


def per_session_stats(df, group_col='exp_moment', rpa_col='rpa',
                      rew_col='total_reward', act_col='n_actions'):
    """Return per-session mean RPAs (averaged over seeds within each session)."""
    return df.groupby(group_col)[rpa_col].mean()


def macro_rpa(df, group_col='exp_moment', rew_col='total_reward', act_col='n_actions'):
    """Global total rewards / total actions (averaged per session first, then sum)."""
    sess = df.groupby(group_col)[[rew_col, act_col]].mean()
    return sess[rew_col].sum() / sess[act_col].sum()


def win_rate(agent_sess_rpa, mouse_rpa):
    """Fraction of sessions where agent mean RPA > mouse RPA."""
    common = agent_sess_rpa.index.intersection(mouse_rpa.index)
    return float((agent_sess_rpa.loc[common] > mouse_rpa.loc[common]).sum()) / len(common)


def ci95(series):
    """95% CI half-width for per-session values."""
    series = pd.Series(series).dropna()
    if len(series) == 0:
        return 0.0
    return round(1.96 * float(series.std()) / np.sqrt(len(series)), 3)


def pct_ceiling(rpa, random_rpa, oracle_rpa):
    """Percent of the random-to-POMCP dynamic range."""
    denom = oracle_rpa - random_rpa
    if denom == 0:
        return 0.0
    return round((float(rpa) - random_rpa) / denom * 100, 1)


def sci_latex(x, sig=2):
    """Format a p-value as LaTeX scientific notation, e.g. 1.3\\times10^{-4}."""
    if x == 0:
        return '0'
    exp = int(np.floor(np.log10(abs(x))))
    mantissa = x / (10 ** exp)
    mantissa_rounded = round(mantissa, sig - 1)
    if mantissa_rounded == 10.0:
        mantissa_rounded = 1.0
        exp += 1
    # Format mantissa: drop trailing zeros after decimal
    m_str = f'{mantissa_rounded:.{sig-1}f}'.rstrip('0').rstrip('.')
    return f'{m_str}\\times10^{{{exp}}}'


def fmt(x, decimals=3):
    """Format a number to given decimal places as a string."""
    return f'{x:.{decimals}f}'


def fmt2(x):
    return fmt(x, 2)


def fmt3(x):
    return fmt(x, 3)


def pct_str(x, decimals=1):
    """Format percentage (pass already-multiplied value)."""
    return f'{x:.{decimals}f}\\%'


# ---------------------------------------------------------------------------
# Statistical tests
# ---------------------------------------------------------------------------
def ttest_paired(vals_a, vals_b, n_bonf=N_BONF):
    """Paired t-test for vals_a vs vals_b on common sessions.
    Returns (t, p_raw, p_bonf, d, n) with sign: positive if a > b.
    """
    if isinstance(vals_a, pd.Series) and isinstance(vals_b, pd.Series):
        common = vals_a.index.intersection(vals_b.index)
        a = vals_a.loc[common].values
        b = vals_b.loc[common].values
    else:
        a, b = np.array(vals_a), np.array(vals_b)
        assert len(a) == len(b), 'Arrays must have equal length'

    t, p_raw = stats.ttest_rel(a, b)
    diff = a - b
    d = float(diff.mean() / diff.std()) if diff.std() > 0 else 0.0
    p_bonf = min(float(p_raw) * n_bonf, 1.0)
    n = len(a)
    return float(t), float(p_raw), p_bonf, d, n


# ---------------------------------------------------------------------------
# Agent performance from HPO CSV
# ---------------------------------------------------------------------------
def hpo_agent_stats(hpo_df, mouse_rpa_sess, oracle_rpa, n_bonf=N_BONF, random_rpa=None):
    """Compute stats for all HPO-tuned agents using their best config.

    pct_ceiling = (RPA - Random) / (Oracle - Random) * 100  when random_rpa is given;
    pct_omni    = RPA / Oracle * 100  (legacy, retained for backwards compatibility).
    """
    hpo_df = add_rpa(hpo_df)
    results = {}
    for model, grp in hpo_df.groupby('model'):
        best_cfg = grp.groupby('hpo_config')['rpa'].mean().idxmax()
        best = grp[grp['hpo_config'] == best_cfg]
        sess_rpa = per_session_stats(best)
        m_rpa = macro_rpa(best)
        wr = win_rate(sess_rpa, mouse_rpa_sess)
        sd = sess_rpa.std()
        pct_omni = m_rpa / oracle_rpa * 100
        pct_ceiling = ((m_rpa - random_rpa) / (oracle_rpa - random_rpa) * 100
                       if random_rpa is not None else None)
        results[model] = {
            'config': best_cfg,
            'macro_rpa': round(m_rpa, 4),
            'sd': round(sd, 4),
            'pct_omni': round(pct_omni, 1),
            'pct_ceiling': round(pct_ceiling, 1) if pct_ceiling is not None else None,
            'win_rate': round(wr * 100, 0),
            'n_sessions': len(sess_rpa),
        }
    return results


# ---------------------------------------------------------------------------
# Main computation
# ---------------------------------------------------------------------------
def compute_all(dfs):
    mouse = dfs['mouse']
    mouse_rpa_sess = mouse.set_index('exp_moment')['rpa']
    mouse_rew_sess = mouse.set_index('exp_moment')['n_rewards']
    n_actions_sess = mouse.set_index('exp_moment')['n_actions']

    V = {}  # all values dict

    # ---- Dataset characteristics ----
    V['NSessions'] = len(mouse)
    V['NAnimals'] = mouse['animal_ID'].nunique()
    V['NMazes'] = mouse['adj_file'].nunique()
    V['NAgents'] = 12  # fixed: 12 agents evaluated (Table 1 count)
    hpo_raw = dfs['hpo']
    hpo_model_free = hpo_raw[hpo_raw['model'] != 'POMCP'] if 'model' in hpo_raw.columns else hpo_raw
    V['NHpoSims'] = int(len(hpo_model_free))
    V['NHpoConfigs'] = int(hpo_model_free.groupby(['model', 'hpo_config']).ngroups)
    V['NHpoSimsAll'] = int(len(hpo_raw))
    V['NHpoConfigsAll'] = int(hpo_raw.groupby(['model', 'hpo_config']).ngroups)
    V['BudgetMin'] = int(mouse['n_actions'].min())
    V['BudgetMax'] = int(mouse['n_actions'].max())
    V['BudgetMean'] = round(float(mouse['n_actions'].mean()), 1)
    V['BudgetMedian'] = int(mouse['n_actions'].median())

    # Per-animal session counts. Keep legacy A/B macros for manuscript
    # compatibility, but derive them from explicit sorted animal IDs.
    animal_counts = mouse.groupby('animal_ID').size().sort_index()
    animal_ids = list(animal_counts.index)
    V['AnimalIdA'] = str(animal_ids[0])
    V['AnimalIdB'] = str(animal_ids[1])
    V['SessAnimalA'] = int(animal_counts.iloc[0])
    V['SessAnimalB'] = int(animal_counts.iloc[1])

    # Per-maze session counts. Keep legacy A/B/C macros for manuscript
    # compatibility, but derive them from explicit sorted maze filenames.
    maze_counts = mouse.groupby('adj_file').size().sort_index()
    maze_files = list(maze_counts.index)
    V['MazeFileA'] = str(maze_files[0])
    V['MazeFileB'] = str(maze_files[1])
    V['MazeFileC'] = str(maze_files[2])
    V['SessMazeA'] = int(maze_counts.iloc[0])
    V['SessMazeB'] = int(maze_counts.iloc[1])
    V['SessMazeC'] = int(maze_counts.iloc[2])

    # Maze topology from upstream maze files, not hardcoded constants.
    maze_stats = []
    for adj_file in maze_files:
        st_pos_file = mouse.loc[mouse['adj_file'] == adj_file, 'st_pos_file'].iloc[0]
        adj_mat, _ = load_maze(
            PROJECT_ROOT / 'data_in' / 'mazes' / adj_file,
            PROJECT_ROOT / 'data_in' / 'mazes' / st_pos_file,
        )
        graph = nx.from_numpy_array(np.asarray(adj_mat))
        nonisolated = [n for n, degree in graph.degree() if degree > 0]
        if nonisolated:
            subgraph = graph.subgraph(nonisolated)
            navigable_nodes = max(nx.connected_components(subgraph), key=len)
            navigable_graph = subgraph.subgraph(navigable_nodes)
        else:
            navigable_graph = graph
        maze_stats.append({
            'adj_file': adj_file,
            'nodes': int(navigable_graph.number_of_nodes()),
            'edges': int(navigable_graph.number_of_edges()),
            'sessions': int(maze_counts.loc[adj_file]),
        })
    node_values = [m['nodes'] for m in maze_stats]
    session_weights = [m['sessions'] for m in maze_stats]
    V['MazeNodeMin'] = int(min(node_values))
    V['MazeNodeMax'] = int(max(node_values))
    V['MazeNodeMeanApprox'] = int(round(float(np.average(node_values, weights=session_weights))))
    V['MazeRewardMeanApprox'] = V['MazeNodeMeanApprox']
    V['MazeNodesA'] = maze_stats[0]['nodes']
    V['MazeEdgesA'] = maze_stats[0]['edges']
    V['MazeNodesB'] = maze_stats[1]['nodes']
    V['MazeEdgesB'] = maze_stats[1]['edges']
    V['MazeNodesC'] = maze_stats[2]['nodes']
    V['MazeEdgesC'] = maze_stats[2]['edges']
    V['RecSacNEpochs'] = 100  # epochs RecSAC trains meta-RL style

    # ---- Mouse performance ----
    mouse_macro = float(mouse['n_rewards'].sum() / mouse['n_actions'].sum())
    V['MouseRpa'] = round(mouse_macro, 3)
    V['MouseRpaSD'] = round(float(mouse_rpa_sess.std()), 3)
    V['MouseMeanRew'] = round(float(mouse['n_rewards'].mean()), 1)

    # ---- Heuristics ----
    hur = add_rpa(dfs['heuristics'])
    rand = hur[hur['model'] == 'Random']
    fwd = hur[hur['model'] == 'Random_forward']
    greedy = hur[hur['model'] == 'Greedy_oracle']
    V['RandomRpa'] = round(float(macro_rpa(rand)), 3)
    V['FwdRpa'] = round(float(macro_rpa(fwd)), 3)
    V['GreedyRpa'] = round(float(macro_rpa(greedy)), 3)

    rand_sess = per_session_stats(rand)
    fwd_sess = per_session_stats(fwd)

    # Random_forward Table-1 stats (added 260501 — for the NoBk-only row)
    _fwd_macro = float(macro_rpa(fwd))
    V['FwdRpaSD']    = round(float(fwd_sess.std()), 3)
    V['FwdRpaCI']    = round(1.96 * float(fwd_sess.std()) / np.sqrt(len(fwd_sess)), 3)
    V['FwdWinRate']  = int(round(win_rate(fwd_sess, mouse_rpa_sess) * 100))

    # ---- POMCP oracle (fewer_sims from HPO CSV) ----
    hpo_df = add_rpa(dfs['hpo'])
    pomcp_fs = hpo_df[(hpo_df['model'] == 'POMCP') & (hpo_df['hpo_config'] == 'fewer_sims')]
    pomcp_macro = float(macro_rpa(pomcp_fs))
    pomcp_sess = per_session_stats(pomcp_fs)
    V['PomcpRpa'] = round(pomcp_macro, 3)
    V['PomcpRpaSD'] = round(float(pomcp_sess.std()), 3)
    V['PomcpWinRate'] = int(round(win_rate(pomcp_sess, mouse_rpa_sess) * 100))
    V['PomcpPctOmni'] = round(pomcp_macro / pomcp_macro * 100, 1)  # 100.0 by definition
    V['PomcpMeanRew'] = round(float(pomcp_fs.groupby('exp_moment')['total_reward'].mean().mean()), 1)
    V['FwdPctOmni']  = round(_fwd_macro / pomcp_macro * 100, 1)

    oracle_rpa = pomcp_macro  # use for % omniscient throughout

    # ---- HPO agents (best config per model) ----
    hpo_stats = hpo_agent_stats(hpo_df[hpo_df['model'] != 'POMCP'], mouse_rpa_sess, oracle_rpa)

    def _hpo(model):
        return hpo_stats.get(model, {})

    # DQN
    dqn_s = _hpo('DQN')
    V['DqnRpa'] = dqn_s.get('macro_rpa', 0)
    V['DqnRpaSD'] = dqn_s.get('sd', 0)
    V['DqnPctOmni'] = dqn_s.get('pct_omni', 0)
    V['DqnWinRate'] = int(dqn_s.get('win_rate', 0))
    dqn_best_cfg = dqn_s.get('config', 'default')
    dqn_best = hpo_df[(hpo_df['model'] == 'DQN') & (hpo_df['hpo_config'] == dqn_best_cfg)]
    dqn_sess_rpa = per_session_stats(dqn_best)

    # DRQN_seq
    drqn_seq_s = _hpo('DRQN_seq')
    V['DrqnSeqRpa'] = drqn_seq_s.get('macro_rpa', 0)
    V['DrqnSeqRpaSD'] = drqn_seq_s.get('sd', 0)
    V['DrqnSeqPctOmni'] = drqn_seq_s.get('pct_omni', 0)
    V['DrqnSeqWinRate'] = int(drqn_seq_s.get('win_rate', 0))
    drqn_seq_best_cfg = drqn_seq_s.get('config', 'default')
    drqn_seq_best = hpo_df[(hpo_df['model'] == 'DRQN_seq') & (hpo_df['hpo_config'] == drqn_seq_best_cfg)]
    drqn_seq_sess_rpa = per_session_stats(drqn_seq_best)

    # DRQN_rand
    drqn_rand_s = _hpo('DRQN_rand')
    V['DrqnRandRpa'] = drqn_rand_s.get('macro_rpa', 0)
    V['DrqnRandRpaSD'] = drqn_rand_s.get('sd', 0)
    V['DrqnRandPctOmni'] = drqn_rand_s.get('pct_omni', 0)
    V['DrqnRandWinRate'] = int(drqn_rand_s.get('win_rate', 0))

    # QRDQN
    qrdqn_s = _hpo('QRDQN')
    V['QrdqnRpa'] = qrdqn_s.get('macro_rpa', 0)
    V['QrdqnRpaSD'] = qrdqn_s.get('sd', 0)
    V['QrdqnPctOmni'] = qrdqn_s.get('pct_omni', 0)
    V['QrdqnWinRate'] = int(qrdqn_s.get('win_rate', 0))

    # TRPO
    trpo_s = _hpo('TRPO')
    V['TrpoRpa'] = trpo_s.get('macro_rpa', 0)
    V['TrpoRpaSD'] = trpo_s.get('sd', 0)
    V['TrpoPctOmni'] = trpo_s.get('pct_omni', 0)
    V['TrpoWinRate'] = int(trpo_s.get('win_rate', 0))

    # PPO
    ppo_s = _hpo('PPO')
    V['PpoRpa'] = ppo_s.get('macro_rpa', 0)
    V['PpoRpaSD'] = ppo_s.get('sd', 0)
    V['PpoPctOmni'] = ppo_s.get('pct_omni', 0)
    V['PpoWinRate'] = int(ppo_s.get('win_rate', 0))

    # A2C
    atwoc_s = _hpo('A2C')
    V['AtwocRpa'] = atwoc_s.get('macro_rpa', 0)
    V['AtwocRpaSD'] = atwoc_s.get('sd', 0)
    V['AtwocPctOmni'] = atwoc_s.get('pct_omni', 0)
    V['AtwocWinRate'] = int(atwoc_s.get('win_rate', 0))

    # RecurrentPPO
    recppo_s = _hpo('RecurrentPPO')
    V['RecPpoRpa'] = recppo_s.get('macro_rpa', 0)
    V['RecPpoRpaSD'] = recppo_s.get('sd', 0)
    V['RecPpoPctOmni'] = recppo_s.get('pct_omni', 0)
    V['RecPpoWinRate'] = int(recppo_s.get('win_rate', 0))

    # ---- RecurrentSAC D=32 ----
    rsac_df = add_rpa(dfs['recsac'])
    rsac_sess = per_session_stats(rsac_df)
    rsac_macro = float(macro_rpa(rsac_df))
    V['RecSacRpa'] = round(rsac_macro, 3)
    V['RecSacRpaSD'] = round(float(rsac_sess.std()), 3)
    V['RecSacPctOmni'] = round(rsac_macro / oracle_rpa * 100, 1)
    V['RecSacWinRate'] = int(round(win_rate(rsac_sess, mouse_rpa_sess) * 100))
    V['RecSacMeanRew'] = round(float(rsac_df.groupby('exp_moment')['total_reward'].mean().mean()), 1)
    V['RecSacXMouse'] = round(rsac_macro / mouse_macro, 1)
    V['RecSacGapPct'] = round((rsac_macro - V['RandomRpa']) / (oracle_rpa - V['RandomRpa']) * 100, 0)

    # ---- RecurrentSAC (causal) ----
    rcausal_df = dfs['recsac_causal']
    rcausal_sess = rcausal_df.groupby('exp_moment')['rpa'].mean()
    rcausal_macro = float(macro_rpa(rcausal_df, rew_col='total_reward', act_col='n_actions')
                          if 'n_actions' in rcausal_df.columns
                          else rcausal_sess.mean())  # fallback using per-session RPA mean
    # Since causal CSV has rpa and total_reward but no n_actions col, use the rpa col directly
    # Macro RPA = mean of per-session means (equivalent since all sessions same budget scaling)
    rcausal_macro = float(rcausal_sess.mean())
    V['RecSacCausalRpa'] = round(rcausal_macro, 3)
    V['RecSacCausalRpaSD'] = round(float(rcausal_df['rpa'].std()), 3)  # raw SD (as in paper)
    V['RecSacCausalPctOmni'] = round(rcausal_macro / oracle_rpa * 100, 1)
    V['RecSacCausalWinRate'] = int(round(win_rate(rcausal_sess, mouse_rpa_sess) * 100))

    # ---- FSC_bio (intermediate agents CSV, default params) ----
    inter_df = add_rpa(dfs['intermediate'])
    fsc_inter = inter_df[inter_df['model'] == 'FSC_bio']
    fsc_sess = per_session_stats(fsc_inter)
    fsc_macro = float(macro_rpa(fsc_inter))
    V['FscBioRpa'] = round(fsc_macro, 3)
    V['FscBioRpaSD'] = round(float(fsc_sess.std()), 3)
    V['FscBioPctOmni'] = round(fsc_macro / oracle_rpa * 100, 1)
    V['FscBioWinRate'] = int(round(win_rate(fsc_sess, mouse_rpa_sess) * 100))

    # ---- POMCP_bio ----
    pbio_df = add_rpa(dfs['pomcp_bio'])
    pbio_sess = per_session_stats(pbio_df)
    pbio_macro = float(macro_rpa(pbio_df))
    V['PomcpBioRpaPooled'] = round(pbio_macro, 3)
    V['PomcpBioRpa'] = round(float(pbio_sess.mean()), 3)   # Table 1 convention
    V['PomcpBioRpaMean'] = round(float(pbio_sess.mean()), 3)
    V['PomcpBioRpaSD'] = round(float(pbio_sess.std()), 3)
    V['PomcpBioPctOmni'] = round(float(pbio_sess.mean()) / oracle_rpa * 100, 1)
    V['PomcpBioWinRate'] = int(round(win_rate(pbio_sess, mouse_rpa_sess) * 100))
    V['PomcpBioRpaOld'] = '0.75'  # pre-rebait-fix value (hardcoded historical)

    # Ratio POMCP_bio / omniscient
    V['PomcpBioPctOmniRatio'] = round(float(pbio_sess.mean()) / oracle_rpa * 100, 1)

    # ---- Pretrained DQN (Table 2) ----
    # Pretraining CSV: DQN default config, prior_sessions_same_maze
    pre_df = dfs['pretrain']
    MODEL_PRETRAIN_CONFIGS = {
        'DQN': 'default',
        'DRQN_seq': 'large_net',
        'DRQN_rand': 'large_net',
    }
    pre_stats = {}
    for model, cfg in MODEL_PRETRAIN_CONFIGS.items():
        m_df = pre_df[(pre_df['model'] == model) & (pre_df['hpo_config'] == cfg)]
        # Display values: per-session mean total_reward (averaged across seeds)
        baseline_sess = m_df[m_df['pretrain_type'] == 'none'].groupby('exp_moment')['total_reward'].mean()
        pretrain_sess = m_df[m_df['pretrain_type'] == 'prior_sessions_same_maze'].groupby('exp_moment')['total_reward'].mean()
        common = baseline_sess.index.intersection(pretrain_sess.index)
        b_rew = baseline_sess.loc[common].values
        p_rew = pretrain_sess.loc[common].values
        diff_rew = p_rew - b_rew
        # Statistical tests (t, p, Cohen's d): use per-session mean RPA so the
        # paired-test treats sessions as the unit of replication regardless of
        # action budget. (260506: the prior code used total_reward here, which
        # weights longer-budget sessions more heavily and gives a different
        # answer; the audit-fix in paper_values.tex flagged the discrepancy.)
        m_df_rpa = add_rpa(m_df)
        baseline_rpa_sess = per_session_stats(
            m_df_rpa[m_df_rpa['pretrain_type'] == 'none'])
        pretrain_rpa_sess = per_session_stats(
            m_df_rpa[m_df_rpa['pretrain_type'] == 'prior_sessions_same_maze'])
        b_rpa = baseline_rpa_sess.loc[common].values
        p_rpa = pretrain_rpa_sess.loc[common].values
        diff_rpa = p_rpa - b_rpa
        t, p_raw = stats.ttest_rel(p_rpa, b_rpa)
        d = float(diff_rpa.mean() / diff_rpa.std()) if diff_rpa.std() > 0 else 0.0
        p_bonf = min(float(p_raw) * N_BONF, 1.0)
        pct_improve = (diff_rew.mean() / baseline_sess.mean()) * 100  # display: pct based on rewards
        pre_stats[model] = {
            'base_mean_rew': round(float(baseline_sess.mean()), 1),
            'pre_mean_rew': round(float(pretrain_sess.mean()), 1),
            'delta': round(float(diff_rew.mean()), 1),
            'pct': round(float(pct_improve), 0),
            'cohen_d': round(float(d), 2),
            't': round(float(t), 2),
            'p_raw': float(p_raw),
            'p_bonf': float(p_bonf),
            'n': len(common),
        }

    # DQN pretrain
    ps = pre_stats['DQN']
    V['DqnBaseMeanRew'] = ps['base_mean_rew']
    V['DqnPreMeanRew'] = ps['pre_mean_rew']
    V['DqnPreDeltaRew'] = ps['delta']
    V['DqnPrePctImprove'] = int(abs(int(ps['pct'])))
    V['DqnPreCohenD'] = ps['cohen_d']
    V['NSessPretrain'] = ps['n']  # 54 sessions with pretrain data

    # DQN pretrained RPA (from HPO CSV using default config, pretrained)
    dqn_pre_df = pre_df[(pre_df['model'] == 'DQN') & (pre_df['hpo_config'] == 'default') &
                        (pre_df['pretrain_type'] == 'prior_sessions_same_maze')]
    dqn_pre_df = add_rpa(dqn_pre_df)
    dqn_pre_sess = per_session_stats(dqn_pre_df)
    dqn_pre_macro = float(macro_rpa(dqn_pre_df))
    V['DqnPreRpa'] = round(dqn_pre_macro, 3)
    V['DqnPreRpaSD'] = round(float(dqn_pre_sess.std()), 3)
    V['DqnPrePctOmni'] = round(dqn_pre_macro / oracle_rpa * 100, 1)
    V['DqnPreWinRate'] = int(round(win_rate(dqn_pre_sess, mouse_rpa_sess) * 100))

    # Mouse mean reward in pretrain subset (54 sessions)
    common_54 = dqn_pre_sess.index
    mouse_rew_54 = mouse_rew_sess.loc[common_54]
    V['MouseMeanRewPretrain'] = round(float(mouse_rew_54.mean()), 1)

    # Gap closing: (DQN_pre - DQN_base) / (POMCP - DQN_base)
    V['DqnPreGapClose'] = round(
        (V['DqnPreMeanRew'] - V['DqnBaseMeanRew']) /
        (V['PomcpMeanRew'] - V['DqnBaseMeanRew']) * 100, 0)

    # DRQN_seq pretrain
    ps_seq = pre_stats['DRQN_seq']
    V['DrqnSeqBaseMeanRew'] = ps_seq['base_mean_rew']
    V['DrqnSeqPreMeanRew'] = ps_seq['pre_mean_rew']
    V['DrqnSeqPreDeltaRew'] = ps_seq['delta']
    V['DrqnSeqPrePctImprove'] = int(abs(int(ps_seq['pct'])))
    V['DrqnSeqPreCohenD'] = ps_seq['cohen_d']

    # DRQN_rand pretrain (same data as seq for this CSV)
    ps_rand = pre_stats['DRQN_rand']
    V['DrqnRandBaseMeanRew'] = ps_rand['base_mean_rew']
    V['DrqnRandPreMeanRew'] = ps_rand['pre_mean_rew']
    V['DrqnRandPreDeltaRew'] = ps_rand['delta']
    V['DrqnRandPrePctImprove'] = int(abs(int(ps_rand['pct'])))
    V['DrqnRandPreCohenD'] = ps_rand['cohen_d']

    # ---- Dose-response ----
    dqn_pre_dose = pre_df[(pre_df['model'] == 'DQN') & (pre_df['hpo_config'] == 'default') &
                          (pre_df['pretrain_type'] == 'prior_sessions_same_maze')]
    dqn_base_dose = pre_df[(pre_df['model'] == 'DQN') & (pre_df['hpo_config'] == 'default') &
                           (pre_df['pretrain_type'] == 'none')]
    dose_agg = dqn_pre_dose.groupby('exp_moment').agg(
        mean_reward=('total_reward', 'mean'),
        n_pretrain=('n_pretrain_sessions', 'first')
    )
    base_agg = dqn_base_dose.groupby('exp_moment')['total_reward'].mean().rename('baseline')
    dose_merged = dose_agg.join(base_agg, how='inner')
    dose_merged['delta'] = dose_merged['mean_reward'] - dose_merged['baseline']
    rho, p_spearman = stats.spearmanr(dose_merged['n_pretrain'], dose_merged['delta'])
    V['DoseSpearmanRho'] = round(float(rho), 3)
    V['DoseSpearmanP'] = round(float(p_spearman), 3)
    V['DoseN'] = int(dose_merged['n_pretrain'].nunique())

    # ---- Statistical tests (7 comparisons, Bonferroni n=7) ----
    # 1. Mouse vs Random (positive t → mouse > random)
    t, p_raw, p_bonf, d, n = ttest_paired(mouse_rpa_sess, rand_sess)
    V['TMouseVsRandom'] = round(t, 2)
    V['PrawMouseVsRandom'] = p_raw
    V['PbonfMouseVsRandom'] = p_bonf
    V['DMouseVsRandom'] = round(d, 2)
    V['NMouseVsRandom'] = n

    # 2. Mouse vs Forward (negative t → forward > mouse)
    t, p_raw, p_bonf, d, n = ttest_paired(mouse_rpa_sess, fwd_sess)
    V['TMouseVsFwd'] = round(t, 2)
    V['PrawMouseVsFwd'] = p_raw
    V['PbonfMouseVsFwd'] = p_bonf
    V['DMouseVsFwd'] = round(d, 2)
    V['NMouseVsFwd'] = n

    # 3. DQN vs Mouse (positive t → DQN > mouse)
    t, p_raw, p_bonf, d, n = ttest_paired(dqn_sess_rpa, mouse_rpa_sess)
    V['TDqnVsMouse'] = round(t, 2)
    V['PrawDqnVsMouse'] = p_raw
    V['PbonfDqnVsMouse'] = p_bonf
    V['DDqnVsMouse'] = round(d, 2)
    V['NDqnVsMouse'] = n

    # 4. DRQN_seq vs Mouse
    t, p_raw, p_bonf, d, n = ttest_paired(drqn_seq_sess_rpa, mouse_rpa_sess)
    V['TDrqnSeqVsMouse'] = round(t, 2)
    V['PrawDrqnSeqVsMouse'] = p_raw
    V['PbonfDrqnSeqVsMouse'] = p_bonf
    V['DDrqnSeqVsMouse'] = round(d, 2)
    V['NDrqnSeqVsMouse'] = n

    # 5. RecSAC vs Mouse
    t, p_raw, p_bonf, d, n = ttest_paired(rsac_sess, mouse_rpa_sess)
    V['TRecSacVsMouse'] = round(t, 2)
    V['PrawRecSacVsMouse'] = p_raw
    V['PbonfRecSacVsMouse'] = p_bonf
    V['DRecSacVsMouse'] = round(d, 2)
    V['NRecSacVsMouse'] = n

    # 6. RecSAC vs POMCP (negative t → POMCP > RecSAC)
    t, p_raw, p_bonf, d, n = ttest_paired(rsac_sess, pomcp_sess)
    V['TRecSacVsPomcp'] = round(t, 2)
    V['PrawRecSacVsPomcp'] = p_raw
    V['PbonfRecSacVsPomcp'] = p_bonf
    V['DRecSacVsPomcp'] = round(d, 2)
    V['NRecSacVsPomcp'] = n

    # 7. DQN pretrained vs baseline (positive t → pretrained > baseline)
    # Uses per-session-mean RPA, matching the rest of the paired tests in this
    # file. (260506: prior code used total_reward here, weighting longer-budget
    # sessions more heavily — see audit-fix in pre_stats above.)
    dqn_pre_rpa_sess = per_session_stats(add_rpa(
        pre_df[(pre_df['model'] == 'DQN') & (pre_df['hpo_config'] == 'default') &
               (pre_df['pretrain_type'] == 'prior_sessions_same_maze')]))
    dqn_base_rpa_sess = per_session_stats(add_rpa(
        pre_df[(pre_df['model'] == 'DQN') & (pre_df['hpo_config'] == 'default') &
               (pre_df['pretrain_type'] == 'none')]))
    t, p_raw, p_bonf, d, n = ttest_paired(dqn_pre_rpa_sess, dqn_base_rpa_sess)
    V['TDqnPreVsBase'] = round(t, 2)
    V['PrawDqnPreVsBase'] = p_raw
    V['PbonfDqnPreVsBase'] = p_bonf
    V['DDqnPreVsBase'] = round(d, 2)
    V['NDqnPreVsBase'] = n

    # DQN pretrained vs Mouse (additional, for prose line 275) — also RPA-based
    t, p_raw, _, d, n = ttest_paired(dqn_pre_rpa_sess, mouse_rpa_sess)
    V['TDqnPreVsMouse'] = round(t, 2)
    V['PrawDqnPreVsMouse'] = p_raw
    V['DDqnPreVsMouse'] = round(d, 2)
    V['NDqnPreVsMouse'] = n

    # ---- Cloning agents (FullFwd, NoBkFullFwd, Clone variants) ----
    # CSV: c260413-103634_cloning_agents.csv from 260413_run_cloning_agents.py
    cln_df = add_rpa(dfs['cloning']) if len(dfs.get('cloning', pd.DataFrame())) > 0 else None
    _cln_sess = {}  # store per-session RPAs for stat tests below
    if cln_df is not None:
        for _model, _key in [
            ('FullFwd',           'FullFwd'),
            ('NoBkFullFwd',       'NoBkFullFwd'),
            ('Clone_ego',         'CloneEgo'),
            ('Clone_allo_real',   'CloneAlloReal'),
            ('Clone_allo_latent', 'CloneAlloLatent'),
        ]:
            sub = cln_df[cln_df['model'] == _model]
            if len(sub) == 0:
                continue
            _sess = per_session_stats(sub)
            _macro = float(macro_rpa(sub))
            V[f'{_key}Rpa'] = round(_macro, 3)
            V[f'{_key}RpaSD'] = round(float(_sess.std()), 3)
            V[f'{_key}WinRate'] = int(round(win_rate(_sess, mouse_rpa_sess) * 100))
            V[f'{_key}PctOmni'] = round(_macro / oracle_rpa * 100, 1)
            _cln_sess[_key] = _sess
    else:
        for _key in ['FullFwd', 'NoBkFullFwd', 'CloneEgo', 'CloneAlloReal', 'CloneAlloLatent']:
            V[f'{_key}Rpa'] = 0.0
            V[f'{_key}RpaSD'] = 0.0
            V[f'{_key}WinRate'] = 0
            V[f'{_key}PctOmni'] = 0.0

    # ---- Cloning / heuristic stat tests vs mouse (Bonferroni n=5, separate pool) ----
    # NOTE: uses n=5 (5 cloning/heuristic comparisons), separate from the 7 primary tests.
    # See internal tracking notes for decision on whether to merge into a single n=12 pool.
    N_BONF_CLONING = 5
    for _key, _prefix in [
        ('FullFwd',         'FullFwdVsMouse'),
        ('NoBkFullFwd',     'NoBkFullFwdVsMouse'),
        ('CloneEgo',        'CloneEgoVsMouse'),
        ('CloneAlloReal',   'CloneAlloRealVsMouse'),
        ('CloneAlloLatent', 'CloneAlloLatentVsMouse'),
    ]:
        if _key in _cln_sess:
            t, p_raw, p_bonf, d, n = ttest_paired(_cln_sess[_key], mouse_rpa_sess,
                                                   n_bonf=N_BONF_CLONING)
            V[f'T{_prefix}'] = round(t, 2)
            V[f'Praw{_prefix}'] = p_raw
            V[f'Pbonf{_prefix}'] = p_bonf
            V[f'D{_prefix}'] = round(d, 2)
            V[f'N{_prefix}'] = n
        else:
            V[f'T{_prefix}'] = 0.0
            V[f'Praw{_prefix}'] = 1.0
            V[f'Pbonf{_prefix}'] = 1.0
            V[f'D{_prefix}'] = 0.0
            V[f'N{_prefix}'] = 0

    # ---- Table 1 CI and %Ceiling macros ----
    # These were previously maintained as manual blocks appended to paper_values.tex.
    # Emit them here so regenerating paper_values.tex does not break Table 1.
    def _hpo_best_sess(model):
        cfg = _hpo(model).get('config', 'default')
        return per_session_stats(hpo_df[(hpo_df['model'] == model) &
                                         (hpo_df['hpo_config'] == cfg)])

    table1_sess = {
        'Mouse': mouse_rpa_sess,
        'Pomcp': pomcp_sess,
        'RecSac': rsac_sess,
        'RecSacCausal': rcausal_sess,
        'DqnPre': dqn_pre_sess,
        'FscBio': fsc_sess,
        'DrqnSeq': drqn_seq_sess_rpa,
        'Dqn': dqn_sess_rpa,
        'DrqnRand': _hpo_best_sess('DRQN_rand'),
        'Qrdqn': _hpo_best_sess('QRDQN'),
        'Trpo': _hpo_best_sess('TRPO'),
        'Ppo': _hpo_best_sess('PPO'),
        'Atwoc': _hpo_best_sess('A2C'),
        'RecPpo': _hpo_best_sess('RecurrentPPO'),
        'PomcpBio': pbio_sess,
        'FullFwd': _cln_sess.get('FullFwd', pd.Series(dtype=float)),
        'NoBkFullFwd': _cln_sess.get('NoBkFullFwd', pd.Series(dtype=float)),
        'CloneEgo': _cln_sess.get('CloneEgo', pd.Series(dtype=float)),
        'CloneAlloReal': _cln_sess.get('CloneAlloReal', pd.Series(dtype=float)),
        'CloneAlloLatent': _cln_sess.get('CloneAlloLatent', pd.Series(dtype=float)),
    }
    for key, sess in table1_sess.items():
        V[f'{key}RpaCI'] = ci95(sess)

    # NNS denominator: switched POMCP -> Greedy_oracle on 2026-05-04 per review.
    # Greedy_oracle (BFS+replan with full graph access) is empirically the
    # strictest oracle on the canonical 60-session set (RPA 0.878 vs POMCP 0.820);
    # using it as the NNS denominator pins all bars in [0, 100%] and aligns the
    # ceiling with "best achievable with full info". POMCP is still cited as the
    # POMDP-canonical baseline elsewhere in the paper.
    nns_oracle_rpa = V['GreedyRpa']
    for key in [
        'Pomcp', 'RecSac', 'RecSacCausal', 'DqnPre', 'FscBio', 'DrqnSeq',
        'Dqn', 'DrqnRand', 'Qrdqn', 'Trpo', 'Ppo', 'Atwoc', 'RecPpo',
        'PomcpBio', 'FullFwd', 'NoBkFullFwd', 'CloneEgo', 'CloneAlloReal',
        'CloneAlloLatent', 'Fwd', 'Mouse',
    ]:
        V[f'{key}PctCeiling'] = pct_ceiling(V[f'{key}Rpa'], V['RandomRpa'], nns_oracle_rpa)
    V['GreedyPctCeiling'] = 100.0
    V['PomcpPctCeiling'] = pct_ceiling(V['PomcpRpa'], V['RandomRpa'], nns_oracle_rpa)

    return V


# ---------------------------------------------------------------------------
# LaTeX formatting
# ---------------------------------------------------------------------------
def to_latex_commands(V):
    """Generate \newcommand lines for all values in V."""
    lines = []
    lines.append('%% Auto-generated by 260405_generate_paper_values.py')
    lines.append('%% Do not edit by hand. Regenerate with: python code/260405_generate_paper_values.py')
    lines.append('')

    def cmd(name, value_str, comment=''):
        c = f'  % {comment}' if comment else ''
        return f'\\newcommand{{\\{name}}}{{{value_str}}}{c}'

    # --- Dataset ---
    lines.append('% === Dataset ===')
    lines.append('% Paper-facing agent labels are defined in PAPER_AGENT_MAP in the generator.')
    lines.append(cmd('NSessions', str(V['NSessions']), '60 mouse sessions'))
    lines.append(cmd('NAnimals', str(V['NAnimals']), '2 mice'))
    lines.append(cmd('NMazes', str(V['NMazes']), '3 maze topologies'))
    lines.append(cmd('NAgents', str(V['NAgents']), '12 agents evaluated'))
    lines.append(cmd('NHpoSims', f"{V['NHpoSims']:,}".replace(',', ','), ''))
    lines.append(cmd('NHpoConfigs', str(V['NHpoConfigs']), ''))
    lines.append(cmd('NHpoSimsAll', f"{V['NHpoSimsAll']:,}".replace(',', ','), 'includes POMCP tuning rows'))
    lines.append(cmd('NHpoConfigsAll', str(V['NHpoConfigsAll']), 'includes POMCP tuning configs'))
    lines.append(cmd('NSessPretrain', str(V['NSessPretrain']), 'sessions with pretrain data'))
    lines.append(cmd('BudgetMin', str(V['BudgetMin']), 'min action budget'))
    lines.append(cmd('BudgetMax', str(V['BudgetMax']), 'max action budget'))
    lines.append(cmd('BudgetMean', fmt(V['BudgetMean'], 1), 'mean action budget'))
    lines.append(cmd('BudgetMedian', str(V['BudgetMedian']), 'median action budget per session (canonical-60)'))
    lines.append(cmd('SessAnimalA', str(V['SessAnimalA']), 'a031 sessions'))
    lines.append(cmd('SessAnimalB', str(V['SessAnimalB']), 'a033 sessions'))
    lines.append(cmd('SessMazeA', str(V['SessMazeA']), 'latMaz100 sessions'))
    lines.append(cmd('SessMazeB', str(V['SessMazeB']), 'latMaz101 sessions'))
    lines.append(cmd('SessMazeC', str(V['SessMazeC']), 'latMaz103 sessions'))
    lines.append(cmd('MazeNodeMin', str(V['MazeNodeMin']), 'min connected nodes across mazes'))
    lines.append(cmd('MazeNodeMax', str(V['MazeNodeMax']), 'max connected nodes'))
    lines.append(cmd('MazeNodeMeanApprox', str(V['MazeNodeMeanApprox']), '~16 weighted mean'))
    lines.append(cmd('MazeRewardMeanApprox', str(V['MazeRewardMeanApprox']), '~16 mean initial rewards'))
    lines.append(cmd('MazeNodesA', str(V['MazeNodesA']), 'latMaz100 nodes'))
    lines.append(cmd('MazeEdgesA', str(V['MazeEdgesA']), 'latMaz100 edges'))
    lines.append(cmd('MazeNodesB', str(V['MazeNodesB']), 'latMaz101 nodes'))
    lines.append(cmd('MazeEdgesB', str(V['MazeEdgesB']), 'latMaz101 edges'))
    lines.append(cmd('MazeNodesC', str(V['MazeNodesC']), 'latMaz103 nodes'))
    lines.append(cmd('MazeEdgesC', str(V['MazeEdgesC']), 'latMaz103 edges'))
    lines.append(cmd('RecSacNEpochs', str(V['RecSacNEpochs']), 'RecSAC meta-RL training epochs'))
    lines.append('')

    # --- Mouse ---
    lines.append('% === Mouse ===')
    lines.append(cmd('MouseRpa', fmt3(V['MouseRpa']), 'macro RPA'))
    lines.append(cmd('MouseRpaSD', fmt3(V['MouseRpaSD']), 'SD of per-session RPAs'))
    lines.append(cmd('MouseMeanRew', fmt(V['MouseMeanRew'], 1), 'mean reward per session'))
    lines.append(cmd('MouseMeanRewPretrain', fmt(V['MouseMeanRewPretrain'], 1), 'mean reward in 54-session pretrain set'))
    lines.append(cmd('MousePctOmni', fmt(V['MouseRpa'] / V['PomcpRpa'] * 100, 1) + '\\%', 'mouse % of omniscient'))
    lines.append('')

    # --- Heuristics ---
    lines.append('% === Heuristics ===')
    lines.append(cmd('RandomRpa', fmt3(V['RandomRpa']), 'Random macro RPA'))
    lines.append(cmd('FwdRpa', fmt3(V['FwdRpa']), 'Forward-random macro RPA'))
    lines.append(cmd('FwdRpaSD', fmt3(V['FwdRpaSD']), 'Forward-random per-session SD'))
    lines.append(cmd('FwdRpaCI', fmt3(V['FwdRpaCI']), 'Forward-random 95\\% CI half-width (n=60)'))
    lines.append(cmd('FwdPctOmni', fmt(V['FwdPctOmni'], 1) + '\\%', 'Forward-random % of omniscient'))
    lines.append(cmd('FwdWinRate', f"{V['FwdWinRate']}\\%", 'Forward-random win rate vs mouse'))
    lines.append(cmd('GreedyRpa', fmt3(V['GreedyRpa']), 'Greedy oracle macro RPA'))
    lines.append('')

    # --- Agents ---
    lines.append('% === POMCP oracle (fewer_sims, c=20) ===')
    lines.append(cmd('PomcpRpa', fmt3(V['PomcpRpa']), 'macro RPA'))
    lines.append(cmd('PomcpRpaSD', fmt3(V['PomcpRpaSD'])))
    lines.append(cmd('PomcpPctOmni', '100.0\\%', 'by definition'))
    lines.append(cmd('PomcpWinRate', str(V['PomcpWinRate']) + '\\%'))
    lines.append(cmd('PomcpMeanRew', fmt(V['PomcpMeanRew'], 1), 'mean reward per session'))
    lines.append(cmd('PomcpXMouse', fmt(V['PomcpRpa'] / V['MouseRpa'], 1) + '\\times', 'POMCP / mouse ratio'))
    lines.append('')

    lines.append('% === RecurrentSAC D=32 ===')
    lines.append(cmd('RecSacRpa', fmt3(V['RecSacRpa']), 'macro RPA'))
    lines.append(cmd('RecSacRpaSD', fmt3(V['RecSacRpaSD'])))
    lines.append(cmd('RecSacPctOmni', fmt(V['RecSacPctOmni'], 1) + '\\%'))
    lines.append(cmd('RecSacWinRate', str(V['RecSacWinRate']) + '\\%'))
    lines.append(cmd('RecSacXMouse', fmt(V['RecSacXMouse'], 1) + '\\times', 'RecSAC / mouse RPA ratio'))
    lines.append(cmd('RecSacGapPct', str(int(V['RecSacGapPct'])) + '\\%', '% of random-to-oracle gap closed'))
    lines.append('')

    lines.append('% === RecurrentSAC (causal) ===')
    lines.append(cmd('RecSacCausalRpa', fmt3(V['RecSacCausalRpa'])))
    lines.append(cmd('RecSacCausalRpaSD', fmt3(V['RecSacCausalRpaSD'])))
    lines.append(cmd('RecSacCausalPctOmni', fmt(V['RecSacCausalPctOmni'], 1) + '\\%'))
    lines.append(cmd('RecSacCausalWinRate', str(V['RecSacCausalWinRate']) + '\\%'))
    lines.append('')

    lines.append('% === DQN pretrained ===')
    lines.append(cmd('DqnPreRpa', fmt3(V['DqnPreRpa'])))
    lines.append(cmd('DqnPreRpaSD', fmt3(V['DqnPreRpaSD'])))
    lines.append(cmd('DqnPrePctOmni', fmt(V['DqnPrePctOmni'], 1) + '\\%'))
    lines.append(cmd('DqnPreWinRate', str(V['DqnPreWinRate']) + '\\%'))
    lines.append('')

    lines.append('% === FSC_bio (Table 1: default params, intermediate agents CSV) ===')
    lines.append(cmd('FscBioRpa', fmt3(V['FscBioRpa'])))
    lines.append(cmd('FscBioRpaSD', fmt3(V['FscBioRpaSD'])))
    lines.append(cmd('FscBioPctOmni', fmt(V['FscBioPctOmni'], 1) + '\\%'))
    lines.append(cmd('FscBioWinRate', str(V['FscBioWinRate']) + '\\%'))
    lines.append('')

    lines.append('% === DRQN_seq ===')
    lines.append(cmd('DrqnSeqRpa', fmt3(V['DrqnSeqRpa'])))
    lines.append(cmd('DrqnSeqRpaSD', fmt3(V['DrqnSeqRpaSD'])))
    lines.append(cmd('DrqnSeqPctOmni', fmt(V['DrqnSeqPctOmni'], 1) + '\\%'))
    lines.append(cmd('DrqnSeqWinRate', str(V['DrqnSeqWinRate']) + '\\%'))
    lines.append('')

    lines.append('% === DQN ===')
    lines.append(cmd('DqnRpa', fmt3(V['DqnRpa'])))
    lines.append(cmd('DqnRpaSD', fmt3(V['DqnRpaSD'])))
    lines.append(cmd('DqnPctOmni', fmt(V['DqnPctOmni'], 1) + '\\%'))
    lines.append(cmd('DqnWinRate', str(V['DqnWinRate']) + '\\%'))
    lines.append('')

    lines.append('% === DRQN_rand ===')
    lines.append(cmd('DrqnRandRpa', fmt3(V['DrqnRandRpa'])))
    lines.append(cmd('DrqnRandRpaSD', fmt3(V['DrqnRandRpaSD'])))
    lines.append(cmd('DrqnRandPctOmni', fmt(V['DrqnRandPctOmni'], 1) + '\\%'))
    lines.append(cmd('DrqnRandWinRate', str(V['DrqnRandWinRate']) + '\\%'))
    lines.append('')

    lines.append('% === QRDQN ===')
    lines.append(cmd('QrdqnRpa', fmt3(V['QrdqnRpa'])))
    lines.append(cmd('QrdqnRpaSD', fmt3(V['QrdqnRpaSD'])))
    lines.append(cmd('QrdqnPctOmni', fmt(V['QrdqnPctOmni'], 1) + '\\%'))
    lines.append(cmd('QrdqnWinRate', str(V['QrdqnWinRate']) + '\\%'))
    lines.append('')

    lines.append('% === TRPO ===')
    lines.append(cmd('TrpoRpa', fmt3(V['TrpoRpa'])))
    lines.append(cmd('TrpoRpaSD', fmt3(V['TrpoRpaSD'])))
    lines.append(cmd('TrpoPctOmni', fmt(V['TrpoPctOmni'], 1) + '\\%'))
    lines.append(cmd('TrpoWinRate', str(V['TrpoWinRate']) + '\\%'))
    lines.append('')

    lines.append('% === PPO ===')
    lines.append(cmd('PpoRpa', fmt3(V['PpoRpa'])))
    lines.append(cmd('PpoRpaSD', fmt3(V['PpoRpaSD'])))
    lines.append(cmd('PpoPctOmni', fmt(V['PpoPctOmni'], 1) + '\\%'))
    lines.append(cmd('PpoWinRate', str(V['PpoWinRate']) + '\\%'))
    lines.append('')

    lines.append('% === A2C ===')
    lines.append(cmd('AtwocRpa', fmt3(V['AtwocRpa'])))
    lines.append(cmd('AtwocRpaSD', fmt3(V['AtwocRpaSD'])))
    lines.append(cmd('AtwocPctOmni', fmt(V['AtwocPctOmni'], 1) + '\\%'))
    lines.append(cmd('AtwocWinRate', str(V['AtwocWinRate']) + '\\%'))
    lines.append('')

    lines.append('% === RecurrentPPO ===')
    lines.append(cmd('RecPpoRpa', fmt3(V['RecPpoRpa'])))
    lines.append(cmd('RecPpoRpaSD', fmt3(V['RecPpoRpaSD'])))
    lines.append(cmd('RecPpoPctOmni', fmt(V['RecPpoPctOmni'], 1) + '\\%'))
    lines.append(cmd('RecPpoWinRate', str(V['RecPpoWinRate']) + '\\%'))
    lines.append('')

    lines.append('% === POMCP_bio ===')
    lines.append(cmd('PomcpBioRpa', fmt3(V['PomcpBioRpaMean']), 'mean per-session RPA (= macro for yoked)'))
    lines.append(cmd('PomcpBioRpaPooled', fmt3(V['PomcpBioRpaPooled']), 'pooled reward/action RPA, not Table 1 convention'))
    lines.append(cmd('PomcpBioRpaSD', fmt3(V['PomcpBioRpaSD'])))
    lines.append(cmd('PomcpBioPctOmni', fmt(V['PomcpBioPctOmni'], 1) + '\\%'))
    lines.append(cmd('PomcpBioWinRate', str(V['PomcpBioWinRate']) + '\\%'))
    lines.append(cmd('PomcpBioRpaOld', V['PomcpBioRpaOld'], 'pre-rebait-fix (buggy) value'))
    lines.append('')

    # --- Stat tests ---
    lines.append('% === Statistical tests (7 comparisons, Bonferroni n=7) ===')
    lines.append('% Positive t: first entity > second entity')

    def stat_cmds(prefix, t, p_raw, p_bonf, d, n, label):
        lines.append(f'% {label}')
        lines.append(cmd(f'T{prefix}', fmt2(abs(t)) if t >= 0 else '-' + fmt2(abs(t))))
        lines.append(cmd(f'TSign{prefix}', '+' if t >= 0 else '-'))  # sign as separate macro
        lines.append(cmd(f'Praw{prefix}', sci_latex(p_raw)))
        lines.append(cmd(f'Pbonf{prefix}', sci_latex(p_bonf)))
        lines.append(cmd(f'D{prefix}', fmt2(d)))
        lines.append(cmd(f'N{prefix}', str(n)))
        lines.append('')

    stat_cmds('MouseVsRandom', V['TMouseVsRandom'], V['PrawMouseVsRandom'], V['PbonfMouseVsRandom'],
              V['DMouseVsRandom'], V['NMouseVsRandom'], 'Mouse vs Random')
    stat_cmds('MouseVsFwd', V['TMouseVsFwd'], V['PrawMouseVsFwd'], V['PbonfMouseVsFwd'],
              V['DMouseVsFwd'], V['NMouseVsFwd'], 'Mouse vs Forward-random')
    stat_cmds('DqnVsMouse', V['TDqnVsMouse'], V['PrawDqnVsMouse'], V['PbonfDqnVsMouse'],
              V['DDqnVsMouse'], V['NDqnVsMouse'], 'DQN vs Mouse')
    stat_cmds('DrqnSeqVsMouse', V['TDrqnSeqVsMouse'], V['PrawDrqnSeqVsMouse'], V['PbonfDrqnSeqVsMouse'],
              V['DDrqnSeqVsMouse'], V['NDrqnSeqVsMouse'], 'DRQN_seq vs Mouse')
    stat_cmds('RecSacVsMouse', V['TRecSacVsMouse'], V['PrawRecSacVsMouse'], V['PbonfRecSacVsMouse'],
              V['DRecSacVsMouse'], V['NRecSacVsMouse'], 'RecurrentSAC vs Mouse')
    stat_cmds('RecSacVsPomcp', V['TRecSacVsPomcp'], V['PrawRecSacVsPomcp'], V['PbonfRecSacVsPomcp'],
              V['DRecSacVsPomcp'], V['NRecSacVsPomcp'], 'RecurrentSAC vs POMCP')
    stat_cmds('DqnPreVsBase', V['TDqnPreVsBase'], V['PrawDqnPreVsBase'], V['PbonfDqnPreVsBase'],
              V['DDqnPreVsBase'], V['NDqnPreVsBase'], 'DQN pretrained vs baseline')

    # Additional (not Bonferroni-corrected)
    lines.append('% DQN pretrained vs Mouse (additional, prose only)')
    lines.append(cmd('TDqnPreVsMouse', fmt2(abs(V['TDqnPreVsMouse']))))
    lines.append(cmd('PrawDqnPreVsMouse', sci_latex(V['PrawDqnPreVsMouse'])))
    lines.append(cmd('DDqnPreVsMouse', fmt2(V['DDqnPreVsMouse'])))
    lines.append(cmd('NDqnPreVsMouse', str(V['NDqnPreVsMouse'])))
    lines.append('')

    # --- Pretraining ---
    lines.append('% === Multi-session pretraining (Table 2) ===')
    lines.append(cmd('DqnBaseMeanRew', fmt(V['DqnBaseMeanRew'], 1), 'DQN baseline mean reward'))
    lines.append(cmd('DqnPreMeanRew', fmt(V['DqnPreMeanRew'], 1), 'DQN pretrained mean reward'))
    lines.append(cmd('DqnPreDeltaRew', ('+' if V['DqnPreDeltaRew'] >= 0 else '') + fmt(V['DqnPreDeltaRew'], 1)))
    lines.append(cmd('DqnPrePctImprove', f'+{V["DqnPrePctImprove"]}\\%'))
    lines.append(cmd('DqnPreCohenD', fmt2(V['DqnPreCohenD'])))
    lines.append(cmd('DqnPreGapClose', str(int(V['DqnPreGapClose'])) + '\\%', '% of DQN-to-POMCP gap closed'))
    lines.append(cmd('DrqnSeqBaseMeanRew', fmt(V['DrqnSeqBaseMeanRew'], 1)))
    lines.append(cmd('DrqnSeqPreMeanRew', fmt(V['DrqnSeqPreMeanRew'], 1)))
    lines.append(cmd('DrqnSeqPreDeltaRew', ('+' if V['DrqnSeqPreDeltaRew'] >= 0 else '') + fmt(V['DrqnSeqPreDeltaRew'], 1)))
    lines.append(cmd('DrqnSeqPrePctImprove', f'+{V["DrqnSeqPrePctImprove"]}\\%'))
    lines.append(cmd('DrqnSeqPreCohenD', fmt2(V['DrqnSeqPreCohenD'])))
    lines.append(cmd('DrqnRandBaseMeanRew', fmt(V['DrqnRandBaseMeanRew'], 1)))
    lines.append(cmd('DrqnRandPreMeanRew', fmt(V['DrqnRandPreMeanRew'], 1)))
    lines.append(cmd('DrqnRandPreDeltaRew', ('+' if V['DrqnRandPreDeltaRew'] >= 0 else '') + fmt(V['DrqnRandPreDeltaRew'], 1)))
    lines.append(cmd('DrqnRandPrePctImprove', f'+{V["DrqnRandPrePctImprove"]}\\%'))
    lines.append(cmd('DrqnRandPreCohenD', fmt2(V['DrqnRandPreCohenD'])))
    lines.append('')

    # --- Dose-response ---
    lines.append('% === Dose-response (DQN pretraining) ===')
    lines.append(cmd('DoseSpearmanRho', fmt3(V['DoseSpearmanRho']), 'Spearman rho'))
    lines.append(cmd('DoseSpearmanP', fmt3(V['DoseSpearmanP']), 'Spearman p-value'))
    lines.append(cmd('DoseN', str(V['DoseN']), 'number of unique pretrain-dose counts'))
    lines.append('')

    # --- Cloning / heuristic agents (Table 1 rows from 260413 run) ---
    lines.append('% === FullFwd / NoBkFullFwd / Clone heuristics (260413 canonical run) ===')
    for _key, _label in [
        ('FullFwd',         'FullFwd'),
        ('NoBkFullFwd',     'NoBkFullFwd'),
        ('CloneEgo',        'Clone ego marginal'),
        ('CloneAlloReal',   'Clone allo-real marginal'),
        ('CloneAlloLatent', 'Clone allo-latent marginal'),
    ]:
        lines.append(cmd(f'{_key}Rpa',      fmt3(V[f'{_key}Rpa']),  f'macro RPA — {_label}'))
        lines.append(cmd(f'{_key}RpaSD',    fmt3(V[f'{_key}RpaSD'])))
        lines.append(cmd(f'{_key}PctOmni',  fmt(V[f'{_key}PctOmni'], 1) + '\\%'))
        lines.append(cmd(f'{_key}WinRate',  str(V[f'{_key}WinRate']) + '\\%'))
        lines.append('')

    # --- Cloning stat tests vs mouse (Bonferroni n=5; see internal notes) ---
    lines.append('% === Cloning/heuristic stat tests vs mouse (Bonferroni n=5; see internal notes) ===')
    lines.append('% NOTE: separate n=5 Bonferroni pool — pending decision on merging with n=7 primary tests')
    for _prefix, _label in [
        ('FullFwdVsMouse',         'FullFwd vs Mouse'),
        ('NoBkFullFwdVsMouse',     'NoBkFullFwd vs Mouse'),
        ('CloneEgoVsMouse',        'Clone_ego vs Mouse'),
        ('CloneAlloRealVsMouse',   'Clone_allo_real vs Mouse'),
        ('CloneAlloLatentVsMouse', 'Clone_allo_latent vs Mouse'),
    ]:
        stat_cmds(_prefix, V[f'T{_prefix}'], V[f'Praw{_prefix}'], V[f'Pbonf{_prefix}'],
                  V[f'D{_prefix}'], V[f'N{_prefix}'], _label)

    # --- Table 1 generated CI and %Ceiling macros ---
    lines.append('% === 95% CI half-widths (1.96 * SD / sqrt(n)) ===')
    for _key, _comment in [
        ('Mouse', 'n=60'),
        ('Pomcp', 'n=60'),
        ('RecSac', 'n=60'),
        ('RecSacCausal', 'n=60'),
        ('DqnPre', 'n=54'),
        ('FscBio', 'n=60'),
        ('DrqnSeq', 'n=60'),
        ('Dqn', 'n=60'),
        ('DrqnRand', 'n=60'),
        ('Qrdqn', 'n=60'),
        ('Trpo', 'n=60'),
        ('Ppo', 'n=60'),
        ('Atwoc', 'n=60'),
        ('RecPpo', 'n=60'),
        ('PomcpBio', 'n=60'),
        ('FullFwd', 'n=60'),
        ('NoBkFullFwd', 'n=60'),
        ('CloneEgo', 'n=60'),
        ('CloneAlloReal', 'n=60'),
        ('CloneAlloLatent', 'n=60'),
    ]:
        lines.append(cmd(f'{_key}RpaCI', fmt3(V[f'{_key}RpaCI']), f'95\\% CI half-width ({_comment})'))
    lines.append('')

    lines.append('% === NNS = (RPA - Random) / (Greedy_oracle - Random) * 100% ===')
    for _key in [
        'Greedy', 'Pomcp', 'RecSac', 'RecSacCausal', 'NoBkFullFwd', 'DqnPre',
        'FscBio', 'Fwd', 'Mouse', 'DrqnSeq', 'Dqn', 'DrqnRand', 'CloneEgo',
        'Qrdqn', 'Trpo', 'Ppo', 'Atwoc', 'RecPpo', 'PomcpBio', 'FullFwd',
        'CloneAlloLatent', 'CloneAlloReal',
    ]:
        lines.append(cmd(f'{_key}PctCeiling', fmt(V[f'{_key}PctCeiling'], 1) + '\\%'))
    lines.append('')

    return '\n'.join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print('=== generate_paper_values.py ===')
    print('Loading data...')
    dfs = load_all()
    print('Computing statistics...')
    V = compute_all(dfs)

    # Print summary
    print('\n=== KEY VALUES ===')
    print(f'  Mouse macro RPA: {V["MouseRpa"]}')
    print(f'  POMCP oracle RPA: {V["PomcpRpa"]}')
    print(f'  RecSAC RPA: {V["RecSacRpa"]}  ({V["RecSacPctOmni"]}% omni)')
    print(f'  DQN RPA: {V["DqnRpa"]}  DQN pretrained: {V["DqnPreRpa"]}')
    print(f'  FSC_bio RPA: {V["FscBioRpa"]}  POMCP_bio: {V["PomcpBioRpa"]}')
    print(f'  Random RPA: {V["RandomRpa"]}  Forward: {V["FwdRpa"]}')
    print(f'\n  Stat tests:')
    print(f'    Mouse > Random: t={V["TMouseVsRandom"]}, p_bonf={V["PbonfMouseVsRandom"]:.2e}, d={V["DMouseVsRandom"]}')
    print(f'    Mouse < Forward: t={V["TMouseVsFwd"]}, p_bonf={V["PbonfMouseVsFwd"]:.2e}, d={V["DMouseVsFwd"]}')
    print(f'    DQN > Mouse: t={V["TDqnVsMouse"]}, p_bonf={V["PbonfDqnVsMouse"]:.2e}, d={V["DDqnVsMouse"]}')
    print(f'    RecSAC > Mouse: t={V["TRecSacVsMouse"]}, p_bonf={V["PbonfRecSacVsMouse"]:.2e}, d={V["DRecSacVsMouse"]}')
    print(f'    POMCP > RecSAC: t={V["TRecSacVsPomcp"]}, p_bonf={V["PbonfRecSacVsPomcp"]:.2e}, d={V["DRecSacVsPomcp"]}')
    print(f'    DQN pre > base: t={V["TDqnPreVsBase"]}, p_bonf={V["PbonfDqnPreVsBase"]:.2e}, d={V["DDqnPreVsBase"]}')
    print(f'\n  Pretrain: DQN base={V["DqnBaseMeanRew"]}, pre={V["DqnPreMeanRew"]}, +{V["DqnPrePctImprove"]}%, gap_close={V["DqnPreGapClose"]}%')
    print(f'  Dose-response: rho={V["DoseSpearmanRho"]}, p={V["DoseSpearmanP"]}, n={V["DoseN"]} doses')

    # Write LaTeX
    latex_str = to_latex_commands(V)
    out_tex = PAPER_DIR / 'paper_values.tex'
    with open(out_tex, 'w') as f:
        f.write(latex_str + '\n')
    print(f'\nWrote: {out_tex}')

    # Write JSON
    out_json = DATA_DIR / f'{TS}_paper_values.json'
    # Convert numpy types for JSON serialization
    V_json = {}
    for k, v in V.items():
        if isinstance(v, (np.integer, np.int64)):
            V_json[k] = int(v)
        elif isinstance(v, (np.floating, np.float64)):
            V_json[k] = float(v)
        else:
            V_json[k] = v
    with open(out_json, 'w') as f:
        json.dump(V_json, f, indent=2)
    print(f'Wrote: {out_json}')


if __name__ == '__main__':
    main()
