"""
Shared experiment configuration and convenience functions.

Centralises constants (BEST_CONFIGS, OBS_CONFIG, ...) and data-loading
patterns that were previously copy-pasted across 15+ runner scripts.

Usage (from any script in this directory)::

    from experiment_config import (
        BEST_CONFIGS, OBS_CONFIG, load_data, build_runner,
        obs_kwargs, run_random_agent, run_forward_random_agent,
    )
"""
import os
import re
from collections import deque
from copy import deepcopy

import numpy as np
import pandas as pd

# Project imports (paths are relative to this file's directory)
import sys
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.dirname(__file__))

from utils_latMaz import get_most_recent_file
from yoked_rl_runner import (
    YokedRLRunner, AlgoConfig, GraphMazeEnv,
    get_adj_states, env_transition_func,
)

# =============================================================================
# Constants
# =============================================================================

# Best HPO config per model: {agent: (config_name, overrides_dict)}
BEST_CONFIGS = {
    # Synced with HPO winners from c260316-171035_hpo_tuning_FIXED.csv
    # (see internal RPA-discrepancy diagnosis notes; not in public release).
    'DQN': ('aggressive_explore', {
        'learning_rate': 5e-4, 'learning_starts': 10, 'train_freq': 1,
        'target_update_interval': 100, 'batch_size': 32,
    }),
    'DRQN_seq': ('aggressive', {
        'epsilon_decay_steps': 100, 'epsilon_end': 0.1,
        'train_every_n_steps': 1, 'target_update_every': 20,
        'learning_rate': 2e-3, 'batch_size': 32, 'seq_len': 5,
        'replay_mode': 'sequential',
    }),
    'DRQN_rand': ('aggressive', {
        'epsilon_decay_steps': 100, 'epsilon_end': 0.1,
        'train_every_n_steps': 1, 'target_update_every': 20,
        'learning_rate': 2e-3, 'batch_size': 32, 'seq_len': 5,
        'replay_mode': 'random',
    }),
    'QRDQN': ('aggressive', {
        'learning_rate': 1e-3, 'exploration_fraction': 0.5,
        'exploration_final_eps': 0.05, 'batch_size': 32,
    }),
    'PPO': ('small_batch', {
        'batch_size': 32, 'n_steps': 64, 'n_epochs': 20, 'learning_rate': 1e-3,
    }),
    'A2C': ('high_ent', {'ent_coef': 0.05, 'learning_rate': 1e-3}),
    'TRPO': ('loose_kl', {
        'target_kl': 0.05, 'n_steps': 32, 'cg_max_steps': 15,
        'learning_rate': 5e-4,
    }),
    'RecurrentPPO': ('high_ent', {
        'ent_coef': 0.05, 'learning_rate': 5e-4, 'n_epochs': 5,
        'lstm_hidden_size': 64, 'n_lstm_layers': 1,
    }),
    'POMCP': ('fewer_sims', {'num_sims': 200, 'max_depth': 30}),
    'BeliefOracle': ('default', {}),
    'RC_GVF': ('default', {}),
    'MOP': ('default', {}),
    'POMCP_bio_egoOnly': ('default', {}),
    # SHELVED (260316): CSCG consistently below Random across all configs.
    # Classes still importable from advanced_agents.py for reproducibility.
    # Do NOT include these in active benchmark runs.
    'CSCG_bio_BFS': ('default', {}),
    'CSCG_bio_POMCP': ('default', {}),
    'POMCP_bio_CSCG': ('default', {}),  # backward compat alias → CSCGBioBFSAgent
    'VarMarkov': ('default', {}),
    'FSC_oracle': ('default', {}),
    'OOI': ('default', {}),
    'FSC_bio': ('default', {}),
    'DreamerLite': ('default', {}),
    'RecurrentSAC': ('default', {}),
    'SSM_SAC': ('default', {}),
}

# Standard observation config for RL agents
OBS_CONFIG = {
    'obs_type': 'ego',
    'action_type': 'ego',
    'include_prev_action': True,
    'include_prev_reward': True,
}

# POMCP agents use different obs (no prev action/reward)
POMCP_OBS_CONFIG = {
    'obs_type': 'ego',
    'action_type': 'ego',
    'include_prev_action': False,
    'include_prev_reward': False,
}

HISTORY_LEN = 0  # LSTM agents handle their own memory

DEFAULT_N_REPEATS = 5

PRETRAIN_AGENTS = ['DQN', 'DRQN_seq', 'DRQN_rand']


# =============================================================================
# Path helpers
# =============================================================================

def project_root():
    """Return the project root directory.

    Layout-agnostic: walks up from this file looking for a marker directory
    (``data_released/`` or ``paper/``). This makes the same source file work
    in three layouts:
      - development:    ``<root>/code/experiment_config.py``  (depth 2)
      - released stage: ``<root>/code/experiment_config.py``              (depth 1)
      - submission:     ``<root>/code/experiment_config.py``                       (depth 1)

    Falls back to the legacy ``..``/``..`` (depth-2) assumption if no marker
    is found, preserving backward compatibility for unusual setups.
    """
    here = os.path.dirname(os.path.abspath(__file__))
    cur = here
    for _ in range(5):  # max 5 levels up
        if (os.path.isdir(os.path.join(cur, 'data_released')) or
                os.path.isdir(os.path.join(cur, 'paper'))):
            return cur
        parent = os.path.dirname(cur)
        if parent == cur:
            break
        cur = parent
    # Legacy fallback: dev layout's two-up ascent
    return os.path.normpath(os.path.join(here, '..', '..'))


def output_path(filename):
    """Return ``<project_root>/data_out/rl_sims/<filename>``."""
    return os.path.join(project_root(), 'data_out', 'rl_sims', filename)


# =============================================================================
# Data loading
# =============================================================================

def load_yoking_df():
    """Load the yoking CSV and add ``exp_moment`` / ``date_part`` columns.

    Searches two paths in priority order:
      1. ``<root>/data_released/yoked_dfs/c*_yoked_sessions.csv``  (public release layout)
      2. ``<root>/yoked_dfs/*animal_to_agent_yoking_info*.csv``    (development layout)

    The first match wins. This makes the canonical replication recipe
    (``python 260405_generate_paper_values.py``) runnable both in the
    development environment and in the released artefact.
    """
    import glob as _glob
    root = project_root()
    candidates = [
        os.path.join(root, 'data_released', 'yoked_dfs', 'c*_yoked_sessions.csv'),
        os.path.join(root, 'yoked_dfs', '*animal_to_agent_yoking_info*.csv'),
    ]
    yoking_path = None
    for pat in candidates:
        matches = sorted(_glob.glob(pat))
        if matches:
            yoking_path = matches[-1]  # most-recent by lexicographic c{ts} sort
            break
    if yoking_path is None:
        raise FileNotFoundError(
            f"No yoking CSV found. Searched: {candidates}. "
            f"Expected `<root>/data_released/yoked_dfs/c*_yoked_sessions.csv` "
            f"OR `<root>/yoked_dfs/*animal_to_agent_yoking_info*.csv`.")
    yoking_df = pd.read_csv(yoking_path, dtype={'animal_ID': str})

    # 260507 v12 (Option B): prefer the explicit `exp_moment` column written
    # by the release-converter (260504_create_released_yoked_df.py). For
    # overnight 24h sessions, the file-write timestamp in csv_data_path's
    # basename differs from the canonical session-START timestamp by
    # 15-48 hours; the explicit exp_moment column captures the session-start
    # so the regen recipe matches the HPO CSV's exp_moment column.
    if 'exp_moment' in yoking_df.columns:
        # Validate: column must be fully populated.
        n_missing = yoking_df['exp_moment'].isna().sum()
        if n_missing > 0:
            import sys
            print('=' * 70, file=sys.stderr)
            print(f'!!! ERROR: {n_missing} rows have NaN exp_moment in yoking CSV     !!!',
                  file=sys.stderr)
            print(f'!!! ({yoking_path})                                              !!!',
                  file=sys.stderr)
            print('=' * 70, file=sys.stderr)
            raise ValueError(
                f'{n_missing} yoking rows have NaN exp_moment; '
                f'cannot proceed with downstream session lookups.')
    else:
        # Fallback path: yoking CSV pre-dates v12 and has no explicit exp_moment.
        # Extract via regex from csv_data_path. NOISY warning so downstream
        # debugging is straightforward — this path mis-identifies overnight
        # 24h sessions (file-write timestamp != session-start timestamp).
        import sys
        print('=' * 70, file=sys.stderr)
        print('!!! WARNING: yoking CSV has no `exp_moment` column            !!!',
              file=sys.stderr)
        print('!!! (legacy file pre-dating v12). Falling back to regex       !!!',
              file=sys.stderr)
        print('!!! extraction from csv_data_path. This MIS-IDENTIFIES         !!!',
              file=sys.stderr)
        print('!!! 19+ overnight 24h sessions whose file-write timestamps    !!!',
              file=sys.stderr)
        print('!!! differ from their session-START timestamps. Regenerate     !!!',
              file=sys.stderr)
        print('!!! the yoking CSV via 260504_create_released_yoked_df.py.    !!!',
              file=sys.stderr)
        print('=' * 70, file=sys.stderr)
        yoking_df['exp_moment'] = yoking_df['csv_data_path'].apply(
            lambda p: re.search(r'(\d{6}-\d{6})', str(p)).group(1)
            if re.search(r'(\d{6}-\d{6})', str(p)) else None)

    yoking_df['date_part'] = yoking_df['exp_moment'].str[:6].astype(int)
    return yoking_df


def load_reward_df():
    """Load the rewarded-states CSV (or ``None`` if missing)."""
    root = project_root()
    rwd_path = get_most_recent_file(
        os.path.join(root, 'data_out', '*rewarded_states*.csv'))
    return pd.read_csv(rwd_path) if rwd_path else None


def filter_sessions(yoking_df, min_date=251001, min_states_visited=50):
    """Return deduplicated list of ``exp_moment`` strings that pass the filter."""
    filtered = yoking_df[
        (yoking_df['date_part'] > min_date) &
        (yoking_df['n_states_visited'] > min_states_visited)
    ]
    return filtered['exp_moment'].drop_duplicates().tolist()


def load_data(min_date=251001, min_states_visited=50):
    """Load yoking + reward data and return filtered sessions.

    Returns
    -------
    yoking_df, rwd_df, sessions
    """
    yoking_df = load_yoking_df()
    rwd_df = load_reward_df()
    sessions = filter_sessions(yoking_df, min_date, min_states_visited)
    return yoking_df, rwd_df, sessions


# =============================================================================
# Runner helpers
# =============================================================================

def build_runner(yoking_df, rwd_df, model_name=None, overrides=None):
    """Build a :class:`YokedRLRunner`, optionally with HPO overrides applied."""
    root = project_root()
    algo = AlgoConfig()
    if model_name is not None and overrides is not None:
        base = deepcopy(getattr(algo, model_name))
        base.update(overrides)
        setattr(algo, model_name, base)
    runner = YokedRLRunner(
        yoking_df=yoking_df,
        rewarded_states_df=rwd_df,
        maze_dir=os.path.join(root, 'data_in', 'mazes'),
        output_dir=os.path.join(root, 'data_out', 'rl_sims'),
        algo_config=algo,
    )
    return runner


def apply_best_configs(agent_names=None):
    """Return an :class:`AlgoConfig` with HPO overrides applied.

    Parameters
    ----------
    agent_names : list[str] or None
        Subset of agents to configure.  *None* means all agents in
        ``BEST_CONFIGS``.
    """
    algo = AlgoConfig()
    names = agent_names if agent_names is not None else list(BEST_CONFIGS.keys())
    for name in names:
        if name not in BEST_CONFIGS:
            continue
        _, overrides = BEST_CONFIGS[name]
        if overrides:
            base = deepcopy(getattr(algo, name))
            base.update(overrides)
            setattr(algo, name, base)
    return algo


def obs_kwargs():
    """Return OBS_CONFIG + HISTORY_LEN as keyword arguments for ``run_single``."""
    return dict(
        obs_type=OBS_CONFIG['obs_type'],
        action_type=OBS_CONFIG['action_type'],
        history_len=HISTORY_LEN,
        include_prev_action=OBS_CONFIG['include_prev_action'],
        include_prev_reward=OBS_CONFIG['include_prev_reward'],
    )


def pomcp_obs_kwargs():
    """Return POMCP_OBS_CONFIG + HISTORY_LEN as kwargs for ``run_single``."""
    return dict(
        obs_type=POMCP_OBS_CONFIG['obs_type'],
        action_type=POMCP_OBS_CONFIG['action_type'],
        history_len=HISTORY_LEN,
        include_prev_action=POMCP_OBS_CONFIG['include_prev_action'],
        include_prev_reward=POMCP_OBS_CONFIG['include_prev_reward'],
    )


# =============================================================================
# Baseline (heuristic) agent functions
# =============================================================================

def run_random_agent(env, seed):
    """Uniformly random valid action selection."""
    rng = np.random.RandomState(seed)
    obs, _ = env.reset()
    total_reward = 0.0
    done = False

    while not done:
        adj_nodes = get_adj_states(env.node_current, env.adj_mat)
        valid_actions = []
        for a in range(env.action_space.n):
            node = env_transition_func(
                env.node_current, a, env.action_type,
                heading_latent=env.heading_latent, heading_real=env.heading_real,
                adj_mat=env.adj_mat, st_positions=env.node_coords,
            )
            if node in adj_nodes:
                valid_actions.append(a)

        if not valid_actions:
            action = rng.randint(env.action_space.n)
        else:
            action = valid_actions[rng.randint(len(valid_actions))]

        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        done = terminated or truncated

    return total_reward


def run_forward_random_agent(env, seed):
    """Random agent that avoids going back to the previous node (unless dead-end)."""
    rng = np.random.RandomState(seed)
    obs, _ = env.reset()
    total_reward = 0.0
    done = False
    prev_node = None

    while not done:
        adj_nodes = get_adj_states(env.node_current, env.adj_mat)
        forward_nodes = (
            [n for n in adj_nodes if n != prev_node]
            if prev_node is not None and len(adj_nodes) > 1
            else list(adj_nodes)
        )

        valid_actions = []
        for a in range(env.action_space.n):
            node = env_transition_func(
                env.node_current, a, env.action_type,
                heading_latent=env.heading_latent, heading_real=env.heading_real,
                adj_mat=env.adj_mat, st_positions=env.node_coords,
            )
            if node in forward_nodes:
                valid_actions.append(a)

        if not valid_actions:
            for a in range(env.action_space.n):
                node = env_transition_func(
                    env.node_current, a, env.action_type,
                    heading_latent=env.heading_latent, heading_real=env.heading_real,
                    adj_mat=env.adj_mat, st_positions=env.node_coords,
                )
                if node in adj_nodes:
                    valid_actions.append(a)

        action = valid_actions[rng.randint(len(valid_actions))] if valid_actions else 0
        prev_node = env.node_current
        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        done = terminated or truncated

    return total_reward


def bfs_nearest_reward(node, adj_mat, rewards):
    """BFS from *node* to nearest node with reward > 0.  Returns path (list of nodes)."""
    if rewards[node] > 0:
        return [node]
    visited = {node}
    queue = deque([(node, [node])])
    while queue:
        current, path = queue.popleft()
        for neighbor in get_adj_states(current, adj_mat):
            if neighbor not in visited:
                visited.add(neighbor)
                new_path = path + [neighbor]
                if rewards[neighbor] > 0:
                    return new_path
                queue.append((neighbor, new_path))
    return []


def _action_to_reach_node(env, target_node):
    """Find which action index moves from current node to *target_node*."""
    for a in range(env.action_space.n):
        node = env_transition_func(
            env.node_current, a, env.action_type,
            heading_latent=env.heading_latent, heading_real=env.heading_real,
            adj_mat=env.adj_mat, st_positions=env.node_coords,
        )
        if node == target_node:
            return a
    return None


def run_greedy_agent(env, seed):
    """Oracle greedy: BFS to nearest reward using full graph+reward knowledge."""
    rng = np.random.RandomState(seed)
    obs, _ = env.reset()
    total_reward = 0.0
    done = False
    current_path = []

    while not done:
        if not current_path:
            path = bfs_nearest_reward(env.node_current, env.adj_mat, env.rwd_per_node)
            if len(path) > 1:
                current_path = path[1:]
            else:
                adj_nodes = get_adj_states(env.node_current, env.adj_mat)
                valid_actions = []
                for a in range(env.action_space.n):
                    node = env_transition_func(
                        env.node_current, a, env.action_type,
                        heading_latent=env.heading_latent, heading_real=env.heading_real,
                        adj_mat=env.adj_mat, st_positions=env.node_coords,
                    )
                    if node in adj_nodes:
                        valid_actions.append(a)
                action = valid_actions[rng.randint(len(valid_actions))] if valid_actions else 0
                obs, reward, terminated, truncated, _ = env.step(action)
                total_reward += reward
                done = terminated or truncated
                continue

        target = current_path[0]
        action = _action_to_reach_node(env, target)
        if action is None:
            current_path = []
            continue

        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        done = terminated or truncated

        if env.node_current == target:
            current_path.pop(0)
            if reward > 0:
                current_path = []

    return total_reward
