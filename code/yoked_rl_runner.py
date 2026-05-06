"""
Yoked RL Simulation Runner

A minimal, modular interface for running RL simulations yoked to mouse behavior data.
Integrates with existing utils_latMaz and utils_latMaz_RL modules from code/code.

Usage:
    import sys
    sys.path.insert(0, '../../code/code')  # Add path to utils modules

    from yoked_rl_runner import YokedRLRunner, SimConfig

    runner = YokedRLRunner(yoking_df, rewarded_states_df)
    config = SimConfig(models=['A2C', 'PPO'], history_lengths=[0, 10])
    results = runner.run(config, target_repeats=10)
"""

import os
import sys
import ast
import re
import gc
import resource
import platform

# Add code to path for utils imports
_this_dir = os.path.dirname(os.path.abspath(__file__))
_code_in_mr = os.path.join(_this_dir, '..', '..', 'code')
if _code_in_mr not in sys.path:
    sys.path.insert(0, _code_in_mr)
from copy import deepcopy
from datetime import datetime
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any
from glob import glob

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces

from stable_baselines3 import A2C, DQN, PPO

# Memory monitoring (optional)
try:
    import psutil
    _PSUTIL_AVAILABLE = True
except ImportError:
    _PSUTIL_AVAILABLE = False

# PyTorch for custom DRQN agents (graceful degradation)
try:
    import torch
    import torch.nn as nn
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

# sb3_contrib agents (graceful degradation if not installed)
try:
    from sb3_contrib import TRPO, RecurrentPPO, QRDQN
    _SB3_CONTRIB_AVAILABLE = True
except ImportError:
    TRPO = None
    RecurrentPPO = None
    QRDQN = None
    _SB3_CONTRIB_AVAILABLE = False
    print("Warning: sb3-contrib not installed. TRPO, QRDQN, RecurrentPPO unavailable.")

from utils_latMaz import (
    load_maze,
    get_adj_states,
    displacement_to_compass_heading,
    get_direction_from_radian,
    allo_radian_map_dict,
    get_ego_direction,
    env_transition_func,
    allo_actions_one_hot_dict,
    ego_actions_one_hot_dict,
    get_now_str,
)
from utils_latMaz_RL import StopAfterFirstEpisode

# BeliefOracleAgent (OnlineBeliefBFS) — does not require pomdp-py
try:
    from advanced_agents import BeliefOracleAgent
except ImportError:
    BeliefOracleAgent = None

# POMCP planning agent (separate module, requires pomdp-py)
try:
    from advanced_agents import POMCPbioAlloAgent, POMCPbioEgoAgent, POMCPBioCSCGAgent, POMCP_AVAILABLE
    POMCPbioCSCGagent = POMCPBioCSCGAgent  # backward-compat alias
    from shelved_agents.pomcp_oracle import POMCPAgent
    _POMCP_AVAILABLE = POMCP_AVAILABLE
except ImportError:
    _POMCP_AVAILABLE = False

# Exploration agents (MOP, RC-GVF, CountBasedExplorer [shelved])
try:
    from exploration_agents import MOPagent, RCGVFAgent, CountBasedExplorerAgent
    _EXPLORATION_AVAILABLE = True
except ImportError:
    _EXPLORATION_AVAILABLE = False

# Intermediate-complexity agents (Markov, FSC, OOI)
try:
    from intermediate_agents import VarMarkovAgent, FSCAgent, OOIAgent, FSCBioAgent
    _INTERMEDIATE_AVAILABLE = True
except ImportError:
    _INTERMEDIATE_AVAILABLE = False

# Tiny RNN agent (Ji-An, Mattar 2025)
try:
    from tiny_rnn_agent import TinyRNNAgent
    _TINYRNN_AVAILABLE = True
except ImportError:
    _TINYRNN_AVAILABLE = False

# Dreamer-lite agent (Hafner et al. 2020, simplified)
try:
    from dreamer_agent import DreamerLiteAgent
    _DREAMER_AVAILABLE = True
except ImportError:
    _DREAMER_AVAILABLE = False

# Recurrent SAC-Discrete (Ni et al. 2022 + Christodoulou 2019)
try:
    from recurrent_sac_agent import RecurrentSACAgent
    _RECSAC_AVAILABLE = True
except ImportError:
    _RECSAC_AVAILABLE = False

# SSM-SAC (S4D-style state-space model)
try:
    from ssm_sac_agent import SSMSACAgent
    _SSM_SAC_AVAILABLE = True
except ImportError:
    _SSM_SAC_AVAILABLE = False


# =============================================================================
# Memory Monitoring
# =============================================================================

class MemoryMonitor:
    """Tracks current RSS and peak RSS with configurable threshold failsafes.

    Uses psutil for current RSS (if available) and resource.getrusage for
    process-lifetime peak RSS. On macOS, ru_maxrss is in bytes; on Linux, KB.
    """

    def __init__(self, limit_gb: float = 8.0, threshold_fraction: float = 0.80,
                 verbose: bool = True):
        self.verbose = verbose
        self._peak_rss_bytes = 0
        self._snapshots = []

        if _PSUTIL_AVAILABLE:
            total_ram = psutil.virtual_memory().total
        else:
            total_ram = 16 * 1024**3  # conservative fallback

        # Effective threshold: min of absolute limit and fraction of total RAM
        frac_limit = int(total_ram * threshold_fraction)
        abs_limit = int(limit_gb * 1024**3)
        self.threshold_bytes = min(frac_limit, abs_limit)

        if self.verbose:
            print(f"[MemoryMonitor] Total RAM: {total_ram / 1024**3:.1f} GB, "
                  f"limit: {self.threshold_bytes / 1024**3:.1f} GB "
                  f"(min of {limit_gb:.1f} GB abs, "
                  f"{frac_limit / 1024**3:.1f} GB @ {threshold_fraction:.0%})")

    def get_current_rss_bytes(self) -> int:
        if _PSUTIL_AVAILABLE:
            return psutil.Process(os.getpid()).memory_info().rss
        return self._get_peak_rss_resource()

    def _get_peak_rss_resource(self) -> int:
        usage = resource.getrusage(resource.RUSAGE_SELF)
        if platform.system() == 'Darwin':
            return usage.ru_maxrss          # bytes on macOS
        else:
            return usage.ru_maxrss * 1024   # KB -> bytes on Linux

    def get_peak_rss_bytes(self) -> int:
        resource_peak = self._get_peak_rss_resource()
        return max(resource_peak, self._peak_rss_bytes)

    def snapshot(self, label: str = "") -> dict:
        current = self.get_current_rss_bytes()
        peak = self.get_peak_rss_bytes()
        if current > self._peak_rss_bytes:
            self._peak_rss_bytes = current

        info = {
            'label': label,
            'rss_mb': current / (1024**2),
            'peak_mb': peak / (1024**2),
            'timestamp': datetime.now().isoformat(),
        }
        self._snapshots.append(info)

        if self.verbose:
            print(f"  [memory] {label}: RSS={info['rss_mb']:.0f} MB, "
                  f"peak={info['peak_mb']:.0f} MB")
        return info

    def check_threshold(self, label: str = "") -> str:
        """Returns 'ok', 'warning' (gc fixed it), or 'critical' (still over)."""
        current = self.get_current_rss_bytes()
        if current < self.threshold_bytes:
            return 'ok'

        if self.verbose:
            print(f"  [memory] WARNING at '{label}': "
                  f"RSS={current / 1024**2:.0f} MB exceeds "
                  f"threshold {self.threshold_bytes / 1024**2:.0f} MB. "
                  f"Running gc.collect()...")

        gc.collect()
        current_after = self.get_current_rss_bytes()
        if self.verbose:
            print(f"  [memory] After gc: RSS={current_after / 1024**2:.0f} MB")

        if current_after < self.threshold_bytes:
            return 'warning'
        return 'critical'

    def get_summary(self) -> dict:
        if not self._snapshots:
            return {}
        rss_values = [s['rss_mb'] for s in self._snapshots]
        return {
            'n_snapshots': len(self._snapshots),
            'min_rss_mb': min(rss_values),
            'max_rss_mb': max(rss_values),
            'final_rss_mb': rss_values[-1],
            'process_peak_mb': self.get_peak_rss_bytes() / (1024**2),
        }


# =============================================================================
# Configuration Classes
# =============================================================================

@dataclass
class MemoryConfig:
    """Configuration for memory monitoring and failsafes."""
    enabled: bool = True
    limit_gb: float = 8.0
    threshold_fraction: float = 0.80
    snapshot_every_n_sims: int = 10
    save_every_n_sims: int = 50
    verbose: bool = True

@dataclass
class SimConfig:
    """Configuration for simulation runs."""
    models: List[str] = field(default_factory=lambda: [
        'A2C', 'DQN', 'PPO', 'TRPO', 'QRDQN', 'RecurrentPPO',
        'DRQN_seq', 'DRQN_rand'])
    history_lengths: List[int] = field(default_factory=lambda: [0, 10, 20])
    observation_configs: List[Dict] = field(default_factory=lambda: [
        {'name': 'ego', 'obs_type': 'ego', 'action_type': 'ego',
         'include_prev_action': True, 'include_prev_reward': True},
        {'name': 'allo', 'obs_type': 'allo_latent', 'action_type': 'allo_latent',
         'include_prev_action': False, 'include_prev_reward': False},
    ])
    seed_base: int = 42
    verbose: bool = False


@dataclass
class AlgoConfig:
    """Hyperparameters for each algorithm type."""
    A2C: Dict = field(default_factory=lambda: {
        'learning_rate': 7e-4, 'n_steps': 5, 'gamma': 0.99,
        'gae_lambda': 1.0, 'ent_coef': 0.01, 'vf_coef': 0.25, 'max_grad_norm': 0.5
    })
    DQN: Dict = field(default_factory=lambda: {
        'learning_rate': 1e-3, 'buffer_size': 50_000, 'learning_starts': 50,
        'batch_size': 64, 'tau': 1.0, 'gamma': 0.99, 'train_freq': 4,
        'gradient_steps': 1, 'target_update_interval': 250
    })
    PPO: Dict = field(default_factory=lambda: {
        'learning_rate': 3e-4, 'n_steps': 128, 'batch_size': 64, 'n_epochs': 10,
        'gamma': 0.99, 'gae_lambda': 0.95, 'clip_range': 0.2,
        'ent_coef': 0.01, 'vf_coef': 0.5, 'max_grad_norm': 0.5
    })
    TRPO: Dict = field(default_factory=lambda: {
        'learning_rate': 1e-3, 'n_steps': 64, 'target_kl': 0.01,
        'cg_max_steps': 10, 'gamma': 0.99,
    })
    QRDQN: Dict = field(default_factory=lambda: {
        'learning_rate': 5e-4, 'buffer_size': 200_000, 'batch_size': 64,
        'gamma': 0.99, 'exploration_fraction': 0.2, 'exploration_final_eps': 0.02,
    })
    RecurrentPPO: Dict = field(default_factory=lambda: {
        'learning_rate': 2.5e-4, 'n_steps': 128, 'batch_size': 64,
        'n_epochs': 10, 'gamma': 0.99, 'gae_lambda': 0.95, 'clip_range': 0.2,
        'ent_coef': 0.01, 'vf_coef': 0.5, 'max_grad_norm': 0.5,
        # LSTM-specific (extracted before model creation)
        'lstm_hidden_size': 64, 'n_lstm_layers': 1,
    })
    DRQN_seq: Dict = field(default_factory=lambda: {
        'hidden_dim': 64, 'lstm_hidden': 64, 'n_lstm_layers': 1,
        'learning_rate': 1e-3, 'gamma': 0.99,
        'epsilon_start': 1.0, 'epsilon_end': 0.05, 'epsilon_decay_steps': 200,
        'seq_len': 10, 'batch_size': 16,
        'train_every_n_steps': 4, 'target_update_every': 50,
        'replay_mode': 'sequential',
    })
    DRQN_rand: Dict = field(default_factory=lambda: {
        'hidden_dim': 64, 'lstm_hidden': 64, 'n_lstm_layers': 1,
        'learning_rate': 1e-3, 'gamma': 0.99,
        'epsilon_start': 1.0, 'epsilon_end': 0.05, 'epsilon_decay_steps': 200,
        'seq_len': 10, 'batch_size': 16,
        'train_every_n_steps': 4, 'target_update_every': 50,
        'replay_mode': 'random',
    })
    POMCP: Dict = field(default_factory=lambda: {
        'max_depth': 100, 'num_sims': 500,
        'exploration_const': 1.0, 'discount': 0.99,
    })
    POMCP_bio: Dict = field(default_factory=lambda: {
        'max_depth': 100, 'num_sims': 500,
        'exploration_const': 1.0, 'discount': 0.99,
    })
    POMCP_novelty: Dict = field(default_factory=lambda: {
        'max_depth': 100, 'num_sims': 500,
        'exploration_const': 1.0, 'discount': 0.99,
        'novelty_rollout': True,
    })
    POMCP_bio_novelty: Dict = field(default_factory=lambda: {
        'max_depth': 100, 'num_sims': 500,
        'exploration_const': 1.0, 'discount': 0.99,
        'novelty_rollout': True,
    })
    POMCP_bio_egoOnly: Dict = field(default_factory=lambda: {
        'max_depth': 100, 'num_sims': 500,
        'exploration_const': 1.0, 'discount': 0.99,
    })
    POMCP_bio_CSCG: Dict = field(default_factory=lambda: {
        'max_depth': 100, 'num_sims': 500,
        'exploration_const': 1.0, 'discount': 0.99,
        'n_clones': 10,
    })
    BeliefOracle: Dict = field(default_factory=lambda: {
        'exploration_bonus': 0.5,
        'optimistic_reward_prior': True,
    })
    RC_GVF: Dict = field(default_factory=lambda: {
        'hidden_dim': 64, 'n_cumulants': 8, 'n_ensemble': 5,
        'intrinsic_coef': 0.1, 'gvf_gamma': 0.9,
        'learning_rate': 1e-3, 'gamma': 0.99,
        'ent_coef': 0.01, 'vf_coef': 0.5, 'max_grad_norm': 0.5,
    })
    MOP: Dict = field(default_factory=lambda: {
        'alpha': 1.0, 'gamma_mop': 0.99, 'max_iter': 500, 'tol': 1e-6,
        'reward_weight': 0.0,
    })
    VarMarkov: Dict = field(default_factory=lambda: {
        'max_depth': 4, 'min_count': 3, 'pseudocount': 1.0,
        'forward_bias': 0.5,
    })
    FSC_oracle: Dict = field(default_factory=lambda: {
        'n_nodes': 4, 'n_cem_iters': 20, 'n_candidates': 50,
        'n_elite': 10, 'n_eval_episodes': 3, 'train_budget': None,
    })
    OOI: Dict = field(default_factory=lambda: {
        'pseudocount': 1.0, 'forward_bias': 1.0,
        'max_corridor_steps': 20,
    })
    FSC_bio: Dict = field(default_factory=lambda: {
        'exploit_timeout': 10, 'trace_decay': 0.8,
        'pseudocount': 1.0, 'forward_bias': 1.0,
        'prevent_backward': True, 'exploit_greediness': 3.0,
    })

    def get(self, model_name: str) -> Dict:
        return getattr(self, model_name, {})


# =============================================================================
# Environment Classes
# =============================================================================

class GraphMazeEnv(gym.Env):
    """Custom gym environment for graph-based maze navigation with PacMan rewards."""

    def __init__(
        self,
        adjacency_matrix: np.ndarray,
        node_xy_coordinate_list: np.ndarray,
        start_node: int,
        rwd_per_node_initial: np.ndarray,
        n_valid_actions_per_episode: int,
        action_type: str,
        obs_type: str,
        min_allowed_rewarded_states: int = 2,
        prevent_reverse: bool = False,
        verbose: bool = False,
    ):
        super().__init__()
        self.verbose = verbose
        self.adj_mat = deepcopy(adjacency_matrix)
        self.node_coords = deepcopy(node_xy_coordinate_list)
        self.n_nodes = len(adjacency_matrix)
        self.action_type = action_type
        self.obs_type = obs_type
        self.n_valid_actions_per_episode = n_valid_actions_per_episode
        self.min_allowed_rewarded_states = min_allowed_rewarded_states
        self.prevent_reverse = prevent_reverse

        # Action space
        if action_type == 'node':
            self.action_space = spaces.Discrete(self.n_nodes)
        else:
            self.action_space = spaces.Discrete(4)

        # Observation space
        if obs_type == 'node_current':
            self.observation_space = spaces.Discrete(self.n_nodes)
        elif obs_type == 'nodes_adj':
            self.observation_space = spaces.MultiBinary(self.n_nodes)
        else:
            self.observation_space = spaces.MultiBinary(4)

        # State initialization
        self.node_start = start_node
        self.rwd_per_node_initial = deepcopy(rwd_per_node_initial)
        self.heading_latent_initial = np.pi / 2
        self.heading_real_initial = np.pi / 2

        # Logging
        self.observation_log = []
        self._reset_state()

    def _reset_state(self):
        self.node_current = self.node_start
        self.node_prior = None
        self.heading_latent = self.heading_latent_initial
        self.heading_real = self.heading_real_initial
        self.rwd_per_node = deepcopy(self.rwd_per_node_initial)
        self.n_valid_actions_taken = 0
        self.last_reward = 0.0

    def _get_observation(self):
        adj_nodes = get_adj_states(self.node_current, self.adj_mat)
        # Filter out backward node when prevent_reverse is on
        if self.prevent_reverse and self.node_prior is not None and len(adj_nodes) > 1:
            adj_nodes = [n for n in adj_nodes if n != self.node_prior]
        adj_allo_latent = [displacement_to_compass_heading(
            self.node_coords[n] - self.node_coords[self.node_current]) for n in adj_nodes]
        adj_ego = [get_ego_direction(
            self.node_coords[self.node_current], self.node_coords[n], self.heading_latent) for n in adj_nodes]

        # Store full observation for logging
        self.obs_current = {
            'reward': float(self.last_reward),
            'node_current': int(self.node_current),
            'nodes_adj': list(adj_nodes),
            'allo_latent': list(adj_allo_latent),
            'ego': list(adj_ego),
        }

        if self.obs_type == 'nodes_adj':
            obs = np.zeros(self.n_nodes, dtype=np.int8)
            obs[list(adj_nodes)] = 1
            return obs
        elif self.obs_type == 'ego':
            obs = np.zeros(4, dtype=np.int8)
            obs[[ego_actions_one_hot_dict[a] for a in adj_ego]] = 1
            return obs
        elif self.obs_type == 'allo_latent':
            obs = np.zeros(4, dtype=np.int8)
            obs[[allo_actions_one_hot_dict[a] for a in adj_allo_latent]] = 1
            return obs
        elif self.obs_type == 'node_current':
            return int(self.node_current)
        else:
            raise ValueError(f"Unknown obs_type: {self.obs_type}")

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self._np_random, seed = gym.utils.seeding.np_random(seed)
        self._reset_state()
        obs = self._get_observation()
        self.observation_log.append(self.obs_current)
        return obs, {}

    def step(self, action):
        # SB3 may pass np.ndarray or np.int64 — ensure plain int
        action = int(action)
        # Get next node
        if self.action_type == 'node':
            node_selected = action
        else:
            node_selected = env_transition_func(
                self.node_current, action, self.action_type,
                heading_latent=self.heading_latent, heading_real=self.heading_real,
                adj_mat=self.adj_mat, st_positions=self.node_coords
            )

        # Check validity
        adj_nodes = get_adj_states(self.node_current, self.adj_mat)
        valid = node_selected in adj_nodes
        # Block reversal when prevent_reverse is on (allow at dead-ends)
        if valid and self.prevent_reverse and self.node_prior is not None:
            if node_selected == self.node_prior and len(adj_nodes) > 1:
                valid = False

        if valid:
            self.node_prior = self.node_current
            self.node_current = node_selected
            self.n_valid_actions_taken += 1

            # Update heading
            disp = self.node_coords[self.node_current] - self.node_coords[self.node_prior]
            self.heading_latent = allo_radian_map_dict[displacement_to_compass_heading(disp)]

            # Get reward (PacMan style) — matches RPi experiment logic:
            #   1. Only rebait+remove when a reward is actually collected
            #   2. Check rebait BEFORE removing consumed state
            #   3. Consumed state is NOT restored after rebait
            # RPi code: if len(rewarded_states) == RESET_WHEN_N_RWDS_REMAINING + 1
            # min_allowed_rewarded_states = RESET_WHEN_N_RWDS_REMAINING + 1
            # so check: sum == min_allowed (NOT min_allowed + 1, that double-counts)
            self.last_reward = float(self.rwd_per_node[self.node_current])
            if self.last_reward > 0:
                if sum(self.rwd_per_node) == self.min_allowed_rewarded_states:
                    self.rwd_per_node = deepcopy(self.rwd_per_node_initial)
                self.rwd_per_node[self.node_current] = 0
        else:
            self.last_reward = 0.0

        # Check termination
        truncated = self.n_valid_actions_taken >= self.n_valid_actions_per_episode

        obs = self._get_observation()
        if valid:
            self.observation_log.append(self.obs_current)

        return obs, self.last_reward, False, truncated, {}


class HistoryConcatWrapper(gym.Wrapper):
    """Wraps environment to concatenate observation history."""

    def __init__(self, env, history_len=1, include_prev_action=False, include_prev_reward=False):
        super().__init__(env)
        self.history_len = history_len
        self.include_prev_action = include_prev_action
        self.include_prev_reward = include_prev_reward

        # Get dimensions
        if isinstance(env.observation_space, spaces.MultiBinary):
            self.obs_dim = env.observation_space.n
        elif isinstance(env.observation_space, spaces.Box):
            self.obs_dim = env.observation_space.shape[0]
        else:
            self.obs_dim = env.observation_space.n
        self._discrete_obs = isinstance(env.observation_space, spaces.Discrete)
        self.action_dim = env.action_space.n

        # Compute total dimension
        total_dim = self.obs_dim * (history_len + 1)
        if include_prev_action:
            total_dim += self.action_dim * history_len
        if include_prev_reward:
            total_dim += history_len
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(total_dim,), dtype=np.float32)

    def _convert_obs(self, obs):
        if self._discrete_obs:
            one_hot = np.zeros(self.obs_dim, dtype=np.float32)
            one_hot[int(obs)] = 1.0
            return one_hot
        return np.asarray(obs, dtype=np.float32).flatten()

    def _build_obs(self):
        parts = list(self._obs_history)
        if self.include_prev_action:
            for a in self._action_history:
                one_hot = np.zeros(self.action_dim, dtype=np.float32)
                one_hot[int(a)] = 1.0
                parts.append(one_hot)
        if self.include_prev_reward:
            parts.append(np.array(self._reward_history, dtype=np.float32))
        return np.concatenate(parts).astype(np.float32)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        obs_conv = self._convert_obs(obs)
        zero = np.zeros(self.obs_dim, dtype=np.float32)
        self._obs_history = [zero.copy() for _ in range(self.history_len)] + [obs_conv]
        self._action_history = [0] * self.history_len
        self._reward_history = [0.0] * self.history_len
        return self._build_obs(), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._obs_history.pop(0)
        self._obs_history.append(self._convert_obs(obs))
        if self.history_len > 0:
            self._action_history.pop(0)
            self._action_history.append(int(action))
            self._reward_history.pop(0)
            self._reward_history.append(float(reward))
        return self._build_obs(), reward, terminated, truncated, info


# =============================================================================
# DRQN Agent (Hausknecht & Stone, 2015 — adapted for graph observations)
# =============================================================================

if _TORCH_AVAILABLE:

    class DRQNNetwork(nn.Module):
        """Q-value network with LSTM for temporal integration.

        Architecture: obs → Linear → ReLU → LSTM → Linear → Q-values
        Replaces the CNN in the original DRQN paper with a linear embedding,
        since our observations are already structured (MultiBinary(4)).
        """

        def __init__(self, obs_dim=4, n_actions=4, hidden_dim=64,
                     lstm_hidden=64, n_lstm_layers=1):
            super().__init__()
            self.lstm_hidden = lstm_hidden
            self.n_lstm_layers = n_lstm_layers
            self.embed = nn.Sequential(nn.Linear(obs_dim, hidden_dim), nn.ReLU())
            self.lstm = nn.LSTM(hidden_dim, lstm_hidden, n_lstm_layers,
                                batch_first=True)
            self.q_head = nn.Linear(lstm_hidden, n_actions)

        def forward(self, obs_seq, hidden=None):
            """Forward pass over a sequence of observations.

            Args:
                obs_seq: (batch, seq_len, obs_dim)
                hidden: (h, c) each (n_layers, batch, lstm_hidden) or None
            Returns:
                q_values: (batch, seq_len, n_actions)
                new_hidden: (h, c)
            """
            x = self.embed(obs_seq)
            lstm_out, new_hidden = self.lstm(x, hidden)
            return self.q_head(lstm_out), new_hidden

        def init_hidden(self, batch_size=1):
            h = torch.zeros(self.n_lstm_layers, batch_size, self.lstm_hidden)
            c = torch.zeros(self.n_lstm_layers, batch_size, self.lstm_hidden)
            return (h, c)

    class EpisodeReplayBuffer:
        """Stores transitions from a single episode, samples subsequences.

        Two modes per Hausknecht & Stone (2015):
          'sequential': bootstrapped sequential updates
          'random': bootstrapped random updates
        """

        def __init__(self):
            self.obs = []
            self.actions = []
            self.rewards = []
            self.next_obs = []
            self.dones = []

        def add(self, obs, action, reward, next_obs, done):
            self.obs.append(obs)
            self.actions.append(action)
            self.rewards.append(reward)
            self.next_obs.append(next_obs)
            self.dones.append(done)

        def __len__(self):
            return len(self.obs)

        def can_sample(self, batch_size, seq_len):
            return len(self) >= seq_len and len(self) >= batch_size

        def sample_sequences(self, batch_size, seq_len, mode='sequential'):
            """Sample batch of subsequences as tensors.

            Hausknecht & Stone (2015) §3.2 distinction, adapted for a single-
            episode buffer (each yoked session is one episode):
              'random'     : batch_size independent uniform sub-sequence starts
                             (zero-init hidden, mini-trajectories anywhere in
                             the buffer).
              'sequential' : ONE random anchor; then batch_size adjacent
                             non-overlapping chunks of length seq_len starting
                             at that anchor. Preserves temporal contiguity
                             across batch elements, mirroring H&S's
                             "bootstrapped sequential updates" intent in the
                             single-episode regime. If the buffer is too short
                             for batch_size * seq_len contiguous samples,
                             fall back to evenly-spaced starts spanning
                             [0, max_start] (still temporally ordered, just
                             with gaps).

            Issue history: prior to 260505 this method ignored `mode` entirely
            and always used the random branch, producing bit-identical outputs
            for DRQN_seq and DRQN_rand under matched seeds (see
            260505_drqn_replay_mode_fix.txt).
            """
            max_start = len(self) - seq_len
            if mode == 'sequential':
                stride = seq_len
                max_anchor = max_start - (batch_size - 1) * stride
                if max_anchor >= 0:
                    anchor = np.random.randint(0, max_anchor + 1)
                    starts = anchor + np.arange(batch_size) * stride
                else:
                    # buffer too small for fully-contiguous batch — fall back
                    # to evenly-spaced ordered starts (still strictly ordered)
                    starts = np.linspace(0, max_start, batch_size).astype(int)
            else:  # 'random'
                starts = np.random.randint(0, max_start + 1, size=batch_size)
            o, a, r, no, d = [], [], [], [], []
            for s in starts:
                sl = slice(s, s + seq_len)
                o.append(self.obs[sl])
                a.append(self.actions[sl])
                r.append(self.rewards[sl])
                no.append(self.next_obs[sl])
                d.append(self.dones[sl])
            return {
                'obs': torch.FloatTensor(np.array(o)),
                'actions': torch.LongTensor(np.array(a)),
                'rewards': torch.FloatTensor(np.array(r)),
                'next_obs': torch.FloatTensor(np.array(no)),
                'dones': torch.FloatTensor(np.array(d, dtype=np.float32)),
            }

    class DRQNAgent:
        """Deep Recurrent Q-Network (Hausknecht & Stone, 2015).

        Pure PyTorch implementation with SB3-compatible API for integration
        with YokedRLRunner.run_single(). Replaces CNN with linear embedding
        for graph-based observations.

        Two variants via replay_mode kwarg:
          'sequential': bootstrapped sequential updates
          'random': bootstrapped random updates
        """

        def __init__(self, policy_str, env, verbose=0, device="cpu",
                     seed=0, **kwargs):
            self.env = env
            self.verbose = verbose
            self.device = torch.device(device)
            torch.manual_seed(seed)
            np.random.seed(seed)

            # Hyperparameters
            self.hidden_dim = kwargs.get('hidden_dim', 64)
            self.lstm_hidden = kwargs.get('lstm_hidden', 64)
            self.n_lstm_layers = kwargs.get('n_lstm_layers', 1)
            self.lr = kwargs.get('learning_rate', 1e-3)
            self.gamma = kwargs.get('gamma', 0.99)
            self.eps_start = kwargs.get('epsilon_start', 1.0)
            self.eps_end = kwargs.get('epsilon_end', 0.05)
            self.eps_decay = kwargs.get('epsilon_decay_steps', 200)
            self.seq_len = kwargs.get('seq_len', 10)
            self.batch_size = kwargs.get('batch_size', 16)
            self.train_freq = kwargs.get('train_every_n_steps', 4)
            self.target_freq = kwargs.get('target_update_every', 50)
            self.replay_mode = kwargs.get('replay_mode', 'sequential')

            # Derive dims from env
            obs_sp = env.observation_space
            if isinstance(obs_sp, spaces.MultiBinary):
                self.obs_dim = obs_sp.n
            elif isinstance(obs_sp, spaces.Box):
                self.obs_dim = obs_sp.shape[0]
            else:
                self.obs_dim = obs_sp.n
            self.n_actions = env.action_space.n

            # Networks
            net_kwargs = dict(obs_dim=self.obs_dim, n_actions=self.n_actions,
                              hidden_dim=self.hidden_dim,
                              lstm_hidden=self.lstm_hidden,
                              n_lstm_layers=self.n_lstm_layers)
            self.q_net = DRQNNetwork(**net_kwargs).to(self.device)
            self.target_net = DRQNNetwork(**net_kwargs).to(self.device)
            self.target_net.load_state_dict(self.q_net.state_dict())
            self.target_net.eval()

            self.optimizer = torch.optim.Adam(self.q_net.parameters(),
                                              lr=self.lr)
            self.replay = EpisodeReplayBuffer()
            self._steps = 0
            self._train_steps = 0

        def _epsilon(self):
            frac = min(1.0, self._steps / max(1, self.eps_decay))
            return self.eps_start + frac * (self.eps_end - self.eps_start)

        @torch.no_grad()
        def _select_action(self, obs, hidden):
            obs_t = torch.FloatTensor(obs).unsqueeze(0).unsqueeze(0).to(
                self.device)
            q_vals, new_hidden = self.q_net(obs_t, hidden)
            q_vals = q_vals.squeeze(0).squeeze(0).cpu().numpy()
            if np.random.random() < self._epsilon():
                action = np.random.randint(self.n_actions)
            else:
                action = int(np.argmax(q_vals))
            return action, new_hidden

        def _train_step(self):
            if not self.replay.can_sample(self.batch_size, self.seq_len):
                return
            batch = self.replay.sample_sequences(
                self.batch_size, self.seq_len, self.replay_mode)
            obs = batch['obs'].to(self.device)
            actions = batch['actions'].to(self.device)
            rewards = batch['rewards'].to(self.device)
            next_obs = batch['next_obs'].to(self.device)
            dones = batch['dones'].to(self.device)

            q_vals, _ = self.q_net(obs, None)
            q_taken = q_vals.gather(2, actions.unsqueeze(-1)).squeeze(-1)

            with torch.no_grad():
                tgt_q, _ = self.target_net(next_obs, None)
                tgt_max = tgt_q.max(dim=2).values
                targets = rewards + self.gamma * tgt_max * (1.0 - dones)

            loss = nn.functional.mse_loss(q_taken, targets)
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), 10.0)
            self.optimizer.step()

            self._train_steps += 1
            if self._train_steps % self.target_freq == 0:
                self.target_net.load_state_dict(self.q_net.state_dict())

        def set_env(self, new_env):
            """Swap environment for pretraining. Keeps network weights, resets replay buffer."""
            self.env = new_env
            self.replay = EpisodeReplayBuffer()

        def learn(self, total_timesteps=None, callback=None, **kwargs):
            """Run one episode with online DRQN training.

            Terminates when env signals truncated=True. The callback arg is
            accepted for SB3 API compatibility but not invoked.
            """
            obs, _ = self.env.reset()
            hidden = self.q_net.init_hidden(1)
            hidden = (hidden[0].to(self.device), hidden[1].to(self.device))

            done = False
            while not done:
                action, hidden = self._select_action(obs, hidden)
                next_obs, reward, terminated, truncated, _ = self.env.step(
                    action)
                done = terminated or truncated

                self.replay.add(
                    np.asarray(obs, dtype=np.float32),
                    int(action), float(reward),
                    np.asarray(next_obs, dtype=np.float32),
                    float(done))
                self._steps += 1
                obs = next_obs

                if self._steps % self.train_freq == 0:
                    self._train_step()

            # Final training passes on complete episode data
            for _ in range(3):
                self._train_step()


# =============================================================================
# Runner Class
# =============================================================================

class YokedRLRunner:
    """Main runner for yoked RL simulations."""

    # Build MODEL_CLASSES dynamically to handle missing dependencies
    MODEL_CLASSES = {'A2C': A2C, 'DQN': DQN, 'PPO': PPO}
    if TRPO is not None:
        MODEL_CLASSES['TRPO'] = TRPO
    if QRDQN is not None:
        MODEL_CLASSES['QRDQN'] = QRDQN
    if RecurrentPPO is not None:
        MODEL_CLASSES['RecurrentPPO'] = RecurrentPPO
    if _TORCH_AVAILABLE:
        MODEL_CLASSES['DRQN_seq'] = DRQNAgent
        MODEL_CLASSES['DRQN_rand'] = DRQNAgent
    if _POMCP_AVAILABLE:
        MODEL_CLASSES['POMCP'] = POMCPAgent
        MODEL_CLASSES['POMCP_bio'] = POMCPbioAlloAgent
        MODEL_CLASSES['POMCP_novelty'] = POMCPAgent
        MODEL_CLASSES['POMCP_bio_novelty'] = POMCPbioAlloAgent
        MODEL_CLASSES['POMCP_bio_egoOnly'] = POMCPbioEgoAgent
        MODEL_CLASSES['POMCP_bio_CSCG'] = POMCPbioCSCGagent
    if BeliefOracleAgent is not None:
        MODEL_CLASSES['BeliefOracle'] = BeliefOracleAgent
    if _EXPLORATION_AVAILABLE:
        MODEL_CLASSES['MOP'] = MOPagent
        MODEL_CLASSES['RC_GVF'] = RCGVFAgent
        MODEL_CLASSES['CountBasedExplorer'] = CountBasedExplorerAgent
    if _INTERMEDIATE_AVAILABLE:
        MODEL_CLASSES['VarMarkov'] = VarMarkovAgent
        MODEL_CLASSES['FSC_oracle'] = FSCAgent
        MODEL_CLASSES['OOI'] = OOIAgent
        MODEL_CLASSES['FSC_bio'] = FSCBioAgent
    if _TINYRNN_AVAILABLE:
        MODEL_CLASSES['TinyRNN'] = TinyRNNAgent
    if _DREAMER_AVAILABLE:
        MODEL_CLASSES['DreamerLite'] = DreamerLiteAgent
    if _RECSAC_AVAILABLE:
        MODEL_CLASSES['RecurrentSAC'] = RecurrentSACAgent
    if _SSM_SAC_AVAILABLE:
        MODEL_CLASSES['SSM_SAC'] = SSMSACAgent

    # Agents that use LSTM memory (skip h_len > 0 in run loop)
    _LSTM_AGENTS = {'RecurrentPPO', 'DRQN_seq', 'DRQN_rand', 'RC_GVF', 'CountBasedExplorer'}

    # Custom agents with own training loop (not SB3 — skip policy_kwargs injection)
    _CUSTOM_AGENTS = {'DRQN_seq', 'DRQN_rand', 'RC_GVF', 'CountBasedExplorer'}

    # Planning agents (skip h_len > 0; don't use gym env or SB3 API)
    _PLANNING_AGENTS = {'POMCP', 'POMCP_bio', 'POMCP_novelty', 'POMCP_bio_novelty', 'POMCP_bio_egoOnly', 'POMCP_bio_CSCG', 'BeliefOracle', 'MOP', 'VarMarkov', 'FSC_oracle', 'OOI', 'FSC_bio', 'TinyRNN', 'DreamerLite', 'RecurrentSAC', 'SSM_SAC'}

    def __init__(
        self,
        yoking_df: pd.DataFrame,
        rewarded_states_df: Optional[pd.DataFrame] = None,
        maze_dir: str = '../data_in/mazes',
        output_dir: str = '../data_out/rl_sims',
        algo_config: Optional[AlgoConfig] = None,
    ):
        self.yoking_df = yoking_df.copy()
        self.maze_dir = maze_dir
        self.output_dir = output_dir
        self.algo_config = algo_config or AlgoConfig()
        os.makedirs(output_dir, exist_ok=True)

        # Build rewarded states lookup
        self.rwd_lookup = {}
        if rewarded_states_df is not None:
            for _, row in rewarded_states_df.iterrows():
                match = re.search(r'(\d{6}-\d{6})', str(row.get('csv_data_path', '')))
                if match:
                    exp_mom = match.group(1)
                    try:
                        states = ast.literal_eval(row['rewarded_states']) if isinstance(row['rewarded_states'], str) else row['rewarded_states']
                        reset_val = int(row.get('reset_when_n_rwds_remaining', 1)) + 1
                        self.rwd_lookup[exp_mom] = (states, reset_val)
                    except:
                        pass

        # Cache for mazes
        self._maze_cache = {}

    def _load_maze(self, adj_file: str, st_pos_file: str) -> Tuple[np.ndarray, np.ndarray]:
        key = (adj_file, st_pos_file)
        if key not in self._maze_cache:
            adj_path = os.path.join(self.maze_dir, adj_file)
            pos_path = os.path.join(self.maze_dir, st_pos_file)
            self._maze_cache[key] = load_maze(adj_path, pos_path)
        return self._maze_cache[key]

    def _get_rewarded_states(self, exp_moment: str, n_nodes: int) -> Tuple[np.ndarray, int]:
        """Get initial reward distribution for a session."""
        if exp_moment in self.rwd_lookup:
            states, reset_val = self.rwd_lookup[exp_moment]
            rwd = np.zeros(n_nodes)
            for s in states:
                if 0 <= s < n_nodes:
                    rwd[s] = 1.0
            return rwd, reset_val
        # Fallback: all nodes rewarded, default rebait threshold
        return np.ones(n_nodes), 2

    def get_pretrain_sessions(self, yoke_row, max_pretrain_sessions=None):
        """Get earlier sessions for same animal + same maze, sorted oldest->newest.

        Ported from legacy get_pretrain_yoke_df_strict(). Requires yoking_df
        to have 'exp_moment', 'animal_ID', 'adj_file', 'st_pos_file' columns.

        Args:
            yoke_row: Row from yoking_df for the target session.
            max_pretrain_sessions: If not None, keep only the most recent N.

        Returns:
            DataFrame of prior sessions sorted ascending by exp_moment.
        """
        animal = yoke_row['animal_ID']
        exp_moment = yoke_row['exp_moment']
        adj_file = yoke_row['adj_file']
        st_pos_file = yoke_row['st_pos_file']

        mask = (
            (self.yoking_df['animal_ID'] == animal) &
            (self.yoking_df['exp_moment'] < exp_moment) &
            (self.yoking_df['adj_file'] == adj_file) &
            (self.yoking_df['st_pos_file'] == st_pos_file)
        )
        pre = self.yoking_df.loc[mask].sort_values('exp_moment')

        if max_pretrain_sessions is not None:
            pre = pre.tail(max_pretrain_sessions)

        return pre

    def get_pretrain_sessions_cross_maze(self, yoke_row,
                                          max_pretrain_sessions=None):
        """Get earlier sessions for same animal, ANY maze, sorted oldest->newest.

        Unlike get_pretrain_sessions() which requires same maze, this includes
        sessions from different mazes. The ego observation space (MultiBinary(4))
        is identical across mazes, so cross-maze pretraining can teach general
        navigation strategies.

        Args:
            yoke_row: Row from yoking_df for the target session.
            max_pretrain_sessions: If not None, keep only the most recent N.

        Returns:
            DataFrame of prior sessions sorted ascending by exp_moment.
        """
        animal = yoke_row['animal_ID']
        exp_moment = yoke_row['exp_moment']

        mask = (
            (self.yoking_df['animal_ID'] == animal) &
            (self.yoking_df['exp_moment'] < exp_moment)
        )
        pre = self.yoking_df.loc[mask].sort_values('exp_moment')

        if max_pretrain_sessions is not None:
            pre = pre.tail(max_pretrain_sessions)

        return pre

    def _create_env(
        self,
        adj_mat: np.ndarray,
        st_positions: np.ndarray,
        start_node: int,
        rewards: np.ndarray,
        n_actions: int,
        obs_type: str,
        action_type: str,
        history_len: int,
        include_prev_action: bool,
        include_prev_reward: bool,
        min_allowed_rewarded_states: int,
        prevent_reverse: bool = False,
    ) -> gym.Env:
        env = GraphMazeEnv(
            adjacency_matrix=adj_mat,
            node_xy_coordinate_list=st_positions,
            start_node=start_node,
            rwd_per_node_initial=rewards,
            n_valid_actions_per_episode=n_actions,
            action_type=action_type,
            obs_type=obs_type,
            min_allowed_rewarded_states=min_allowed_rewarded_states,
            prevent_reverse=prevent_reverse,
        )
        if history_len > 0:
            env = HistoryConcatWrapper(
                env, history_len=history_len,
                include_prev_action=include_prev_action,
                include_prev_reward=include_prev_reward
            )
        return env

    def run_single(
        self,
        model_name: str,
        adj_mat: np.ndarray,
        st_positions: np.ndarray,
        start_node: int,
        rewards: np.ndarray,
        n_actions: int,
        obs_type: str,
        action_type: str,
        history_len: int,
        include_prev_action: bool,
        include_prev_reward: bool,
        min_allowed_rewarded_states: int,
        seed: int,
        prevent_reverse: bool = False,
    ) -> float:
        """Run a single simulation and return total reward."""
        np.random.seed(seed)

        # Planning agents (POMCP, POMCP_bio): model-based, no gym env
        if model_name in self._PLANNING_AGENTS:
            kwargs = deepcopy(self.algo_config.get(model_name))
            AgentClass = self.MODEL_CLASSES[model_name]
            agent = AgentClass(**kwargs)
            total_reward = agent.run_episode(
                adj_mat=adj_mat,
                node_positions=st_positions,
                start_node=start_node,
                rewarded_nodes=rewards,
                n_actions=n_actions,
                min_rewarded_states=min_allowed_rewarded_states,
                seed=seed,
                prevent_reverse=prevent_reverse,
            )
            del agent
            gc.collect()
            return total_reward

        # RL agents: use gym env + SB3 or custom training loop
        env = self._create_env(
            adj_mat, st_positions, start_node, rewards, n_actions,
            obs_type, action_type, history_len,
            include_prev_action, include_prev_reward, min_allowed_rewarded_states,
            prevent_reverse=prevent_reverse,
        )

        ModelClass = self.MODEL_CLASSES[model_name]
        kwargs = deepcopy(self.algo_config.get(model_name))

        # Determine policy string and handle LSTM-specific kwargs
        if model_name in self._CUSTOM_AGENTS:
            # Custom agents (DRQN, RC_GVF, CountBasedExplorer): pass all kwargs directly
            policy_str = "MlpLstmPolicy"
        elif model_name in self._LSTM_AGENTS:
            # RecurrentPPO: extract LSTM kwargs into SB3 policy_kwargs
            policy_str = "MlpLstmPolicy"
            lstm_hidden = kwargs.pop('lstm_hidden_size', 64)
            n_lstm = kwargs.pop('n_lstm_layers', 1)
            kwargs['policy_kwargs'] = {
                'lstm_hidden_size': lstm_hidden,
                'n_lstm_layers': n_lstm,
                'shared_lstm': True,
                'enable_critic_lstm': False,
            }
        else:
            policy_str = "MlpPolicy"

        # Fix total_timesteps: SB3 uses this to compute exploration schedules
        # (e.g., DQN epsilon decay). With 1e12, epsilon never decays from 1.0.
        # Set to n_actions so epsilon decays over the actual episode length.
        total_ts = max(n_actions, 1000)

        # Cap n_steps for on-policy agents: if n_steps > episode length, the
        # policy never updates during the episode (default TRPO n_steps=2048)
        if 'n_steps' in kwargs and kwargs['n_steps'] > n_actions:
            kwargs['n_steps'] = max(n_actions // 2, 8)

        model = ModelClass(policy_str, env, verbose=0, device="cpu", seed=seed, **kwargs)
        model.learn(total_timesteps=total_ts, callback=[StopAfterFirstEpisode()])

        # Get base env for observation log
        base_env = env.env if hasattr(env, 'env') else env
        total_reward = sum(obs.get('reward', 0) for obs in base_env.observation_log)

        # Explicit cleanup to free memory
        try:
            env.close()
        except Exception:
            pass
        del model
        del env
        gc.collect()

        return total_reward

    def _make_model(self, model_name, env, n_actions, seed, algo_config=None):
        """Create a fresh model for the given env. Returns (model, total_ts).

        Factored out of run_single() so run_single_pretrained() can reuse it.
        """
        cfg = algo_config or self.algo_config
        ModelClass = self.MODEL_CLASSES[model_name]
        kwargs = deepcopy(cfg.get(model_name))

        if model_name in self._CUSTOM_AGENTS:
            policy_str = "MlpLstmPolicy"
        elif model_name in self._LSTM_AGENTS:
            policy_str = "MlpLstmPolicy"
            lstm_hidden = kwargs.pop('lstm_hidden_size', 64)
            n_lstm = kwargs.pop('n_lstm_layers', 1)
            kwargs['policy_kwargs'] = {
                'lstm_hidden_size': lstm_hidden,
                'n_lstm_layers': n_lstm,
                'shared_lstm': True,
                'enable_critic_lstm': False,
            }
        else:
            policy_str = "MlpPolicy"

        total_ts = max(n_actions, 1000)

        if 'n_steps' in kwargs and kwargs['n_steps'] > n_actions:
            kwargs['n_steps'] = max(n_actions // 2, 8)

        model = ModelClass(policy_str, env, verbose=0, device="cpu",
                           seed=seed, **kwargs)
        return model, total_ts

    def run_single_pretrained(
        self,
        model_name: str,
        target_row: pd.Series,
        pretrain_rows: pd.DataFrame,
        obs_type: str,
        action_type: str,
        history_len: int,
        include_prev_action: bool,
        include_prev_reward: bool,
        seed: int,
        algo_config: Optional[AlgoConfig] = None,
        prevent_reverse: bool = False,
    ) -> Tuple[float, int]:
        """Train on prior sessions, then evaluate on target session.

        Args:
            model_name: Agent name ('DQN', 'DRQN_seq', 'DRQN_rand').
            target_row: yoking_df row for the target session.
            pretrain_rows: DataFrame of prior sessions (from get_pretrain_sessions).
            obs_type: Observation type (e.g. 'ego').
            action_type: Action type (e.g. 'ego').
            history_len: History concat length (0 for LSTM agents).
            include_prev_action: Include previous action in obs.
            include_prev_reward: Include previous reward in obs.
            seed: Random seed.
            algo_config: Optional override for algo hyperparameters.

        Returns:
            (total_reward, n_pretrain_sessions) tuple.
        """
        np.random.seed(seed)
        cfg = algo_config or self.algo_config
        is_custom = model_name in self._CUSTOM_AGENTS
        model = None
        prev_env = None
        n_pretrain = len(pretrain_rows)

        # --- Pretraining phase ---
        for i, (_, ptr_row) in enumerate(pretrain_rows.iterrows()):
            adj_mat, st_positions = self._load_maze(
                ptr_row['adj_file'], ptr_row['st_pos_file'])
            ptr_exp = ptr_row['exp_moment']
            rewards, reset_val = self._get_rewarded_states(
                ptr_exp, len(st_positions))
            n_actions = int(ptr_row['n_states_visited']) - 1
            start_node = int(ptr_row['start_state'])

            env = self._create_env(
                adj_mat, st_positions, start_node, rewards, n_actions,
                obs_type, action_type, history_len,
                include_prev_action, include_prev_reward, reset_val,
                prevent_reverse=prevent_reverse)

            if model is None:
                model, total_ts = self._make_model(
                    model_name, env, n_actions, seed, cfg)
            else:
                total_ts = max(n_actions, 1000)
                model.set_env(env)

            if prev_env is not None:
                try:
                    prev_env.close()
                except Exception:
                    pass

            if is_custom:
                model.learn()
            else:
                model.learn(total_timesteps=total_ts,
                            callback=[StopAfterFirstEpisode()],
                            reset_num_timesteps=False)
            prev_env = env

        # --- Target session phase ---
        adj_mat, st_positions = self._load_maze(
            target_row['adj_file'], target_row['st_pos_file'])
        target_exp = target_row['exp_moment']
        rewards, reset_val = self._get_rewarded_states(
            target_exp, len(st_positions))
        n_actions = int(target_row['n_states_visited']) - 1
        start_node = int(target_row['start_state'])

        env = self._create_env(
            adj_mat, st_positions, start_node, rewards, n_actions,
            obs_type, action_type, history_len,
            include_prev_action, include_prev_reward, reset_val,
            prevent_reverse=prevent_reverse)

        if model is None:
            # No pretraining rows — create fresh model
            model, total_ts = self._make_model(
                model_name, env, n_actions, seed, cfg)
        else:
            total_ts = max(n_actions, 1000)
            model.set_env(env)

        if prev_env is not None:
            try:
                prev_env.close()
            except Exception:
                pass

        if is_custom:
            model.learn()
        else:
            model.learn(total_timesteps=total_ts,
                        callback=[StopAfterFirstEpisode()],
                        reset_num_timesteps=False)

        # Extract reward from observation log
        base_env = env.env if hasattr(env, 'env') else env
        total_reward = sum(obs.get('reward', 0) for obs in base_env.observation_log)

        # Cleanup
        try:
            env.close()
        except Exception:
            pass
        del model
        del env
        gc.collect()

        return total_reward, n_pretrain

    def run_single_target_pretrain(
        self,
        model_name: str,
        adj_mat: np.ndarray,
        st_positions: np.ndarray,
        start_node: int,
        rewards: np.ndarray,
        n_actions: int,
        obs_type: str,
        action_type: str,
        history_len: int,
        include_prev_action: bool,
        include_prev_reward: bool,
        min_allowed_rewarded_states: int,
        seed: int,
        n_pretrain_episodes: int = 10,
        algo_config: Optional[AlgoConfig] = None,
        prevent_reverse: bool = False,
    ) -> Tuple[float, int]:
        """Train on the target session for N episodes, then evaluate on the same session.

        Unlike run_single_pretrained() which trains on *different* (prior) sessions,
        this variant trains on the *same* session repeatedly. This tests whether
        multi-episode experience on the identical maze/reward/budget helps.

        The agent's weights and replay buffer persist across pretrain episodes.
        The environment is reset (same maze, start, rewards, budget) each episode.

        Args:
            model_name: Agent name ('DQN', 'DRQN_seq', 'DRQN_rand').
            adj_mat, st_positions, start_node, rewards, n_actions: Session params.
            obs_type, action_type, history_len: Observation config.
            include_prev_action, include_prev_reward: History augmentation.
            min_allowed_rewarded_states: PacMan rebait threshold.
            seed: Random seed.
            n_pretrain_episodes: Number of training episodes before evaluation.
            algo_config: Optional override for algo hyperparameters.
            prevent_reverse: Block backward actions.

        Returns:
            (total_reward_on_eval_episode, n_pretrain_episodes) tuple.
        """
        np.random.seed(seed)
        cfg = algo_config or self.algo_config
        is_custom = model_name in self._CUSTOM_AGENTS

        env = self._create_env(
            adj_mat, st_positions, start_node, rewards, n_actions,
            obs_type, action_type, history_len,
            include_prev_action, include_prev_reward,
            min_allowed_rewarded_states,
            prevent_reverse=prevent_reverse,
        )

        model, total_ts = self._make_model(
            model_name, env, n_actions, seed, cfg)

        # --- Pretrain phase: N episodes on the same session ---
        for _ep in range(n_pretrain_episodes):
            if is_custom:
                model.learn()
            else:
                model.learn(total_timesteps=total_ts,
                            callback=[StopAfterFirstEpisode()],
                            reset_num_timesteps=False)

        # --- Eval phase: one final episode ---
        # Clear observation log so we only capture the eval episode's reward
        base_env = env.env if hasattr(env, 'env') else env
        base_env.observation_log = []

        if is_custom:
            model.learn()
        else:
            model.learn(total_timesteps=total_ts,
                        callback=[StopAfterFirstEpisode()],
                        reset_num_timesteps=False)

        total_reward = sum(
            obs.get('reward', 0) for obs in base_env.observation_log)

        # Cleanup
        try:
            env.close()
        except Exception:
            pass
        del model
        del env
        gc.collect()

        return total_reward, n_pretrain_episodes

    def run(
        self,
        config: SimConfig,
        target_repeats: int = 10,
        session_filter: Optional[List[str]] = None,
        memory_config: Optional[MemoryConfig] = None,
    ) -> pd.DataFrame:
        """Run simulations for all configurations with memory failsafes."""
        mem_cfg = memory_config or MemoryConfig()
        mem_monitor = MemoryMonitor(
            limit_gb=mem_cfg.limit_gb,
            threshold_fraction=mem_cfg.threshold_fraction,
            verbose=mem_cfg.verbose,
        ) if mem_cfg.enabled else None

        results = []
        skipped_agents = set()
        sessions = session_filter or self.yoking_df['exp_moment'].unique().tolist()

        total = (len(sessions) * len(config.models) *
                 len(config.history_lengths) * len(config.observation_configs) * target_repeats)
        print(f"Running up to {total} simulations...")

        if mem_monitor:
            mem_monitor.snapshot("run_start")

        sim_count = 0
        timestamp_str = get_now_str()
        intermediate_path = os.path.join(
            self.output_dir, f"{timestamp_str}_rl_results_PARTIAL.csv")

        for exp_moment in sessions:
            yoke_row = self.yoking_df[
                self.yoking_df['exp_moment'] == exp_moment].iloc[0]
            adj_mat, st_positions = self._load_maze(
                yoke_row['adj_file'], yoke_row['st_pos_file'])
            rewards, reset_val = self._get_rewarded_states(
                exp_moment, len(st_positions))
            n_actions = int(yoke_row['n_states_visited']) - 1
            start_node = int(yoke_row['start_state'])

            for model in config.models:
                if model in skipped_agents:
                    continue

                if model not in self.MODEL_CLASSES:
                    print(f"WARNING: {model} not available "
                          f"(missing sb3-contrib?). Skipping.")
                    skipped_agents.add(model)
                    continue

                for h_len in config.history_lengths:
                    # LSTM/planning agents have their own memory; skip h_len > 0
                    if (model in self._LSTM_AGENTS
                            or model in self._PLANNING_AGENTS) and h_len > 0:
                        continue

                    for obs_cfg in config.observation_configs:
                        for rep in range(target_repeats):
                            sim_count += 1

                            # Memory check before each simulation
                            if mem_monitor and sim_count > 1:
                                status = mem_monitor.check_threshold(
                                    f"before sim {sim_count} ({model})")
                                if status == 'critical':
                                    print(
                                        f"MEMORY CRITICAL: Skipping remaining "
                                        f"sims for '{model}'. "
                                        f"Completed {sim_count - 1} so far.")
                                    skipped_agents.add(model)
                                    break

                            if model in skipped_agents:
                                break

                            # Unique seed per configuration
                            seed = (config.seed_base + rep
                                    + hash(f"{model}_{obs_cfg['name']}_{h_len}")
                                    % 100000)

                            reward = self.run_single(
                                model_name=model,
                                adj_mat=adj_mat,
                                st_positions=st_positions,
                                start_node=start_node,
                                rewards=rewards,
                                n_actions=n_actions,
                                obs_type=obs_cfg['obs_type'],
                                action_type=obs_cfg['action_type'],
                                history_len=h_len,
                                include_prev_action=obs_cfg.get(
                                    'include_prev_action', False),
                                include_prev_reward=obs_cfg.get(
                                    'include_prev_reward', False),
                                min_allowed_rewarded_states=reset_val,
                                seed=seed,
                            )

                            results.append({
                                'exp_moment': exp_moment,
                                'animal_ID': yoke_row.get('animal_ID', ''),
                                'model': model,
                                'history_len': h_len,
                                'config_name': obs_cfg['name'],
                                'repeat_idx': rep,
                                'total_reward': reward,
                                'n_actions': n_actions,
                                'timestamp': datetime.now().isoformat(),
                                'seed': seed,
                            })

                            # Periodic memory snapshot
                            if (mem_monitor
                                    and mem_cfg.snapshot_every_n_sims > 0
                                    and sim_count % mem_cfg.snapshot_every_n_sims == 0):
                                mem_monitor.snapshot(
                                    f"sim {sim_count}/{total} ({model})")

                            # Periodic intermediate save
                            if (mem_cfg.save_every_n_sims > 0
                                    and sim_count % mem_cfg.save_every_n_sims == 0
                                    and results):
                                pd.DataFrame(results).to_csv(
                                    intermediate_path, index=False)
                                if config.verbose:
                                    print(f"  Saved intermediate "
                                          f"({len(results)} rows)")

                            if config.verbose and sim_count % 10 == 0:
                                print(f"  Completed {sim_count}/{total}")

                        if model in skipped_agents:
                            break
                    if model in skipped_agents:
                        break

        df = pd.DataFrame(results)

        # Final save
        final_path = os.path.join(
            self.output_dir, f"{timestamp_str}_rl_results.csv")
        df.to_csv(final_path, index=False)
        print(f"Saved {len(df)} results to: {final_path}")

        # Clean up intermediate file
        if os.path.exists(intermediate_path):
            os.remove(intermediate_path)

        # Memory summary
        if mem_monitor:
            mem_monitor.snapshot("run_end")
            summary = mem_monitor.get_summary()
            print(f"\n[Memory Summary] "
                  f"Peak RSS: {summary.get('process_peak_mb', 0):.0f} MB, "
                  f"Final RSS: {summary.get('final_rss_mb', 0):.0f} MB, "
                  f"Max observed: {summary.get('max_rss_mb', 0):.0f} MB")

        if skipped_agents:
            print(f"\nWARNING: Agents skipped due to memory: "
                  f"{skipped_agents}")

        return df


# =============================================================================
# Convenience Functions
# =============================================================================

def quick_validate(df: pd.DataFrame) -> Dict[str, Any]:
    """Quick validation of simulation results."""
    mode_val = df['total_reward'].mode().iloc[0] if len(df) > 0 else 0
    mode_pct = 100 * (df['total_reward'] == mode_val).sum() / len(df) if len(df) > 0 else 0

    return {
        'n_sims': len(df),
        'mean_reward': df['total_reward'].mean(),
        'max_reward': df['total_reward'].max(),
        'min_reward': df['total_reward'].min(),
        'mode_pct': mode_pct,
        'status': 'OK' if mode_pct < 10 else 'WARNING: possible ceiling effect',
    }
