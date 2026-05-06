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
from stable_baselines3.common.callbacks import BaseCallback

import gymnasium as gym
from gymnasium.spaces import Graph, MultiBinary, Discrete
from gymnasium.wrappers import TransformAction, TransformObservation

from stable_baselines3.ppo import MlpPolicy
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3 import TD3
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import ts2xy
from stable_baselines3.common.noise import NormalActionNoise
from sb3_contrib import TRPO, CrossQ, RecurrentPPO, TQC
from stable_baselines3 import A2C, DDPG, DQN, PPO, HER, HerReplayBuffer

sys.path.append('../')
print('251222 Author: assume utils_latMaz is in the same directory as the the utils_latMaz_analysis.py file. If false change the sys.path.append line accordingly.')

from utils_latMaz import (get_most_recent_file, deserialize_str, get_moment_strings, load_maze, get_now_str, get_adj_states, 
    displacement_to_compass_heading, get_direction_from_radian, allo_radian_map_dict, get_ego_direction, get_verified_st_traj, 
    move_df_col_to_leftmost, chk_type)


def parse_exp_csv_to_observations_WD(csv_data_path):
    """ Parse the experiment CSV file to one row per STATE with observations.
    Note: initial csv file is one row per ACTION with observations either lacking or in a hard to parse format."""
    data = pd.read_csv('data_in/1_experiment_csvs/' + os.path.basename(csv_data_path))
    adj_file = data.iloc[0]['adjacency_file']
    st_pos_file = data.iloc[0]['st_positions_file']
    assert data.adjacency_file.nunique() == 1, 'Author: there should be only one adjacency file per experiment'
    assert data.st_positions_file.nunique() == 1, 'Author: there should be only one state positions file per experiment'
    data.lights_on = data.lights_on.apply(chk_type)
    data.choice = data.choice.apply(chk_type)
    data['obs_allo_via_lights'] = data.lights_on.apply(convert_lights_on_to_allo_real_chars)
    adj_mat, st_pos = load_maze('data_in/mazes/' + adj_file, 'data_in/mazes/' + st_pos_file)
    data['action_rewarded'] = False
    data.loc[data.reward_time.notna(), 'action_rewarded'] = data.reward_time.notna().astype(bool)
    assert data.start_state.nunique() == 1, 'Author: there should be only one start state per experiment'
    st = data.iloc[0]['start_state']
    adj_states = get_adj_states(st, adj_mat)
    heading_real = np.pi / 2
    heading_latent = np.pi / 2
    observations_adj_nodes_allo_latent = deepcopy([displacement_to_compass_heading(st_pos[adj_node] - st_pos[st]) for adj_node in adj_states])
    observations_adj_nodes_allo_real = deepcopy([get_direction_from_radian(np.mod(allo_radian_map_dict[obs] + (heading_real - heading_latent), 2 * np.pi)) for obs in observations_adj_nodes_allo_latent])
    observations_adj_node_ego = deepcopy([get_ego_direction(st_pos[st], st_pos[adj_node], heading_latent) for adj_node in adj_states])
    obs_ego_col_list = [tuple(observations_adj_node_ego)]
    obs_allo_latent_col_list = [tuple(observations_adj_nodes_allo_latent)]
    obs_allo_real_col_list = [tuple(observations_adj_nodes_allo_real)]
    try:
        reward_duration = data.iloc[0]['reward_duration']
    except KeyError:
        print(f'Author: no reward_duration column found, setting to np.nan\n{csv_data_path}')
        reward_duration = np.nan
    start_row_df = pd.DataFrame([{'time': np.nan, 'action_idx': np.nan, 'lights_on': np.nan, 'state': st, 'rewarded_states': data.iloc[0].rewarded_states, 'choice': np.nan, 'action': np.nan, 'reward_time': np.nan, 'n_rewards': 0, 'mouse_rotation': np.nan, 'maze_rotation': np.nan, 'note': np.nan, 'adjacency_file': adj_file, 'st_positions_file': st_pos_file, 'start_state': st, 'reward_duration': reward_duration, 'action_rewarded': np.nan, 'obs_ego': obs_ego_col_list[0], 'obs_allo_real': obs_allo_real_col_list[0], 'obs_allo_latent': obs_allo_latent_col_list[0]}])
    st_prior = st
    action_ego_col_list = []
    action_allo_real_col_list = []
    action_allo_latent_col_list = []
    for _i, row in data.iterrows():
        assert row.choice[0] in row.obs_allo_via_lights, f'Author: invalid choice in obs_allo_real at row {_i}\n{row}'
        st = row['state']
        assert st != st_prior, 'Author: this st should never equal st_prior; if failure occurs, consider pass by reference issue'
        action_ego_col_list.append(row['action'])
        action_allo_real_col_list.append(row['choice'])
        action_allo_latent_col_list.append(displacement_to_compass_heading(st_pos[st] - st_pos[st_prior]))
        adj_states = get_adj_states(st, adj_mat)
        heading_latent = allo_radian_map_dict[displacement_to_compass_heading(st_pos[st] - st_pos[st_prior])]
        if _i % 2 == 1:
            heading_real = heading_latent
        else:
            heading_real = heading_latent + np.pi
        ' get observations '
        observations_adj_nodes_allo_latent = deepcopy([displacement_to_compass_heading(st_pos[adj_node] - st_pos[st]) for adj_node in adj_states])
        observations_adj_nodes_allo_real = deepcopy([get_direction_from_radian(np.mod(allo_radian_map_dict[obs] + (heading_real - heading_latent), 2 * np.pi)) for obs in observations_adj_nodes_allo_latent])
        observations_adj_node_ego = deepcopy([get_ego_direction(st_pos[st], st_pos[adj_node], heading_latent) for adj_node in adj_states])
        st_prior = st
        obs_ego_col_list.append(tuple(observations_adj_node_ego))
        obs_allo_latent_col_list.append(tuple(observations_adj_nodes_allo_latent))
        obs_allo_real_col_list.append(tuple(observations_adj_nodes_allo_real))
    data['action_ego'] = action_ego_col_list
    data['action_allo_latent'] = action_allo_latent_col_list
    data['action_allo_real'] = action_allo_real_col_list
    data['obs_ego'] = obs_ego_col_list[1:]
    data['obs_allo_real'] = obs_allo_real_col_list[1:]
    data['obs_allo_latent'] = obs_allo_latent_col_list[1:]
    exp_df_action_rows = pd.concat([start_row_df, data]).reset_index(drop=True)
    exp_df_state_rows = exp_df_action_rows[['time', 'action_idx', 'state', 'choice', 'action', 'obs_ego', 'obs_allo_real', 'obs_allo_latent', 'action_allo_real', 'action_allo_latent', 'action_ego', 'reward_time', 'n_rewards', 'action_rewarded']].copy()
    exp_df_state_rows['action_idx'] = exp_df_state_rows.action_idx.shift(-1)
    exp_df_state_rows['action'] = exp_df_state_rows.action.shift(-1)
    exp_df_state_rows['choice'] = exp_df_state_rows.choice.shift(-1)
    exp_df_state_rows['action_ego'] = exp_df_state_rows.action_ego.shift(-1)
    exp_df_state_rows['action_allo_real'] = exp_df_state_rows.action_allo_real.shift(-1)
    exp_df_state_rows['action_allo_real'] = exp_df_state_rows.action_allo_real.apply(lambda x: x[0] if type(x) == list else np.nan)
    exp_df_state_rows['action_allo_latent'] = exp_df_state_rows.action_allo_latent.shift(-1)
    exp_df_state_rows['reward_time'] = exp_df_state_rows.reward_time.shift(-1)
    exp_df_state_rows['n_rewards'] = exp_df_state_rows.n_rewards.shift(-1)
    exp_df_state_rows['action_rewarded'] = exp_df_state_rows.action_rewarded.shift(-1)
    a = [set(e) for e in exp_df_state_rows.obs_allo_real.to_list()]
    b = data.obs_allo_via_lights.to_list()
    for e_a, e_b in zip(a, b):
        assert e_a == e_b, f'Author: {e_a} != {e_b}'
    return (exp_df_state_rows, exp_df_action_rows)

def convert_lights_on_to_allo_real_chars(lights_on_list):
    avail_allo_real_chars = set()
    if lights_on_list[0]:
        avail_allo_real_chars.add('N')
    if lights_on_list[1]:
        avail_allo_real_chars.add('S')
    if lights_on_list[2]:
        avail_allo_real_chars.add('W')
    if lights_on_list[3]:
        avail_allo_real_chars.add('E')
    return avail_allo_real_chars
convert_lights_on_to_allo_real_chars([True, False, True, True])

def get_obs_action_prob_dict_MR(csv_data_path, representation, verbose=False):
    """ computes combined state-action probabilities from mouse data
    Args
        -representation [str]: either 'ego' or 'allo_real' or 'allo_latent'
        -csv_data_path [str]: path to the mouse experiment csv file
    Returns  ## add the new columns to the dataframe
        -obs_action_dict [dict]: a nested dict of {obs: {action: probability}} where obs is either 'ego' or 'allo' depending on the representation
    """

    def normalize_group(g):
        actions = g[action_str].values
        probs = g['prob'].values.astype(np.float64)
        probs = probs / probs.sum()
        return dict(zip(actions, probs))

    def is_hashable(v):
        try:
            hash(v)
            return True
        except TypeError:
            return False
    action_str = f'action_{representation}'
    obs_str = f'obs_{representation}'
    exp_df_state_rows, exp_df_action_rows = parse_exp_csv_to_observations_WD(csv_data_path)
    del exp_df_action_rows
    exp_df_state_rows['choice'] = exp_df_state_rows['choice'].apply(lambda x: chk_type(x))
    exp_df_state_rows['choice'] = exp_df_state_rows.choice.apply(lambda x: x[0] if type(x) == list else np.nan)
    assert all(exp_df_state_rows['choice'].dropna().reset_index(drop=True) == exp_df_state_rows['action_allo_real'].dropna().reset_index(drop=True)), f"Author: choice and action_allo_real should be identical but got \n{exp_df_state_rows['choice']} \nvs \n{exp_df_state_rows['action_allo_real']}"
    exp_df_state_rows[obs_str] = exp_df_state_rows[obs_str].apply(lambda x: frozenset(x) if type(x) == tuple else np.nan)
    assert not exp_df_state_rows[obs_str].isna().any(), f'Author: unexpected non-tuple values in {obs_str}'
    assert not exp_df_state_rows['choice'].iloc[:-1].isna().any(), 'Author: unexpected NaN values in choice column (excluding last row)'
    assert pd.isna(exp_df_state_rows['choice'].iloc[-1]), 'Author: expected to find a NaN for the last state cuz no action was taken'
    exp_df_state_rows = exp_df_state_rows.iloc[:-1].reset_index(drop=True)
    assert exp_df_state_rows[obs_str].map(is_hashable).all(), 'Non-hashable obs values remain'
    assert exp_df_state_rows[action_str].map(is_hashable).all(), f'Non-hashable action values remain: {action_str}; {exp_df_state_rows[action_str]}'
    joint_obs_action_counts = exp_df_state_rows.groupby([obs_str, action_str], as_index=True).size().rename('action_obs_count').reset_index()
    marg_obs_counts = exp_df_state_rows.groupby(obs_str).size().rename('obs_count').reset_index()
    obs_action_df = pd.merge(joint_obs_action_counts, marg_obs_counts, on=obs_str, how='inner')
    obs_action_df['prob'] = obs_action_df['action_obs_count'] / obs_action_df['obs_count']
    obs_action_df['num_actions_avail'] = obs_action_df[obs_str].apply(lambda x: len(x))
    obs_action_dict = obs_action_df.groupby(obs_str).apply(normalize_group, include_groups=False).to_dict()
    if verbose:
        return (obs_action_dict, {'obs_action_df': obs_action_df, 'exp_df_state_rows': exp_df_state_rows})
    else:
        return obs_action_dict

def get_latent_state_counts_dict(mouse_latent_states_visited, adj_mat):
    all_states = np.arange(len(adj_mat))
    assert type(mouse_latent_states_visited) == list, 'Author: mouse_latent_states_visited should be a list'
    return {int(s): mouse_latent_states_visited.count(s) for s in all_states}

def get_policy_param_dict(yoke_row, sim_recipe_row, adj_mat, n_actions_shifted_bw_window):
    policy_category, policy_func, representation, avoid_reversal, seq_length = (sim_recipe_row.policy_category, sim_recipe_row.policy_func, sim_recipe_row.representation, sim_recipe_row.avoid_reversal, sim_recipe_row.seq_length)
    if policy_category == 'random':
        prms_dict = {'avoid_reversal': avoid_reversal}
    elif policy_category == 'action_biased':
        action_prob_dict_allo_real = chk_type(yoke_row.prob_dict_allo_real)[seq_length, n_actions_shifted_bw_window]
        action_prob_dict_allo_latent = chk_type(yoke_row.prob_dict_allo_latent)[seq_length, n_actions_shifted_bw_window]
        action_prob_dict_ego = chk_type(yoke_row.prob_dict_ego)[seq_length, n_actions_shifted_bw_window]
        prms_dict = {'seq_length': seq_length, 'action_prob_dict_allo_real': action_prob_dict_allo_real, 'action_prob_dict_allo_latent': action_prob_dict_allo_latent, 'action_prob_dict_ego': action_prob_dict_ego}
    elif policy_category == 'observation_action_biased':
        if representation == 'ego':
            obs_action_nested_dict = get_obs_action_prob_dict_MR(yoke_row.csv_data_path, 'ego')
        elif representation == 'allo_real':
            obs_action_nested_dict = get_obs_action_prob_dict_MR(yoke_row.csv_data_path, 'allo_real')
        elif representation == 'allo_latent':
            obs_action_nested_dict = get_obs_action_prob_dict_MR(yoke_row.csv_data_path, 'allo_latent')
        prms_dict = {'obs_action_nested_dict': obs_action_nested_dict}
    elif policy_category == 'latent_state_biased':
        assert type(yoke_row.states_visited) == list, 'Author: yoke_row.states_visited should be a list'
        prms_dict = {'node_counts_dict': get_latent_state_counts_dict(yoke_row.states_visited, adj_mat)}
    else:
        raise NotImplementedError(f'Author: unexpected policy category: {policy_category}')
    prms_dict.update({'representation': representation})
    policy_dict_ = {'policy_category': policy_category, 'func': policy_func, 'prms': prms_dict, 'func_name': policy_func.__name__}
    return policy_dict_

def convert_numpy_types(obj):
    """Recursively convert numpy types in a nested dict or list to native Python types."""
    if isinstance(obj, dict):
        return {convert_numpy_types(k): convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(_i) for _i in obj]
    elif isinstance(obj, tuple):
        return tuple((convert_numpy_types(_i) for _i in obj))
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, (frozenset, set)):
        return obj
    else:
        return obj

def convert_frozenset_keys(obj):
    """Recursively convert frozenset keys in dicts to sorted tuples. Leaves values untouched."""
    if isinstance(obj, dict):
        new_dict = {}
        for k, v in obj.items():
            new_key = tuple(sorted(k)) if isinstance(k, frozenset) else k
            new_dict[new_key] = convert_frozenset_keys(v)
        return new_dict
    elif isinstance(obj, list):
        return [convert_frozenset_keys(_i) for _i in obj]
    elif isinstance(obj, tuple):
        return tuple((convert_frozenset_keys(_i) for _i in obj))
    else:
        return obj
    
## overriding sb3 monitor.py to allow more customization and avoid annoying changes in future versions
__all__ = ['Monitor', 'ResultsWriter', 'get_monitor_files', 'load_results']

import csv
import json
from typing import Any, Optional, SupportsFloat, Union
import pandas
from gymnasium.core import ActType, ObsType

class Monitor_MR(gym.Wrapper[ObsType, ActType, ObsType, ActType]):
    """
    A monitor wrapper for Gym environments, it is used to know the episode reward, length, time and other data.

    :param env: The environment
    :param filename: the location to save a log file, can be None for no log
    :param allow_early_resets: allows the reset of the environment before it is done
    :param reset_keywords: extra keywords for the reset call,
        if extra parameters are needed at reset
    :param info_keywords: extra information to log, from the information return of env.step()
    :param override_existing: appends to file if ``filename`` exists, otherwise
        override existing files (default)
    """
    EXT = f'{get_now_str(hms=True)}_monitor.csv'

    def __init__(self, env: gym.Env, filename: Optional[str]=None, allow_early_resets: bool=True, reset_keywords: tuple[str, ...]=(), info_keywords: tuple[str, ...]=(), override_existing: bool=True):
        super().__init__(env=env)
        self.t_start = time()
        self.results_writer = None
        if filename is not None:
            env_id = env.spec.id if env.spec is not None else None
            self.results_writer = ResultsWriter(filename, header={'t_start': self.t_start, 'env_id': str(env_id)}, extra_keys=reset_keywords + info_keywords, override_existing=override_existing)  # EXT = "monitor.csv"
        self.reset_keywords = reset_keywords
        self.info_keywords = info_keywords
        self.allow_early_resets = allow_early_resets
        self.rewards: list[float] = []
        self.needs_reset = True
        self.episode_returns: list[float] = []
        self.episode_lengths: list[int] = []
        self.episode_times: list[float] = []
        self.total_steps = 0
        self.current_reset_info: dict[str, Any] = {}

    def reset(self, **kwargs) -> tuple[ObsType, dict[str, Any]]:
        """
        Calls the Gym environment reset. Can only be called if the environment is over, or if allow_early_resets is True

        :param kwargs: Extra keywords saved for the next episode. only if defined by reset_keywords
        :return: the first observation of the environment
        """
        if not self.allow_early_resets and (not self.needs_reset):
            raise RuntimeError('Tried to reset an environment before done. If you want to allow early resets, wrap your env with Monitor(env, path, allow_early_resets=True)')
        self.rewards = []
        self.needs_reset = False
        for key in self.reset_keywords:
            value = kwargs.get(key)
            if value is None:
                raise ValueError(f'Expected you to pass keyword argument {key} into reset')
            self.current_reset_info[key] = value
        return self.env.reset(**kwargs)

    def step(self, action: ActType) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        """
        Step the environment with the given action
  # extra info about the current episode, that was passed in during reset()
        :param action: the action
        :return: observation, reward, terminated, truncated, information
        """
        if self.needs_reset:
            raise RuntimeError('Tried to step environment that needs reset')
        observation, reward, terminated, truncated, info = self.env.step(action)
        self.rewards.append(float(reward))
        if terminated or truncated:
            self.needs_reset = True
            ep_rew = sum(self.rewards)
            ep_len = len(self.rewards)
            # ep_info = {'r': round(ep_rew, 6), 'l': ep_len, 't': round(time.time() - self.t_start, 6)}
            ep_info = {'r': round(ep_rew, 6), 'l': ep_len, 't': round(time() - self.t_start, 6)}
            for key in self.info_keywords:
                ep_info[key] = info[key]
            self.episode_returns.append(ep_rew)
            self.episode_lengths.append(ep_len)
            # self.episode_times.append(time.time() - self.t_start)
            self.episode_times.append(time() - self.t_start)
            ep_info.update(self.current_reset_info)
            if self.results_writer:
                self.results_writer.write_row(ep_info)
            info['episode'] = ep_info
        self.total_steps = self.total_steps + 1
        return (observation, reward, terminated, truncated, info)

    def close(self) -> None:
        """
        Closes the environment
        """
        super().close()
        if self.results_writer is not None:
            self.results_writer.close()

    def get_total_steps(self) -> int:
        """
        Returns the total number of timesteps
        :return:
        """
        return self.total_steps

    def get_episode_rewards(self) -> list[float]:
        """
        Returns the rewards of all the episodes
        :return:
        """
        return self.episode_returns

    def get_episode_lengths(self) -> list[int]:
        """
        Returns the number of timesteps of all the episodes
        :return:
        """
        return self.episode_lengths

    def get_episode_times(self) -> list[float]:
        """
        Returns the runtime in seconds of all the episodes
        :return:
        """
        return self.episode_times

class LoadMonitorResultsError(Exception):
    """
    Raised when loading the monitor log fails.
    """
    pass

class ResultsWriter:
    """
    A result writer that saves the data from the `Monitor` class
    :param filename: the location to save a log file. When it does not end in
        the string ``"monitor.csv"``, this suffix will be appended to it
    :param header: the header dictionary object of the saved csv
    :param extra_keys: the extra information to log, typically is composed of
        ``reset_keywords`` and ``info_keywords``
    :param override_existing: appends to file if ``filename`` exists, otherwise
        override existing files (default)
    """

    def __init__(self, filename: str='', header: Optional[dict[str, Union[float, str]]]=None, extra_keys: tuple[str, ...]=(), override_existing: bool=True):
        if header is None:
            header = {}
        if not filename.endswith(Monitor_MR.EXT):
            if os.path.isdir(filename):
                filename = os.path.join(filename, Monitor_MR.EXT)
            else:
                filename = filename + '.' + Monitor_MR.EXT
        filename = os.path.realpath(filename)
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        mode = 'w' if override_existing else 'a'
        self.file_handler = open(filename, f'{mode}t', newline='\n')
        self.logger = csv.DictWriter(self.file_handler, fieldnames=('r', 'l', 't', *extra_keys))
        if override_existing:
            self.file_handler.write(f'#{json.dumps(header)}\n')
            self.logger.writeheader()
        self.file_handler.flush()

    def write_row(self, epinfo: dict[str, float]) -> None:
        """
        Write row of monitor data to csv log file.
        :param epinfo: the information on episodic return, length, and time
        """
        if self.logger:
            self.logger.writerow(epinfo)
            self.file_handler.flush()

    def close(self) -> None:
        """
        Close the file handler
        """
        self.file_handler.close()

def get_monitor_files(path: str) -> list[str]:
    """
    get all the monitor files in the given path

    :param path: the logging folder
    :return: the log files
    """
    return glob(os.path.join(path, '*' + Monitor_MR.EXT))

def load_results(path: str) -> pandas.DataFrame:
    """  # Create (if any) missing filename directories
    Load all Monitor logs from a given directory path matching ``*monitor.csv``
    :param path: the directory path containing the log file(s)  # Append mode when not overriding existing file
    :return: the logged data
    """  # Prevent newline issue on Windows, see GH issue #692
    monitor_files = get_monitor_files(path)
    if len(monitor_files) == 0:
        raise LoadMonitorResultsError(f'No monitor files of the form *{Monitor_MR.EXT} found in {path}')
    data_frames, headers = ([], [])
    for file_name in monitor_files:
        with open(file_name) as file_handler:
            first_line = file_handler.readline()
            assert first_line[0] == '#'
            header = json.loads(first_line[1:])
            data_frame = pandas.read_csv(file_handler, index_col=None)
            headers.append(header)
            data_frame['t'] = data_frame['t'] + header['t_start']
        data_frames.append(data_frame)
    data_frame = pandas.concat(data_frames)
    data_frame.sort_values('t', inplace=True)
    data_frame.reset_index(inplace=True)
    data_frame['t'] = data_frame['t'] - min((header['t_start'] for header in headers))
    return data_frame



class SaveOnBestTrainingRewardCallback_MR(BaseCallback):
    """
    Callback for saving a model (the check is done every ``check_freq`` steps)
    based on the training reward (in practice, we recommend using ``EvalCallback``).

    :param check_freq: (int)
    :param log_dir: (str) Path to the folder where the model will be saved.
      It must contains the file created by the ``Monitor`` wrapper.
    :param verbose: (int)
    """

    def __init__(self, check_freq: int, log_dir: str, verbose=1):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_dir = os.path.join(log_dir, f"{get_now_str(hms=False)}_best_models")
        self.save_name=f"{get_now_str(hms=True)}_best_model"  # file stem (SB3 adds .zip if not present)
        self.best_mean_reward = -np.inf

    def _init_callback(self) -> None:
        # Create folder if needed
        # if self.save_dir is not None:
        os.makedirs(self.save_dir, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            # Retrieve training reward
            x, y = ts2xy(load_results(self.log_dir), "timesteps")
            if len(x) > 0:
                # Mean training reward over the last 100 episodes
                mean_reward = np.mean(y[-100:])
                if self.verbose > 0:
                    print(f"Num timesteps: {self.num_timesteps}")
                    print(
                        f"Best mean reward: {self.best_mean_reward:.2f} - Last mean reward per episode: {mean_reward:.2f}"
                    )
                # New best model, you could save the agent here
                if mean_reward > self.best_mean_reward:
                    self.best_mean_reward = mean_reward
                    result_output_fullpath=os.path.join(self.save_dir, self.save_name)
                    # Example for saving best model
                    if self.verbose > 0:
                        print(f"Saving new best model to {result_output_fullpath}.zip")
                    self.model.save(result_output_fullpath)
        return True
    
class StopAfterFirstEpisode(BaseCallback):
    """ 251009 Author: newer and critical to ensure only a single episode."""
    def __init__(self, verbose=0):
        super().__init__(verbose)  ## GPT: self.locals is a dict injected by the learner; it contains 'dones' in VecEnv mode
        self._episode_done = False

    def _on_step(self) -> bool:
        dones = self.locals.get('dones')
        if dones is None:
            done = self.locals.get('done') or self.locals.get('terminal')  # self.locals["dones"] exists in vectorized and non-vectorized settings
            if done:
                self._episode_done = True
        elif bool(np.any(dones)):  # (Rare) non-vectorized path: some algos also expose single 'done' or 'terminal'
            self._episode_done = True  # non-vectorized: check env’s last info tuple
        return not self._episode_done  # VecEnv path (typical): stop if ANY env finished an episode this step  # Returning False tells SB3 to stop training now
