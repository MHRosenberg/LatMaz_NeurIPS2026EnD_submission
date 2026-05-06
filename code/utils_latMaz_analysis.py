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
from collections import Counter
from datetime import timedelta

sys.path.append('./')
print('251221 Author: assume utils_latMaz is in the same directory as the the utils_latMaz_analysis.py file. If false change the sys.path.append line accordingly.')

from utils_latMaz import (get_most_recent_file, deserialize_str, get_moment_strings, load_maze, get_now_str, get_adj_states, 
    displacement_to_compass_heading, get_direction_from_radian, allo_radian_map_dict, get_ego_direction, get_verified_st_traj, 
    move_df_col_to_leftmost, chk_type)

def parse_exp_csv_to_observations_WD(csv_data_path):
    """ Parse the experiment CSV file to one row per STATE with observations.
    Note: initial csv file is one row per ACTION with observations either lacking or in a hard to parse format. """
    data = pd.read_csv('data_in/1_experiment_csvs/' + os.path.basename(csv_data_path))
    adj_file = data.iloc[0]['adjacency_file']
    st_pos_file = data.iloc[0]['st_positions_file']
    assert data.adjacency_file.nunique() == 1, 'Author: there should be only one adjacency file per experiment'
    assert data.st_positions_file.nunique() == 1, 'Author: there should be only one state positions file per experiment'
    data.lights_on = data.lights_on.apply(chk_type)
    data.choice = data.choice.apply(chk_type)
    data['obs_allo_via_lights'] = data.lights_on.apply(convert_lights_on_to_allo_real_chars)  ## convert strings back to python datatypes
    adj_mat, st_pos = load_maze('data_in/mazes/' + adj_file, 'data_in/mazes/' + st_pos_file)
    data['action_rewarded'] = False
    data.loc[data.reward_time.notna(), 'action_rewarded'] = data.reward_time.notna().astype(bool)  ## create a nicer formatting for the available options
    assert data.start_state.nunique() == 1, 'Author: there should be only one start state per experiment'
    st = data.iloc[0]['start_state']
    adj_states = get_adj_states(st, adj_mat)
    ## initialize both latent and real heading to north
    heading_real = np.pi / 2
    heading_latent = np.pi / 2  ## 1. if the action is rewarded
    observations_adj_nodes_allo_latent = deepcopy([displacement_to_compass_heading(st_pos[adj_node] - st_pos[st]) for adj_node in adj_states])  # data['action_rewarded'] = (data['n_rewards'] - data['n_rewards'].shift()) > 0   # Author: Author3 approach neglects the first row; might have other issues
    observations_adj_nodes_allo_real = deepcopy([get_direction_from_radian(np.mod(allo_radian_map_dict[obs] + (heading_real - heading_latent), 2 * np.pi)) for obs in observations_adj_nodes_allo_latent])  # Initialize full column
    observations_adj_node_ego = deepcopy([get_ego_direction(st_pos[st], st_pos[adj_node], heading_latent) for adj_node in adj_states])
    obs_ego_col_list = [tuple(observations_adj_node_ego)]
    obs_allo_latent_col_list = [tuple(observations_adj_nodes_allo_latent)]  
    obs_allo_real_col_list = [tuple(observations_adj_nodes_allo_real)]
    try:
        reward_duration = data.iloc[0]['reward_duration']
    except KeyError:
        print(f'Author: no reward_duration column found, setting to np.nan\n{csv_data_path}')  
        reward_duration = np.nan

    ## add start state
    start_row_df = pd.DataFrame([{'time': np.nan, 'action_idx': np.nan, 'lights_on': np.nan, 'state': st, 'rewarded_states': data.iloc[0].rewarded_states, 'choice': np.nan, 'action': np.nan, 'reward_time': np.nan, 'n_rewards': 0, 'mouse_rotation': np.nan, 'maze_rotation': np.nan, 'note': np.nan, 'adjacency_file': adj_file, 'st_positions_file': st_pos_file, 'start_state': st, 'reward_duration': reward_duration, 'action_rewarded': np.nan, 'obs_ego': obs_ego_col_list[0], 'obs_allo_real': obs_allo_real_col_list[0], 'obs_allo_latent': obs_allo_latent_col_list[0]}])
    st_prior = st  ## compute initial observations 
    action_ego_col_list = []  ## allocentric latent
    action_allo_real_col_list = []  ## allocentric real    
    action_allo_latent_col_list = []  ## egocentric
    for i, row in data.iterrows():
        assert row.choice[0] in row.obs_allo_via_lights, f'Author: invalid choice in obs_allo_real at row {i}\n{row}'
        st = row['state']
        assert st != st_prior, 'Author: this st should never equal st_prior; if failure occurs, consider pass by reference issue'
        action_ego_col_list.append(row['action'])
        action_allo_real_col_list.append(row['choice'])
        action_allo_latent_col_list.append(displacement_to_compass_heading(st_pos[st] - st_pos[st_prior]))
        adj_states = get_adj_states(st, adj_mat)
        heading_latent = allo_radian_map_dict[displacement_to_compass_heading(st_pos[st] - st_pos[st_prior])]
        if i % 2 == 1:
            heading_real = heading_latent
        else:
            heading_real = heading_latent + np.pi
        ' get observations '
        observations_adj_nodes_allo_latent = deepcopy([displacement_to_compass_heading(st_pos[adj_node] - st_pos[st]) for adj_node in adj_states])
        observations_adj_nodes_allo_real = deepcopy([get_direction_from_radian(np.mod(allo_radian_map_dict[obs] + (heading_real - heading_latent), 2 * np.pi)) for obs in observations_adj_nodes_allo_latent])  ## 2. what are the available options?
        observations_adj_node_ego = deepcopy([get_ego_direction(st_pos[st], st_pos[adj_node], heading_latent) for adj_node in adj_states])
        st_prior = st  # heading_real += np.pi # 250826 Author: incorrect version for some reason...
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
    exp_df_state_rows = exp_df_action_rows[['time', 'action_idx', 'state', 'choice', 'action', 'obs_ego', 'obs_allo_real', 'obs_allo_latent', 'action_allo_real', 'action_allo_latent', 'action_ego', 'reward_time', 'n_rewards', 'action_rewarded']].copy()  ## todo after NeurIPS: rename these upstream in experiment code
    exp_df_state_rows['action_idx'] = exp_df_state_rows.action_idx.shift(-1)
    exp_df_state_rows['action'] = exp_df_state_rows.action.shift(-1)
    exp_df_state_rows['choice'] = exp_df_state_rows.choice.shift(-1)
    exp_df_state_rows['action_ego'] = exp_df_state_rows.action_ego.shift(-1)
    exp_df_state_rows['action_allo_real'] = exp_df_state_rows.action_allo_real.shift(-1)
    exp_df_state_rows['action_allo_real'] = exp_df_state_rows.action_allo_real.apply(lambda x: x[0] if type(x) == list else np.nan)
    exp_df_state_rows['action_allo_latent'] = exp_df_state_rows.action_allo_latent.shift(-1)  # defined by the current sensor poked relative to prior sensor poked only
    exp_df_state_rows['reward_time'] = exp_df_state_rows.reward_time.shift(-1)
    exp_df_state_rows['n_rewards'] = exp_df_state_rows.n_rewards.shift(-1)  # heading_real += np.pi # 250826 Author: PRIOR but incorrect version for some reason... 
    exp_df_state_rows['action_rewarded'] = exp_df_state_rows.action_rewarded.shift(-1)  ## 250826 Author: new version that appears correct
    a = [set(e) for e in exp_df_state_rows.obs_allo_real.to_list()]
    b = data.obs_allo_via_lights.to_list()
    for e_a, e_b in zip(a, b):
        assert e_a == e_b, f'Author: {e_a} != {e_b}'
    return (exp_df_state_rows, exp_df_action_rows)
    # print('st; heading_latent; heading_real:', st, get_direction_from_radian(np.mod(heading_latent, 2*np.pi)), get_direction_from_radian(np.mod(heading_real, 2*np.pi)))

def convert_lights_on_to_allo_real_chars(lights_on_list):  # print('st; heading_latent:', st, get_direction_from_radian(np.mod(heading_latent, 2*np.pi)))
    avail_allo_real_chars = set()  # print('st; heading_real:', st, get_direction_from_radian(np.mod(heading_real, 2*np.pi)))
    if lights_on_list[0]:
        avail_allo_real_chars.add('N')
    if lights_on_list[1]:  ## allocentric latent
        avail_allo_real_chars.add('S')
    if lights_on_list[2]:  ## allocentric real
        avail_allo_real_chars.add('W')
    if lights_on_list[3]:  ## egocentric
        avail_allo_real_chars.add('E')
    return avail_allo_real_chars
# convert_lights_on_to_allo_real_chars([True, False, True, True])  ## prep for the next iteration

def get_obs_action_prob_dict_MR(csv_data_path, representation, verbose=False):  # heading_real += np.pi # 250826 Author: also incorrect version for some reason..
    """ computes combined state-action probabilities from mouse data  ## append to the results
    Args
        -representation [str]: either 'ego' or 'allo_real' or 'allo_latent'
        -csv_data_path [str]: path to the mouse experiment csv file
    Returns
        -obs_action_dict [dict]: a nested dict of {obs: {action: probability}} where obs is either 'ego' or 'allo' depending on the representation  ## add the new columns to the dataframe
    """

    def normalize_group(g):
        actions = g[action_str].values  ## 250827 Author: note the [1:] to skip the first entry which corresponds to the first row; we explicitly indexed into [0] when we defined the first row (start_row_df) above 
        probs = g['prob'].values.astype(np.float64)
        probs = probs / probs.sum()
        return dict(zip(actions, probs))

    def is_hashable(v):  ### state in row is the state AFTER the action; 250827 Author: not currently using this version
        try:
            hash(v)
            return True  ## state in row is the state BEFORE the action (via shifting the action-related columns up by one row)
        except TypeError:
            return False
    action_str = f'action_{representation}'
    obs_str = f'obs_{representation}'
    exp_df_state_rows, exp_df_action_rows = parse_exp_csv_to_observations_WD(csv_data_path)
    del exp_df_action_rows  ## legacy nomencalture prior to 250827 unclear terminology; leaving in briefly
    exp_df_state_rows['choice'] = exp_df_state_rows['choice'].apply(lambda x: chk_type(x))
    exp_df_state_rows['choice'] = exp_df_state_rows.choice.apply(lambda x: x[0] if type(x) == list else np.nan)
    assert all(exp_df_state_rows['choice'].dropna().reset_index(drop=True) == exp_df_state_rows['action_allo_real'].dropna().reset_index(drop=True)), f"Author: choice and action_allo_real should be identical but got \n{exp_df_state_rows['choice']} \nvs \n{exp_df_state_rows['action_allo_real']}"  ## new 250827 clarifications
    exp_df_state_rows[obs_str] = exp_df_state_rows[obs_str].apply(lambda x: frozenset(x) if type(x) == tuple else np.nan)
    assert not exp_df_state_rows[obs_str].isna().any(), f'Author: unexpected non-tuple values in {obs_str}'
    assert not exp_df_state_rows['choice'].iloc[:-1].isna().any(), 'Author: unexpected NaN values in choice column (excluding last row)'  ## Author: convert choice to string if it's a list; NOTE: the original experiment code saves a list with the pin IDs as well; we need the nan option for the final row which has no action
    assert pd.isna(exp_df_state_rows['choice'].iloc[-1]), 'Author: expected to find a NaN for the last state cuz no action was taken'
    exp_df_state_rows = exp_df_state_rows.iloc[:-1].reset_index(drop=True)
    assert exp_df_state_rows[obs_str].map(is_hashable).all(), 'Non-hashable obs values remain'  ## other action-related columns 
    assert exp_df_state_rows[action_str].map(is_hashable).all(), f'Non-hashable action values remain: {action_str}; {exp_df_state_rows[action_str]}'
    joint_obs_action_counts = exp_df_state_rows.groupby([obs_str, action_str], as_index=True).size().rename('action_obs_count').reset_index()
    marg_obs_counts = exp_df_state_rows.groupby(obs_str).size().rename('obs_count').reset_index()
    obs_action_df = pd.merge(joint_obs_action_counts, marg_obs_counts, on=obs_str, how='inner')
    obs_action_df['prob'] = obs_action_df['action_obs_count'] / obs_action_df['obs_count']  ## sanity check that rotation issue was actually fixed
    obs_action_df['num_actions_avail'] = obs_action_df[obs_str].apply(lambda x: len(x))
    obs_action_dict = obs_action_df.groupby(obs_str).apply(normalize_group, include_groups=False).to_dict()
    if verbose:
        return (obs_action_dict, {'obs_action_df': obs_action_df, 'exp_df_state_rows': exp_df_state_rows})
    else:
        return obs_action_dict  ## alternate sanity check (untested)
    # sa=exp_df_state_rows.obs_allo_real.map(frozenset)
def get_latent_state_counts_dict(mouse_latent_states_visited, adj_mat):  # sb=data.obs_allo_via_lights.map(frozenset)
    all_states = np.arange(len(adj_mat))
    assert type(mouse_latent_states_visited) == list, 'Author: mouse_latent_states_visited should be a list'  # mismatch=(sa!=sb)
    return {int(s): mouse_latent_states_visited.count(s) for s in all_states}  # if mismatch.any():
    #     i=mismatch.idxmax()
def get_policy_param_dict(yoke_row, sim_recipe_row, adj_mat, n_actions_shifted_bw_window):  #     raise AssertionError(f"row {i}: {sa[i]} != {sb[i]}")
    policy_category, policy_func, representation, avoid_reversal, seq_length = (sim_recipe_row.policy_category, sim_recipe_row.policy_func, sim_recipe_row.representation, sim_recipe_row.avoid_reversal, sim_recipe_row.seq_length)
    if policy_category == 'random':
        prms_dict = {'avoid_reversal': avoid_reversal}
    elif policy_category == 'action_biased':
        action_prob_dict_allo_real = chk_type(yoke_row.prob_dict_allo_real)[seq_length, n_actions_shifted_bw_window]
        action_prob_dict_allo_latent = chk_type(yoke_row.prob_dict_allo_latent)[seq_length, n_actions_shifted_bw_window]  # initialize to empty set
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
## get ego action probability given state 
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
        return [convert_numpy_types(i) for i in obj]
    elif isinstance(obj, tuple):
        return tuple((convert_numpy_types(i) for i in obj))
    elif isinstance(obj, np.floating):
        return float(obj)  # action_str = 'action' if representation == 'ego' else 'choice' # see raw experiment csv file for the column names, e.g # legacy approach prior to 250827
    elif isinstance(obj, np.integer):  # new 250827 approach to differentiate allo_real vs allo_latent
        return int(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)  # Note just using the the state rows here
    elif isinstance(obj, (frozenset, set)):  # free up memory and explicitly indicate we won't be using it for now (250827 Author)
        return obj  # Author: convert choice to list if it's a string due to serialization to csv
    else:  # Author: convert choice to string if it's a list; NOTE: the original experiment code saves a list with the pin IDs as well
        return obj  # exp_df_state_rows['action_allo_real'] = exp_df_state_rows.action_allo_real.apply(lambda x: x[0] if type(x) == list else np.nan) # Author: convert choice to string if it's a list; NOTE: the original experiment code saves a list with the pin IDs as well

def convert_frozenset_keys(obj):  ## checking legacy column name (choice) data matches my new column name (action_allo_real) data
    """Recursively convert frozenset keys in dicts to sorted tuples. Leaves values untouched."""
    if isinstance(obj, dict):
        new_dict = {}
        for k, v in obj.items():
            new_key = tuple(sorted(k)) if isinstance(k, frozenset) else k  ## verify that no NaNs remain in the observations
            new_dict[new_key] = convert_frozenset_keys(v)
        return new_dict  # assert not exp_df_action_rows[obs_str].isna().any(), f"Author: unexpected non-tuple values in {obs_str}"
    elif isinstance(obj, list):  # assert not exp_df_state_rows['choice'].isna().any(), f"Author: unexpected NaN values in choice column"
        return [convert_frozenset_keys(i) for i in obj]
    elif isinstance(obj, tuple):  ## verify last row's action is NaN then discard it from consideration
        return tuple((convert_frozenset_keys(i) for i in obj))
    else:
# nested_dict_allo_real, debug_info_allo_real = get_obs_action_prob_dict_MR(TEST_CSV_PATH, 'allo_real', verbose=True)
# nested_dict_allo_latent, debug_info_allo_latent = get_obs_action_prob_dict_MR(TEST_CSV_PATH, 'allo_latent', verbose=True)
# nested_dict_ego, debug_info_ego = get_obs_action_prob_dict_MR(TEST_CSV_PATH, 'ego', verbose=True)
## strong bias of right. then forward
# print(nested_dict)
# ## further separate by if previous same action given the same state is rewarded 
# exp_df['prev_action_given_state'] = exp_df.groupby(['opts_ego'])['action'].shift()
# exp_df['prev_reward_given_state'] = exp_df.groupby(['opts_ego'])['action_rewarded'].shift()
# exp_df['action_switched'] = exp_df['prev_action_given_state'] != exp_df['action']
# ## no evidence of win-stay lose shift (both state-dependent and globally)
# exp_df.groupby(['opts_ego', 'prev_reward_given_state'], as_index=False)['action_switched'].mean()
        return obj  ## count the occurrences of each (obs, action) pair and each obs  # n obs-action pairs / n obs for that representation  ## generate probability dict based on action_summary  ## change var names for convenience  ## pass policy category specific parameters to the policy function prms dict  ## 250827 Author: why should seq_length be np.nan ever? --> replacing condition w/ assert  # action_prob_dict_allo_real = chk_type(yoke_row.prob_dict_allo)[(seq_length, n_actions_shifted_bw_window)] if seq_length is not np.nan else np.nan  # assert seq_length in yoked_sequence_lengths, f"Author: unexpected seq_length: {seq_length}"  ## 250826 Author: adding support for both of these  ## define the policy for the agent for this simulation  # Only convert frozenset keys if needed (e.g., to str) — depends on your use case


def check_n_states_match_per_exp(per_exp_result_df):
    per_exp_result_df['n_states_mouse'] = per_exp_result_df['states_visited_mouse'].apply(lambda x: len(chk_type(x))) 
    per_exp_result_df['n_states_agent'] = per_exp_result_df['states_visited_agent'].apply(lambda x: len(chk_type(x)))
    per_exp_result_df['n_states_equal'] = per_exp_result_df['n_states_mouse'] == per_exp_result_df['n_states_agent']
    assert len(per_exp_result_df.query("n_states_equal == False")) == 0, "Author: n_states_mouse and n_states_agent should match for all rows"
    for exp_moment in per_exp_result_df.exp_moment.unique():
        exp_moment_df = per_exp_result_df.query(f"exp_moment == '{exp_moment}'")
        assert exp_moment_df.n_states_mouse.nunique() == 1, f"Author: n_states_mouse should be constant for exp_moment {exp_moment}"
        assert exp_moment_df.n_states_agent.nunique() == 1, f"Author: n_states_agent should be constant for exp_moment {exp_moment}"
        assert all(exp_moment_df.n_states_mouse == exp_moment_df.n_states_agent), f"Author: n_states_mouse and n_states_agent should match for exp_moment {exp_moment}"
        print(f"Author: n_states_mouse and n_states_agent match for exp_moment {exp_moment}: {exp_moment_df.n_states_mouse.unique()[0]} states")
    print('Author: all checks passed for n_states_mouse and n_states_agent matching')

def check_sim_rewarded_inds_vs_n_states_total(per_exp_result_df):
    for row_idx, row in per_exp_result_df.iterrows():
        assert row.rewarded_action_inds_agent[-1] <= row.n_states_visited, f"Author: row {row_idx}: last rewarded agent action index ({row.rewarded_action_inds_agent[-1]}) should not exceed number of states visited ({row.n_states_visited})"
        assert row.rewarded_action_inds_mouse[-1] <= row.n_states_visited, f"Author: row {row_idx}: last rewarded mouse action index ({row.rewarded_action_inds_mouse[-1]}) should not exceed number of states visited ({row.n_states_visited})"

def extract_valid_moment(path):
    moment = get_moment_strings(path, str_separator='_')
    # moment = get_moment_strings(path)
    if isinstance(moment, str) and len(moment) == 13:
        return moment
    moment = get_moment_strings(path, str_separator='-')
    # moment = get_moment_strings(path)
    assert len(moment) == 13, f"Author: moment string is still not 13 characters long: {moment}"
    return moment

def reformat_fixed_policy_sim_df(sim_df, check_all_sim_results_homogenous=True):
    """ add additional columns to fixed policy sim df for easier analysis
    check_all_sim_results_homogenous: if True, checks that the number of simulations types etc is the same across all of the loaded data
    """
    print(f'{get_now_str(hms=True)}: converting policy dicts from strings to dicts')
    sim_df.policy_dict = sim_df.policy_dict.apply(chk_type) # convert string to dict

    ## convert from absolute to relative path
    print(f'{get_now_str(hms=True)}: converting absolute to relative paths')
    # def chop_before_data_in(x):
    #     count = x.count('/data_in')
    #     assert count == 1, f"Expected exactly one '/data_in', found {count} in path: {x}"
    #     return './data_in' + x.split('/data_in', 1)[1]
    # sim_df.csv_data_path = sim_df.csv_data_path.apply(chop_before_data_in)
    counts = sim_df.csv_data_path.str.count('/data_in')
    bad = counts != 1
    assert not bad.any(), f"Expected exactly one '/data_in' in each path, bad rows: {sim_df.csv_data_path[bad].tolist()}"
    sim_df['csv_data_path'] = ('./data_in' + sim_df.csv_data_path.str.split('/data_in', n=1).str[1])
    sim_df['csv_filename'] = sim_df.csv_data_path.apply(lambda x: os.path.basename(x)) # keep only the file name, not the full path

    ## unpack the policy_dict into separate columns
    print(f'{get_now_str(hms=True)}: unpacking policy_dict into separate columns')
    # sim_df['policy_category'] = sim_df.policy_dict.apply(lambda x: x['policy_category'])
    # sim_df['avoid_reversal'] = sim_df.policy_dict.apply(lambda x: x['prms']['avoid_reversal'] if 'avoid_reversal' in x['prms'] else False)
    # sim_df['sequence_length'] = sim_df.policy_dict.apply(lambda x: x['prms']['seq_length'] if 'seq_length' in x['prms'] else 1)
    # sim_df['representation'] = sim_df.policy_dict.apply(lambda x: x['prms']['representation'])
    def _extract_policy_fields(d):
        try: 
            prms = d['prms']
            avoid_reversal = prms.get('avoid_reversal', False)
            sequence_length = prms.get('seq_length', 1)

            return d['policy_category'], prms['representation'], avoid_reversal, sequence_length
        except TypeError as e:
            print(f"Error extracting policy fields: {e} for dict: {d}")

            return None, None, None, None
    policy_tuples = sim_df.policy_dict.apply(_extract_policy_fields)
    (sim_df['policy_category'], sim_df['representation'], sim_df['avoid_reversal'], sim_df['sequence_length']) = zip(*policy_tuples)

    ## 260107 Author: new GPT assert to try to catch downstream headaches early
    bad=sim_df["policy_category"].isna()|sim_df["representation"].isna()
    assert not bad.any(), ("260107 new GPT: failed to extract policy fields for some rows (policy_category/representation is NA). "
                           "Example policy_dict values:\n"
                           f"{sim_df.loc[bad,'policy_dict'].head(10).tolist()}")

    ## create a complete simulation description string column
    print(f'{get_now_str(hms=True)}: creating full_sim_desc column')
    # sim_df['full_sim_desc'] = sim_df.apply(lambda row: f"{row.policy_cate gory};{row.representation};avoidReversal-{row.avoid_reversal};seqLength-{row.sequence_length}", axis=1)
    sim_df['full_sim_desc'] = (
        sim_df['policy_category'].astype(str) + ';' + sim_df['representation'].astype(str) + ';avoidReversal-' 
        + sim_df['avoid_reversal'].astype(str) + ';seqLength-' + sim_df['sequence_length'].astype(str))

    ## convert results back into python variables
    print(f'{get_now_str(hms=True)}: converting actions back to lists from strings')
    # sim_df.states_visited = sim_df.states_visited.apply(chk_type) # convert string to list
    # sim_df.actions_taken_allo_latent = sim_df.actions_taken_allo_latent.apply(chk_type) #
    # sim_df.actions_taken_allo_real = sim_df.actions_taken_allo_real.apply(chk_type) #
    # sim_df.actions_taken_ego = sim_df.actions_taken_ego.apply(chk_type) # convert string to list
    sim_df.states_visited = sim_df.states_visited.map(chk_type) # convert string to list
    sim_df.actions_taken_allo_latent = sim_df.actions_taken_allo_latent.map(chk_type) #
    sim_df.actions_taken_allo_real = sim_df.actions_taken_allo_real.map(chk_type) #
    sim_df.actions_taken_ego = sim_df.actions_taken_ego.map(chk_type) # convert string to list


    ## extract the moment string from the csv_data_path
    print(f'{get_now_str(hms=True)}: extracting moment strings from paths')
    moment_strs = []
    for path in sim_df.csv_data_path.to_list():
        # moment_str = get_moment_strings(path, str_separator='_') # legacy underscore format
        moment_str = get_moment_strings(path) # legacy underscore format
        if type(moment_str) == str and len(moment_str) == 6: 
            # moment_str = get_moment_strings(path, str_separator='-') # overwrite w/ new dash format extraction
            moment_str = get_moment_strings(path) # overwrite w/ new dash format extraction
        moment_strs.append(moment_str)
    sim_df['moment'] = moment_strs

    ## extract the animal ID from the csv_data_path
    print(f'{get_now_str(hms=True)}: extracting animal IDs from paths')
    sim_df['animal_ID'] = sim_df.csv_data_path.apply(lambda x: x.split('_')[-2])
    
    ## 251228 Author: new trying to align to RL as far upstream as convenient
    sim_df['policy_class'] = 'fixed'
    sim_df['obs_type'] = sim_df['representation'].astype(str)
    sim_df['action_type'] = sim_df['representation'].astype(str)
    sim_df['start_state'] = sim_df.apply(lambda row: row.states_visited[0], axis=1)
    
    # sim_df['agent_model'] = sim_df['policy_category'].astype(str) # 260107 Author: prior to 
    ## 260107 Author: new GPT early standardization (might break other things...)
    sim_df["agent_model"]=sim_df["policy_category"].astype("string").str.replace("_"," ").str.strip()
    bad=sim_df["agent_model"].isna()|sim_df["agent_model"].isin(["<NA>","nan","None",""])
    assert not bad.any(), ("Author: fixed-policy agent_model must be non-missing after reformat. "
                           f"Example full_sim_desc: {sim_df.loc[bad,'full_sim_desc'].head(10).tolist()}")
    
    ## 251228 Author: columns to add to align to RL sims (remaining unmatched ... at least here)
    # {'adj_file', , 'result_creation_moment', 'actions_taken_allo_latent', 'total_reward', 'obs_log_path', 'st_pos_file', 'n_valid_actions_per_episode', 'states_visited'}

    ## run some checks
    if check_all_sim_results_homogenous:
        print(f'{get_now_str(hms=True)}: running checks on simulation data')
        ## all simulation types have the same count
        assert sim_df.full_sim_desc.value_counts().nunique() == 1, f"Author: unequal number of simulation types for sim_df: {sim_df}"
        ## all csv files have same number of sims
        assert sim_df.csv_filename.value_counts().nunique() == 1, "Author: the number of sims per csv file is not consistent, check the data"
        ## all policy categories have the same count
        assert sim_df.policy_category.value_counts().nunique() == 1, "Author: unequal number of simulation types!"

    # sim_df.to_csv(f"./data_out/{get_now_str()}_{output_name_str}.csv", index=False)
    return sim_df

def extract_rewarded_states_from_usr_params(data_to_match, exp_user_param_paths=None):

    if isinstance(data_to_match, pd.DataFrame):
        # print('251227 Author: asssuming input is a df with a csv_data_path column')
        csv_path_list = sorted(list(data_to_match.csv_data_path.unique()))
    elif isinstance(data_to_match, str):
        # print('251227 Author: asssuming input is a string from a csv_data_path column')
        csv_path_list = [data_to_match]

    if not exp_user_param_paths: 
        exp_user_param_paths = sorted(glob(f"./data_in/0_raw_exp_dirs_from_RPi/**/*usr_params*.csv", recursive=True))

    rewarded_states_rows = []
    for csv_data_path in tqdm(csv_path_list):
        csv_data_path = os.path.normpath(csv_data_path)

        # Step 1: Match to user params df
        # moment_strs = get_moment_strings(csv_data_path, str_separator='_') # legacy underscore format
        moment_strs = get_moment_strings(csv_data_path, str_separator='-') # legacy underscore format

        # if type(moment_strs) == str and len(moment_strs) == 6: 
        #     # moment_strs = get_moment_strings(csv_data_path, str_separator='-') # overwrite w/ new dash format extraction
        #     moment_strs = get_moment_strings(csv_data_path) # overwrite w/ new dash format extraction
        print(csv_data_path); print(moment_strs)

        ## 251213 Author: relaxed the selection to nearest minute
        matching_usr_param_paths = [p for p in exp_user_param_paths if moment_strs in p or moment_strs.replace('_', '-') in p] # attempt primary match
        # matching_usr_param_paths = [p for p in exp_user_param_paths if moment_strs[:-2] in p ] # attempt primary match

        if len(matching_usr_param_paths) == 0: # retry if no matches found by adding one second to the date string to catch cases where the two files were created at different seconds
            try:
                dt_fmt = "%y%m%d-%H%M%S"
                dt_obj = datetime.strptime(moment_strs, dt_fmt)
                dt_obj += timedelta(seconds=1)
                moment_strs_retry = dt_obj.strftime(dt_fmt)
                matching_usr_param_paths = [p for p in exp_user_param_paths if moment_strs_retry in p or moment_strs_retry.replace('_', '-') in p]
                if len(matching_usr_param_paths) > 0: print(f"Author: usr param path found by adding 1 second to data date string:{moment_strs} → {moment_strs_retry}")
            except Exception as e: print(f"Retry parse failed for {moment_strs}: {e}")

        if len(matching_usr_param_paths) == 0: 
            # accept failure
            matching_usr_param_paths = 'none matched'
            rewarded_states_rows.append({'csv_data_path': csv_data_path, 'usr_param_path': matching_usr_param_paths, 'rewarded_states': 'tbd'})
        else:
            assert len(matching_usr_param_paths) == 1, f"Expected exactly one match for {csv_data_path}, found: {matching_usr_param_paths}"
            assert len(matching_usr_param_paths) == 1, f"Expected exactly one match for {csv_data_path}, found: {matching_usr_param_paths}"
            # Step 2: Verify both are from the same parent directory
            usr_param_path = matching_usr_param_paths[0]

            if not os.path.exists(usr_param_path):
                print(f"File not found: {usr_param_path} --> attempting to swap out the parent path")
                usr_param_path = usr_param_path.replace('<DATA_ROOT>/', '<DATA_ROOT>/')

            # print(glob(f"{os.path.dirname(usr_param_path)}/*{moment_strs}*data.csv"))
            assert len(glob(f"{os.path.dirname(usr_param_path)}/*{moment_strs}*data.csv")) == 1, f"Expected exactly one match for {usr_param_path}, found: {glob(f'{os.path.dirname(usr_param_path)}/*{moment_strs}*data.csv')}"

            # Step 3: Verify rewarded states match expected pattern
            param_df = pd.read_csv(usr_param_path)
            if 'INITIAL_REWARDED_STATES' in param_df.param.values: rwd_states = chk_type(param_df.loc[param_df.param == 'INITIAL_REWARDED_STATES', 'setting'].item())
            elif 'REWARDED_STATES' in param_df.param.values: rwd_states = chk_type(param_df.loc[param_df.param == 'REWARDED_STATES', 'setting'].item())
            else: raise ValueError(f"No rewarded state key found in param file: {usr_param_path}")

            reset_when_n_rwds_remaining = chk_type(param_df.loc[param_df.param == 'RESET_WHEN_N_RWDS_REMAINING', 'setting'].item()) if 'RESET_WHEN_N_RWDS_REMAINING' in param_df.param.values else np.nan # 251224 Author: important more specific parsing of reward rebaiting

            rewarded_states_rows.append({'csv_data_path': csv_data_path, 'usr_param_path': usr_param_path, 'rewarded_states': rwd_states, 'reset_when_n_rwds_remaining': reset_when_n_rwds_remaining})

        print('\n')

    rewarded_states_df = pd.DataFrame(rewarded_states_rows)
    os.makedirs('./data_out', exist_ok=True)
    rewarded_states_df.to_csv(f"./data_out/c{get_now_str()}_rewarded_states_df.csv", index=False)
    return rewarded_states_df


def sanity_check_latent_vs_real_allo_actions(sim_df_reloaded, run_mode='fast'):
    """ sanity check that the latent vs real allocentric actions match expected pattern: on even steps: should match vs on odd steps: should be flipped (N<->S, E<->W) """
    if run_mode == 'fast':
        FLIP = {'N': 'S', 'S': 'N', 'E': 'W', 'W': 'E'}

        # cols = ["states_visited", "actions_taken_allo_latent", "actions_taken_allo_real"]
        cols = ["actions_taken_allo_latent", "actions_taken_allo_real"]
        it = sim_df_reloaded[cols].itertuples(index=True, name=None)  # (idx, states, latent, real)

        # for row_idx, states, latent, real in tqdm(it, total=len(sim_df_reloaded)):
        for row_idx, latent, real in tqdm(it, total=len(sim_df_reloaded)):
            ## basic shape checks (optional but usually helpful for debugging)
            n_lat = len(latent)
            n_real = len(real)
            if n_lat != n_real:
                raise AssertionError(f"Length mismatch at row {row_idx}: latent={n_lat}, real={n_real}")

            ## even steps: must match
            for j in range(0, n_lat, 2):
                a_lat = latent[j]
                a_real = real[j]
                if a_lat != a_real:
                    # s = states[j] if j < len(states) else None
                    raise AssertionError(
                        # f"Mismatch at row {row_idx}, step {j} (even), state={s}: "
                        f"latent={a_lat} real={a_real}")

            ## odd steps: must be flipped
            for j in range(1, n_lat, 2):
                a_lat = latent[j]
                exp = FLIP.get(a_lat)
                if exp is None:
                    # s = states[j] if j < len(states) else None
                    raise AssertionError(
                        # f"Unknown latent action at row {row_idx}, step {j}, state={s}: {a_lat!r}"
                        f"Unknown latent action at row {row_idx}, step {j}, : {a_lat!r}"
                        )
                a_real = real[j]
                if a_real != exp:
                    # s = states[j] if j < len(states) else None
                    raise AssertionError(
                        # f"Mismatch at row {row_idx}, step {j} (odd), state={s}: "
                        f"latent={a_lat} expected_real={exp} got_real={a_real}")
    
    elif run_mode == 'naive':
        for i, row in tqdm(sim_df_reloaded.iterrows()):
            # for j, (s, aa_latent, aa_real, ae) in enumerate(zip(row.states_visited, row.actions_taken_allo_latent, row.actions_taken_allo_real, row.actions_taken_ego)):
            for j, (aa_latent, aa_real) in enumerate(zip(row.actions_taken_allo_latent, row.actions_taken_allo_real)):
                # print(f"state: {s}, allo action (latent): {aa_latent}, allo action (real): {aa_real}, ego action: {ae}")

                if j % 2 == 0:
                    ## should match
                    assert aa_latent == aa_real, f"Author: mismatch between latent and real allo actions at step {i}: {aa_latent} vs {aa_real}"
                else:
                    ## should be flipped
                    if aa_latent == 'N':
                        assert aa_real == 'S', f"Author: mismatch between latent and real allo actions at step {i}: {aa_latent} vs {aa_real}"
                    elif aa_latent == 'S':
                        assert aa_real == 'N', f"Author: mismatch between latent and real allo actions at step {i}: {aa_latent} vs {aa_real}"
                    elif aa_latent == 'E':
                        assert aa_real == 'W', f"Author: mismatch between latent and real allo actions at step {i}: {aa_latent} vs {aa_real}"
                    elif aa_latent == 'W':
                        assert aa_real == 'E', f"Author: mismatch between latent and real allo actions at step {i}: {aa_latent} vs {aa_real}"
    print("Author: sanity check passed: latent vs real allocentric actions match expected pattern")


# 251221 Author: stable for fixed policy sims; had issues w/ RL sims (fixed now?)
# def get_n_rewards_obtained_by_simulations(sim_df, rewarded_states_df, min_rewarded_nodes_allowed_by_env=2):

#     sim_df['csv_data_path'] = sim_df.csv_data_path.map(os.path.normpath)
#     rewarded_states_df['csv_data_path'] = rewarded_states_df.csv_data_path.map(os.path.normpath)

#     ## stronger: prove the grouped rows cover the full DF
#     _group_sizes = sim_df.groupby(['animal_ID','csv_data_path'], sort=False, dropna=True).size()
#     assert int(_group_sizes.sum()) == int(len(sim_df)), f"Author: groupby(dropna=True) dropped rows: grouped_sum={int(_group_sizes.sum())} vs n_rows={int(len(sim_df))}"

#     per_exp_result_rows = []
#     for (a_ID, csv_data_path), animal_x_session_df in sim_df.groupby(['animal_ID', 'csv_data_path'], sort=False, dropna=True):
#         print(f"processing sims for animal: {a_ID}; session: {os.path.basename(csv_data_path)}")

#         ## load the source mouse data
#         assert os.path.exists(csv_data_path), f"Author: csv_data_path {csv_data_path} does not exist, check the path or the data"
#         mouse_exp_df = pd.read_csv(csv_data_path)
#         states_visited_mouse = get_verified_st_traj(mouse_exp_df)
#         rewarded_action_inds_mouse = mouse_exp_df[mouse_exp_df.reward_time.notna()].action_idx.to_list()

#         for sim_idx, row in sim_df.query(f"animal_ID == '{a_ID}' and csv_data_path == '{csv_data_path}'").iterrows(): # REVERSE IT BACK HERE... FIX THIS PATH ISSUE!
#             sdf = row.full_sim_desc
#             repeat_group_idx = row.repeat_group_idx
#             assert type(sdf) == str, f"Author: full_sim_desc should be a string, but got {type(sdf)}"
#             assert type(repeat_group_idx) == int, f"Author: repeat_group_idx should be an integer, but got {type(repeat_group_idx)}"

#             # states_visited_agent = list(chk_type(row.states_visited)) # Author: this might have numpy ints instead of vanilla python ints
#             states_visited_agent = [int(x) for x in chk_type(row.states_visited)] # Author: if you want pure int instead of numpy int
#             assert type(states_visited_agent) == list, f"Author (early test): states_visited should be a list, but got {type(states_visited_agent)}: {states_visited_agent}"
#             assert isinstance(states_visited_agent[0], (int, np.integer)), f'Author (early test): states_visited_agent element was {type(states_visited_agent[0])} instead of int'

#             """ determine rewards here """
#             # rewarded_states_current = deepcopy(REWARDED_STATES_INITIAL) # legacy hard-coded rewarded states
#             n_csv_paths = len(rewarded_states_df.query(f"csv_data_path == '{csv_data_path}'"))
#             assert len (rewarded_states_df.query(f"csv_data_path == '{csv_data_path}'")) == 1, \
#                 f"Author: expected exactly one rewarded states entry for {csv_data_path}, but got {n_csv_paths}"
#             rewarded_states_initial = rewarded_states_df.query(f"csv_data_path == '{csv_data_path}'").rewarded_states.item()
#             rewarded_states_current = deepcopy(chk_type(rewarded_states_initial))
#             if type(rewarded_states_current) != list and rewarded_states_current == 'tbd':
#                 print(f"skipping detailed rwd determination for {csv_data_path} due to tbd rewarded states or other issue for rewarded_states_current: {rewarded_states_current}")
#                 n_rewards_obtained = 'tbd'
#                 rewarded_action_inds_agent = 'tbd'
#                 print(f"251208 Author: skipping processing of ambiguous reward config data for now... ie not adding it to the per_exp_result_df")

#                 assert a_ID not in ['a031', 'a033'], f"Author: unexpected tbd rewarded states for animal_ID {a_ID} for : {csv_data_path}" 
#                 continue
#             else:
#                 assert type(rewarded_states_current) == list, f"Author: rewarded_states_current should be a list, but got {type(rewarded_states_current)}"
#                 # n_rewards_obtained = 0
#                 rewarded_action_inds_agent = []
#                 for i, s in enumerate(states_visited_agent[1:]):
#                     if s in rewarded_states_current:
#                         # n_rewards_obtained += 1
#                         rewarded_action_inds_agent.append(i)
#                         rewarded_states_current.remove(s)
#                     if len(rewarded_states_current) == min_rewarded_nodes_allowed_by_env: 
#                         rewarded_states_current = deepcopy(rewarded_states_initial)

#             ## truncate the states_visited lists to the same length; this handles an edge case where n-actions-mouse-took is not evenly divisible by n-steps-per-sequence for the biased agent
#             initial_n_states_visited_mouse = len(states_visited_mouse)
#             initial_n_states_visited_agent = len(states_visited_agent)
#             assert type(states_visited_mouse) == list and type(states_visited_agent) == list, \
#                 f"Author: states_visited_mouse and states_visited_agent should be lists, but got {type(states_visited_mouse)} and {type(states_visited_agent)}"
#             while len(states_visited_agent) > len(states_visited_mouse): 
#                 # print('truncating agent states_visited')
#                 states_visited_agent.pop()
#             while len(states_visited_agent) < len(states_visited_mouse): 
#                 # print('truncating mouse states_visited')
#                 states_visited_mouse.pop()
#             assert len(states_visited_agent) == len(states_visited_mouse), \
#                 f"Author: states_visited_agent and states_visited_mouse should have the same length, but got {len(states_visited_agent)} and {len(states_visited_mouse)}"

#             ## Author: generates hella prints... maybe useful for debugging...
#             # if initial_n_states_visited_agent != len(states_visited_agent) or initial_n_states_visited_mouse != len(states_visited_mouse):
#             #     print(f"initial_n_states_visited_mouse: {initial_n_states_visited_mouse}, initial_n_states_visited_agent: {initial_n_states_visited_agent}")
#             #     print(f"final_n_states_visited_mouse: {len(states_visited_mouse)}, final_n_states_visited_agent ({sdf}): {len(states_visited_agent)}\n")

#             assert type(states_visited_agent) == list, f"Author (late test): states_visited should be a list, but got {type(states_visited_agent)}: {states_visited_agent}"
#             assert isinstance(states_visited_agent[0], (int, np.integer)), f'Author (late test): states_visited_agent element was {type(states_visited_agent[0])} instead of int'
#             rewarded_action_inds_agent = [idx for idx in rewarded_action_inds_agent if idx < len(states_visited_agent)-1]  
#             rewarded_action_inds_mouse = [idx for idx in rewarded_action_inds_mouse if idx < len(states_visited_mouse)-1]

#             """ check assumptions on data here """
#             ## equal length trajectories
#             assert len(states_visited_agent) == len(states_visited_mouse), \
#                 f"Author: states_visited_agent and states_visited_mouse should have the same length, but got {len(states_visited_agent)} and {len(states_visited_mouse)}"
#             ## check that the truncation is larger than we expect
#             assert abs(initial_n_states_visited_mouse - initial_n_states_visited_agent) < 6, \
#                 f"Author: abs(initial_n_states_visited_mouse - initial_n_states_visited_agent) == {abs(initial_n_states_visited_mouse - initial_n_states_visited_agent)}"
#             ## check that the first state is the same
#             assert states_visited_agent[0] == states_visited_mouse[0], f"Author: first state should be the same, but got {states_visited_agent[0]} and {states_visited_mouse[0]}"
#             ## verify that the initial available rewards were extracted/defined properly
#             assert len(rewarded_states_initial) > 0, f"Author: rewarded_states_initial was found to be an empty list..."
#             ## check that mouse n rewards obtained matches the len of the rewarded_action_inds_mouse
#             assert len(rewarded_action_inds_mouse) == int(mouse_exp_df.n_rewards.iloc[-1]), \
#                 f"Author: len(rewarded_action_inds_mouse) should match mouse_exp_df.n_rewards, but got {len(rewarded_action_inds_mouse)} and {int(mouse_exp_df.n_rewards.iloc[-1])}"
#             assert rewarded_action_inds_agent[-1] <= len(states_visited_agent)-1, \
#                 f"Author: row {sim_idx}: last rewarded agent action index ({rewarded_action_inds_agent[-1]}) should not exceed number of states visited -1 ({len(states_visited_agent)-1})"
#             assert rewarded_action_inds_mouse[-1] <= len(states_visited_agent)-1, \
#                 f"Author: row {sim_idx}: last rewarded mouse action index ({rewarded_action_inds_mouse[-1]}) should not exceed number of states visited -1 ({len(states_visited_agent)-1})"

#             if 'moment' not in row.index:
#                 exp_moment = row.exp_moment
#             else: 
#                 exp_moment = row.moment
#             assert pd.notna(exp_moment), "exp_moment is NaN after fallback"

#             ## store the results
#             per_exp_result_rows.append({'exp_moment': exp_moment,
#                                         'repeat_group_idx': repeat_group_idx,
#                                         'initial_rewards': rewarded_states_initial,
#                                         'n_states_visited': len(states_visited_agent),
#                                         'n_mouse_states_dropped_to_align': initial_n_states_visited_mouse - len(states_visited_mouse),
#                                         'n_agent_states_dropped_to_align': initial_n_states_visited_agent - len(states_visited_agent),

#                                         'full_sim_desc': row.full_sim_desc, 
#                                         'n_rewards_obtained_agent': len(rewarded_action_inds_agent),
#                                         'rewarded_action_inds_agent': rewarded_action_inds_agent,
#                                         'states_visited_agent': states_visited_agent,

#                                         'animal_ID': a_ID,
#                                         'n_rewards_obtained_mouse': len(rewarded_action_inds_mouse),    
#                                         'rewarded_action_inds_mouse': rewarded_action_inds_mouse,
#                                         'states_visited_mouse': states_visited_mouse})

#     per_exp_result_df = pd.DataFrame(per_exp_result_rows)
#     assert len(per_exp_result_df) > 0, f'Author: no results found in the per_exp_result_df: {per_exp_result_df}'
#     per_exp_result_df = per_exp_result_df.sort_values(['animal_ID', 'full_sim_desc', 'repeat_group_idx'])

#     # per_exp_result_df['mouse_2_agent_reward_ratio'] = per_exp_result_df.n_rewards_obtained_mouse / per_exp_result_df.n_rewards_obtained_agent # division by zero issues. hmmmmmmm
#     # per_exp_result_df.to_csv(f"./data_out/c{get_now_str()}_per_exp_result_df-{sim_type_str}.csv", index=False)
#     per_exp_result_df.to_csv(f"./data_out/c{get_now_str()}_per_exp_result_df.csv", index=False)
#     return per_exp_result_df

## 251222 Author: new GPT approach, attempting to speed up the processing by avoiding repeated .query() calls and redundant path normalization
def get_n_rewards_obtained_by_simulations_fast(sim_df, rewarded_states_df, *, verbose=True, save_csv=True):
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

            ## 251229 Author: columns in RL results not these
            ##{'start_state', 'action_type', 'n_valid_actions_per_episode', 'actions_taken_allo_real', 'adj_file', 'actions_taken_ego', 'total_reward', 'agent_model', 'states_visited', 'st_pos_file', 'actions_taken_allo_latent', 'result_creation_moment', 'obs_log_path', 'obs_type'}

            per_exp_result_rows.append({
                    "exp_moment": exp_moment,
                    "repeat_group_idx": int(repeat_group_idx),
                    "initial_rewards": init_list,
                    "n_states_visited": min_len,
                    "n_mouse_states_dropped_to_align": n_mouse_dropped,
                    "n_agent_states_dropped_to_align": n_agent_dropped,
                    "policy_class":  row[pos["policy_class"]],
                    "full_sim_desc": sdf,
                    "n_rewards_obtained_agent": len(rewarded_action_inds_agent),
                    "rewarded_action_inds_agent": rewarded_action_inds_agent,
                    "states_visited_agent": states_visited_agent,
                    "animal_ID": a_ID,
                    "n_rewards_obtained_mouse": len(rewarded_action_inds_mouse),
                    "rewarded_action_inds_mouse": rewarded_action_inds_mouse,
                    "states_visited_mouse": states_visited_mouse,
                    
                    ## 251229 adding more columns, but these might not be necessary. The above def are. 
                    'csv_data_path': csv_data_path,
                    })

    per_exp_result_df = pd.DataFrame(per_exp_result_rows)
    if len(per_exp_result_df) == 0:
        raise AssertionError("Author: no results found in per_exp_result_df")

    per_exp_result_df = per_exp_result_df.sort_values(["animal_ID", "full_sim_desc", "repeat_group_idx"])

    if save_csv:
        per_exp_result_df.to_csv(f"./data_out/c{get_now_str()}_per_exp_result_df.csv", index=False)

    return per_exp_result_df
