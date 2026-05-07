import os

import pandas as pd
# from google.colab import drive
from glob import glob
import re
import collections
from datetime import datetime
import numpy as np
from copy import deepcopy
import imageio
from matplotlib import pyplot as plt
from PIL import Image
import plotly.graph_objects as go
# import plotly.io as pio
import ast
try: import RPi.GPIO as GPIO
except Exception as e: print(e); print('RPi.GPIO not imported, which is probably ok if you are not running on a Raspberry Pi...')
from time import sleep
import os
import math
from time import time
import time
import shutil
import hashlib
from collections import defaultdict
import heapq

##  251231 deserialize entire df
def deserialize_df_MR(df):
    """ convert serialized strings back into python datatypes via ast.safe_eval under the hood """
    for col in df.columns:
        try:
            df[col] = df[col].apply(safe_eval)
        except Exception as e:
            print(f"unable to deserialize col {col} of your df due to: {e}")
    return df

from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import floyd_warshall
""" graph/maze relationship functions """
def get_dist_matrix(adj_mat): return floyd_warshall(csgraph=csr_matrix(adj_mat), directed=False, return_predecessors=True)[0] # first output is the distance matrix; the second is predecessors which I don't understand...

def dijkstra(adj_matrix, start_node, end_node):
    start_node = int(start_node)
    end_node = int(end_node)
    n = len(adj_matrix)
    distances = [float('inf')] * n
    distances[start_node] = 0
    pq = [(0, start_node)]

    while pq:
        cur_distance, cur_node = heapq.heappop(pq)

        # Skip if distance in the priority queue is greater than the known distance
        if cur_distance > distances[cur_node]:
            continue

        # Stop if the end node is reached
        if cur_node == end_node:
            return cur_distance

        for neighbor in range(n):
            weight = adj_matrix[cur_node][neighbor]
            if weight == 0:  # Skip if there is no edge
                continue

            new_distance = cur_distance + weight
            if new_distance < distances[neighbor]:
                distances[neighbor] = new_distance
                heapq.heappush(pq, (new_distance, neighbor))

    return float('inf')  # Return infinity if the end node is not reachable

def move_df_col_to_leftmost(df, col_name): 
    print('move_df_col_to_leftmost() is not an in-place operation currently; assign the result back to the input df')
    return df[[col_name] + [col for col in df.columns if col != col_name]]

def remove_consecutive_duplicates(original_list, return_inds_kept=False):
    new_list = [original_list[0]]  # Initialize the new list with the first element
    inds_kept = [0]
    for i in range(1, len(original_list)):
        if original_list[i] != original_list[i-1]:
            new_list.append(original_list[i])
            inds_kept.append(i)
    if return_inds_kept: return new_list, inds_kept
    else: return new_list

def zero_pad_numbers(start, end=None, padded_str_length=3): # 250123: minor mod to return the contents of the list instead of the list itself, if len is 1
    if end is None: end = start
    if isinstance(start, np.ndarray): start = list(start) # convert numpy to list
    if isinstance(start, list): result = [str(s).zfill(padded_str_length) for s in start]
    else: result = [str(n).zfill(padded_str_length) for n in range(start, end + 1)]
    if len(result) == 1: return result[0]
    else: return result

# def parse_exp_csv_to_observations_WD(csv_data_path):
#     """ Parse the experiment CSV file to one row per STATE with observations.
#     Note: initial csv file is one row per ACTION with observations either lacking or in a hard to parse format.
#     """

#     data = pd.read_csv('data_in/1_experiment_csvs/' + os.path.basename(csv_data_path))
#     adj_file = data.iloc[0]['adjacency_file']
#     st_pos_file = data.iloc[0]['st_positions_file']
#     assert data.adjacency_file.nunique() == 1, 'there should be only one adjacency file per experiment'
#     assert data.st_positions_file.nunique() == 1, 'there should be only one state positions file per experiment'
    
#     adj_mat, st_pos = load_maze('data_in/mazes/' + adj_file, 'data_in/mazes/' + st_pos_file)
    
#     ## 1. if the action is rewarded
#     # data['action_rewarded'] = (data['n_rewards'] - data['n_rewards'].shift()) > 0   # Author3 approach neglects the first row; might have other issues
#     data['action_rewarded'] = False  # Initialize full column
#     data.loc[data.reward_time.notna(), 'action_rewarded'] = data.reward_time.notna().astype(bool)

#     ## add start state
#     assert data.start_state.nunique() == 1, 'there should be only one start state per experiment'
#     st = data.iloc[0]['start_state']
#     adj_states = get_adj_states(st, adj_mat)
    
#     ## initialize both latent and real heading to north
#     heading_real = np.pi/2 
#     heading_latent = np.pi/2
#     ## compute initial observations
#     observations_adj_nodes_allo_latent = deepcopy([displacement_to_compass_heading(st_pos[adj_node] - st_pos[st]) for adj_node in adj_states]) ## allocentric latent
#     observations_adj_nodes_allo_real = deepcopy([get_direction_from_radian(np.mod(allo_radian_map_dict[obs] + (heading_real - heading_latent), 2*np.pi)) for obs in observations_adj_nodes_allo_latent]) ## allocentric real    
#     observations_adj_node_ego = deepcopy([get_ego_direction(st_pos[st], st_pos[adj_node], heading_latent) for adj_node in adj_states]) ## egocentric

#     obs_ego = [tuple(observations_adj_node_ego)]
#     obs_allo_real = [tuple(observations_adj_nodes_allo_real)]
#     obs_allo_latent = [tuple(observations_adj_nodes_allo_latent)]

#     try: reward_duration = data.iloc[0]['reward_duration']
#     except KeyError: 
#         print(f"no reward_duration column found, setting to np.nan\n{csv_data_path}")
#         reward_duration = np.nan

#     start_row_df = pd.DataFrame([{'time': np.nan, 'action_idx': np.nan, 'lights_on': np.nan, 'state': st, 'rewarded_states': data.iloc[0].rewarded_states, 
#                  'choice': np.nan, 'action': np.nan, 'reward_time': np.nan, 'n_rewards': 0, 'mouse_rotation': np.nan, 'maze_rotation': np.nan, 
#                  'note': np.nan, 'adjacency_file': adj_file, 'st_positions_file': st_pos_file, 'start_state': st, 'reward_duration': reward_duration,
#                  'action_rewarded': np.nan, 'obs_ego': obs_ego[0], 'obs_allo_real': obs_allo_real[0], 'obs_allo_latent': obs_allo_latent[0]}])

#     ## 2. what are the available options?
#     st_prior = st
#     for i, row in data.iterrows():
#         st = row['state']
#         adj_states = get_adj_states(st, adj_mat)
        
#         heading_real += np.pi
#         heading_latent = allo_radian_map_dict[displacement_to_compass_heading(st_pos[st] - st_pos[st_prior])]
#         st_prior = st

#         ## allocentric latent
#         observations_adj_nodes_allo_latent = deepcopy([displacement_to_compass_heading(st_pos[adj_node] - st_pos[st]) for adj_node in adj_states])
#         ## allocentric real
#         observations_adj_nodes_allo_real = deepcopy([get_direction_from_radian(np.mod(allo_radian_map_dict[obs] + (heading_real - heading_latent), 2*np.pi)) for obs in observations_adj_nodes_allo_latent])    
#         ## egocentric
#         observations_adj_node_ego = deepcopy([get_ego_direction(st_pos[st], st_pos[adj_node], heading_latent) for adj_node in adj_states])

#         obs_ego.append(tuple(observations_adj_node_ego))
#         obs_allo_latent.append(tuple(observations_adj_nodes_allo_latent))
#         obs_allo_real.append(tuple(observations_adj_nodes_allo_real))

#     data['obs_ego'] = obs_ego[1:] 
#     data['obs_allo_real'] = obs_allo_real[1:]
#     data['obs_allo_latent'] = obs_allo_latent[1:]

#     ## state in row is the state AFTER the action
#     exp_df_action_rows = pd.concat([start_row_df, data]) 
    
#     ## state in row is the state BEFORE the action
#     exp_df_state_rows = exp_df_action_rows[['time', 'action_idx', 'state', 'choice', 'action', 'obs_ego', 'obs_allo_real', 'obs_allo_latent', 'reward_time', 'n_rewards', 'action_rewarded']].copy()
#     exp_df_state_rows['action_idx'] = exp_df_state_rows.action_idx.shift(-1)
#     exp_df_state_rows['action'] = exp_df_state_rows.action.shift(-1)
#     exp_df_state_rows['choice'] = exp_df_state_rows.choice.shift(-1)
#     exp_df_state_rows['reward_time'] = exp_df_state_rows.reward_time.shift(-1)
#     exp_df_state_rows['n_rewards'] = exp_df_state_rows.n_rewards.shift(-1)
#     exp_df_state_rows['action_rewarded'] = exp_df_state_rows.action_rewarded.shift(-1)

#     return exp_df_state_rows, exp_df_action_rows

"""
Maze environment definitions and transformations
"""

#### state transitions 
## important dictionaries
cart_to_orient_dict = {(0, 1): 'N', (0, -1): 'S', (-1, 0): 'W', (1, 0): 'E'} # defining the mapping between cartesian displacement and agent orientation

ego_radian_map_dict = {'F': 0.0, 'B': np.pi, 'L': np.pi/2, 'R': 3*np.pi/2} # egocentric conventions for chars to radians and back

# ego_radian_map_dict.update({v: k for k, v in ego_radian_map_dict.items()}, mapping_dict = ego_radian_map_dict) # note values are also keys --> it gives you whatever format wasn't what you passed in
ego_radian_map_dict.update({v: k for k, v in ego_radian_map_dict.items()}) # note values are also keys --> it gives you whatever format wasn't what you passed in

allo_radian_map_dict = {'N': np.pi/2, 'S': 3*np.pi/2, 'W': np.pi, 'E': 0.0} # allocentric conventions for chars to radians and back
allo_radian_map_dict.update({v: k for k,v in allo_radian_map_dict.items()})

## one hot versions (added 250128)
ego_actions_one_hot_dict = {0: 'F', 1: 'B', 2: 'L', 3: 'R'}
ego_actions_one_hot_dict.update({v: k for k,v in ego_actions_one_hot_dict.items()})
allo_actions_one_hot_dict = {0: 'N', 1: 'S', 2: 'W', 3: 'E'}
allo_actions_one_hot_dict.update({v: k for k,v in allo_actions_one_hot_dict.items()})

def get_direction_from_radian(radian, tolerance=1e-5):
    radian = radian % (2 * np.pi)  # Normalize to [0, 2π)
    if np.isclose(radian, 2 * np.pi, atol=tolerance): ## Snap to 0.0 if extremely close to 2π
        radian = 0.0 
    for key_radian, char in allo_radian_map_dict.items():
        if isinstance(key_radian, float) and np.isclose(radian, key_radian, atol=tolerance): 
            return char
    raise ValueError(f"No matching direction found for radian {radian} within tolerance {tolerance} for radian_map_dict: {allo_radian_map_dict}")

def displacement_to_compass_heading(displacement): return cart_to_orient_dict[tuple(np.sign(displacement.flatten()).tolist())]

def get_ego_direction(current_point, candidate_point, current_heading):
    '''Args:
        current_point: a (x,y) location in latent position of the current state.
        candidate_point: a (x,y) location in latent position of the new state.
        current heading: 250618: is this allo or ego???? the current heading of the agent in radians, as defined by the ego_radian_map_dict (Forwards = 0, Backwards = π, Left = π/2, Right = 3π/2).
    Returns: forward, backward, left, or right as defined by the ego_radian_map_dict.
    '''
    # print(f'current_point: {current_point}, candidate_point: {candidate_point}, angle_to_candidate: {angle_to_candidate} radians')
    # print(f'current_point: {current_point}, candidate_point: {candidate_point}, current_heading: {current_heading} radians')
    current_point, candidate_point = np.array(current_point).flatten(), np.array(candidate_point).flatten()
    angle_to_candidate = np.arctan2(candidate_point[1] - current_point[1], candidate_point[0] - current_point[0])
    assert isinstance(current_heading, (float, np.floating)) and isinstance(angle_to_candidate, (float, np.floating)), f'angle_to_candidate and current_heading must be a float-like but got {type(angle_to_candidate)} and {type(current_heading)}, respectively'
    relative_angle = angle_to_candidate - current_heading # current_heading in radians (see ego_radian_map_dict which converts between ego and radians)
    return ego_radian_map_dict[np.mod(relative_angle, 2*np.pi)]  # Normalize to [0, 2π)

def get_adj_states(state, adj_mat):
    state = int(state)
    # return np.argwhere(adj_mat[state] == 1).flatten()
    return [int(state) for state in np.argwhere(adj_mat[state] == 1).flatten()] # 250622: attempting to solve string to dict conversion error

## modified to support 1 hot encodings for RL agents
def env_transition_func(current_state, action, representation, adj_mat=None, st_positions=None, heading_real=None, heading_latent=None): # the transition function
    
    current_state = int(current_state)
    accessible_states = get_adj_states(current_state, adj_mat=adj_mat) # accessible states
    action2state_dict = None # clear previous version if it exists to avoid confusion/corruption
    if representation == 'ego': 
        assert heading_latent is not None, "Heading latent must be provided for ego"
        if type(heading_latent) == str: 
            heading_latent = allo_radian_map_dict[heading_latent]
        action2state_dict = {get_ego_direction(st_positions[current_state], st_positions[acc_st], heading_latent): acc_st for acc_st in accessible_states} # equivalent to current observations and available actions
        if type(action) != str: # convert to one hot for RL agents
            action = ego_actions_one_hot_dict[action]

    elif representation == 'allo_latent': 
        action2state_dict = {displacement_to_compass_heading(st_positions[acc_st] - st_positions[current_state]): acc_st for acc_st in accessible_states} # equivalent to current observations and available actions
        if type(action) != str: # convert to one hot for RL agents
            action = allo_actions_one_hot_dict[action]

    elif representation == 'allo_real':
        assert heading_real is not None and heading_latent is not None, "Heading real and latent must be provided for allo_real"
        observations_adj_nodes_allo_latent = deepcopy([displacement_to_compass_heading(st_positions[acc_st] - st_positions[current_state]) for acc_st in accessible_states])
        
        ## convert heading latent back to radian if it is a string
        if type(heading_latent) == str: 
            heading_latent = allo_radian_map_dict[heading_latent]
        assert type(observations_adj_nodes_allo_latent[0]) == str, f"obs_allo_latent must be a string but got {type(observations_adj_nodes_allo_latent[0])}"
        assert isinstance(heading_real, (float, np.floating)) and isinstance(heading_latent, (float, np.floating)), f'heading_real and heading_latent must be floats but got {type(heading_real)}, {type(heading_latent)}'
        obs_real2latent_dict = {get_direction_from_radian(np.mod(allo_radian_map_dict[obs_allo_latent] + (heading_real - heading_latent), 2*np.pi)): obs_allo_latent 
                             for obs_allo_latent in observations_adj_nodes_allo_latent}
        
        action2state_dict = {}
        for obs_real, obs_latent in obs_real2latent_dict.items():
            for acc_st in accessible_states:
                if displacement_to_compass_heading(st_positions[acc_st] - st_positions[current_state]) == obs_latent:
                    action2state_dict[obs_real] = acc_st

        if type(action) != str: # convert to one hot for RL agents
            action = allo_actions_one_hot_dict[action]
        
    try: 
        return action2state_dict[action]
    except KeyError as e: 
        return 'action_unavailable'

def load_maze(adj_path, st_pos_path):
    """Load the adjacency matrix + state positions for a maze.

    Layout-aware path resolution (added 260507 v12): if the given path
    doesn't resolve as-is, try the canonical released location
    ``data_released/mazes/<basename>`` and the legacy dev location
    ``data_in/mazes/<basename>``. Lets the same yoking dataframe drive
    both the dev pipeline and the released artifact without hand-edits
    to call sites (utils_latMaz_analysis.py, utils_latMaz_RL.py, the
    yoked_rl_runner default, and the figure scripts that hardcoded
    'data_in/mazes/').
    """
    def _resolve(p):
        if os.path.exists(p):
            return p
        base = os.path.basename(p)
        here = os.path.dirname(os.path.abspath(__file__))
        cur = here
        for _ in range(5):
            for cand_dir in ('data_released/mazes', 'data_in/mazes'):
                cand = os.path.join(cur, cand_dir, base)
                if os.path.exists(cand):
                    return cand
            parent = os.path.dirname(cur)
            if parent == cur:
                break
            cur = parent
        return p  # let pandas raise the original error

    st_positions = pd.read_csv(_resolve(st_pos_path)).to_numpy()[:,1:] # row index matches the state indices; each row is [x_position, y_position]
    adj_df = pd.read_csv(_resolve(adj_path), header=None)

    adj_df = adj_df.fillna(0) # map nans to zeros
    adj_df = adj_df.astype(int) # convert floats (NaNs are imported as float) to int
    adj_mat = adj_df.to_numpy() + adj_df.to_numpy().T # make the adjacency matrix symmetric and convert to numpy
    adj_df = pd.DataFrame(adj_mat) # update this to be symmetric across the diagonal like the numpy version

    # return adj_df, st_positions
    return adj_mat, st_positions

""" 240802: new standard (ToDo: port back to exp script)
src: 240802_latent_maze_exp-consistentRwdLoc-mop0full.py"""
def plot_maze(adj_mat, st_positions, title='', rwd_states=None, current_position=None, filename = "unspecified",
              output_dir= './data_out', close=True, show=False, save=True):

    ## plot the lines connecting nodes
    for i, row in enumerate(adj_mat):
        connected_states = np.argwhere(adj_mat[i]==1).flatten().tolist()
        pt_1 = st_positions[i] # [x, y]
        for j in connected_states:
            pt_2 = st_positions[j]
            plt.plot([pt_1[0], pt_2[0]], [pt_1[1], pt_2[1]], '-o', c='k', linewidth=4)

    ## label the nodes
    for node_idx, (x,y) in enumerate(st_positions):
        plt.annotate(node_idx, (x, y), textcoords="offset points", xytext=(9, 7), ha='center')

    try:
        ## color the reward
        if type(rwd_states) == list:
            for rwd_state in rwd_states:
                # print(f'rwd_state: {rwd_state}')
                # print(f"st_positions[rwd_state]: {st_positions[rwd_state]}")
                plt.plot(st_positions[rwd_state][0], st_positions[rwd_state][1], 'o', markersize=12, color='deepskyblue')
        else: plt.plot(st_positions[rwd_states][0], st_positions[rwd_states][1], 'o', markersize=12, color='deepskyblue') # this is a misnomer cuz there is only one rwd state in this condition
    except IndexError as e:
        print(e); print('no rwds found -> attempting to skip this')
        # import pdb; pdb.set_trace()

    ## plot current position of agent/mouse

    if current_position is not None: # 250605: why would there be more than one current position?
    #     for sp in current_position: plt.plot(st_positions[sp][0], st_positions[sp][1], 'o', markersize=13, color='red')
        plt.plot(st_positions[current_position][0], st_positions[current_position][1], 'o', markersize=13, color='red')

    # title = f"{title}\nrwds at: {rwd_states}"
    title = f"{title}"

    plt.title(title, fontsize=11)
    plt.axis('off')
    plt.tight_layout()
    plt.axis('equal')

    if save:
        if not os.path.exists(output_dir): os.mkdir(output_dir)
        plt.savefig(f"{output_dir}/{filename}.png", dpi=150)

    if show: plt.show()
    if close: plt.close('all')

def copy_checksum_del_dir(original_dir_path, destination_parent_folder, checksum_algor='md5'): # 241004 untested GPT
    os.makedirs(destination_parent_folder, exist_ok=True)  # Ensure destination parent folder exists
    destination_dir_path = os.path.join(destination_parent_folder, os.path.basename(original_dir_path))  # Destination directory path

    print(f'Copying directory: {original_dir_path}\nTo: {destination_dir_path}')
    os.makedirs(destination_dir_path, exist_ok=True)  # Create the destination directory if it doesn't exist

    for root, dirs, files in os.walk(original_dir_path):  # Walk through all files in the directory
        rel_path = os.path.relpath(root, original_dir_path)  # Relative path from original directory
        target_dir = os.path.join(destination_dir_path, rel_path)  # Corresponding target directory path
        os.makedirs(target_dir, exist_ok=True)  # Create target directories as needed
        for file in files:
            file_source_path = os.path.join(root, file)  # Full source file path
            file_dest_dir = os.path.join(destination_dir_path, rel_path)  # Destination directory for the file
            copy_checksum_del_file(file_source_path, file_dest_dir, checksum_algor)  # Call file-level function

    print(f'Deleting original directory: {original_dir_path}')
    shutil.rmtree(original_dir_path)  # Recursively delete the original directory after all files are processed
# Example usage:
# copy_checksum_del_dir("path/to/original/dir", "path/to/destination/folder")
        
def copy_checksum_del_file(original_file_path, destination_folder, checksum_algor='md5'): # 240903: custom safer move involving checksum verification
    os.makedirs(destination_folder, exist_ok=True) # Ensure destination folder exists
    assert destination_folder[-1] == '/', "Destination folder path must end with a '/'"
    destination_file_path = os.path.join(destination_folder, os.path.basename(original_file_path)) # Define path for the copied file
    print(f'copying: {original_file_path}\nto: {destination_file_path}') 
    shutil.copy2(original_file_path, destination_file_path) # Copy the file
    
    def calculate_checksum(file_path, algorithm="md5"): # 241004: md5 seems to be ok and faster
        if algorithm == "sha256": checksum = hashlib.sha256()
        elif algorithm == "md5": checksum = hashlib.md5()
        elif algorithm == "xxhash": checksum = xxhash.xxh64() # Much faster for non-cryptographic integrity checks
        else: raise ValueError(f"Unsupported algorithm: {algorithm}")
        with open(file_path, 'rb') as f:
            for block in iter(lambda: f.read(4096), b""): checksum.update(block)
        return checksum.hexdigest()
    
    print('verifying checksums...')
    original_checksum = calculate_checksum(original_file_path, algorithm=checksum_algor) 
    copied_checksum = calculate_checksum(destination_file_path, algorithm=checksum_algor)
    if original_checksum != copied_checksum: raise ValueError("Checksum mismatch! File copy verification failed.") # Verify checksums
    
    print('checksums match! deleting original file...')
    
    try:
        os.remove(original_file_path)  # Delete the original file
        if not os.path.exists(original_file_path): print(f"Original file {original_file_path} was deleted successfully.")
        else: print(f"Failed to delete the original file {original_file_path}.")
    except Exception as e: print(f"Error occurred while trying to delete the file: {e}"); import pdb; pdb.set_trace()
# Example usage:
# copy_and_verify_file("path/to/original/file.txt", "path/to/destination/folder")

def extract_animal_IDs(text): ## 241014: from utils_ephys.py
    pattern = r'a(\d{3})'
    print(f'searching pattern: {pattern}')
    matches = re.findall(pattern, text)
    print(f'found in {text}:\n', matches)
    if len(matches) == 1: return matches[0]
    else: return matches

# def get_moment_strings(string_input, force_output_as_list=False): # 250901 copied from utils_ephys_n_decoding.py (250606: updated to version from utils_ephys.py)
#     assert type(string_input) is str, '241013: input must be a string'
#     pattern = r'\b\d{6}(?:-\d{6}|[a-z])?\b'
#     # Find all matches in the input string
#     matches = re.split(r'[_]', os.path.basename(string_input))  # Split by underscore first
#     results = []
#     for match in matches:
#         valid_match = re.match(pattern, match)
#         if valid_match: results.append(valid_match.group(0))
#     if len(results) == 1 and not force_output_as_list: return results[0]
#     else: return results

def get_moment_strings(string_input, str_separator='-', use_filename_only=True, return_as_single_str=True): # 241015: from utils_ephys.py
    assert type(string_input) is str, '241013: input must be a string'
    if use_filename_only: string_input = os.path.basename(string_input)
    if str_separator == '-': pattern = r'\b\d{6}(?:-\d{6}|[a-z])?\b'
    elif str_separator == '_': pattern = r'\b\d{6}(?:_\d{6}|[a-z])?\b'
    # Find all matches in the input string
    matches = re.split(r'[_]', string_input)  # Split by underscore first
    results = []
    for match in matches:
        valid_match = re.match(pattern, match)
        if valid_match: results.append(valid_match.group(0))
    if len(results) == 1: return results[0]
    else: 
        if return_as_single_str: return '-'.join(results)
        else: return results
## Test cases
# print(get_datetime_strings('240903_240530-171942_LED_crossings.feather'))  # Expected: ['240903', '240530-171942']
# print(get_datetime_strings('240530-171942_LED_crossings.feather'))         # Expected: ['240530-171942']
# print(get_datetime_strings('240530_LED_crossings.feather'))                # Expected: ['240530']

## 240827: plot counts for a given experiment csv path
def plot_session_counts(csv_path, min_actions_per_exp=10, data_included=['choice', 'state', 'action', 'n_rewards'], 
                        x_label='date&time', y_label='cumulative count', show_mpl=False, save_mpl=True, show_plty=False, save_plty=True):
    print(csv_path)

    df = pd.read_csv(csv_path)
    if len(df) < min_actions_per_exp: print(f"skipping {csv_path} due to insufficient states"); return # exclude the shortest experiments
    adj_path, st_pos_path = df.adjacency_file.unique()[0], df.st_positions_file.unique()[0]
    print(f"csv_path: {csv_path}\nadj_path: {adj_path}\nst_pos_path: {st_pos_path}")
    
    # # adj_mat, st_positions = load_maze(adj_path, st_pos_path)
    # try: adj_mat, st_positions = load_maze(adj_path, st_pos_path)
    # except FileNotFoundError as e: 
    #     print(e); print('retrying w reasonable expected ./data_in/ path')
    #     adj_mat, st_positions = load_maze(f"./data_in/{adj_path}", f"./data_in/{st_pos_path}")
    
    print('states visited:', df.state.to_list())
    assert len(df.rewarded_states) == len(df.state), 'downstream analysis assumes these should match!'
    animal_ID = extract_animal_IDs(csv_path)
    print(f'animal_ID: {animal_ID}')
    if type(animal_ID) == list: animal_ID = animal_ID[0]
    exp_date_time = '_'.join(os.path.basename(csv_path).split('_')[:2]) # extracts the yymmdd_hhmmss unique experiment date-time str
    print(f'exp_date: {exp_date_time}')

    try: xs_all = [datetime.strptime(ds, '%y%m%d_%H%M%S') for ds in df.time.to_list()] # convert to datetime objects 
    except ValueError as e:
        print(e); print('retrying w alternate datetime format')
        xs_all = [datetime.strptime(ds, '%y%m%d-%H%M%S') for ds in df.time.to_list()]


    ## initialize the plots
    plt.figure(figsize=(10, 5)) # Matplotlib plot
    fig = go.Figure() # plotly plot 
    fig.update_layout(autosize=False, width=1000, height=800,paper_bgcolor='black',    # Sets the background color of the paper
        plot_bgcolor='black',     # Sets the background color of the plotting area
        font=dict(color='white'),  # Sets the font color to white for better readability on a black background)
        xaxis=dict(showgrid=False),  # Hides the grid lines on the x-axis
        yaxis=dict(showgrid=False),   # Hides the grid lines on the y-axis
        hoverlabel=dict(bgcolor='rgba(0, 0, 0, 0)'),  # Sets the background color of the hover label to transparent
        hovermode='x unified')

    for data_stream in data_included: # iterate over the data streams to plot
        
        """ select data here! """
        print(f'data_stream: {data_stream}')
        if data_stream == 'choice': 
            input_list = [chk_type(e)[0] for e in df.choice.to_list()]; dict_approach = True
            line_type_mpl = ':'
            line_type_plty = 'dot'
        elif data_stream in ['state', 'action']: 
            input_list = df[data_stream].to_list(); dict_approach = True
            if data_stream == 'state': 
                line_type_mpl = '-'
                line_type_plty = 'solid'
            elif data_stream == 'action':
                line_type_mpl = '--'
                line_type_plty = 'dash'
        if data_stream == 'n_rewards': 
            hit_inds = df[~df.reward_time.isna()].action_idx.to_list(); dict_approach = False
            line_type_mpl = '-'
            line_type_plty = 'solid'

        if dict_approach:
            e_hit_dict = get_element_hit_inds(input_list) # get the indices of each unique element's occurence in the list
            for e, hit_inds in e_hit_dict.items(): 
                xs_these = np.array(xs_all)[hit_inds] 
                ys = np.arange(len(xs_these))
                ## plot
                plt.plot(xs_these, ys, marker='o', ms=2, linestyle=line_type_mpl, label=f'{data_stream} {e}') # add line to mpl plot
                fig.add_trace(go.Scatter(x=xs_these, y=ys, mode='lines+markers', line=dict(dash=line_type_plty), name=f'{data_stream} {e}')) # add line to Plotly plot
        else:
            xs_these = np.array(xs_all)[hit_inds]
            ys = np.arange(len(xs_these))
            ## plot
            if data_stream == 'n_rewards': 
                plt.plot(xs_these, ys, color='cyan', ms=2, marker='o', linestyle=line_type_mpl, label=f'{data_stream}') # add line to mpl plot
                fig.add_trace(go.Scatter(x=xs_these, y=ys, line=dict(dash=line_type_plty, color='cyan'), mode='lines+markers', name=f'{data_stream}')) # add line to Plotly plot
            else: 
                plt.plot(xs_these, ys, marker='o', ms=2, linestyle=line_type_mpl, label=f'{data_stream}')
                fig.add_trace(go.Scatter(x=xs_these, y=ys, mode='lines+markers', line=dict(dash=line_type_plty), name=f'{data_stream}')) # add line to Plotly plot
                
    TITLE = f'{os.path.basename(csv_path)}'
    FILE_NAME = f'{exp_date_time}_{animal_ID}_session_counts'
    
    plt.xlabel(x_label) ## matplotlib formatting
    plt.ylabel(y_label)
    plt.title(TITLE)
    plt.xticks(rotation=45)
    plt.legend(fontsize=6, loc='upper left', bbox_to_anchor=(1, 1))
    # plt.tight_layout(pad=2) # squishes many plots...

    # ax = plt.gca()  # Get the current axis
    # ax.set_aspect(2/1, adjustable='datalim')  # Enforce the 2:1 aspect ratio

    # ax = plt.gca()  # Get the current axis
    # ax.set_xlim([min(xs_all), max(xs_all)])
    # ax.set_ylim([min(ys), max(ys)])

    # # Adjust aspect ratio by fixing the data ratio or the axis ratio
    # ax.set_aspect(2, adjustable='datalim')  # Adjust this to enforce a 2:1 ratio with

    fig.update_layout( ## plotly formatting
        title=TITLE,
        xaxis_title=x_label,
        yaxis_title=y_label,
        legend_title_text='', 
        legend_font=dict(size=10))
    os.makedirs('./a_visualizations/session_counts', exist_ok=True)
    if save_mpl: plt.savefig(f'./a_visualizations/session_counts/{FILE_NAME}.png', dpi=200)
    if show_mpl: plt.show()
    if save_plty: fig.write_html(f'./a_visualizations/session_counts/{FILE_NAME}.html')
    if show_plty: fig.show()
    

def get_element_hit_inds(input_list): # 240827: specifies the indices at which an list element occurs as a dictionary over the set of unique elements
    element_set = list(set(input_list)) # count these
    element_hit_inds_dict = {e: [] for e in element_set}
    for e_i in element_set:
        for j, e_j in enumerate(input_list):
            if e_j == e_i: element_hit_inds_dict[e_i].append(j)
    return element_hit_inds_dict

""" 
Code offloaded from the experiment control script: 240728_latent_maze_exp-consistentRwdLoc-mop0full.py 
"""
def setup_gpio_pins(nose_poke_pins, led_pins, motor_info_dict, door_type='5 pin stepper motors'):
    if door_type=='5 pin stepper motors': 
        all_used_GPIO_pins = sorted(nose_poke_pins + led_pins + list(np.array(motor_info_dict['step_pins']).flatten()))
        print('all GPIO pins currently in use:', all_used_GPIO_pins)
        assert len(all_used_GPIO_pins) == len(set(all_used_GPIO_pins)), 'At least one GPIO pin is being reused by the code which is sketch...'

        ## Initialize the Raspberry Pi GPIO pins to the correct values
        GPIO.setmode(GPIO.BCM)
        for pin in nose_poke_pins:
            print(f'initializing NOSE_POKE_PINS: pin {pin}')
            GPIO.setup(pin, GPIO.IN, pull_up_down=GPIO.PUD_DOWN) # original #GPIO.setup(pin, GPIO.IN, pull_up_down=GPIO.PUD_UP) # attempting opposite polarity
        print('\nMR: NOTE: the physical pull up resister warning has been investigated in some detail and deemed NO cause for concern. :) \n')

        for pin in led_pins:
            print(f'initializing LED_PINS: pin {pin}')
            GPIO.setup(pin, GPIO.OUT)
            GPIO.output(pin, GPIO.LOW)

        reward_pins = list(motor_info_dict['port_dir_pin_dict'].values())
        for pin in reward_pins:
            print(f"initializing reward pins ({motor_info_dict['port_dir_pin_dict'].keys()}): pin {pin}")
            GPIO.setup(pin, GPIO.OUT)
            GPIO.output(pin, GPIO.LOW)

        for motor_idx, motorPins in enumerate(motor_info_dict['step_pins']):
            for pin in motorPins:
                print(f'initializing motor STEP_PINS: motor {motor_idx} pin {pin}')
                GPIO.setup(pin,GPIO.OUT)
                GPIO.output(pin, False)
    else:
        raise NotImplementedError('door_type not yet implemented')
    print('GPIO setup complete')

def display_new_data(new_data, data_columns): 
    print(); 
    for d, c in zip(new_data, data_columns): 
        if c not in []: print(f"{c}: {d}")
    print()
    
def export_data_to_csv(data, data_columns, file_path):
    data_df = pd.DataFrame(data, columns=data_columns)
    data_df.to_csv(file_path, index=False)
    print(f"data saved to: {file_path}")

'''Assorted less common functionality associated with setup/calibration'''
#### this GPT code seems to verify that all of the pins are of the same type and doing the same thing on the software side...
# def func_to_str(func_code):
    # func_dict = {GPIO.IN: 'GPIO.IN', GPIO.OUT: 'GPIO.OUT', GPIO.SPI: 'GPIO.SPI', GPIO.I2C: 'GPIO.I2C', GPIO.HARD_PWM: 'GPIO.HARD_PWM', GPIO.SERIAL: 'GPIO.SERIAL', GPIO.UNKNOWN: 'GPIO.UNKNOWN',}
    # return func_dict.get(func_code, 'not in dict...')
# for port in ['N', 'S', 'W', 'E']:   
        # try:
            # pin = PORT_DIR_PIN_DICT[port]
            # GPIO.output(PORT_DIR_PIN_DICT[port], GPIO.HIGH)
            # original_mode = GPIO.gpio_function(pin) 
            # GPIO.setup(pin, GPIO.IN)
            # pin_state = GPIO.input(pin)
            # print(f"Pin: {pin}: {pin_state}, {func_to_str(original_mode)}")
            # GPIO.setup(pin, original_mode)
            # GPIO.output(PORT_DIR_PIN_DICT[port], GPIO.LOW)
        # except Exception as e:
            # print(f"failed to read the pin: {e}")
# ## debugging
# print('running rwd pin debug code:')
# ID = 1
# for i in range(100):
    # GPIO.output(ID, GPIO.HIGH)
    # print('on')
    # sleep(CLICK_DURATION) #change the time to whatever gives the correct amount of water
    # print(f'clicking: pin {ID} for {CLICK_DURATION} seconds')
    # GPIO.output(ID, GPIO.LOW)
    # print('off')
    # sleep(CLICK_DURATION*2)

'''Experiment operation helper functions
'''
def lightControl(showLights):
    if showLights[0]: GPIO.output(LED_PINS[0], GPIO.HIGH)
    else: GPIO.output(LED_PINS[0], GPIO.LOW) 
    
    if showLights[1]: GPIO.output(LED_PINS[1], GPIO.HIGH)
    else: GPIO.output(LED_PINS[1], GPIO.LOW)
    
    if showLights[2]: GPIO.output(LED_PINS[2], GPIO.HIGH)
    else: GPIO.output(LED_PINS[2], GPIO.LOW)
    
    if showLights[3]: GPIO.output(LED_PINS[3], GPIO.HIGH)
    else: GPIO.output(LED_PINS[3], GPIO.LOW)

def await_input():
    print('waiting for the next poke')
    while True:
        for pin in NOSE_POKE_PINS: # this is defined on the calling code side; pass in explicitly if an error occurs
            #sleep(1)
            if GPIO.input(pin) == GPIO.HIGH:
                print(f'poke registered: {pin}: {GPIO.input(pin)}, {NOSE_POKE_PIN_DICT[pin]}')
                return [NOSE_POKE_PIN_DICT[pin], pin]

## approach (240820: neither of us can understand each other's approach... but I verified that Author2's works)
def options_to_true_ego(options, current_State, mouse_rotation, maze_rotation, st_positions): # st_positions defined on the calling code side; pass in explicitly if an error occurs
    true_ego = [] #length same as options, has arrays with [pos1, pos2, state]
    rot_dict = {0: ["left", "right", "forwards", "backwards"], 90: ["backwards", "forwards", "left", "right"], 
                180: [ "right", "left", "backwards", "forwards"], 270: ["forwards", "backwards", "right", "left"]}
    pos2options = rot_dict[mouse_rotation]
    if mouse_rotation == maze_rotation: pos1options = ["W", "E", "N", "S"]             #(0,0), (90,90), (180, 180), (270, 270)
    elif abs(mouse_rotation-90) == maze_rotation: pos1options = ["S", "N", "W", "E"]   #(0,90), (90,0), (180, 90), (270,180)
    elif (mouse_rotation+180)%360 == maze_rotation: pos1options = ["E", "W", "S", "N"] #(0, 180), (90, 270), (180,0), (270, 90)
    else: pos1options = ["N", "S", "E", "W"]                                          #(0, 270), (90, 180), (180, 270), (270, 0)
        
    for state in options: # Max 3 states in options 240131: why max 3? Is this outdated from the binary code?
        if (st_positions[state][0] - st_positions[current_State][0] < 0): true_ego.append([pos1options[0], pos2options[0], state]) #If West
        elif (st_positions[state][0] - st_positions[current_State][0] > 0): true_ego.append([pos1options[1], pos2options[1], state]) # If East
        elif (st_positions[state][1] - st_positions[current_State][1] > 0): true_ego.append([pos1options[2], pos2options[2], state]) #If North
        elif (st_positions[state][1] - st_positions[current_State][1] < 0): true_ego.append([pos1options[3], pos2options[3], state]) #If South
    return true_ego

## randomly select new nodes to reward
def update_rewarded_nodes(reached_node, rewarded_nodes, all_nodes):
    # Remove reached node from rewarded_nodes
    print(f"rewarded nodes before deletion: {rewarded_nodes}", type(rewarded_nodes))
    rewarded_nodes = np.delete(rewarded_nodes, np.where(np.array(rewarded_nodes) == reached_node))
    print(f"rewarded nodes after deletion: {rewarded_nodes}", type(rewarded_nodes))

    available_nodes = np.setdiff1d(all_nodes, rewarded_nodes) # Get the nodes that are not yet in rewarded_nodes

    while True: # Randomly select a new unique node
        new_node = np.random.choice(available_nodes, 1)[0]
        if new_node != reached_node: break # enforce a strictly new node not just a random one

    rewarded_nodes = np.append(rewarded_nodes, new_node) # Add new node to rewarded_nodes
    rewarded_nodes = np.sort(rewarded_nodes) # Sort for readability (optional)
    print(f'swapping{reached_node} for {new_node}\nall rewarded nodes: {rewarded_nodes}')
    return rewarded_nodes

"""
Mouse agent stats computations for "yoking"/matching
"""
def chunk_list_into_windows(input_list=None, chunk_size=None, n_per_shift=None): return [input_list[i:i + chunk_size] for i in range(0, len(input_list) - chunk_size + 1, n_per_shift)]

def count_occurences_in_list(input):
    elements = sorted(list(input))
    try: counts = {item: elements.count(item) for item in sorted(list(set(elements)))}
    except TypeError as e:
        elements = [''.join(e) for e in elements]
        counts = {item: elements.count(item) for item in sorted(list(set(elements)))}
    return counts

def get_n_action_prob_dict(input_list, n_actions_per_window, n_actions_shifted):
    """ 240811: whether the args should match reflects subtle hypotheses about how the animal might learn;
            Note: throws away the final states if the list isn't evenly divisible"""
    action_window_list = chunk_list_into_windows(input_list=input_list, chunk_size=n_actions_per_window, n_per_shift=n_actions_shifted)
    action_chunk_counts_dict = count_occurences_in_list(action_window_list)
    action_prob_dict = {k: v/sum(action_chunk_counts_dict.values()) for k, v in action_chunk_counts_dict.items()}
    assert abs(sum(action_prob_dict.values()) - 1) < 0.000001, 'action_prob_dict does not sum to 1'
    return action_prob_dict

def convert_action_words_to_chars(list_of_actions):
    ego_action_str2char = {'right': 'R', 'left': 'L', 'forwards': 'F', 'backwards': 'B'} # convert out of legacy format from keys to values
    try: return [ego_action_str2char[a] for a in list_of_actions]
    except Exception as e:
        print(e); print('240614: unable to convert legacy format data, hopefully cuz the format is already correct...')
        return list_of_actions

import re
import ast
from typing import Any

## Precompile optional, but nice if you call this a lot
_int_pattern = re.compile(r"np\.(?:int|int8|int16|int32|int64|uint8|uint16|uint32|uint64)\((-?\d+)\)")

## Float literal:  1, 1., .5, 1.23, 1e3, 1.0e-3, etc.
_float_pattern = re.compile(
    r"np\.(?:float|float16|float32|float64|float_)\("
    r"(-?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)"
    r"\)")

def deserialize_str(obj: Any) -> Any:
    """
    Safely deserialize strings that may contain numpy scalar constructors like
    '[np.int64(1), np.float64(2.5)]' into normal Python datatypes.

    - If `obj` is not a string, it is returned unchanged.
    - If `obj` is a string, we:
        1) strip known 'np.<int/float>(<literal>)' wrappers,
        2) attempt ast.literal_eval,
        3) return the original string on failure.
    """
    if not isinstance(obj, str):
        return obj

    s_clean = obj

    # Strip np.*int* wrappers: np.int64(1) -> 1, np.uint16(-3) -> -3
    s_clean = _int_pattern.sub(r"\1", s_clean)

    # Strip np.*float* wrappers: np.float64(1.5) -> 1.5
    s_clean = _float_pattern.sub(r"\1", s_clean)

    try:
        return ast.literal_eval(s_clean)
    except (ValueError, SyntaxError):
        # Not a pure literal / collection of literals; keep the original string
        return obj
chk_type = deserialize_str # to maintain legacy compatibility

# Define the colormap to use (e.g., 'turbo')
from matplotlib.cm import get_cmap
# from torch.utils.data import TensorDataset, DataLoader

from tqdm import tqdm

#### 240804 move/organize these below later:

def extract_animal_IDs(text):
    pattern = r'a(\d{3})'
    # print(f'searching pattern: {pattern}')
    matches = re.findall(pattern, text)
    # print(f'found in {text}:\n', matches)
    if len(matches) == 1: return matches[0]
    else: return matches

def get_verified_st_traj(exp_df, verbose=False):
    assert len(exp_df.start_state.unique()) == 1, 'start_state not unique: definitly a problem!'
    if exp_df.start_state.unique()[0] != exp_df.state.to_list()[0]:
        if verbose: print('Warning: start_state not found in state list --> added it')
        return [int(exp_df.start_state.unique()[0])] + exp_df.state.to_list()
    else: return exp_df.state.to_list()

####

""" Pandas operations
"""
def reposition_df_column(orig_df=None, col_name=None, new_col_idx=None):
    assert orig_df is not None and col_name is not None and new_col_idx is not None, '240619: you need to provide the df, the column, and the new column index!'
    df = deepcopy(orig_df)
    col = df.pop(col_name)
    df.insert(new_col_idx, col.name, col)
    return df


""" Logging and I/O functions
"""
class Timer:
    """ NOTE: the timer starts upon instantiation of the object
            - time returned is in units of SECONDS
            - time_func options: time.time, time.perf_counter, time.perf_counter_ns (minimal testing between these thus far... 240327)
    """
    def __init__(self, description='', time_func=time.perf_counter_ns):
        self.time_func = time_func
        self.times = []
        self.durations = []
        self.descriptions = []
        self.start_time = self.start(description=description) ## Start timer immediately

    def start(self, description=''):
        """Start or restart the timer."""
        start_time = self.time_func()
        self.times.append(get_now_str(hms=True, ms=True))
        self.descriptions.append(f"start_{description}")
        self.start_time = start_time
        return start_time

    def stop(self, description=''):
        """Stop the timer, calculate elapsed time, and optionally store it with a description."""
        
        # if self.start_time is None: raise ValueError("Timer has not been started.")
        if self.start_time is None: 
            # raise ValueError("Timer has not been started.")
            print("Timer has not been started -> ignoring stop command")
            return None
        
        elapsed_time = self.time_func() - self.start_time
        if self.time_func==time.perf_counter_ns: elapsed_time = elapsed_time * 1e-9 # convert to seconds if in ns
        self.times.append(get_now_str(hms=True, ms=True))
        self.durations.append(elapsed_time)
        self.descriptions.append(f"stop_{description}")
        self.start_time = None  # Reset start_time to indicate the timer isn't running
        return elapsed_time

    def print(self):
        duration_idx = 0
        for i, (time, description) in enumerate(zip(self.times, self.descriptions)):
            if np.mod(i, 2)==0: print_str = f'time: {time}; {description}'
            elif np.mod(i, 2)==1:
                duration = self.durations[duration_idx]
                print_str = f'time: {time}; {description}\nduration: {duration:.4f}'
                duration_idx += 1
            print(print_str)

def get_most_recent_file(pattern):
    list_of_files = glob(pattern) # * means all if need specific format then *.csv
    # latest_file = min(list_of_files, key=os.path.getctime)
    latest_file = sorted(list_of_files)[-1]
    print(latest_file)
    return latest_file

def get_elapsed_mins(curr_time_stamp, prev_time_stamp):
    """Calculates the elapsed minutes between two timestamps in the format "yymmdd_hhmmss"
    Args:
        curr_time_stamp: The current timestamp.
        prev_time_stamp: The previous timestamp.
    Returns:
        The elapsed minutes between the two timestamps."""
    try:
        curr_datetime = datetime.strptime(curr_time_stamp, "%y%m%d_%H%M%S")
        prev_datetime = datetime.strptime(prev_time_stamp, "%y%m%d_%H%M%S")
        delta = curr_datetime - prev_datetime
    except ValueError as e: 
        curr_datetime = datetime.strptime(curr_time_stamp, "%y%m%d-%H%M%S")
        prev_datetime = datetime.strptime(prev_time_stamp, "%y%m%d-%H%M%S")
        delta = curr_datetime - prev_datetime
    except Exception as e:
        print(e)
        delta = curr_time_stamp - prev_time_stamp
    return delta.total_seconds() / 60

def str2time(time_str, time_format="%y%m%d_%H%M%S"): return datetime.strptime(time_str, time_format)

def extract_leading_time_string(input_string): return "_".join(os.path.basename(input_string).split('_')[:2])
# def png_path_to_time_str(png_full_path): return "_".join(os.path.basename(png_full_path).split('_')[:2]) # identical?

def get_now_str(hms = False, ms=False):
    if ms: return datetime.now().strftime('%Y%m%d-%H%M%S-%f')[:-3]
    elif hms:return datetime.now().strftime("%y%m%d-%H%M%S")
    else: return datetime.now().strftime("%y%m%d")

def save_params(namespace, extra_info_in_name=''): # USE THIS
    # up_df = save_user_params_to_df(globals)
    # up_df = save_user_params_to_df(vars)
    up_df = save_user_params_to_df(namespace)
    print(up_df)
    try: param_path = f'{PROJECT_DIR}/param_records'
    except NameError as e: print(e); print('using ./ for PROJECT_DIR'); param_path = './param_records'
    if not os.path.exists(param_path):
        os.mkdir(param_path)
    fn = f'{get_now_str(hms=True)}_usr_params-{extra_info_in_name}.csv'
    up_df.to_csv(f'{param_path}/{fn}')
    # print('Make sure to run the user parameter cell before this cell!')
    print(f'User parameters saved to: {param_path} as {fn}')
    return up_df

def save_user_params_to_df(variables):
    """ Save all ALL_CAP user paramters to a pandas dataframe (CALLED BY save_params; USE THAT INSTEAD!)
    Args:
        variables - must be set to globals or vars
    Returns:
        Pandas dataframe with columns param and setting, for each respective key and value
    """
    var_name = None # these need to be preallocated to avoid confusing it during the for loop variable creation
    val = None
    user_param_dict = {}
    for (var_name, val) in variables.items(): # use this if you pass in vars or globals in as a dictionary
    # for (var_name, val) in variables().items(): # use this if you pass in vars or globals in as a function (doesn't work when this code is called from a different script/notebook)
        if var_name.isupper(): # user parameters are in ALL_CAPS by convention
            if val is None: # prevents variables set to None as showing up as blank in the df/csv
                val = 'None'
            user_param_dict[var_name] = val # add variable names as dict keys and their settings as dict values
    user_param_dict = collections.OrderedDict(sorted(user_param_dict.items())) # make the dictionary alphabetical
    df = pd.DataFrame(user_param_dict.items(), columns = ['param', 'setting']) # convert the dict to a pandas df
    return df

def show_all_user_params(namespace):
    """ Prints all variables in the ALL_CAPS format
    Args:
        namespace - MUST use dir()
    """
    print('ALL_CAPS format variables in memory:')
    for name in namespace:
    # for name in list(namespace().keys()):
        # if not name.startswith('_'):
            # del globals()[name]
        if name.upper() == name and re.search('[a-zA-Z]', name): # check if variable is ALL_CAPS format
            print(f'{name}')

def del_user_params(namespace):
    """ Delete ALL_CAPS user parameters from namespace
    Args:
        namespace - MUST use globals
    """
    for name in list(namespace().keys()):
        # if not name.startswith('_'):
        # if not name.startswith('_'):
            # del globals()[name]
        if name.upper() == name and re.search('[a-zA-Z]', name): # check if variable is ALL_CAPS format
            print(f'deleting {name}')
            del globals()[name]

def get_user_param(df, param_name_as_str):
    setting = df.loc[df['param'] == param_name_as_str]['setting'].values
    if setting is None or len(setting) == 0:
        return 'param name not found'
    elif type(setting) == np.ndarray:
        return setting[0]  # assumes your df has param and setting columns
    else:
        raise NotImplementedError("Extraction of this datatype is not yet implemented")
## example usage
# up_df = save_params(extra_info_in_name='decoding_notebook')
# # dir()
# show_all_user_params(dir())
# get_user_param(up_df, 'PROJECT_DIR')

def count_files(directory, extension):
  """
  Counts the number of files with the given extension in the given directory.

  Args:
    directory: The directory to search.
    extension: The file extension to count.

  Returns:
    The number of files with the given extension in the directory.
  """
  return len(glob(os.path.join(directory, f"*.{extension.replace('.','')}")))
# # Example usage:
# (legacy Google Drive count_files call removed for double-blind review)
# print(f"Number of CSV files: {num_csv_files}")
# get_now_str(include_hms = False)

""" list operations (todo update these from 240810 simulation code)
"""
# def overlapping_chunks(list_, chunk_size, n_per_shift=1): return [list_[i:i + chunk_size] for i in range(0, len(list_) - chunk_size + 1, n_per_shift)]
# Example usage
# lst = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
# n = 3
# k = 1
# print(overlapping_chunks(lst, n, k))  # Output: [[1, 2, 3], [2, 3, 4], [3, 4, 5], [4, 5, 6], [5, 6, 7], [6, 7, 8], [7, 8, 9]]
# overlapping_chunks(animal_data_dir_dict['a007'], 2)

# def count_occurences_in_list(input):
#     elements = sorted(list(input))
#     counts = {item: elements.count(item) for item in sorted(list(set(elements)))}
#     return counts

def concatenate_gifs(gif_list, output_gif):
    images = []
    first_gif = gif_list[0]

    # Read the first GIF and get its dimensions
    first_reader = imageio.get_reader(first_gif)
    first_frame = first_reader.get_data(0)
    first_height, first_width = first_frame.shape[:2]
    first_reader.close()

    for gif in gif_list:
        reader = imageio.get_reader(gif)
        for frame in reader:
            # Convert frame to an image and resize
            frame_image = Image.fromarray(frame)
            resized_frame = frame_image.resize((first_width, first_height), Image.ANTIALIAS)
            resized_frame_array = np.array(resized_frame)
            # Ensure the resized frame has 3 color channels
            if resized_frame_array.shape[2] == 4:
                resized_frame_array = resized_frame_array[:, :, :3]
            images.append(resized_frame_array)
        reader.close()

    # Save the concatenated frames as a new GIF
    imageio.mimsave(output_gif, images, format='GIF', duration=0.1)

""" Generate yoking data from animal experiments for agent simulations 
"""

def list_to_item_count_dict(list_):
    d={}
    for element in list_:
        if element in d: d[element]+=1
        else: d[element]=1
    return d

def count_dict_to_proportions_dict(count_dict):
    total_count = np.sum(list(count_dict.values()))
    return {k: v/total_count for k,v in count_dict.items()}

def nested_list_to_item_count_dict(list_):
    list_ = list(list_)
    d={}
    for element in list_:
        chunk_str = ''.join(element)
        if chunk_str in d: d[chunk_str]+=1
        else: d[chunk_str]=1
    return d
