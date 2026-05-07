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
import plotly.io as pio
import ast
try: import RPi.GPIO as GPIO
except Exception as e: print(e); print('RPi.GPIO not imported, which is probably ok if you are not running on a Raspberry Pi...')
from time import sleep
import os
import math
from time import time
import time
import subprocess

## 240827: plot counts for a given experiment csv path
def plot_session_counts(csv_path, min_actions_per_exp=10, data_included=['choice', 'state', 'action', 'n_rewards'], 
                        x_label='date&time', y_label='cumulative count', show_mpl=False, save_mpl=True, show_plty=False, save_plty=True, output_dir='./data_out'):
    print(csv_path)

    df = pd.read_csv(csv_path)
    # if len(df) < min_actions_per_exp: print(f"skipping {csv_path} due to insufficient states"); return # exclude the shortest experiments
    # adj_path, st_pos_path = df.adjacency_file.unique()[0], df.st_positions_file.unique()[0]
    # print(f"csv_path: {csv_path}\nadj_path: {adj_path}\nst_pos_path: {st_pos_path}")
    
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
    
    ## todo: generalize this
    # exp_date_time = '_'.join(os.path.basename(csv_path).split('_')[:2]) # extracts the yymmdd_hhmmss unique experiment date-time str
    exp_date_time = os.path.basename(csv_path).split('_')[0] # extracts the yymmdd_hhmmss unique experiment date-time str
    
    print(f'exp_date: {exp_date_time}')

    try: xs_all = [datetime.strptime(ds, '%y%m%d_%H%M%S') for ds in df.time.to_list()] # convert to datetime objects 
    except ValueError as e: 
        try: xs_all = [datetime.strptime(ds, '%y%m%d-%H%M%S') for ds in df.time.to_list()] # alternate format
        except Exception as e:
            print(e)
            import pdb; print('try to debug it now... '); pdb.set_trace()
    

    ## initialize the plots
    plt.figure(figsize=(10, 5)) # Matplotlib plot
    fig = go.Figure() # plotly plot 
    fig.update_layout(autosize=False, width=1000, height=800,paper_bgcolor='black',    # Sets the background color of the paper
        plot_bgcolor='black',     # Sets the background color of the plotting area
        font=dict(color='white'),  # Sets the font color to white for better readability on a black background)
        xaxis=dict(showgrid=False),  # Hides the grid lines on the x-axis
        yaxis=dict(showgrid=False),   # Hides the grid lines on the y-axis
        hoverlabel=dict(bgcolor='rgba(0, 0, 0, 0)'),  # Sets the background color of the hover label to transparent
        hovermode='closest')

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
    FILE_NAME = f'{exp_date_time}_a{animal_ID}_session_counts'
    
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
    
    if save_mpl: plt.savefig(f'{output_dir}/{FILE_NAME}.png', dpi=200)
    if show_mpl: plt.show()
    if save_plty: fig.write_html(f'{output_dir}/{FILE_NAME}.html')
    if show_plty: fig.show()
    return plt.gcf(), plt.gca()

def get_element_hit_inds(input_list): # 240827: specifies the indices at which an list element occurs as a dictionary over the set of unique elements
    element_set = list(set(input_list)) # count these
    element_hit_inds_dict = {e: [] for e in element_set}
    for e_i in element_set:
        for j, e_j in enumerate(input_list):
            if e_j == e_i: element_hit_inds_dict[e_i].append(j)
    return element_hit_inds_dict

def show_new_pngs_in_dir(png_dir, bash_script_path='./display_newest_image-autoMove.sh'):
    # Start the bash script and return the process object
    process = subprocess.Popen(['bash', bash_script_path, '-d', png_dir])
    return process

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
def lightControl(showLights, led_pins):
    if showLights[0]: GPIO.output(led_pins[0], GPIO.HIGH)
    else: GPIO.output(led_pins[0], GPIO.LOW) 
    
    if showLights[1]: GPIO.output(led_pins[1], GPIO.HIGH)
    else: GPIO.output(led_pins[1], GPIO.LOW)
    
    if showLights[2]: GPIO.output(led_pins[2], GPIO.HIGH)
    else: GPIO.output(led_pins[2], GPIO.LOW)
    
    if showLights[3]: GPIO.output(led_pins[3], GPIO.HIGH)
    else: GPIO.output(led_pins[3], GPIO.LOW)

def await_input(nose_poke_pins, nose_poke_pin_dict):
    print('waiting for the next poke')
    while True:
        for pin in nose_poke_pins: # this is defined on the calling code side; pass in explicitly if an error occurs
            #sleep(1)
            if GPIO.input(pin) == GPIO.HIGH:
                print(f'poke registered: {pin}: {GPIO.input(pin)}, {nose_poke_pin_dict[pin]}')
                return [nose_poke_pin_dict[pin], pin]

## approach (240820: neither of us can understand each other's approach... but I verified that Author2's works)
def options_to_true_ego(options, current_State, mouse_rotation, maze_rotation, st_positions): # st_positions defined on the calling code side; pass in explicitly if an error occurs
    true_ego = [] #length same as options, has arrays with [pos1, pos2, state]
    rot_dict = {0: ["left", "right", "forwards", "backwards"], 90: ["backwards", "forwards", "left", "right"], 180: [ "right", "left", "backwards", "forwards"], 270: ["forwards", "backwards", "right", "left"]}
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
        ego_actions = df.action.tolist()

"""
Maze environment definitions and transformations
"""

#### state transitions 

## important dictionaries
cart_to_orient_dict = {(0, 1): 'N', (0, -1): 'S', (-1, 0): 'W', (1, 0): 'E'} # defining the mapping between cartesian displacement and agent orientation

ego_radian_map_dict = {'F': 0, 'B': np.pi, 'L': np.pi/2, 'R': 3*np.pi/2} # egocentric conventions for chars to radians and back
ego_radian_map_dict.update({v: k for k, v in ego_radian_map_dict.items()}, mapping_dict = ego_radian_map_dict) # note values are also keys --> it gives you whatever format wasn't what you passed in

allo_radian_map_dict = {'N': np.pi/2, 'S': 3*np.pi/2, 'W': np.pi, 'E': 0} # allocentric conventions for chars to radians and back
allo_radian_map_dict.update({v: k for k,v in allo_radian_map_dict.items()})

def displacement_to_compass_heading(displacement): return cart_to_orient_dict[tuple(np.sign(displacement).tolist())]

def get_ego_direction(current_point, candidate_point, current_heading):
    angle_to_candidate = np.arctan2(candidate_point[1] - current_point[1], candidate_point[0] - current_point[0])
    relative_angle = angle_to_candidate - current_heading
    return ego_radian_map_dict[np.mod(relative_angle, 2*np.pi)]  # Normalize to [0, 2π)

def get_adj_states(state, adj_mat=None):
    state = int(state)
    return np.argwhere(adj_mat[state] == 1).flatten()

def env_transition_func(current_state, action, type_, heading=None, adj_mat=None, st_positions=None): # the transition function
    current_state = int(current_state)
    accessible_states = get_adj_states(current_state, adj_mat=adj_mat) # accessible states
    if type_ == 'ego': available_states = {get_ego_direction(st_positions[current_state], st_positions[acc_st], heading): acc_st for acc_st in accessible_states}
    elif type_ == 'allo': available_states = {displacement_to_compass_heading(st_positions[acc_st] - st_positions[current_state]): acc_st for acc_st in accessible_states}
    try: return available_states[action]
    except KeyError as e: return 'action_unavailable'

def load_maze(adj_path, st_pos_path):
    st_positions = pd.read_csv(f'{st_pos_path}').to_numpy()[:,1:] # row index matches the state indices; each row is [x_position, y_position]
    adj_df = pd.read_csv(f'{adj_path}', header=None)

    adj_df = adj_df.fillna(0) # map nans to zeros
    adj_df = adj_df.astype(int) # convert floats (NaNs are imported as float) to int
    adj_mat = adj_df.to_numpy() + adj_df.to_numpy().T # make the adjacency matrix symmetric and convert to numpy
    adj_df = pd.DataFrame(adj_mat) # update this to be symmetric across the diagonal like the numpy version

    # return adj_df, st_positions
    return adj_mat, st_positions

def plot_maze(adj_mat, st_positions, title='', rwd_states=None, current_position=None, current_position_marker_size=40, filename = "unspecified",
              output_dir= './data_out', close=True, show=False, save=True):
                  
    WIDTH_PIXELS = 1024
    HEIGHT_PIXELS = 600
    DPI = 150
    
    plt.figure(figsize=(WIDTH_PIXELS/DPI, HEIGHT_PIXELS/DPI))

    ## plot the lines connecting nodes
    for i, row in enumerate(adj_mat):
        connected_states = np.argwhere(adj_mat[i]==1).flatten().tolist()
        pt_1 = st_positions[i] # [x, y]
        for j in connected_states:
            pt_2 = st_positions[j]
            plt.plot([pt_1[0], pt_2[0]], [pt_1[1], pt_2[1]], '-o', c='k', linewidth=5)

    ## label the nodes
    for node_idx, (x,y) in enumerate(st_positions):
        plt.annotate(node_idx, (x, y), textcoords="offset points", xytext=(9, 7), ha='center')

    ## plot current position of agent/mouse
    if current_position is not None: plt.plot(st_positions[current_position][0], st_positions[current_position][1], 'o', markersize=current_position_marker_size, color='black')

    try:
        ## color the reward
        if type(rwd_states) == list:
            for rwd_state in rwd_states:
                # print(f'rwd_state: {rwd_state}')
                # print(f"st_positions[rwd_state]: {st_positions[rwd_state]}")
                plt.plot(st_positions[rwd_state][0], st_positions[rwd_state][1], '*', markersize=38, color='deepskyblue')
        else: plt.plot(st_positions[rwd_states][0], st_positions[rwd_states][1], 'o', markersize=12, color='deepskyblue') # this is a misnomer cuz there is only one rwd state in this condition
    except IndexError as e:
        print(e); print('no rwds found -> attempting to skip this')
        # import pdb; pdb.set_trace()

    # title = f"{title}\nrwds at: {rwd_states}"
    title = f"{title}"

    plt.title(title, fontsize=11)
    plt.axis('off')
    plt.tight_layout()
    plt.axis('equal')

    if save:
        if not os.path.exists(output_dir): os.mkdir(output_dir)
        plt.savefig(f"{output_dir}/{filename}.png", dpi=DPI)

    if show: plt.show()
    if close: plt.close('all')

def chk_type(x):
    """ safe eval strings back into python datatypes or return the original object if unable to convert"""
    try:
        return ast.literal_eval(x)
    except (ValueError, SyntaxError):
        return x

# Define the colormap to use (e.g., 'turbo')
from matplotlib.cm import get_cmap
# from torch.utils.data import TensorDataset, DataLoader

from tqdm import tqdm

#### 240804 move/organize these below later:

def extract_animal_IDs(text):
    pattern = r'a(\d{3})'
    print(f'searching pattern: {pattern}')
    matches = re.findall(pattern, text)
    print(f'found in {text}:\n', matches)
    if len(matches) == 1: return matches[0]
    else: return matches

def get_verified_st_traj(exp_df):
    assert len(exp_df.start_state.unique()) == 1, 'start_state not unique: definitly a problem!'
    if exp_df.start_state.unique()[0] != exp_df.state.to_list()[0]:
        print('Warning: start_state not found in state list --> added it')
        return [exp_df.start_state.unique()[0]] + exp_df.state.to_list()
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
        print('test')

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

""" save variables defined with the ALL_CAPS_N_UNDERSCORES convention to a pandas dataframe as a csv"""
def save_params(namespace=globals(), output_dir='./', extra_info_in_name='', now_str=None): # USE THIS
    if now_str is None:
        now_str = get_now_str(hms=True)
    # up_df = save_user_params_to_df(globals)
    up_df = save_user_params_to_df(namespace)
    print(up_df)
    param_path = f'{output_dir}'
    os.makedirs(param_path, exist_ok=True)
    if extra_info_in_name is not None: extra_info_in_name = '-' + extra_info_in_name
    fn = f'{now_str}_usr_params{extra_info_in_name}.csv'
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
    # for (var_name, val) in variables().items():
    for (var_name, val) in variables.items():
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
# num_csv_files = count_files("/content/drive/MyDrive/a_science_redacted/behavior/a_latent_maze/experiment_out/decoding_notebook/20230614-151817_latent_maze_decoding_notebook-mouse_15_session_1", "csv")
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

# count_dict_to_proportions_dict(list_to_item_count_dict(d_df.action.tolist()))

# def get_n_back_dict(input_list, chunk_size=2, shift=1, max_n_back=50):
#     n_back_dict = {}
#     overlapping_chunks(input_list, chunk_size, shift)
#     assert input_list == list(np.array(overlapping_chunks(input_list, 1, 1)).flatten()), 'failure here is highly alarming!'

#     for chunk_size in range(1, max_n_back+1): # initial attempt to run up to the full sequence length lead to memory-related crashes even w/ 50 gb ram
#         # print(chunk_size)
#         chunks = overlapping_chunks(input_list, chunk_size, shift)
#         # print(chunks)
#         n_back_dict[chunk_size] = chunks
#     return n_back_dict



## outdated: replace w/ equivalent from 240813b_latMaz_simulations-DEV.ipynb or later
# def get_n_back_probability_dict(n_back_dict, max_allowed_sum_prob_dev_from_1 = 0.000001):
#     n_back_prob_dict = {}
#     for k, v in n_back_dict.items():

#         if len(v) == 0: continue

#         # print(k, v)
#         proportions_dict = count_dict_to_proportions_dict(nested_list_to_item_count_dict(v))
#         # print(proportions_dict)
#         prob_sum = np.sum(np.array(list(proportions_dict.values())))
#         # print('sum of probabilities: ', prob_sum)
#         assert np.abs(prob_sum - 1) < max_allowed_sum_prob_dev_from_1, f'Probabilities do not sum to 1 ({prob_sum})!'
#         # n_back_prob_dict[k] = proportions_dict
#         n_back_prob_dict[str(k)] = proportions_dict
#         # print()
#     return n_back_prob_dict

""" outdated/reference generate simulation state trajectories
"""
# d = {}; d[(0, 1)] = 'N'; d[(0, -1)] = 'S'; d[(-1, 0)] = 'W'; d[(1, 0)] = 'E' # defining the mapping between cartesian coord and agent orientation
# cart_to_orient_dict = d
# def get_neighbors(state, adj_mat=None): return np.argwhere(adj_mat[state] == 1).flatten()
# def displacement_to_orientation(displacement, mapping_dict=cart_to_orient_dict): return mapping_dict[tuple(np.sign(displacement).tolist())]

# ego_radian_map_dict = {'F': 0, 'B': np.pi, 'L': np.pi/2, 'R': 3*np.pi/2}
# ego_radian_map_dict.update({v: k for k, v in ego_radian_map_dict.items()})
# def get_ego_direction(current_point, candidate_point, current_heading):
#     angle_to_candidate = np.arctan2(candidate_point[1] - current_point[1], candidate_point[0] - current_point[0])
#     relative_angle = angle_to_candidate - current_heading
#     return ego_radian_map_dict[np.mod(relative_angle, 2*np.pi)]  # Normalize to [0, 2π)

# def standardize_dict(dct, all_desired_keys): # add in missing choices (keys) and assign their probability (values) to 0. 
#     if set(all_desired_keys) == set(dct.keys()): return dct
#     else:
#         for k in set(list(all_desired_keys)) - set(list(dct.keys())): dct[k] = 0 # add the missing keys and assign the value to 0 probability
#     return dct

# def take_biased_action(current_state, action, type_, heading= None, adj_mat=None, st_positions=None):
#     accessible_states = get_neighbors(current_state, adj_mat=adj_mat) # accessible states
#     if type_ == 'ego': available_actions = {get_ego_direction(st_positions[current_state], st_positions[acc_st], heading): acc_st for acc_st in accessible_states}
#     elif type_ == 'allo': available_actions = {displacement_to_orientation(st_positions[acc_st] - st_positions[current_state]): acc_st for acc_st in accessible_states}
#     try: return available_actions[action]
#     except KeyError as e: return 'action_unavailable'

# # def generate_trajectory_legacy(loc_start=None, trajectory_len=None, adj_mat=None, st_positions=None, ego_action_biases=None, allo_arm_biases=None, initial_heading=np.pi/2):
# #     prior_state = None
# #     heading = initial_heading # defined such that east is 0 and north is pi/2
# #     traj = [loc_start]
# #     for i in range(trajectory_len-1): # -1 cuz we preallocated with the first state outside of the loop
# #     # for i in tqdm.tqdm(range(trajectory_len-1)): # -1 cuz we preallocated with the first state outside of the loop
# #         current_state = traj[i]
# #         accessible_states = get_neighbors(current_state, adj_mat=adj_mat) # accessible states

# #         assert ego_action_biases is None or allo_arm_biases is None, 'code is not currently setup to match BOTH ego and allo animal biases...'
# #         if ego_action_biases is None and allo_arm_biases is None:
# #             if len(accessible_states) > 1: valid_next_states = accessible_states[accessible_states != prior_state] # prevent reversals in most cases; TODO: make this a flexible bias param
# #             else: valid_next_states = accessible_states # allow reversal if at a dead end
# #             traj.append(np.random.choice(valid_next_states)) # select next action among all those that don't prematurely reverse
# #             # prior_state = traj[-2]
# #         elif ego_action_biases is not None:
# #             while True: # keep iterating until the key is found
# #                 ego_direction_chosen = np.random.choice(['F', 'B', 'L', 'R'], size=1, replace=True, p=[ego_action_biases['F'], ego_action_biases['B'], ego_action_biases['L'], ego_action_biases['R']])[0]
# #                 state_option_directions = {get_ego_direction(st_positions[current_state], st_positions[acc_st], heading): acc_st for acc_st in accessible_states}
# #                 try:
# #                     state_chosen = state_option_directions[ego_direction_chosen]
# #                     break
# #                 except KeyError as e: pass
# #             traj.append(state_chosen)
# #         elif allo_arm_biases is not None:
# #             while True:
# #                 action_heading_chosen = np.random.choice(['N', 'S', 'W', 'E'], size=1, replace=True, p=[allo_arm_biases['N'], allo_arm_biases['S'], allo_arm_biases['W'], allo_arm_biases['E']])[0]
# #                 state_option_headings = {displacement_to_orientation(st_positions[acc_st] - st_positions[current_state]): acc_st for acc_st in accessible_states}
# #                 try:
# #                     state_chosen = state_option_headings[action_heading_chosen]
# #                     break
# #                 except KeyError as e: pass
# #             traj.append(state_chosen)
# #         prior_state = traj[-2]
# #     return traj

# # def generate_trajectory(loc_start=None, trajectory_len=None, adj_mat=None, st_positions=None, ego_action_biases=None, allo_arm_biases=None, initial_heading=np.pi/2, max_attempts=1000):
# #     assert ego_action_biases is not None or allo_arm_biases is not None, 'see generate_trajectory for the case where neither biases are provided...'
# #     assert not (ego_action_biases is not None and allo_arm_biases is not None), 'code is not currently setup to match BOTH ego and allo animal biases...'
    
# #     if ego_action_biases is not None: 
# #         n_back_dict = standardize_dict(ego_action_biases, ['F', 'B', 'L', 'R'])
# #         type_ = 'ego'
# #     elif allo_arm_biases is not None: 
# #         n_back_dict = standardize_dict(allo_arm_biases, ['N', 'S', 'W', 'E'])
# #         type_ = 'allo'
# #     n_actions_per_selection = len(np.unique([len(k) for k in n_back_dict.keys()]))
# #     assert n_actions_per_selection == 1, 'the n-back dictionary appears to have keys of different lengths...'
    
# #     attempt_idx = 0
# #     heading = initial_heading # defined such that east is 0 and north is pi/2
# #     traj = [loc_start]
# #     while len(traj) < trajectory_len:
# #         current_state = traj[-1] # this one should be valid
# #         current_state_temp = deepcopy(current_state) # this version is used to test the validity of the action sequence before committing to adding it to the trajectory        
# #         attempted_action_groups = []
# #         while True: # keep iterating until the key is found
# #             assert len(set(attempted_action_groups)) != len(n_back_dict.keys()), 'all possible action sequences have been attempted... check the n_back_dict for errors...'
# #             candidate_action_seq = np.random.choice(list(n_back_dict.keys()), size=1, replace=True, p=list(n_back_dict.values()))[0] # get candidate next action sequence
# #             attempted_action_groups.append(candidate_action_seq)

# #             chosen_state_group = []
# #             for action in candidate_action_seq:
# #                 state_chosen = take_biased_action(current_state_temp, action, type_, heading= heading, adj_mat=adj_mat, st_positions=st_positions)
# #                 if state_chosen == 'action_unavailable': break # abort this candidate_action_seq
# #                 chosen_state_group.append(state_chosen)
# #                 current_state_temp = state_chosen
# #             if 'action_unavailable' not in chosen_state_group: break # success criterion
# #             attempt_idx += 1
# #             if attempt_idx > max_attempts: 
# #                 traj.append('failed to satisfy candidate_action_seq befre max_attempts reached')
# #                 print(f'failed to satisfy candidate_action_seq befre max_attempts reached in state: {traj[-2]}')
# #                 sys.exit('halting when max_attempts reached... comment out to let it to rerun')
# #                 return traj
# #         traj.extend(chosen_state_group)    
# #     return traj

# def load_maze(adj_path, st_pos_path):
#     try: 
#         st_positions = pd.read_csv(f'./data_in/{st_pos_path}').to_numpy()[:,1:] # row index matches the state indices; each row is [x_position, y_position]
#         adj_df = pd.read_csv(f'./data_in/{adj_path}', header=None)
#     except Exception as e:
#         print(e)
#         print('Failed to load the maze data. Check that the file paths are correct and that the files are in the correct format.')
#         st_positions = pd.read_csv(f'{st_pos_path}').to_numpy()[:,1:] # row index matches the state indices; each row is [x_position, y_position]
#         adj_df = pd.read_csv(f'{adj_path}', header=None)
        
#     adj_df = adj_df.fillna(0) # map nans to zeros
#     adj_df = adj_df.astype(int) # convert floats (NaNs are imported as float) to int
#     adj_mat = adj_df.to_numpy() + adj_df.to_numpy().T # make the adjacency matrix symmetric and convert to numpy
#     adj_df = pd.DataFrame(adj_mat) # update this to be symmetric across the diagonal like the numpy version

#     ## optional saving
#     # with open(f"{MAZE_ENV_CSV.replace('.csv', '')}.npy", 'wb') as f: np.save(f, adj_mat)
#     # with open(f"{STATE_POSITION_CSV.replace('.csv', '')}.npy", 'wb') as f: np.save(f, st_positions)

#     # return adj_df, st_positions
#     return adj_mat, st_positions

