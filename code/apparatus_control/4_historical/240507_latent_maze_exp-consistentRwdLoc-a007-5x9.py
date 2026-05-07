
"""
Latent Maze Experiment

-runs the experiment

Authors: anonymized for review

Legacy file is located at
    (Colab URL redacted for double-blind review)
"""
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import matplotlib

import RPi.GPIO as GPIO
from time import sleep
import datetime
import os
import pandas as pd
from copy import deepcopy
import math
from time import time
import sys
import time
import pdb

'''
USER_PARAMETERS, i.e. experiment config
'''
# use: 'set motors', 'calibrate rwds', or 'run experiment'
RUN_MODE = 'run experiment'
# RUN_MODE = 'set motors' 
# RUN_MODE = 'calibrate rwds'

#### RPi pin assignments
## NOTE: THESE NUMBERS ARE GPIO# not RPI PINS!!!!!
NOSE_POKE_PINS = [2, 10, 11, 14] #These values need to be changed to whichever pins correlate with the input from the beam breaks
LED_PINS = [3, 9, 0, 15] #These values need to be changed to whichever pins correlate with the output to the LEDs
PORT_DIR_PIN_DICT = {'N': 21, 'S': 1, 'W': 7, 'E': 8} # GPIO pin values for the solenoids/rewards associated with each arm
# PORT_DIR_PIN_DICT = {'N': 8, 'S': 1, 'W': 7, 'E': 21} # GPIO pin values for the solenoids/rewards associated with each arm
# PORT_DIR_PIN_DICT = {'N': 8, 'S': 7, 'W': 1, 'E': 21} # GPIO pin values for the solenoids/rewards associated with each arm

POKE_PIN_TO_RWD_PIN_MAP = {2: 21, 10: 1, 11: 7, 14: 8}
# POKE_PIN_TO_RWD_PIN_MAP = {2: 8, 10: 1, 11: 7, 14: 21}
# POKE_PIN_TO_RWD_PIN_MAP = {2: 8, 10: 7, 11: 1, 14: 21}

NOSE_POKE_PIN_DICT = {NOSE_POKE_PINS[0]: 'N', NOSE_POKE_PINS[1]: 'S', NOSE_POKE_PINS[2]: 'W', NOSE_POKE_PINS[3]: 'E'} ## map port locations to RPi pins
ARM_TO_MOTOR_IDX_DICT = {'N': 0, 'S': 1, 'W': 2, 'E': 3}

#### state transition delays
SAME_PORT_MIN_REPOKE_INTERVAL = 60 #240415 45 -> 60; 240304 2nd sessions --> 30 -> 45 240301 45 -> 30 # 240226 30 -> 45; 240224 15 --> 30; 240222 10 -> 12 
MIN_INTER_POKE_INTERVAL = 0.5 # in seconds
MAX_ALLOWED_POKE_DOOR_DELAY = 6 # ie purposely end the experiment if this delay is too long (cuz animals will be able to enter inactive arms before the doors close)

#### motors (for doors blocking invalid arms)
STEP_PINS = [[17,18,27,22],[23,24,25,4],[13,12,6,5],[20,26,16,19]] # motors: list of lists, 1 list per motor ### original m0, 1, 2, 3
MOTOR_CLOSED_POS_DEFAULT = 0.0  # Default position as a fraction of STEPS_PER_FULL_ROTATION
MOTORS_CLOSED_AT_START = [0, 1, 2, 3]  # A list of motor indexes to operate on, default is all motors

# MOTOR_DIRECTION = [1, -1, 1, 1] # THIS IS CONSISTENT DIRECTION, LEAVE FOR REFERENCE! 1 is the default direction; -1 reverses that; N S W E (i.e. aiming for same order conventions as elsewhere)
MOTOR_DIRECTION = [1, -1, 1, -1] # 1 is the default direction; -1 reverses that; N S W E (i.e. aiming for same order conventions as elsewhere) ALL CORRECT BUT SOUTH
MOTOR_DIRECTION = [1, 1, 1, -1] # 1 is the default direction; -1 reverses that; N S W E (i.e. aiming for same order conventions as elsewhere)

STEPS_PER_FULL_ROTATION = 1024 
INTERACTIVE_ROTATION = 0.15 # Give as fraction of a full rotation
DOOR_OPEN_MOTOR_POS = int(STEPS_PER_FULL_ROTATION * INTERACTIVE_ROTATION)

# MOTOR_STEP_MS_DELAY = 2/float(1000) # device example/recommended value: 10/float(1000); presumably in ms; ~2/float(1000) appears to be min before it jams
MOTOR_STEP_DELAY = 2/float(1000) # in seconds device example/recommended value: 10/float(1000); presumably in ms; ~2/float(1000) appears to be min before it jams

## STEP_SPEED details:
##   240131: example code suggest that 1 and 2 are valid speeds (240201: verified? 3 with 2 ms MOTOR_STEP_DELAY appears to be jagged and not faster). 
##   Negative values reverse the direction. 
STEP_SPEED = -2 

#### input/output paths
PROJECT_DIR = '/home/anonymized-user/Desktop/latent_maze-RPI'
DATA_OUT_DIR = './data_out'

#### Select a specific latent maze selection


# MAZE_SELECTED = 'small less aliased' # only east closed
# MAZE_SELECTED = 'small binary' # only north open
# MAZE_SELECTED = 'eLife binary'
# MAZE_SELECTED = '240218'
MAZE_SELECTED = '240507-5x9'
if MAZE_SELECTED == '240507-5x9':
    MAZE_ENV_CSV = '240510_latent_maze_003b_5x9-adjacency.csv'
    STATE_POSITION_CSV = '240507_latent_maze_003b_5x9-state_positions.csv'
    START_STATE = 0
    RWD_STATE = 7 # not used if selecting locations randomly via REWARDED_STATES = 'random' below
elif MAZE_SELECTED == '240218':
    MAZE_ENV_CSV = '240218_intuition_maze_003-adjacency.csv'
    STATE_POSITION_CSV = '240218_intuition_maze_003-state_positions.csv'
    START_STATE = 0
    RWD_STATE = 7 # not used if selecting locations randomly via REWARDED_STATES = 'random' below

elif MAZE_SELECTED == 'small less aliased': ## less aliased graph 001
    MAZE_ENV_CSV = '230921_latent_maze_002_less_aliased_n_no_deadends-adjacency_to_csv.csv'
    STATE_POSITION_CSV = '230921_latent_maze_002_less_aliased_n_no_deadends-state_positions.csv'
    START_STATE = 3
    RWD_STATE = 7

elif MAZE_SELECTED == 'small binary': ## small binary maze
    MAZE_ENV_CSV = '220918_3_level_binary_tree-adjacency.csv' # only need to define the upper triangle
    STATE_POSITION_CSV = '220918_3_level_binary_tree-state_positions.csv'
    START_STATE = 15
    RWD_STATE = 3

elif MAZE_SELECTED == 'eLife binary': ## eLife '21 binary maze
    MAZE_ENV_CSV = '2201028_6_level_binary_tree-adjacency.csv' # only need to define the upper triangle
    STATE_POSITION_CSV = '2201023_6_level_binary_tree-state_positions.csv'
    START_STATE = 123
    RWD_STATE = 4
else:
    sys.exit('MAZE_SELECTED not found!')

#### rewards: specify a list of states to reward or 'random' (note: that this then uses PROPORTION_STATES_REWARDED to select random states to reward)
# REWARDED_STATES = [RWD_STATE, START_STATE, 2, 12, 11] 
# REWARDED_STATES = [RWD_STATE] 
# REWARDED_STATES = 'random'
# REWARDED_STATES = 'pseudo random (without replacement)' 
# REWARDED_STATES = 'side alternation w/ refinement ver2.5' 
REWARDED_STATES = 'stable long path'

if REWARDED_STATES == 'stable long path':
    rewarded_states = [0, 1, 10, 19, 20, 11, 2, 3, 12, 13, 14, 15, 24, 25, 34, 35, 44]
    INITIAL_REWARDED_STATES = deepcopy(rewarded_states)
elif REWARDED_STATES == 'random': 
    PROPORTION_STATES_REWARDED = 0.5  # Replace with the proportion you want, between 0 and 1
    n_select = max(math.floor(n_states * PROPORTION_STATES_REWARDED), 1) # Calculate the number of random integers to select
    rewarded_states = list(np.sort(np.random.choice(all_nodes, size=n_select, replace=False)))  # Randomly select unique integers
    raise NotImplementedError('check below that random rewards are properly defined i.e. is there still asdf as a string?')
elif REWARDED_STATES == 'pseudo random (without replacement)':
    PROPORTION_STATES_REWARDED = 1.0
    # EXCLUDED_STATES = [0, 1, 9, 10, 11, 18, 19, 20, 27, 28, 29, 30, 36, 37, 38] # moderate 
    # EXCLUDED_STATES = [0, 1, 2, 3, 9, 10, 11, 18, 19, 20, 27, 28, 29, 30, 31, 36, 37, 38, 39, 40] # harder 
    # EXCLUDED_STATES = [0, 1, 2, 3, 4, 9, 10, 11, 12, 13, 18, 19, 20, 21, 22, 27, 28, 29, 30, 31, 36, 37, 38, 39, 40] # 240414: isn't this a misnomer? 240304: east 20 only
    EXCLUDED_STATES = [0, 1, 2, 4, 9, 11, 18, 21, 22, 27, 28, 31, 36, 37, 38, 39, 40] # 240304: east side + path to east 
    INITIAL_REWARDED_STATES = list(set(np.arange(45)) - set(EXCLUDED_STATES))
    REPLENISHED_REWARDED_STATES_0 = list(set(EXCLUDED_STATES))
    REPLENISHED_REWARDED_STATES_1 = INITIAL_REWARDED_STATES
elif REWARDED_STATES == 'side alternation w/ refinement ver0':
    EAST_20_STATES = sorted(list(set(np.arange(5,8+1)))+list(set(np.arange(14, 17+1)))+list(set(np.arange(23, 26+1)))+list(set(np.arange(32,35+1)))+list(set(np.arange(41,44+1))))
    
    MIDLINE_5 = {4, 13, 22, 31, 40}
    MIDLINE_15 = sorted(list(np.array([list(np.arange(3+i,39+i+1, 9)) for i in range(0,2+1)]).flatten()))
    
    WEST_20_STATES = sorted(list(set(np.arange(0, 44+1)) - set( EAST_20_STATES) - MIDLINE_5))
    
    SHAPING_PATH = [10, 19, 20, 11, 2, 3, 12, 13, 29, 30, 39, 40, 31]
    
    INITIAL_REWARDED_STATES = deepcopy(EAST_20_STATES + SHAPING_PATH)
    REPLENISHED_REWARDED_STATES_0 = deepcopy(WEST_20_STATES)
    REPLENISHED_REWARDED_STATES_1 = sorted(list(set(deepcopy(INITIAL_REWARDED_STATES)) - set(MIDLINE_15)))
    REPLENISHED_REWARDED_STATES_2 = sorted(list(set(deepcopy(REPLENISHED_REWARDED_STATES_0)) - set(MIDLINE_15)))
    
    RESET_WHEN_N_RWDS_REMAINING = 0
    
elif REWARDED_STATES == 'side alternation w/ refinement ver1':
    EAST_20_STATES = sorted(list(np.array([list(np.arange(5+i,41+i+1, 9)) for i in range(0,3+1)]).flatten()))
    MIDLINE_15 = sorted(list(np.array([list(np.arange(3+i,39+i+1, 9)) for i in range(0,2+1)]).flatten()))
    MIDLINE_5 = {4, 13, 22, 31, 40}
    WEST_20_STATES = sorted(list(set(np.arange(0, 44+1)) - set(EAST_20_STATES) - MIDLINE_5))
    SHAPING_PATH = [10, 19, 20, 11, 2, 3, 12, 13, 14, 5, 23, 32, 29, 30, 39, 40, 31]
    
    EAST_15_STATES = sorted(list(set(EAST_20_STATES) - set(MIDLINE_15)))
    WEST_15_STATES = sorted(list(set(WEST_20_STATES) - set(MIDLINE_15)))
    
    INITIAL_REWARDED_STATES = deepcopy(EAST_15_STATES + SHAPING_PATH)
    REPLENISHED_REWARDED_STATES_0 = deepcopy(WEST_15_STATES)
    REPLENISHED_REWARDED_STATES_1 = sorted(EAST_15_STATES)
    REPLENISHED_REWARDED_STATES_2 = sorted(WEST_15_STATES)
    
    RESET_WHEN_N_RWDS_REMAINING = 0
    
elif REWARDED_STATES == 'side alternation w/ refinement ver2':
    EAST_20_STATES = sorted(list(np.array([list(np.arange(5+i,41+i+1, 9)) for i in range(0,3+1)]).flatten()))
    MIDLINE_15 = sorted(list(np.array([list(np.arange(3+i,39+i+1, 9)) for i in range(0,2+1)]).flatten()))
    MIDLINE_5 = {4, 13, 22, 31, 40}
    WEST_20_STATES = sorted(list(set(np.arange(0, 44+1)) - set(EAST_20_STATES) - MIDLINE_5))
    
    SHAPING_PATH = [10, 20, 3, 14, 32, 30, 32]
    
    EAST_10_STATES = sorted(list(set(EAST_20_STATES) - set(MIDLINE_15) - set(np.arange(6, 42+1, 9))))
    WEST_10_STATES = sorted(list(set(WEST_20_STATES) - set(MIDLINE_15) - set(np.arange(2, 38+1, 9))))
    
    INITIAL_REWARDED_STATES = deepcopy(EAST_10_STATES + SHAPING_PATH)
    REPLENISHED_REWARDED_STATES_0 = deepcopy(WEST_10_STATES)
    REPLENISHED_REWARDED_STATES_1 = sorted(EAST_10_STATES)
    REPLENISHED_REWARDED_STATES_2 = sorted(WEST_10_STATES)
    
    RESET_WHEN_N_RWDS_REMAINING = 2
    
elif REWARDED_STATES == 'side alternation w/ refinement ver2.5':
    EAST_20_STATES = sorted(list(np.array([list(np.arange(5+i,41+i+1, 9)) for i in range(0,3+1)]).flatten()))
    MIDLINE_15 = sorted(list(np.array([list(np.arange(3+i,39+i+1, 9)) for i in range(0,2+1)]).flatten()))
    MIDLINE_5 = {4, 13, 22, 31, 40}
    WEST_20_STATES = sorted(list(set(np.arange(0, 44+1)) - set(EAST_20_STATES) - MIDLINE_5))
    
    SHAPING_PATH = [10, 20, 3, 14, 32, 30, 32]
    
    EAST_10_STATES = sorted(list(set(EAST_20_STATES) - set(MIDLINE_15) - set(np.arange(6, 42+1, 9))))
    WEST_10_STATES = sorted(list(set(WEST_20_STATES) - set(MIDLINE_15) - set(np.arange(2, 38+1, 9))))
    
    INITIAL_REWARDED_STATES = deepcopy(EAST_10_STATES + SHAPING_PATH)
    REPLENISHED_REWARDED_STATES_0 = deepcopy(WEST_10_STATES)
    REPLENISHED_REWARDED_STATES_1 = sorted(EAST_10_STATES)
    REPLENISHED_REWARDED_STATES_2 = sorted(WEST_10_STATES)
    
    RESET_WHEN_N_RWDS_REMAINING = 3
    
elif REWARDED_STATES == 'side alternation w/ refinement ver3':
    EAST_20_STATES = sorted(list(np.array([list(np.arange(5+i,41+i+1, 9)) for i in range(0,3+1)]).flatten()))
    MIDLINE_15 = sorted(list(np.array([list(np.arange(3+i,39+i+1, 9)) for i in range(0,2+1)]).flatten()))
    MIDLINE_5 = {4, 13, 22, 31, 40}
    WEST_20_STATES = sorted(list(set(np.arange(0, 44+1)) - set(EAST_20_STATES) - MIDLINE_5))
    
    SHAPING_PATH = [10, 20, 3, 14, 32, 30, 32]
    
    EAST_5_STATES = sorted(list(set(EAST_20_STATES) - set(MIDLINE_15) - set(np.arange(6, 42+1, 9)) - set(np.arange(7, 43+1, 9))))
    WEST_5_STATES = sorted(list(set(WEST_20_STATES) - set(MIDLINE_15) - set(np.arange(2, 38+1, 9)) - set(np.arange(1, 37+1, 9))))
    
    INITIAL_REWARDED_STATES = deepcopy(EAST_5_STATES + SHAPING_PATH)
    REPLENISHED_REWARDED_STATES_0 = deepcopy(WEST_5_STATES)
    REPLENISHED_REWARDED_STATES_1 = sorted(EAST_5_STATES)
    REPLENISHED_REWARDED_STATES_2 = sorted(WEST_5_STATES)
    
    RESET_WHEN_N_RWDS_REMAINING = 1

rewarded_states = deepcopy(INITIAL_REWARDED_STATES) # initialize the actual first set of rewarded states
print('INITIAL_REWARDED STATES: ', type(rewarded_states), rewarded_states)

    
REWARD_DURATION = 0.015 # 240517: 0.02 -> 0.015 240514: 0.05 -> 0.01 (maybe too low) -> 0.02; 240512: 0.15 -> 0.05; 240510: 0.15-> 0.05; 240417: 0.15 --> 0.1; 240416: 0.3 -> 0.15; 240415: 0.15 -> 0.3; 240414: 0.075 -> 0.15; 240301: 0.1 -> 0.075; 240229 0.15 -> 0.1; 240227 0.07 -> 0.15; 240224 0.07 -> 0.05; 240223 (after a002): 0.1 -> 0.07; 240222: 0.15 -> 0.1 solenoid open time in seconds corresponding to water delivered

'''
Setup the experiment: load the specified latent maze and configure the RPi pins
'''
## load specific latent maze
st_positions = pd.read_csv(f'{PROJECT_DIR}/data_in/{STATE_POSITION_CSV}').to_numpy()[:,1:] # row index matches the state indices; each row is [x_position, y_position]
adj_df = pd.read_csv(f'{PROJECT_DIR}/data_in/{MAZE_ENV_CSV}', header=None)
adj_df = adj_df.fillna(0) # map nans to zeros
adj_df = adj_df.astype(int) # convert floats (NaNs are imported as float) to int
adj_mat = adj_df.to_numpy() + adj_df.to_numpy().T # make the adjacency matrix symmetric and convert to numpy
adj_df = pd.DataFrame(adj_mat) # update this to be symmetric across the diagonal like the numpy version
## save the latent maze
# with open(f"{MAZE_ENV_CSV.replace('.csv', '')}.npy", 'wb') as f: np.save(f, adj_mat)
# with open(f"{STATE_POSITION_CSV.replace('.csv', '')}.npy", 'wb') as f: np.save(f, st_positions)

all_used_GPIO_pins = sorted(NOSE_POKE_PINS + LED_PINS + list(np.array(STEP_PINS).flatten()))
print('all GPIO pins currently in use:', all_used_GPIO_pins)
assert len(all_used_GPIO_pins) == len(set(all_used_GPIO_pins)), 'At least one GPIO pin is being reused by the code which is sketch...'

## Initialize the Raspberry Pi GPIO pins to the correct values
GPIO.setmode(GPIO.BCM)
for pin in NOSE_POKE_PINS:
    print(f'initializing NOSE_POKE_PINS: pin {pin}')
    GPIO.setup(pin, GPIO.IN, pull_up_down=GPIO.PUD_DOWN) # original #GPIO.setup(pin, GPIO.IN, pull_up_down=GPIO.PUD_UP) # attempting opposite polarity
print('\nMR: NOTE: the physical pull up resister warning has been investigated in some detail and deemed NO cause for concern. :) \n')

for pin in LED_PINS:
    print(f'initializing LED_PINS: pin {pin}')
    GPIO.setup(pin, GPIO.OUT)
    GPIO.output(pin, GPIO.LOW)

reward_pins = list(PORT_DIR_PIN_DICT.values())
for pin in reward_pins:
    print(f'initializing reward pins ({PORT_DIR_PIN_DICT.keys()}): pin {pin}')
    GPIO.setup(pin, GPIO.OUT)
    GPIO.output(pin, GPIO.LOW)

for motor_idx, motorPins in enumerate(STEP_PINS):
    for pin in motorPins:
        print(f'initializing motor STEP_PINS: motor {motor_idx} pin {pin}')
        GPIO.setup(pin,GPIO.OUT)
        GPIO.output(pin, False)
print('GPIO setup complete')

'''
Assorted less common functionality associated with setup/calibration
'''
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

#### port calibration
if RUN_MODE == 'calibrate rwds':
    # REWARD_DURATION = 0.01 # secs
    print('testing/calibrating ports: listen for 4 clicks')
    # sleep(5)
    N_REPS = 1
    for j in range(N_REPS):
        for port in ['N', 'S', 'W', 'E']:
        # for port in ['S']:
            for i in range(1):
                GPIO.output(PORT_DIR_PIN_DICT[port], GPIO.HIGH)
                print('on')
                sleep(REWARD_DURATION) #change the time to whatever gives the correct amount of water
                print(f'clicking: {port} via pin {PORT_DIR_PIN_DICT[port]} for {REWARD_DURATION} seconds')
                GPIO.output(PORT_DIR_PIN_DICT[port], GPIO.LOW)
                print('off')
                sleep(0.5)
    # sys.exit('rerun after commenting out the port calibration code.')

'''
Misc / house keeping
'''
def get_now_str(include_hms=False, include_ms=False):
    now = datetime.datetime.now()
    if include_hms and include_ms: return now.strftime('%y%m%d_%H%M%S') + '_' + now.strftime('%f')[:3]
    elif include_hms: return now.strftime('%y%m%d_%H%M%S')
    else: return now.strftime('%y%m%d')
    
def display_new_data(new_data, data_columns): 
    print(); 
    for d, c in zip(new_data, data_columns): 
        if c not in []: print(f"{c}: {d}")
    print()
    
def export_data_to_csv(data, data_columns, file_path):
    data_df = pd.DataFrame(data, columns=data_columns)
    data_df.to_csv(file_path, index=False)
    print(f"data saved to: {file_path}")
    
def plot_maze(adj_mat=adj_mat, title='', rwd_states=None, special_points=None, name = "test", output_dir= './data_out'):
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

    ## color the reward
    if type(rwd_states) == list: 
        for rwd_state in rwd_states:
            # print(f'rwd_state: {rwd_state}')
            # print(f"st_positions[rwd_state]: {st_positions[rwd_state]}") 
            plt.plot(st_positions[rwd_state][0], st_positions[rwd_state][1], 'o', color='cyan')
    else: plt.plot(st_positions[rwd_states][0], st_positions[rwd_states][1], 'o', color='cyan') # this is a misnomer cuz there is only one rwd state in this condition
    
    ## plot special points (list of nodes)
    if special_points is not None:
        for sp in special_points:
            plt.plot(st_positions[sp][0], st_positions[sp][1], 'o', color='red')
    
    # title = f"{title}\nrwds at: {rwd_states}"
    title = f"{title}"
    
    plt.title(title)
    plt.axis('equal')
    plt.axis('off')
    plt.savefig(f"{output_dir}/{get_now_str(include_hms=True, include_ms=False)}_{name}.png")
    plt.close('all')
    
'''
Experiment operation helper functions
'''

def lightControl(showLights):
    if showLights[0]: GPIO.output(LED_PINS[0], GPIO.HIGH)
    if showLights[1]: GPIO.output(LED_PINS[1], GPIO.HIGH)
    if showLights[2]: GPIO.output(LED_PINS[2], GPIO.HIGH)
    if showLights[3]: GPIO.output(LED_PINS[3], GPIO.HIGH)

def await_input():
    print('waiting for the next poke')
    while True:
        for pin in NOSE_POKE_PINS:
            #sleep(1)
            if GPIO.input(pin) == GPIO.HIGH:
                print(f'poke registered: {pin}: {GPIO.input(pin)}, {NOSE_POKE_PIN_DICT[pin]}')
                return [NOSE_POKE_PIN_DICT[pin], pin]

## approach
def options_to_true_ego(options, current_State, mouse_rotation, maze_rotation):
    true_ego = [] #length same as options, has arrays with [pos1, pos2, state]

    rot_dict = {0: ["left", "right", "forwards", "backwards"], 90: ["backwards", "forwards", "left", "right"], 180: [ "right", "left", "backwards", "forwards"], 270: ["forwards", "backwards", "right", "left"]}
    pos2options = rot_dict[mouse_rotation]

    if mouse_rotation == maze_rotation:              #(0,0), (90,90), (180, 180), (270, 270)
        pos1options = ["W", "E", "N", "S"]
    elif abs(mouse_rotation-90) == maze_rotation:    #(0,90), (90,0), (180, 90), (270,180)
        pos1options = ["S", "N", "W", "E"]
    elif (mouse_rotation+180)%360 == maze_rotation:  #(0, 180), (90, 270), (180,0), (270, 90)
        pos1options = ["E", "W", "S", "N"]
    else:                                           #(0, 270), (90, 180), (180, 270), (270, 0)
        pos1options = ["N", "S", "E", "W"]

    for state in options: #Max 3 states in options 240131: why max 3? Is this outdated from the binary code?
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

    # Get the nodes that are not yet in rewarded_nodes
    available_nodes = np.setdiff1d(all_nodes, rewarded_nodes)

    # Randomly select a new unique node
    while True:
        new_node = np.random.choice(available_nodes, 1)[0]
        if new_node != reached_node: break # enforce a strictly new node not just a random one

    # Add new node to rewarded_nodes
    rewarded_nodes = np.append(rewarded_nodes, new_node)

    # Sort for readability (optional)
    rewarded_nodes = np.sort(rewarded_nodes)
    
    print(f'swapping{reached_node} for {new_node}\nall rewarded nodes: {rewarded_nodes}')

    return rewarded_nodes

'''
motor/door control  
'''
motorsPos = [0,0,0,0] # initialize the motor/door positions; 240201: are these the safest values?

## global variables assigned and modified below
cumulative_steps_from_closed = [0, 0, 0, 0] # Global variables to track the steps from closed position (which is now the default)
motor_open_positions = deepcopy(cumulative_steps_from_closed)
all_motors = list(np.arange(len(STEP_PINS)))
selected_motors = deepcopy(all_motors)

## Order in which to drive the 4 wires for each motor; a different approach MIGHT work, but this appears to be complicated and maybe undesirable
STEPPER_SEQ = [[1,0,0,1], [1,0,0,0], [1,1,0,0], [0,1,0,0], [0,1,1,0], [0,0,1,0], [0,0,1,1], [0,0,0,1]]

def motorControl(motor,n,StepDir): # Set StepDir 1 or 2 for clockwise and -1 or -2 for counterclockwise
    StepDir = StepDir * MOTOR_DIRECTION[motor]
    StepCounter = 0
    motorsPos[motor] += (n * StepDir)
    StepCount = len(STEPPER_SEQ)
    for a in range(int(n/abs(StepDir))):
        for pin in range(0, 4):
            xpin = STEP_PINS[motor][pin]
            if STEPPER_SEQ[int(StepCounter)][pin] != 0:
                # print(" Enable GPIO %i" %(xpin))
                GPIO.output(xpin, True)
            else:
                GPIO.output(xpin, False)
        StepCounter += StepDir #if StepDir has magnitude 2, spins at 2x speed
        if (StepCounter>=StepCount):
            StepCounter = 0
        if (StepCounter<0):
            StepCounter = StepCount+StepDir
        time.sleep(MOTOR_STEP_DELAY)
            
def motorsSetRot(motorsPosNew, motors):
    global cumulative_steps_from_closed
    # Enforce input to be list even if single values
    if not isinstance(motors, list): motors = [motors]
    if not isinstance(motorsPosNew, list): motorsPosNew = [motorsPosNew] * len(motors)  # Replicate the position for each motor
    for index, motor in enumerate(motors):
        try:
            ## Calculate rotation needed for each motor
            rotNeeded = motorsPosNew[index] - cumulative_steps_from_closed[motor]
            # print(f"Setting motor: {motor} to position: {motorsPosNew[index]} via rotNeeded: {rotNeeded}")
            if rotNeeded != 0: 
                motorControl(motor, abs(rotNeeded), STEP_SPEED * rotNeeded/abs(rotNeeded))  # motor #, steps away, step size and direction
                cumulative_steps_from_closed[motor] += rotNeeded
        except IndexError as e: print(f"Error: {e}. Check if the motors and motorsPosNew lists are correctly paired.")
        except Exception as e:
            print(e)
            import pdb; pdb.set_trace()

# StepCounter = 0 # 240131: why is this here? why is it commented? I'm not sure what the correct usage is...

def moveToDefaultPosition(motors):
    defaultPosSteps = [int(STEPS_PER_FULL_ROTATION * MOTOR_CLOSED_POS_DEFAULT)] * 4  # Calculate default position in steps for each motor (hence *4)
    motorsSetRot(defaultPosSteps, motors)
    
def parse_motor_selection(selection):
    if selection == 'all': return [0, 1, 2, 3]
    else: return [int(digit) for digit in selection if digit.isdigit() and int(digit) < 4]

MOTOR_INSTRUCTIONS = """
Motor Control Instructions:
1. Select Motors: choose which motors to adjust by typing 'm' followed by the motor number(s) or type 'all' to select all of the motors.
   - Example: 'm0' for motor 0, 'm12' for motors 1 and 2, 'all' for all of them.
2. Close gate: c
3. Open Gates: o (letter not numeral) (UNTESTED!)
4. Step Motors: s
   - This will move the motor(s) in small predefined increments.
5. Set Position: set (UNTESTED!)
   - To set the current position as the new opened position for the selected motor(s), type 'set'.
   - This records the number of steps from the closed position to the current position.
Please follow these instructions to control the motors. The numerals correspond to the motor indices, which should be determined based on your hardware setup.
"""
def stepMotorsInteractive():
    global selected_motors
    global cumulative_steps_from_closed # for all the motors; rename this eventually to reflect this
    global motor_open_positions
    global all_motors

    try:
        while True:
            print(MOTOR_INSTRUCTIONS)
            print(f"\ncumulative_steps_from_closed: {cumulative_steps_from_closed}\nselected_motors: {selected_motors}\nmotor_open_positions: {motor_open_positions}")
            userInput = input("Enter command: ")
            command = userInput[0].lower()  # Get the command character
            motor_input = userInput[1:]  # Get the selection string, if any
            if command == 'm':
                selected_motors = parse_motor_selection(motor_input)
                print(f"Selected motors: {selected_motors}")
            elif userInput == 'all':
                print('here')
                selected_motors = deepcopy(all_motors)
                print(f"Selected motors: {selected_motors}")
            elif command == 'c':  # Close command, which is now the default position
                newPos = [MOTOR_CLOSED_POS_DEFAULT] * len(selected_motors)
                print(f"setting motors: {selected_motors} to position: {newPos}")
                motorsSetRot(newPos, selected_motors)
                for motor in selected_motors: cumulative_steps_from_closed[motor] = 0 # Reset cumulative steps for selected motors
            elif command == 'o':  # Open command
                newPos = deepcopy(motor_open_positions)
                motorsSetRot(newPos, selected_motors)
                # Set cumulative steps to the open position for selected motors
                for motor in selected_motors: cumulative_steps_from_closed[motor] = int(STEPS_PER_FULL_ROTATION * MOTOR_CLOSED_POS_DEFAULT)
            elif command == 's':  # Step command
                step_amount = int(STEPS_PER_FULL_ROTATION * INTERACTIVE_ROTATION)
                print('step_amount', step_amount)
                for motor in selected_motors:
                    this_motor_cumulative_steps_from_closed = cumulative_steps_from_closed[motor]
                    print('pre this_motor_cumulative_steps_from_closed: ', this_motor_cumulative_steps_from_closed)
                    new_position = step_amount + this_motor_cumulative_steps_from_closed
                    print('new_position: ', new_position) 
                    motorsSetRot(new_position, motor)
                    # cumulative_steps_from_closed[motor] = new_position
                    print('post cumulative_steps_from_closed: ', cumulative_steps_from_closed)
                    print('post this_motor_cumulative_steps_from_closed: ', this_motor_cumulative_steps_from_closed)
            elif command == 'set':
                print(f"Setting current positions as the closed positions for motors: {selected_motors}")
                for motor in selected_motors: motor_open_positions[motor] = cumulative_steps_from_closed[motor]
            elif command == 'f':
                print("Final positions accepted.")
                break # exit this user input loop!
            else: print('invalid user input!')
    except KeyboardInterrupt:
        print("\nExiting due to keyboard interrupt. NOTE: this is  not the intended usage...")
    finally:
        # Reset or stop all motors
        print("Resetting motors to safe state...")
        all_motors = list(np.arange(len(STEP_PINS)))
        motorsSetRot([MOTOR_CLOSED_POS_DEFAULT]*4, all_motors)  # Reset to safe position
        GPIO.cleanup()
            
#### motor/door positioning
moveToDefaultPosition(MOTORS_CLOSED_AT_START)

'''
Motors / doors: adjustment, debugging, or manual override.
    Uncomment to use.
'''
if RUN_MODE == 'set motors':
    stepMotorsInteractive() ## adjust the limits of the rotation for the given application
    print('cumulative_steps_from_closed: ', cumulative_steps_from_closed)
    print('motor_open_positions: ', motor_open_positions)        
    sys.exit('halting here for now. Comment this sys.exit() line to allow the experiment to actually run')

'''
Main experiment
'''
try: 
    experiment_started = False # allows graceful exit (see NOTE immediately below; this is used in the 'except' block)
    ## get animal ID from user input (NOTE: included in try so that the user can exit gracefully before entering an animal ID)
    while True:
        ANIMAL_ID = input("Enter the animal's ID number as 3 digits (zero padded): ")
        if ANIMAL_ID: break
        else: print('ID number can not be empty!')
    
    ## Create output directories
    new_data_dir = DATA_OUT_DIR + '_a' + ANIMAL_ID
    if not os.path.exists(DATA_OUT_DIR): os.mkdir(DATA_OUT_DIR)
    output_dir = f'{DATA_OUT_DIR}/{get_now_str(include_hms=True, include_ms=False)}_a{ANIMAL_ID}'
    os.mkdir(output_dir)
    
    ## Initialize the array and proportion
    n_states = len(st_positions) 
    all_nodes = np.arange(n_states)
    
    ## 240414: moving upstream
    # if REWARDED_STATES == 'pseudo random (without replacement)': rewarded_states = INITIAL_REWARDED_STATES
    # elif REWARDED_STATES != 'random': rewarded_states = REWARDED_STATES
    # else: # randomly selected 
        # n_select = max(math.floor(n_states * PROPORTION_STATES_REWARDED), 1) # Calculate the number of random integers to select
        # rewarded_states = list(np.sort(np.random.choice(all_nodes, size=n_select, replace=False)))  # Randomly select unique integers
    
    '''
    Run experiment 
    '''    
    experiment_started = True
    print('Starting experiment:\n   use control c to abort the experiment and save the data')
    plt.close('all')
    # plot_maze(adj_mat=adj_mat, rwd_states=rewarded_states, special_points=[START_STATE])
    
    #### initialize the agent/mouse in the maze
    n_rewards = 0
    n_reward_replenishments = 0
    maze_rotation = 0
    
    # prior_state = None
    state_tm1 = None # aka prior state
    state_tm1_end_time = 0  
    
    # prior_choice = None
    choice_tm1 = None
    
    state_t = START_STATE
    
    # state_tm1 = None # 240131: this should be state_tm2 instead, right?
    state_tm2 = None # 
    
    mouse_rotation = 0
    choice_t = [None, None]
    data = [] # list of rows formatted as lists ( --> pandas df --> csv file)
    data_columns = ['time', 'action_idx', 'lights_on', 'state', 'rewarded_states', 'choice', 'action', 
        'reward_time', 'n_rewards', 'mouse_rotation', 'maze_rotation', 'note', 'adjacency_file', 'st_positions_file', 'start_state']
    action_idx = 0
    poke_time = None
    door_time = time.time()
    while True:
        tic = time.time()
        
        plot_maze(adj_mat=adj_mat, title=f'a{ANIMAL_ID}', rwd_states=rewarded_states, special_points=[state_t], name = f"a{ANIMAL_ID}_latSt_{action_idx:04}", output_dir=output_dir)
        
        ## display world cues
        options = options_to_true_ego(np.argwhere(adj_mat[state_t]==1).flatten().tolist(), state_t, mouse_rotation, maze_rotation)
        
        toc = time.time()
        print(f"plotting and option computation time: {toc-tic}")
        
        print(f"State: {state_t}")
        
        
        tic = time.time()
        lights_on=[False, False, False, False]
        
        #### determine available options/arms
        doors_opened = []
        for option in options:
            if option[0] == "N":
                lights_on[0] = True
                doors_opened.append(0)
            if option[0] == "S":
                lights_on[1] = True
                doors_opened.append(1)
            if option[0] == "W":
                lights_on[2] = True
                doors_opened.append(2)
            if option[0] == "E":
                lights_on[3] = True
                doors_opened.append(3)
        lightControl(lights_on)
        
        toc = time.time()
        print(f"light control time: {toc-tic}")
        
        tic = time.time()
        ## change door configuration
        doors_closed = list(set(all_motors) - set(doors_opened))
        motorsSetRot(MOTOR_CLOSED_POS_DEFAULT, doors_closed) # close unavailable arms
        motorsSetRot(DOOR_OPEN_MOTOR_POS, doors_opened) # open available arms
        
        toc = time.time()
        print(f"door open and closing time: {toc-tic}")
        
        ## check for weird timing delay between poke and door changes
        if poke_time is not None: 
            door_time = time.time()
            poke_door_delay = door_time - poke_time
            # assert poke_door_delay < MAX_ALLOWED_POKE_DOOR_DELAY, f'poke door delay greater than 1 second: {poke_door_delay}'
            # if poke_door_delay > MAX_ALLOWED_POKE_DOOR_DELAY: print(f'poke door delay greater than {MAX_ALLOWED_POKE_DOOR_DELAY} seconds: {poke_door_delay}')
            print(f'poke door delay: {poke_door_delay} seconds')
    
        tic = time.time()
        ## get mouse action
        while True:
            print("awaiting input")
            print(options)
            print(motorsPos)
            choice_t = await_input()
            poke_time = time.time() 
            print(choice_t)
            action = None            
            if choice_t != choice_tm1 or poke_time - state_tm1_end_time > SAME_PORT_MIN_REPOKE_INTERVAL: break
            else: 
                print(f'animal repoked too early; TODO: log these states')
                sleep(MIN_INTER_POKE_INTERVAL)
        
        state_t = 'invalid' # assume invalid until selected otherwise
        for option in options:
            if choice_t[0] == option[0]:
                action = option[1]
                state_t = option[2]
        if state_t == 'invalid': ## abort experiment if the animal managed to poke a blocked/dead/inactive port.
            export_data_to_csv(data, data_columns, f'{output_dir}/{get_now_str(include_hms=True)}_a{ANIMAL_ID}_data.csv') ## export the data to a pandas dataframe csv
            print("state_tm1", state_tm1)
            print("state_t", state_t)
            print(f"options: {options}\nchoice_t: {choice_t}")
            sys.exit('Poke registered into dead arm. This is supposed to be impossible! There is presumably a bug or the mouse made it past a closed door...')
            
        toc = time.time()
        print(f"getting mouse action time: {toc-tic}")
        
        tic = time.time()
        """ 240221 temporary hack: close the door to the arm that was just poked so the animal doesn't escape, 
                then allow the rest of the loop to occur to set the other doors at a more leisurely pace
        """
        poked_arm_door_closed = ARM_TO_MOTOR_IDX_DICT[choice_t[0]]
        motorsSetRot(MOTOR_CLOSED_POS_DEFAULT, poked_arm_door_closed)
        toc = time.time()
        print(f"door locking in mouse time: {toc-tic}")
        
        if action == "forwards":
            mouse_rotation += 0 #comment out for allocentric
            maze_rotation += 180 #comment out for semi-egocentric
        elif action == "backwards":
            mouse_rotation += 180 #comment out for allocentric
            maze_rotation += 0 #comment out for semi-egocentric
        elif action == "left":
            mouse_rotation += 270 #comment out for allocentric
            maze_rotation += 90  #comment out for semi-egocentric
        elif action == "right":
            mouse_rotation += 90 #comment out for allocentric
            maze_rotation += 270 #comment out for semi-egocentric
        maze_rotation %= 360 #comment out for semi-egocentric
        mouse_rotation %= 360 #comment out for allocentric      
        
        # tic = time.time()
        ## reward entrance into one of the rewarded_states
        reward_time = 'NA'
        if state_t in rewarded_states: # rwd just for being in that state
            print(f"\nMouse rewarded for reaching state {state_t} in rewarded_states ({rewarded_states})")
            #Open reward for __ seconds
            print(f"choice: {choice_t}")
            print(f"reward dispensed from: {POKE_PIN_TO_RWD_PIN_MAP[choice_t[1]]}\n")
            GPIO.output(POKE_PIN_TO_RWD_PIN_MAP[choice_t[1]], GPIO.HIGH)
            sleep(REWARD_DURATION) #change the time to whatever gives the correct amount of water
            GPIO.output(POKE_PIN_TO_RWD_PIN_MAP[choice_t[1]], GPIO.LOW)
            reward_time = get_now_str(include_hms=True)
            n_rewards +=1
            
            ## 240507 new version: do nothing to the list of rewards
            
            
            ## mid version
            # print(f'pre update rewarded states: {rewarded_states}')
            # if REWARDED_STATES == 'asdfasfd':
                # rewarded_states = list(update_rewarded_nodes(state_t, rewarded_states, all_nodes))            
            # elif REWARDED_STATES == 'pseudo random (without replacement)':
                # if len(rewarded_states) == RESET_WHEN_N_RWDS_REMAINING + 1: # + 1 cuz the animal has reached this reward location but we haven't yet removed it from the list 
                    # print("\n\n\nANIMAL FOUND ALL STATES! :D\n\n\n")
                    # if np.mod(n_reward_replenishments, 2) == 0: rewarded_states = REPLENISHED_REWARDED_STATES_0 # animal has found all rewards! --> rebaiting all states
                    # else: rewarded_states = REPLENISHED_REWARDED_STATES_1
                    # n_reward_replenishments += 1
                # if state_t in rewarded_states: rewarded_states.remove(state_t)
            # elif 'side alternation w/ refinement' in REWARDED_STATES:
                # if len(rewarded_states) == RESET_WHEN_N_RWDS_REMAINING + 1: # see above rationale
                    # print("\n\n\nANIMAL FOUND ALL STATES! :D\n\n\n")
                    # if n_reward_replenishments == 0: rewarded_states = deepcopy(REPLENISHED_REWARDED_STATES_0) + deepcopy(rewarded_states) # animal has found all rewards! --> rebaiting states
                    # elif n_reward_replenishments == 1: rewarded_states = deepcopy(REPLENISHED_REWARDED_STATES_1) + deepcopy(rewarded_states) # animal has found all rewards! --> rebaiting states
                    # elif n_reward_replenishments == 2: 
                        # print('completed reward cycle!')
                        # rewarded_states = deepcopy(REPLENISHED_REWARDED_STATES_2) + deepcopy(rewarded_states) # animal has found all rewards! --> rebaiting states
                        # n_reward_replenishments = 0 # this will immediately be incremented to 1 below tho
                    # n_reward_replenishments += 1
                # if state_t in rewarded_states: rewarded_states.remove(state_t)    
            # print(f'post update rewarded states: {rewarded_states}')
            
            ## very early version
            # if REWARDED_STATES != 'random': rewarded_states = REWARDED_STATES
            # else: # randomly selected 
                # n_select = max(math.floor(n_states * PROPORTION_STATES_REWARDED), 1) # Calculate the number of random integers to select
                # rewarded_states = list(np.sort(np.random.choice(all_nodes, size=n_select, replace=False)))  # Randomly select unique integers
        
        ## immediately turn off the lights after the poke so that the action seems causal but try to prevent premature re-pokes
        # for led_pin in LED_PINS: GPIO.output(led_pin, GPIO.LOW)
        # sleep(MIN_INTER_POKE_INTERVAL) 
        
        # toc = time.time()
        # print(f"reward dispensing time: {toc-tic}")
        
        # tic = time.time()
        if state_tm1 == state_t: ## abort experiment if the animal managed to poke a blocked/dead/inactive port.
            export_data_to_csv(data, data_columns, f'{output_dir}/{get_now_str(include_hms=True)}_a{ANIMAL_ID}_data.csv') ## export the data to a pandas dataframe csv
            print("state_tm1", state_tm1)
            print("state_t", state_t)
            print(f'\n\nWARNING: state_tm1 ({state_tm1}) == state_tstate_tm1 ({state_tstate_tm1}) == state_t\n\n')
            # sys.exit('Poke registered into dead arm. This is supposed to be impossible! There is presumably a bug or the mouse made it past a closed door...')
            note = 'warning: state_tm1 == state_t'
        else: note = '' 
        
        ## end of experiment iteration house keeping
        new_data = [get_now_str(include_hms=True), action_idx, lights_on, state_t, rewarded_states, choice_t, action, reward_time, 
            n_rewards, mouse_rotation, maze_rotation, note, MAZE_ENV_CSV, STATE_POSITION_CSV, START_STATE]
        data.append(new_data) # update the data list
        
        display_new_data(new_data, data_columns) # display the new data
        
        ## advance the state, choice, and respective histories
        state_tm1 = deepcopy(state_t)
        state_tm2 = deepcopy(state_tm1)
        choice_tm1 = deepcopy(choice_t)
        state_tm1_end_time = time.time()
        action_idx += 1
        
        # toc = time.time()
        # print(f"book keeping time: {toc-tic}")

except KeyboardInterrupt:
    print("Resetting motors to safe state...")
    all_motors = list(np.arange(len(STEP_PINS)))
    motorsSetRot([MOTOR_CLOSED_POS_DEFAULT]*4, all_motors)  # Reset to safe position
    GPIO.cleanup()
    if experiment_started: export_data_to_csv(data, data_columns, f'{output_dir}/{get_now_str(include_hms=True)}_a{ANIMAL_ID}_data.csv') ## export the data to a pandas dataframe csv
    else: print('\nexperiment aborted before generating data --> no data saved')

finally:
    print("Resetting motors to safe state...")
    all_motors = list(np.arange(len(STEP_PINS)))
    motorsSetRot([MOTOR_CLOSED_POS_DEFAULT]*4, all_motors)  # Reset to safe position
    GPIO.cleanup()
    print('GPIO pins supposedly reset back to their default/initial state --> exiting!')
    if experiment_started: export_data_to_csv(data, data_columns, f'{output_dir}/{get_now_str(include_hms=True)}_a{ANIMAL_ID}_data.csv') ## export the data to a pandas dataframe csv
    else: print('\nexperiment aborted before generating data --> no data saved')
    

    
