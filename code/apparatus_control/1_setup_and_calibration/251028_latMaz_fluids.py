
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
# RUN_MODE = 'run experiment' # 240410 use separate script to descrease chances of error
# RUN_MODE = 'set motors' # 240410 use separate script to descrease chances of error
# RUN_MODE = 'calibrate rwds'

#### RPi pin assignments
## NOTE: THESE NUMBERS ARE GPIO# not RPI PINS!!!!!
NOSE_POKE_PINS = [2, 10, 11, 14] #These values need to be changed to whichever pins correlate with the input from the beam breaks
LED_PINS = [3, 9, 0, 15] #These values need to be changed to whichever pins correlate with the output to the LEDs
PORT_DIR_PIN_DICT = {'N': 21, 'S': 1, 'W': 7, 'E': 8} # GPIO pin values for the solenoids/rewards associated with each arm
POKE_PIN_TO_RWD_PIN_MAP = {2: 21, 10: 1, 11: 7, 14: 8}
NOSE_POKE_PIN_DICT = {NOSE_POKE_PINS[0]: 'N', NOSE_POKE_PINS[1]: 'S', NOSE_POKE_PINS[2]: 'W', NOSE_POKE_PINS[3]: 'E'} ## map port locations to RPi pins
ARM_TO_MOTOR_IDX_DICT = {'N': 0, 'S': 1, 'W': 2, 'E': 3}

#### state transition delays
SAME_PORT_MIN_REPOKE_INTERVAL = 45 #240304 2nd sessions --> 30 -> 45 240301 45 -> 30 # 240226 30 -> 45; 240224 15 --> 30; 240222 10 -> 12 
MIN_INTER_POKE_INTERVAL = 0.5 # in seconds
MAX_ALLOWED_POKE_DOOR_DELAY = 8 # ie purposely end the experiment if this delay is too long (cuz animals will be able to enter inactive arms before the doors close)

#### motors (for doors blocking invalid arms)
STEP_PINS = [[17,18,27,22],[23,24,25,4],[13,12,6,5],[20,26,16,19]] # motors: list of lists, 1 list per motor ### original m0, 1, 2, 3
MOTOR_CLOSED_POS_DEFAULT = 0.0  # Default position as a fraction of STEPS_PER_FULL_ROTATION
MOTORS_CLOSED_AT_START = [0, 1, 2, 3]  # A list of motor indexes to operate on, default is all motors

# MOTOR_DIRECTION = [1, -1, 1, 1] # THIS IS CONSISTENT DIRECTION, LEAVE FOR REFERENCE! 1 is the default direction; -1 reverses that; N S W E (i.e. aiming for same order conventions as elsewhere)
MOTOR_DIRECTION = [1, -1, 1, -1] # 1 is the default direction; -1 reverses that; N S W E (i.e. aiming for same order conventions as elsewhere) ALL CORRECT BUT SOUTH
MOTOR_DIRECTION = [1, 1, 1, -1] # 1 is the default direction; -1 reverses that; N S W E (i.e. aiming for same order conventions as elsewhere)
    
# REWARD_DURATION = SET BELOW FOR CALIBRATION 0.075 # 240301: 0.1 -> 0.075; 240229 0.15 -> 0.1; 240227 0.07 -> 0.15; 240224 0.07 -> 0.05; 240223 (after a002): 0.1 -> 0.07; 240222: 0.15 -> 0.1 solenoid open time in seconds corresponding to water delivered

'''
Setup the experiment: load the specified latent maze and configure the RPi pins
'''

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


"""
241113 calibration

N: 200 p at 0.0625 = 2.9 mL --> 0.0145 mL/rwd
S: 200 p at 0.05 = 3.15 mL --> 0.01575 mL/rwd
W: 200 pulses at 0.055 = 3.1 mL --> 0.0155 mL/rwd 
E: 200 p at 0.0575 = 3.3 mL --> 0.0165 mL/rwd
"""

RUN_MODE = 'flush'
# RUN_MODE = 'calibrate rwds'

sleep(1)
#### port flushing
if RUN_MODE == 'flush':
    try: 
        PORTS_FLUSHED = ['N', 'S', 'W', 'E']
        # PORTS_FLUSHED = ['W']
        # PORTS_FLUSHED = ['N',]
        N_REPS = 100 # CHANGE HERE
        print('testing/calibrating ports: listen for 4 clicks')
        for i in range(N_REPS):
            for port in PORTS_FLUSHED:
                GPIO.output(PORT_DIR_PIN_DICT[port], GPIO.HIGH)
                print('on')
                print(f'opening: {port} via pin {PORT_DIR_PIN_DICT[port]} for flushing')
                
                sleep(3)
            
                GPIO.output(PORT_DIR_PIN_DICT[port], GPIO.LOW)
                sleep(0.1)
            print('off')
            sleep(0.1)

    except KeyboardInterrupt:
        for port in ['N', 'S', 'W', 'E']: GPIO.output(PORT_DIR_PIN_DICT[port], GPIO.LOW)
        print("Resetting motors to safe state...")
        all_motors = list(np.arange(len(STEP_PINS)))
        GPIO.cleanup()
        

    finally:
        for port in ['N', 'S', 'W', 'E']: GPIO.output(PORT_DIR_PIN_DICT[port], GPIO.LOW)
        print("Resetting motors to safe state...")
        all_motors = list(np.arange(len(STEP_PINS)))
        GPIO.cleanup()
        print('GPIO pins supposedly reset back to their default/initial state --> exiting!')
        

elif RUN_MODE == 'calibrate rwds':
    
    N_CYCLES = 1 # repeats of all ports
    
    
    N_REPS = 25 # CHANGE HERE # repeats for single port
    # N_REPS = 50
    # N_REPS = 1 # CHANGE HERE

    # PORTS_SELECTED = ['N']
    # PORTS_SELECTED = ['S']
    # PORTS_SELECTED = ['W']
    # PORTS_SELECTED = ['E']
    PORTS_SELECTED = ['N', 'S', 'W', 'E']

    RWD_PROP_OF_INIT = 1.0
    REWARD_DURATION = {'N': RWD_PROP_OF_INIT * 0.067, 'S': RWD_PROP_OF_INIT * 0.06, 'W': RWD_PROP_OF_INIT * 0.062, 'E': RWD_PROP_OF_INIT * 0.066} #  water
    
    print('testing/calibrating ports: listen for 4 clicks')
    sleep(1)
    for j in range(N_CYCLES):
        for port in PORTS_SELECTED:
        # for port in ['N']:
            for i in range(N_REPS):
                GPIO.output(PORT_DIR_PIN_DICT[port], GPIO.HIGH)
                print('on')
                
                # sleep(REWARD_DURATION) #change the time to whatever gives the correct amount of water
                sleep(REWARD_DURATION[port]) #change the time to whatever gives the correct amount of water
                
                GPIO.output(PORT_DIR_PIN_DICT[port], GPIO.LOW)
                print(f'clicking: {port} via pin {PORT_DIR_PIN_DICT[port]} for {REWARD_DURATION} seconds')

                print('off')
                sleep(0.5)

sys.exit('Use the designated script (NOT THIS) to run the actual experiment, if satisfied with the rewards.')

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

    ## plot special points (list of nodes)
    if special_points is not None:
        for sp in special_points:
            plt.plot(st_positions[sp][0], st_positions[sp][1], 'o', color='red')

    ## color the reward
    if type(rwd_states) == list: 
        for rwd_state in rwd_states:
            # print(f'rwd_state: {rwd_state}')
            # print(f"st_positions[rwd_state]: {st_positions[rwd_state]}") 
            plt.plot(st_positions[rwd_state][0], st_positions[rwd_state][1], 'o', color='cyan')
    else: plt.plot(st_positions[rwd_states][0], st_positions[rwd_states][1], 'o', color='cyan') # this is a misnomer cuz there is only one rwd state in this condition
    
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

    if REWARDED_STATES == 'pseudo random (without replacement)': rewarded_states = INITIAL_REWARDED_STATES
    elif REWARDED_STATES != 'random': rewarded_states = REWARDED_STATES
    else: # randomly selected 
        n_select = max(math.floor(n_states * PROPORTION_STATES_REWARDED), 1) # Calculate the number of random integers to select
        rewarded_states = list(np.sort(np.random.choice(all_nodes, size=n_select, replace=False)))  # Randomly select unique integers
    
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
            
            print(f'pre update rewarded states: {rewarded_states}')
            if REWARDED_STATES == 'asdfasfd':
                rewarded_states = list(update_rewarded_nodes(state_t, rewarded_states, all_nodes))            
            elif REWARDED_STATES == 'pseudo random (without replacement)':
                if len(rewarded_states) == 1: 
                    print("\n\n\nANIMAL FOUND ALL STATES! :D\n\n\n")
                    if np.mod(n_reward_replenishments, 2) == 0: rewarded_states = REPLENISHED_REWARDED_STATES_0 # animal has found all rewards! --> rebaiting all states
                    else: rewarded_states = REPLENISHED_REWARDED_STATES_1
                    n_reward_replenishments += 1
                if state_t in rewarded_states: rewarded_states.remove(state_t)
            print(f'post update rewarded states: {rewarded_states}')
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
    for port in ['N', 'S', 'W', 'E']: GPIO.output(PORT_DIR_PIN_DICT[port], GPIO.LOW)
    print("Resetting motors to safe state...")
    all_motors = list(np.arange(len(STEP_PINS)))
    motorsSetRot([MOTOR_CLOSED_POS_DEFAULT]*4, all_motors)  # Reset to safe position
    GPIO.cleanup()
    if experiment_started: export_data_to_csv(data, data_columns, f'{output_dir}/{get_now_str(include_hms=True)}_a{ANIMAL_ID}_data.csv') ## export the data to a pandas dataframe csv
    else: print('\nexperiment aborted before generating data --> no data saved')

finally:
    for port in ['N', 'S', 'W', 'E']: GPIO.output(PORT_DIR_PIN_DICT[port], GPIO.LOW)
    print("Resetting motors to safe state...")
    all_motors = list(np.arange(len(STEP_PINS)))
    motorsSetRot([MOTOR_CLOSED_POS_DEFAULT]*4, all_motors)  # Reset to safe position
    GPIO.cleanup()
    print('GPIO pins supposedly reset back to their default/initial state --> exiting!')
    if experiment_started: export_data_to_csv(data, data_columns, f'{output_dir}/{get_now_str(include_hms=True)}_a{ANIMAL_ID}_data.csv') ## export the data to a pandas dataframe csv
    else: print('\nexperiment aborted before generating data --> no data saved')
    

    
