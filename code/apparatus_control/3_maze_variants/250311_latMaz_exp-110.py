""" Latent Maze Experiment script: runs the experiment
Authors: anonymized for review
Version history:
    - Initial draft (Colab URL redacted for double-blind review)
    - Code/experiments before 240820 did not use the 'utils_latMaz.py' file """

## load functions from utils_latMaz.py
import sys
sys.path.append('./')
print('240820: I assume utils_latMaz is in the same directory as the notebook')
from utils_latMaz import *

'''USER_PARAMETERS, i.e. experiment config. All settings defined via ALL_CAPS variables below'''
# use: 'set motors', 'calibrate rwds', or 'run experiment'
RUN_MODE = 'run experiment'
# RUN_MODE = 'set motors' 
# RUN_MODE = 'calibrate rwds'

## input/output paths
PROJECT_DIR = '/home/anonymized-user/Desktop/latent_maze-RPI' # this file and the utils_latMaz.py file should be in this directory
DATA_OUT_DIR = './data_out'

#### RPi pin assignments
## NOTE: THESE NUMBERS ARE GPIO not RPI PINS!!!!!
NOSE_POKE_PINS = [2, 10, 11, 14] #These values need to be changed to whichever pins correlate with the input from the beam breaks
NOSE_POKE_PIN_DICT = {NOSE_POKE_PINS[0]: 'N', NOSE_POKE_PINS[1]: 'S', NOSE_POKE_PINS[2]: 'W', NOSE_POKE_PINS[3]: 'E'} ## map port locations to RPi pins
PORT_DIR_PIN_DICT = {'N': 21, 'S': 1, 'W': 7, 'E': 8} # GPIO pin values for the solenoids/rewards associated with each arm
POKE_PIN_TO_RWD_PIN_MAP = {2: 21, 10: 1, 11: 7, 14: 8}
ARM_TO_MOTOR_IDX_DICT = {'N': 0, 'S': 1, 'W': 2, 'E': 3}
LED_PINS = [3, 9, 0, 15] #These values need to be changed to whichever pins correlate with the output to the LEDs

#### state transition delays
SAME_PORT_MIN_REPOKE_INTERVAL = 90 # 30 until 250205: 241118 45 -; 45 <- 240906 60; #240415 45 -> 60; 240304 2nd sessions --> 30 -> 45 240301 45 -> 30 # 240226 30 -> 45; 240224 15 --> 30; 240222 10 -> 12 
MIN_INTER_POKE_INTERVAL = 0.5 # in seconds; 240820: 0.5 -> 2
MAX_ALLOWED_POKE_DOOR_DELAY = 8 # 250306: 6 crashed it... 240906; 240820: 6 -> 3; ie purposely end the experiment if this delay is too long (cuz animals will be able to enter inactive arms before the doors close)

#### motors (for doors blocking invalid arms)
STEP_PINS = [[17,18,27,22],[23,24,25,4],[13,12,6,5],[20,26,16,19]] # motors: list of lists, 1 list per motor ### original m0, 1, 2, 3
MOTOR_CLOSED_POS_DEFAULT = 0.0  # Default position as a fraction of STEPS_PER_FULL_ROTATION
MOTORS_CLOSED_AT_START = [0, 1, 2, 3]  # A list of motor indexes to operate on, default is all motors
MOTOR_DIRECTION = MOTOR_DIRECTION = [-1, 1, -1, -1] # half under and half over: [1, 1, 1, -1] # 1 is the default direction; -1 reverses that; N S W E (i.e. aiming for same order conventions as elsewhere)
STEPS_PER_FULL_ROTATION = 1024 
# INTERACTIVE_ROTATION = 0.35 # Give as fraction of a full rotation # until 250305
INTERACTIVE_ROTATION = 0.3 # Give as fraction of a full rotation
DOOR_OPEN_MOTOR_POS = int(STEPS_PER_FULL_ROTATION * INTERACTIVE_ROTATION)
MOTOR_STEP_DELAY = 2/float(1000) # in seconds; device example/recommended value: 10/float(1000); presumably in ms; ~2/float(1000) appears to be min before it jams
STEP_SPEED = -2 # 240131: example code suggest that 1 and 2 are valid speeds (240201: verified? 3 with 2 ms MOTOR_STEP_DELAY appears to be jagged and not faster). Negative values reverse the direction. 
PRE_DOOR_MOVE_DELAY = 0.15
POST_DOOR_MOVE_DELAY = 0.7# secs: 240905: trying to prevent doors jamming due to 1 door not being out of the way of the next one that's moving. 

""" maze environment specification """
#### Select a specific latent maze selection and define it via the if/elifs below
# MAZE_SELECTED = 'small less aliased' # only east closed
# MAZE_SELECTED = 'small binary' # only north open
# MAZE_SELECTED = 'eLife binary'
# MAZE_SELECTED = '240218'
# MAZE_SELECTED = '240507-5x9'
# MAZE_SELECTED = '240722-mop0full'
# MAZE_SELECTED = '240824-mop0half'
# MAZE_SELECTED = '240901-mop001half'
# MAZE_SELECTED = '250228-mop001half-circleGone'
MAZE_SELECTED = '250305-mop002'

if MAZE_SELECTED == '250305-mop002':
    MAZE_ENV_CSV = '250310_latent_maze_011-5x5-adjacency.csv'
    STATE_POSITION_CSV = '240507_latent_maze_004-5x5-state_positions.csv'
    # START_STATE = 0
    START_STATE = 18
    RWD_STATE = None # not used if selecting locations randomly via REWARDED_STATES = 'random' below
# if MAZE_SELECTED == '250228-mop001half-circleGone':
#     MAZE_ENV_CSV = '250228_latMaz007-mp002circleGone-adjacency.csv'
#     STATE_POSITION_CSV = '240901_latMaz007-mop001half-state_positions.csv'
#     # START_STATE = 20
#     START_STATE = 29
    # RWD_STATE = None # not used if selecting locations randomly via REWARDED_STATES = 'random' below
# elif MAZE_SELECTED == '240901-mop001half':
    # MAZE_ENV_CSV = '240901_latMaz007-mop001half-adjacency.csv'
    # STATE_POSITION_CSV = '240901_latMaz007-mop001half-state_positions.csv'
    # # START_STATE = 20
    # START_STATE = 40
    # RWD_STATE = None # not used if selecting locations randomly via REWARDED_STATES = 'random' below
# elif MAZE_SELECTED == '240824-mop0half':
#     MAZE_ENV_CSV = '240824_latent_maze_006-mop0half-adjacency.csv'
#     STATE_POSITION_CSV = '240824_latent_maze_006-mop0half-state_positions.csv'
#     # START_STATE = 17
#     # START_STATE = 19
#     # START_STATE = 6
#     # START_STATE = 22
#     START_STATE = 23
#     RWD_STATE = None # not used if selecting locations randomly via REWARDED_STATES = 'random' below
# elif MAZE_SELECTED == '240722-mop0full':
#     MAZE_ENV_CSV = '240722_latent_maze_005-mop0full-adjacency.csv'
#     STATE_POSITION_CSV = '240722_latent_maze_005-mop0full-state_positions.csv'
#     START_STATE = 17
#     RWD_STATE = 7 # not used if selecting locations randomly via REWARDED_STATES = 'random' below
# elif MAZE_SELECTED == '240507-5x9':
#     MAZE_ENV_CSV = '240510_latent_maze_003b_5x9-adjacency.csv'
#     STATE_POSITION_CSV = '240507_latent_maze_003b_5x9-state_positions.csv'
#     START_STATE = 0
#     RWD_STATE = 7 # not used if selecting locations randomly via REWARDED_STATES = 'random' below
# elif MAZE_SELECTED == '240218':
#     MAZE_ENV_CSV = '240218_intuition_maze_003-adjacency.csv'
#     STATE_POSITION_CSV = '240218_intuition_maze_003-state_positions.csv'
#     START_STATE = 0
#     RWD_STATE = 7 # not used if selecting locations randomly via REWARDED_STATES = 'random' below
# elif MAZE_SELECTED == 'small less aliased': ## less aliased graph 001
#     MAZE_ENV_CSV = '230921_latent_maze_002_less_aliased_n_no_deadends-adjacency_to_csv.csv'
#     STATE_POSITION_CSV = '230921_latent_maze_002_less_aliased_n_no_deadends-state_positions.csv'
#     START_STATE = 3
#     RWD_STATE =12
# elif MAZE_SELECTED == 'small binary': ## small binary maze
#     MAZE_ENV_CSV = '220918_3_level_binary_tree-adjacency.csv' # only need to define the upper triangle
#     STATE_POSITION_CSV = '220918_3_level_binary_tree-state_positions.csv'
#     START_STATE = 15
#     RWD_STATE = 3
# elif MAZE_SELECTED == 'eLife binary': ## eLife '21 binary maze
#     MAZE_ENV_CSV = '2201028_6_level_binary_tree-adjacency.csv' # only need to define the upper triangle
#     STATE_POSITION_CSV = '2201023_6_level_binary_tree-state_positions.csv'
#     START_STATE = 123
#     RWD_STATE = 4
else: sys.exit('MAZE_SELECTED not found!')

""" reward parameters """
RWD_PROP_OF_INIT = 0.8
REWARD_DURATION = {'N': RWD_PROP_OF_INIT * 0.0736, 'S': RWD_PROP_OF_INIT * 0.0556, 'W': RWD_PROP_OF_INIT * 0.0634, 'E': RWD_PROP_OF_INIT * 0.0580} # 250217 see fluid script and online doc instead# until 250202 {'N': RWD_PROP_OF_INIT * 0.0646, 'S': RWD_PROP_OF_INIT * 0.0476, 'W': RWD_PROP_OF_INIT * 0.0532, 'E': RWD_PROP_OF_INIT * 0.0523} # <- 240901; {'N': 0.1707, 'S': 0.1167, 'W': 0.1342, 'E': 0.1434}  <- 240828; {'N': 0.2308, 'S': 0.1555, 'W': 0.1807, 'E': 0.1924} # in seconds

# REWARD_DURATION = 0.4 # 240722: 0.015 -> 0.1; 240517: 0.02 -> 0.015 240514: 0.05 -> 0.01 (maybe too low) -> 0.02; 240512: 0.15 -> 0.05; 240510: 0.15-> 0.05; 240417: 0.15 --> 0.1; 240416: 0.3 -> 0.15; 240415: 0.15 -> 0.3; 240414: 0.075 -> 0.15; 240301: 0.1 -> 0.075; 240229 0.15 -> 0.1; 240227 0.07 -> 0.15; 240224 0.07 -> 0.05; 240223 (after a002): 0.1 -> 0.07; 240222: 0.15 -> 0.1 solenoid open time in seconds corresponding to water delivered

#### rewards: specify a list of states to reward or 'random' (note: that this then uses PROPORTION_STATES_REWARDED to select random states to reward)
# REWARDED_STATES = [RWD_STATE, START_STATE, 2, 12, 11] 
# REWARDED_STATES = [RWD_STATE] 
# REWARDED_STATES = 'random'
# REWARDED_STATES = 'pseudo random (without replacement)' 
# REWARDED_STATES = 'side alternation w/ refinement ver2.5' 
# REWARDED_STATES = 'stable long path'
# REWARDED_STATES = 'fully specified; replenish only after all found'
REWARDED_STATES = 'all; replenish only after RESET_WHEN_N_RWDS_REMAINING'
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
elif REWARDED_STATES == 'fully specified; replenish only after all found':
    ##(240724)
    # WEST_HALF_REWARDED_STATES = [0, 5, 29, 34]
    # EAST_HALF_REWARDED_STATES = [35, 38, 67, 71] 
    ##(240726 - )
    # WEST_HALF_REWARDED_STATES = [0, 13, 5, 36, 29, 28]
    # EAST_HALF_REWARDED_STATES = [35, 38, 66, 67]
    ##240728 - 2 per section
    # WEST_HALF_REWARDED_STATES = [0, 13, 3, 16, 23, 34]
    # EAST_HALF_REWARDED_STATES = [35, 52, 37, 54, 66, 67]
    ##240728 - 4 per section
    WEST_HALF_REWARDED_STATES = [0, 2, 11, 13, 3, 5, 14, 16, 23, 28, 29, 34]
    EAST_HALF_REWARDED_STATES = [39, 40, 47, 48, 37, 38, 53, 54, 65, 66, 67, 68]
    FREEWAY_REWARDED_STATES = [18, 19, 21, 55, 56, 59]
    INITIAL_REWARDED_STATES = deepcopy(WEST_HALF_REWARDED_STATES + EAST_HALF_REWARDED_STATES + FREEWAY_REWARDED_STATES)
    REPLENISHED_REWARDED_STATES_0 = deepcopy(INITIAL_REWARDED_STATES)
    REPLENISHED_REWARDED_STATES_1 = deepcopy(INITIAL_REWARDED_STATES)
elif REWARDED_STATES == 'all; replenish only after RESET_WHEN_N_RWDS_REMAINING': ## 240824    
    INITIAL_REWARDED_STATES = list(set(list(np.arange(24+1))) - set([2, 3, 4, 8, 9, 10, 14]))
    REPLENISHED_REWARDED_STATES_0 = deepcopy(INITIAL_REWARDED_STATES)
    RESET_WHEN_N_RWDS_REMAINING = 1
    N_RWD_REPLENISHMENTS = 20
    
rewarded_states = deepcopy(INITIAL_REWARDED_STATES) # initialize the actual first set of rewarded states
print('INITIAL_REWARDED STATES: ', type(rewarded_states), rewarded_states)

'''Setup the experiment: load the specified latent maze and configure the RPi pins'''
adj_path = f'{PROJECT_DIR}/data_in/{MAZE_ENV_CSV}'
st_pos_path = f'{PROJECT_DIR}/data_in/{STATE_POSITION_CSV}'
adj_mat, st_positions = load_maze(adj_path, st_pos_path) ## load specific latent maze
motor_info_dict = {'step_pins': STEP_PINS, 'port_dir_pin_dict': PORT_DIR_PIN_DICT}
setup_gpio_pins(NOSE_POKE_PINS, LED_PINS, motor_info_dict) ## setup the RPi
                
'''
motor/door control (240820: not moving this to utils_latMaz.py yet due to the use of global variables) 
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
                sleep(PRE_DOOR_MOVE_DELAY) # 240906 added to prevent doors jamming during closure...
                motorControl(motor, abs(rotNeeded), STEP_SPEED * rotNeeded/abs(rotNeeded))  # motor #, steps away, step size and direction
                cumulative_steps_from_closed[motor] += rotNeeded
                sleep(POST_DOOR_MOVE_DELAY) # 240906 added to prevent doors jamming during closure...
        except IndexError as e: print(f"Error: {e}. Check if the motors and motorsPosNew lists are correctly paired.")
        except Exception as e:
            print(e)
            import pdb; pdb.set_trace()

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

if RUN_MODE == 'set motors': # Motors / doors: adjustment, debugging, or manual override.
    stepMotorsInteractive() ## adjust the limits of the rotation for the given application
    print('cumulative_steps_from_closed: ', cumulative_steps_from_closed)
    print('motor_open_positions: ', motor_open_positions)        
    sys.exit('halting here for now. Comment this sys.exit() line to allow the experiment to actually run')
elif RUN_MODE == 'calibrate rwds': # port calibration
    # REWARD_DURATION = 0.01 # secs
    print('testing/calibrating ports: listen for clicks')
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
    sys.exit('rerun after commenting out the port calibration code.')
else: print('running experiment')

''' Main experiment '''
try: 
    already_saved = False
    experiment_started = False # allows graceful exit (see NOTE immediately below; this is used in the 'except' block)
    while True: # get animal ID from user input (NOTE: included in try so that the user can exit gracefully before entering an animal ID)
        ANIMAL_ID = input("Enter the animal's ID number as 3 digits (zero padded): ")
        if ANIMAL_ID: break
        else: print('ID number can not be empty!')
    
    ## Create output directories
    output_dir = f'{DATA_OUT_DIR}/{get_now_str(hms=True, ms=False)}_a{ANIMAL_ID}'
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"output dir: {output_dir}")
    bash_img_display_process = show_new_pngs_in_dir(output_dir) # start bash script to monitor the output directory and display new images to the small screen in the behavior camera's view
    
    ## Initialize the array and proportion; 240820: only used in some legacy reward configurations
    n_states = len(st_positions) 
    all_nodes = np.arange(n_states)
    
    ''' Run experiment '''    
    experiment_started = True
    print('Starting experiment:\n   use control c to abort the experiment and save the data')
    plt.close('all')
    
    #### initialize the agent/mouse in the maze
    n_rewards = 0
    n_reward_replenishments = 0
    maze_rotation = 0
    state_tm1 = None # aka prior state (t minus 1)
    state_tm1_end_time = 0  
    choice_tm1 = None # i.e. prior choice (t minus 1)
    state_t = START_STATE
    state_tm2 = None 
    
    mouse_rotation = 0
    choice_t = [None, None]
    data = [] # list of rows formatted as lists (later converted --> pandas df --> csv file)
    data_columns = ['time', 'action_idx', 'lights_on', 'state', 'rewarded_states', 'choice', 'action', 
        'reward_time', 'n_rewards', 'mouse_rotation', 'maze_rotation', 'note', 'adjacency_file', 'st_positions_file', 'start_state', 'reward_duration']
    action_idx = 0
    poke_time = None
    door_time = time.time()
    poked_arm_door_closed = None
    while True:
        note = ''
        tic = time.time()
        state_start_time = get_now_str(hms=True)
        plot_maze(adj_mat, st_positions, title=f'date/time: {state_start_time}\nmouse: a{ANIMAL_ID}; action: {action_idx:04}', 
                  rwd_states=rewarded_states, current_position=state_t, current_position_marker_size=40, filename = f"{state_start_time}_a{ANIMAL_ID}_latSt{action_idx:04}", output_dir=output_dir)

        ## display world cues
        options = options_to_true_ego(np.argwhere(adj_mat[state_t]==1).flatten().tolist(), state_t, mouse_rotation, maze_rotation, st_positions)
        toc = time.time()
        # print(f"plotting and option computation time: {toc-tic}")
        # print(f"State: {state_t}")
        tic = time.time()
        lights_on=[False, False, False, False]
        
        ## determine available options/arms
        doors_opened = []
        for option in options:
            if option[0] == "N": lights_on[0] = True; doors_opened.append(0)
            if option[0] == "S": lights_on[1] = True; doors_opened.append(1)
            if option[0] == "W": lights_on[2] = True; doors_opened.append(2)
            if option[0] == "E": lights_on[3] = True; doors_opened.append(3)
        lightControl(lights_on, LED_PINS) # turn on the lights for the available arms
        toc = time.time()
        # print(f"light control time: {toc-tic}")
        
        ## 241118: remove arm poked from list of open doors and handle that separately
        if poked_arm_door_closed is not None: doors_opened = list(set(doors_opened) - set([poked_arm_door_closed]))
        
        ## change door configuration
        tic = time.time()
        doors_closed = list(set(all_motors) - set(doors_opened))
        motorsSetRot(MOTOR_CLOSED_POS_DEFAULT, doors_closed) # close unavailable arms
        motorsSetRot(DOOR_OPEN_MOTOR_POS, doors_opened) # open available arms
        
        if poked_arm_door_closed is not None: motorsSetRot(DOOR_OPEN_MOTOR_POS, [poked_arm_door_closed])  # 241118: opening poke arm after other doors
        
        toc = time.time()
        # print(f"door open and closing time: {toc-tic}")
        ## check for weird timing delay between poke and door changes
        if poke_time is not None: 
            door_time = time.time()
            poke_door_delay = door_time - poke_time
            assert poke_door_delay < MAX_ALLOWED_POKE_DOOR_DELAY, f'poke door delay greater than 1 second: {poke_door_delay}'
            # print(f'poke door delay: {poke_door_delay} seconds')
    
        ## get mouse action
        tic = time.time()
        while True:
            # print(f"awaiting input\n  options: {options}\n  motorsPos: {motorsPos}")
            choice_t = await_input(NOSE_POKE_PINS, NOSE_POKE_PIN_DICT)
            poke_time = time.time() 
            print(choice_t)
            if choice_t != choice_tm1 or poke_time - state_tm1_end_time > SAME_PORT_MIN_REPOKE_INTERVAL: break
            else: 
                # print(f'animal repoked too early; TODO: log these states')
                sleep(MIN_INTER_POKE_INTERVAL)
        
        """ update the state of the mouse """
        state_t = 'invalid' # assume invalid until selected otherwise
        action = None
        arm_chosen = choice_t[0]
        for option in options:
            if arm_chosen == option[0]:
                action = option[1]
                state_t = option[2]
        if state_t == 'invalid': ## abort experiment if the animal managed to poke a blocked/dead/inactive port.
            export_data_to_csv(data, data_columns, f'{output_dir}/{get_now_str(hms=True)}_a{ANIMAL_ID}_data.csv') ## export the data to a pandas dataframe csv
            print("state_tm1", state_tm1); print("state_t", state_t); print(f"options: {options}\nchoice_t: {choice_t}")
            sys.exit('Poke registered into dead arm. This is supposed to be impossible! There is presumably a bug or the mouse made it past a closed door...')            
        toc = time.time()
        # print(f"getting mouse action time: {toc-tic}")
        
        """ 240820: this works well; leave it in indefinitely. 240221 temporary hack: close the door to the arm that was just poked so the animal doesn't escape, then allow the rest of the loop to occur to set the other doors at a more leisurely pace """
        tic = time.time()
        poked_arm_door_closed = ARM_TO_MOTOR_IDX_DICT[choice_t[0]]
        motorsSetRot(MOTOR_CLOSED_POS_DEFAULT, poked_arm_door_closed)
        toc = time.time(); #print(f"door locking in mouse time: {toc-tic}")
        
        ## update the orientation of the mouse and maze
        """ 240820: angles are defined in the opposite direction to what I expected, ie all code... --> I checked this and it's correct via 240820_test_ZL_mappings.ipynb"""
        if action == "forwards":
            mouse_rotation += 0 #comment out for allocentric
            maze_rotation += 180 #comment out for semi-egocentric # 240820: what is semi-egocentric?
        elif action == "backwards":
            mouse_rotation += 180 #comment out for allocentric
            maze_rotation += 0 
        elif action == "left":
            mouse_rotation += 270 #comment out for allocentric
            maze_rotation += 90  
        elif action == "right":
            mouse_rotation += 90 #comment out for allocentric
            maze_rotation += 270 
        maze_rotation %= 360 # normalize to 0-360 degrees
        mouse_rotation %= 360 
        
        ## reward entrance into one of the rewarded_states
        reward_time = 'NA'
        if state_t in rewarded_states: # rwd just for being in that state
            print(f"\nMouse rewarded for reaching state {state_t} in rewarded_states ({rewarded_states})"); print(f"choice: {choice_t}"); print(f"reward dispensed from: {POKE_PIN_TO_RWD_PIN_MAP[choice_t[1]]}\n")
            ## Open reward for REWARD_DURATION seconds
            GPIO.output(POKE_PIN_TO_RWD_PIN_MAP[choice_t[1]], GPIO.HIGH) 
            
            try: sleep(REWARD_DURATION[arm_chosen]) # assume we're using the calibrated port durations which are normalized to give the same rwd volume via dict
            except: sleep(REWARD_DURATION) # change the time to whatever gives the correct amount of water
            
            GPIO.output(POKE_PIN_TO_RWD_PIN_MAP[choice_t[1]], GPIO.LOW)
            reward_time = get_now_str(hms=True)
            n_rewards +=1
            
            ## 240507 new version: do nothing to the list of rewards

            # n_select = max(math.floor(n_states * PROPORTION_STATES_REWARDED), 1) # Calculate the number of random integers to select
                # rewarded_states = list(np.sort(np.random.choice(all_nodes, size=n_select, replace=False)))  # Randomly select unique integers
            
            ## mid version
            print(f'pre update rewarded states: {rewarded_states}')
            if REWARDED_STATES == 'random': rewarded_states = list(update_rewarded_nodes(state_t, rewarded_states, all_nodes)) # only selects new rewarded states from non-rewarded states            
            elif REWARDED_STATES == 'random-legacy': # doesn't consider which rewards are already rewarded etc
                n_select = max(math.floor(n_states * PROPORTION_STATES_REWARDED), 1) # Calculate the number of random integers to select
                rewarded_states = list(np.sort(np.random.choice(all_nodes, size=n_select, replace=False)))  # Randomly select unique integers
            elif REWARDED_STATES == 'pseudo random (without replacement)':
                if len(rewarded_states) == RESET_WHEN_N_RWDS_REMAINING + 1: # + 1 cuz the animal has reached this reward location but we haven't yet removed it from the list 
                    print("\n\n\nANIMAL FOUND ALL STATES! :D\n\n\n")
                    if np.mod(n_reward_replenishments, 2) == 0: rewarded_states = REPLENISHED_REWARDED_STATES_0 # animal has found all rewards! --> rebaiting all states
                    else: rewarded_states = REPLENISHED_REWARDED_STATES_1
                    n_reward_replenishments += 1
                if state_t in rewarded_states: rewarded_states.remove(state_t)
            elif 'side alternation w/ refinement' in REWARDED_STATES:
                if len(rewarded_states) == RESET_WHEN_N_RWDS_REMAINING + 1: # + 1 cuz the animal has reached this reward location but we haven't yet removed it from the list 
                    print("\n\n\nANIMAL FOUND ALL STATES! :D\n\n\n")
                    if n_reward_replenishments == 0: rewarded_states = deepcopy(REPLENISHED_REWARDED_STATES_0) + deepcopy(rewarded_states) # animal has found all rewards! --> rebaiting states
                    elif n_reward_replenishments == 1: rewarded_states = deepcopy(REPLENISHED_REWARDED_STATES_1) + deepcopy(rewarded_states) 
                    elif n_reward_replenishments == 2: 
                        print('completed reward cycle!')
                        rewarded_states = deepcopy(REPLENISHED_REWARDED_STATES_2) + deepcopy(rewarded_states) # animal has found all rewards! --> rebaiting states
                        n_reward_replenishments = 0 # this will immediately be incremented to 1 below tho
                    n_reward_replenishments += 1
                if state_t in rewarded_states: rewarded_states.remove(state_t)    
            elif REWARDED_STATES == 'all; replenish only after RESET_WHEN_N_RWDS_REMAINING':
                # if len(rewarded_states) == RESET_WHEN_N_RWDS_REMAINING + 1: # + 1 cuz the animal has reached this reward location but we haven't yet removed it from the list 
                if len(rewarded_states) <= RESET_WHEN_N_RWDS_REMAINING + 1: # + 1 cuz the animal has reached this reward location but we haven't yet removed it from the list                     
                    print("\n\n\nANIMAL FOUND ALL STATES! :D\n\n\n")
                    if n_reward_replenishments < N_RWD_REPLENISHMENTS: rewarded_states = deepcopy(REPLENISHED_REWARDED_STATES_0)
                    # if np.mod(n_reward_replenishments, 2) == 0: rewarded_states = REPLENISHED_REWARDED_STATES_0 # animal has found all rewards! --> rebaiting all states
                    # else: rewarded_states = REPLENISHED_REWARDED_STATES_1
                    n_reward_replenishments += 1
                if state_t in rewarded_states: rewarded_states.remove(state_t)
            else: rewarded_states = REWARDED_STATES
            print(f'post update rewarded states: {rewarded_states}')
            
        ## immediately turn off the lights after the poke so that the action seems causal but try to prevent premature re-pokes
        # for led_pin in LED_PINS: GPIO.output(led_pin, GPIO.LOW)
        # sleep(MIN_INTER_POKE_INTERVAL) 
        
        if state_tm1 == state_t: ## abort experiment if the animal managed to poke a blocked/dead/inactive port.
            export_data_to_csv(data, data_columns, f'{output_dir}/{get_now_str(hms=True)}_a{ANIMAL_ID}_data.csv') ## export the data to a pandas dataframe csv
            print("state_tm1", state_tm1)
            print("state_t", state_t)
            print(f'\n\nWARNING: state_tm1 ({state_tm1}) == state_t ({state_t}) == state_t\n\n')
            sys.exit('Poke registered into dead arm. This is supposed to be impossible! There is presumably a bug or the mouse made it past a closed door...') 
        
        ## end of experiment iteration house keeping
        new_data = [get_now_str(hms=True), action_idx, lights_on, state_t, deepcopy(rewarded_states), choice_t, action, reward_time, n_rewards, mouse_rotation, maze_rotation, note, MAZE_ENV_CSV, STATE_POSITION_CSV, START_STATE, REWARD_DURATION]
        data.append(new_data) # update the data list
        display_new_data(new_data, data_columns) # display the new data
        
        ## advance the state, choice, and respective histories
        state_tm1 = deepcopy(state_t)
        state_tm2 = deepcopy(state_tm1)
        choice_tm1 = deepcopy(choice_t)
        state_tm1_end_time = time.time()
        action_idx += 1

        """ Experiment termination, saving, and cleanup; 240820: except vs finally code left duplicated due to possible complexities associated with the global motor variables """
except KeyboardInterrupt:
    print("Experiment terminated via KeyboardInterrupt --> resetting motors to safe state...")
    all_motors = list(np.arange(len(STEP_PINS)))
    motorsSetRot([MOTOR_CLOSED_POS_DEFAULT]*4, all_motors)  # Reset to safe position
    GPIO.cleanup()
    if experiment_started:
        csv_data_path = f'{output_dir}/{get_now_str(hms=True)}_a{ANIMAL_ID}_data.csv' 
        export_data_to_csv(data, data_columns, csv_data_path) ## export the data to a pandas dataframe csv
        save_params(namespace=globals(), output_dir=output_dir, extra_info_in_name='including_rwd_details')
        plot_session_counts(csv_data_path, output_dir=output_dir)
    else: print('\nexperiment aborted before generating data --> no data saved')
    
    # Ensure the bash process is terminated
    if bash_img_display_process.poll() is None:  # Check if the process is still running
        print('Terminating the bash process...')
        bash_img_display_process.terminate()  # Send SIGTERM
        try:
            bash_img_display_process.wait(timeout=5)  # Wait for process to terminate
        except subprocess.TimeoutExpired:
            print('Bash process did not terminate in time. Killing it...')
            bash_img_display_process.kill()  # Force kill if not terminated
    already_saved = True

finally:
    print("Experiment terminated via finally (possible issue?) --> resetting motors to safe state...")
    if not already_saved:
        all_motors = list(np.arange(len(STEP_PINS)))
        motorsSetRot([MOTOR_CLOSED_POS_DEFAULT]*4, all_motors)  # Reset to safe position
        GPIO.cleanup()
        print('GPIO pins supposedly reset back to their default/initial state --> exiting!')
        if experiment_started:
            csv_data_path = f'{output_dir}/{get_now_str(hms=True)}_a{ANIMAL_ID}_data.csv'  
            export_data_to_csv(data, data_columns, csv_data_path) ## export the data to a pandas dataframe csv
            save_params(namespace=globals(), output_dir=output_dir, extra_info_in_name='including_rwd_details')
            plot_session_counts(csv_data_path, output_dir=output_dir)
        else: print('\nexperiment aborted before generating data --> no data saved')
        
        # Ensure the bash process is terminated
        if bash_img_display_process.poll() is None:  # Check if the process is still running
            print('Terminating the bash process...')
            bash_img_display_process.terminate()  # Send SIGTERM
            try:
                bash_img_display_process.wait(timeout=5)  # Wait for process to terminate
            except subprocess.TimeoutExpired:
                print('Bash process did not terminate in time. Killing it...')
                bash_img_display_process.kill()  # Force kill if not terminated
    

    
