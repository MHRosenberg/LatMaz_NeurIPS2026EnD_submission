from utils_latMaz import *
import numbers


## just for debugging purposes
def has_i_equal_i_minus_2(lst):
    return any(lst[i] == lst[i - 2] for i in range(2, len(lst)))
## test cases
# print(has_i_equal_i_minus_2([3, 1, 3]))      # True
# print(has_i_equal_i_minus_2([1, 2, 3, 4, 3]))   # False
# print(has_i_equal_i_minus_2([1, 2, 3, 4, 5]))   # False

#### 250825 Author: found a issue in the parse_exp_csv_to_observations_WD function that makes we worried about the heading computations in v3. 
## Explicitly aligning this approach to that of the parse_exp_csv_to_observations_WD.
def generate_trajectory_v4(start_state, n_states_visited, adj_mat, st_positions, policy_dict, initial_heading=np.pi/2):
    """ attempting a maximally generalized approach:
            Pass a generic all-inclusive policy_dict dictionary into this function.
                Required keys:
                    'func' - a policy function that returns a tuple of (next_states, next_actions_allo, next_actions_ego) and reads all arguments via a **kwarg approach via prms
                    'prms' - a dictionary of parameters to pass to the policy function via the **kwarg approach
            The policy_dict['func'] then uses only what it needs from the input dictionary policy_dict['prms'] sample """

    ## initialize agent heading in latent maze (compute real heading below)
    heading_latent_radians_initial = initial_heading

    actions_taken_ego = []
    actions_taken_allo_latent = []
    actions_taken_allo_real = []
    states_visited = [start_state]
    while len(states_visited) < n_states_visited:
        # print(f"\n{policy_dict['policy_category']}; state {len(states_visited)} of {n_states_visited}")
        
        ## assign current state
        st_t = deepcopy(states_visited[-1]) # assign the current state to the last of the states_visited
        
        """ compute headings """
        ## assign current heading_latent
        # if len(actions_taken_allo) == 0: 
        if len(states_visited) == 1: 
            assert len(actions_taken_allo_latent) == len(actions_taken_allo_real) == len(actions_taken_ego) == 0 , 'Author: if the number of states visited is only one, then no actions should have been taken yet'
            heading_latent_radians = heading_latent_radians_initial
        else: heading_latent_radians = allo_radian_map_dict[actions_taken_allo_latent[-1]] # the agent's heading is always the same as the last allocentric action taken

        ## assign current heading_real via parity trick to compute real heading (ie the physical arm that would need to be poked in the actual physical maze)
        if len(states_visited) % 2 == 1: # Author different than parse_exp_csv_to_observations_WD cuz we handled the first row separately upfront; same logic: flips back and forth so we can just use the parity 
            heading_real_radians = heading_latent_radians
        else: 
            heading_real_radians = heading_latent_radians + np.pi
        heading_real_radians = np.mod(heading_real_radians, 2*np.pi) # avoid gradual accumulation of heading

        if policy_dict['policy_category'] == 'random':
            if len(states_visited) > 1: policy_dict['prms']['st_tm1'] = deepcopy(states_visited[-2]) # if the agent has visited at least 2 states, set the previous state to the second to last state
            else: policy_dict['prms']['st_tm1'] = 'does not yet exist'

        elif policy_dict['policy_category'] in ['latent_state_biased', 'observation_action_biased']:
            adj_states = get_adj_states(st_t, adj_mat)
            policy_dict['prms']['adj_states'] = deepcopy(adj_states) 

            ## compute observations necessary for next policy step via updating the policy_dict['prms'] 
            if policy_dict['policy_category'] == 'observation_action_biased':
                
                #### compute initial observations
                ## compute latent and real allocentric observations
                observations_adj_nodes_allo_latent = deepcopy([displacement_to_compass_heading(st_positions[adj_node] - st_positions[st_t]) for adj_node in adj_states]) 
                observations_adj_nodes_allo_real = deepcopy([get_direction_from_radian(np.mod(allo_radian_map_dict[obs] + (heading_real_radians - heading_latent_radians), 2*np.pi)) for obs in observations_adj_nodes_allo_latent]) ## allocentric real    
                ## compute egocentric observations
                observations_adj_nodes_ego = deepcopy([get_ego_direction(st_positions[st_t], st_positions[adj_node], heading_latent_radians) for adj_node in adj_states]) 

                # policy_dict['prms']['obs_allo'], policy_dict['prms']['obs_ego'] = observations_adj_nodes_allo_latent, observations_adj_nodes_ego # 250826 Author: prior version doesn't robustly differentiate b/w allo_real and allo_latent
                policy_dict['prms']['obs_allo_latent'] = observations_adj_nodes_allo_latent
                policy_dict['prms']['obs_allo_real'] = observations_adj_nodes_allo_real
                policy_dict['prms']['obs_ego'] = observations_adj_nodes_ego

        elif policy_dict['policy_category'] == 'action_biased':
            pass # 250826 Author: checking but I think the policy_dict['prms'] are already set up correctly for this policy
            # policy_dict['prms']['heading_latent'] = heading_latent_radians
            # policy_dict['prms']['heading_real'] = heading_real_radians
         
        else: raise NotImplementedError('Author: currently only supporting the observation_action_biased policy in this function...') 
        
        """
        EXECUTE POLICY: take action(s) and get new states; returns: a tuple of ([ego actions], [allo actions], [states]), often w/ lists of len 1
        """
        # print('n states visited: ', len(states_visited))

        new_states, new_actions_allo_latent, new_actions_allo_real, new_actions_ego = policy_dict['func'](st_t=st_t, heading_latent=heading_latent_radians, heading_real=heading_real_radians, adj_mat=adj_mat, st_positions=st_positions, **policy_dict['prms'])
        

        states_visited.extend(new_states)
        actions_taken_ego.extend(new_actions_ego) ## store the results
        actions_taken_allo_latent.extend(new_actions_allo_latent)
        actions_taken_allo_real.extend(new_actions_allo_real)
        
        ## update the heading_latent (heading_real updated earlier in the loop)
        assert len(states_visited) >= 2, f'Author: expected at least 2 states visited.. states_visted: {states_visited}'
        heading_latent_radians = displacement_to_compass_heading(st_positions[states_visited[-1]] - st_positions[states_visited[-2]])

        # if 'avoid_reversal' in policy_dict['prms'].keys() and policy_dict['prms']['avoid_reversal'] and has_i_equal_i_minus_2(states_visited): 
        #     # print('reversal prevention might be violated'); print(f'states_visited: {states_visited}'); print(f"st_t: {st_t}; new_states: {new_states}; actions_allo: {actions_allo}; actions_ego: {actions_ego}")
        #     print('reversal prevention might be violated'); print(f'states_visited: {states_visited}'); print(f"st_t: {st_t}; new_states: {new_states}")
        #     import pdb; pdb.set_trace() # debugging point to check the reversal prevention logic

        ## sanity check the length of the returned lists
        assert len(states_visited) - 1 == len(actions_taken_ego) == len(actions_taken_allo_latent) == len(actions_taken_allo_real), 'Author: the histories have gotten out of sync somehow...'

    return states_visited, actions_taken_allo_latent, actions_taken_allo_real, actions_taken_ego


def policy_action_biased_trunc_v3_minimal(**prms):
    MISSING_KEY_PROVIDED_PROB = 0.01 # probability assigned for missing keys WHEN the agent is stuck

    def truncate_action_seq_last_first(action_seq_prob_dict): # Author3 approach 
        merged = defaultdict(float)
        for k, v in action_seq_prob_dict.items():
            if len(k) > 1: merged[k[:-1]] += v
            else: merged[k] += v
        merged = {k: v/sum(merged.values()) for k, v in merged.items()}
        return dict(merged)

    ## select the appropriate action probability dictionary based on the representation    
    if prms['representation'] == 'allo_latent':
        action_prob_dict = prms['action_prob_dict_allo_latent']
    elif prms['representation'] == 'allo_real':
        action_prob_dict = prms['action_prob_dict_allo_real']
    elif prms['representation'] == 'ego':
        action_prob_dict = prms['action_prob_dict_ego']
    
    attempted_action_groups = []
    while True: # keep attempting actions until the environment supports it

        next_states = []
        next_actions_allo_latent = []
        next_actions_allo_real = []
        next_actions_ego = []

        ## truncate the action sequences (from the end) if all have been attempted
        run_dict_sanity_checks = True
        if len(set(attempted_action_groups)) == len(action_prob_dict.keys()):
            action_prob_dict = truncate_action_seq_last_first(action_prob_dict)
            attempted_action_groups = [] # reset the attempted action groups since we have truncated the action sequences 
        else: run_dict_sanity_checks = False

        ## check that the agent is not stuck such that no yoked-mouse action sequences are valid (rare but possible)
        if run_dict_sanity_checks:
            dct = deepcopy(action_prob_dict)
            key_lengths = [len(k) for k in dct.keys()]
            assert len(set(key_lengths)) == 1, f'Author: action sequence lengths are not all the same: {key_lengths}'
            key_length = key_lengths[0]
            n_keys = len(dct.keys())
            if key_length == 1 and n_keys < 4:
                
                if 'allo' in prms['representation']: 
                    missing_keys = list(set(['N', 'S', 'W', 'E']) - set(dct.keys()))
                elif 'ego' in prms['representation']: 
                    missing_keys = list(set(['F', 'B', 'L', 'R']) - set(dct.keys()))

                for key in missing_keys: 
                    dct[key] = MISSING_KEY_PROVIDED_PROB
                dct = {k: v/sum(dct.values()) for k, v in dct.items()} # normalize the probabilities
                assert len(dct.keys()) == 4, f'Author: failed to find 4 keys as expected: {dct.keys()}'
            
            action_prob_dict = dct ## update the prms with the modified action_prob_dict

        ## get a candidate action sequence (might include violations that are checked below)
        while True: 
            candidate_action_seq = np.random.choice(list(action_prob_dict.keys()), size=1, replace=True, p=list(action_prob_dict.values()))[0] # get candidate next action sequence
            if candidate_action_seq not in attempted_action_groups: 
                break
        attempted_action_groups.append(candidate_action_seq)
        
        ## sanity check the candidate action sequence
        if 'allo' in prms['representation']: 
            assert not set(candidate_action_seq) - set('NSEW'), f'Author: non-empty set suggests allo vs ego mismatch: candidate {set(candidate_action_seq)} vs expected NSEW' 
        elif 'ego' in prms['representation']: 
            assert not set(candidate_action_seq) - set('FBLR'), f'Author: non-empty set suggests allo vs ego mismatch: candidate {set(candidate_action_seq)} vs expected FBLR'    
        assert len(candidate_action_seq) > 0, f"Author: candidate_action_seq is empty for representation {prms['representation']}. This should not happen."
        
        #### check that the candidate action sequence is supported by the environment, i.e. that the appropriate maze graph edges exist
        
        ## current state from the calling side
        st_t = int(prms['st_t'])
        assert type(st_t) == int, f'Author: prms["st_t"] should be an integer but got {type(prms["st_t"])}'
        st_positions = deepcopy(prms['st_positions'])
        ## get the current headings from the calling side
        heading_real_radian = deepcopy(prms['heading_real'])
        heading_latent_radian = deepcopy(prms['heading_latent'])
        heading_real_latent_diff = np.mod(heading_real_radian - heading_latent_radian, 2*np.pi)

        # heading_real_radian = np.mod(next_heading_latent_radian + (heading_real_radian - heading_latent_radian), 2*np.pi)
        accept_candidate_action_seq = True
        for i, action in enumerate(candidate_action_seq):

            assert type(heading_latent_radian) != str and type(heading_real_radian) != str, f'Author: heading_latent_radian and heading_real should be a float but got {type(heading_latent_radian)} and {type(heading_real_radian)}; probably needs to be converted backs'
            st_tp1 = env_transition_func(st_t, action, prms['representation'], adj_mat=prms['adj_mat'], st_positions=prms['st_positions'],
                                    heading_real=heading_real_radian, heading_latent=heading_latent_radian)
            if st_tp1 == 'action_unavailable': # abort and resample the candidate_action_seq if a violation is detected by the transition function
                ## sanity checks
                if 'allo' in prms['representation']: 
                    assert set(candidate_action_seq) - set('NSWE') == set(), f'Author: non-empty set suggests representation mismatch: candidate {set(candidate_action_seq)}'
                elif 'ego' in prms['representation']:
                    assert set(candidate_action_seq) - set('FBLR') == set(), f'Author: non-empty set suggests representation mismatch: candidate {set(candidate_action_seq)}'
                accept_candidate_action_seq = False
                break

            next_actions_ego.append(get_ego_direction(st_positions[st_t], st_positions[st_tp1], heading_latent_radian))
            next_heading_latent_str = displacement_to_compass_heading(st_positions[st_tp1] - st_positions[st_t])
            next_actions_allo_latent.append(next_heading_latent_str) # use the str version here

            ## convert the next heading to real coordinates
            assert type(next_heading_latent_str) == str, f'Author: next_heading_latent_str should be a string but got {type(next_heading_latent_str)}'
            next_heading_latent_radian = allo_radian_map_dict[next_heading_latent_str]
            assert type(heading_latent_radian) != str and type(heading_real_radian) != str, f'Author: heading_latent_radian and heading_real should be a float but got {type(heading_latent_radian)} and {type(heading_real_radian)}; probably needs to be converted backs'
            
            heading_real_latent_diff = np.mod(heading_real_radian - heading_latent_radian, 2*np.pi)
            next_heading_real_radian = np.mod(next_heading_latent_radian + heading_real_latent_diff, 2*np.pi)

            next_heading_real_str = get_direction_from_radian(np.mod(next_heading_latent_radian + (heading_real_latent_diff), 2*np.pi))
            # print(f'heading_real_str: {next_heading_real_str}')
            next_actions_allo_real.append(next_heading_real_str)
            
            ## sanity check the action vs representation stuff makes sense
            assert type(heading_latent_radian) != str and type(heading_real_radian) != str, f'Author: heading_latent_radian and heading_real should be a float but got {type(heading_latent_radian)} and {type(heading_real_radian)}; probably needs to be converted backs'
            if prms['representation'] == 'ego':
                inferred_action_ego = get_ego_direction(st_positions[st_t], st_positions[st_tp1], heading_latent_radian)
                assert action == inferred_action_ego, f'Author: action {action} should agree w/ computed inferred action_ego {inferred_action_ego}' 
            elif prms['representation'] == 'allo_latent':
                inferred_action_allo_latent = displacement_to_compass_heading(st_positions[st_tp1] - st_positions[st_t])
                assert action == inferred_action_allo_latent, f'Author: action {action} should agree w/ computed inferred_action_allo_latent {inferred_action_allo_latent}'
            elif prms['representation'] == 'allo_real':
                # inferred_action_allo_real = get_direction_from_radian(np.mod(allo_radian_map_dict[action] + (heading_real_radian - heading_latent_radian), 2*np.pi))
                inferred_action_allo_real = get_direction_from_radian(np.mod(next_heading_real_radian, 2*np.pi))
                
                # assert action == inferred_action_allo_real, f'Author: action {action} should agree w/ computed inferred_action_allo_real {inferred_action_allo_real}'
                if action != inferred_action_allo_real: 
                    print(f'WARNING: action {i} of candidate sequence does not agree w/ computed inferred_action_allo_real {action} vs {inferred_action_allo_real}, respectively.')

            ## updates for the next loop iteration
            heading_latent_radian = next_heading_latent_radian
            heading_real_radian = next_heading_real_radian + np.pi
            next_states.append(int(st_tp1)) # store the new state
            assert type(st_tp1) == int, f'Author: st_tp1 should be an integer but got {type(st_tp1)}'
            st_t = st_tp1 # advance the current state to the new state to prepare for the next loop iteration

        if accept_candidate_action_seq: 
            return next_states, next_actions_allo_latent, next_actions_allo_real, next_actions_ego


def policy_latent_state_biased(**prms):

    node_counts_dict = prms['node_counts_dict']
    adj_nodes = prms['adj_states']
    st_t = prms['st_t']

    adj_node_counts_dict = {k: v for k, v in node_counts_dict.items() if k in adj_nodes} # filter out the nodes that are not adjacent to the current state
    counts_per_adj_node = adj_node_counts_dict.values()

    #### compute the probabilities of selecting available nodes
    if sum(counts_per_adj_node) == 0: ## set to equal probabilities, if the mouse never visited the available nodes
        adj_node_prob_dict = {k: 1/len(adj_node_counts_dict) for k, _ in adj_node_counts_dict.items()} 
    else: ## normalize the probabilities, if the sum is not zero
        adj_node_prob_dict = {k: v/sum(adj_node_counts_dict.values()) for k, v in adj_node_counts_dict.items()} 

    st_tp1 = np.random.choice(list(adj_node_prob_dict.keys()), p=list(adj_node_prob_dict.values())) # select the next state based on the probabilities of the adjacent nodes
    
    ## 250827 Author: murky legacy approach leaving for a bit longer for reference
    # if len(adj_node_prob_dict) > 0: st_tp1 = np.random.choice(list(adj_node_prob_dict.keys()), p=list(adj_node_prob_dict.values())) # select the next state based on the probabilities of the adjacent nodes
    # else: st_tp1 = np.random.choice(list(prms['adj_states'])) # if no valid nodes, select among the adjacent nodes uniformly if 

    action_allo_latent = displacement_to_compass_heading(prms['st_positions'][st_tp1] - prms['st_positions'][st_t])
    action_allo_real = get_direction_from_radian(np.mod(allo_radian_map_dict[action_allo_latent] + (prms['heading_real'] - prms['heading_latent']), 2*np.pi))
    action_ego = get_ego_direction(prms['st_positions'][st_t], prms['st_positions'][st_tp1], prms['heading_latent'])
    return [int(st_tp1)], [action_allo_latent], [action_allo_real], [action_ego]


def policy_obs_action_biased(**prms):
    ## note: use "get" approach for default values for **prms

    obs = prms[f"obs_{prms['representation']}"] # either allo or ego observation based on the representation
    # print(f'obs: {obs}')
    
    if frozenset(obs) not in prms['obs_action_nested_dict'].keys(): # edge case where the observation is not in the dictionary
        # print(f"WARNING: obs {obs} not found in the obs_action_nested_dict keys: {prms['obs_action_nested_dict'].keys()}. Sampling uniformly among adjacent states.")
        sample_adj_states_uniformly = True
    else: # default case where the observation is in the dictionary
        # print('obs found in obs_action_nested_dict')
        prob_obs_action_dict = deepcopy(prms['obs_action_nested_dict'][frozenset(obs)])
        sample_adj_states_uniformly = False
        
    while True:
        
        if not sample_adj_states_uniformly: 
            action = np.random.choice(list(prob_obs_action_dict.keys()), p=list(prob_obs_action_dict.values()))
        
            ## compute the next state based on the action selected
            # st_tp1 = env_transition_func(prms['st_t'], str(action), prms['representation'], heading=prms['heading'], adj_mat=prms['adj_mat'], st_positions=prms['st_positions']) # 250828 Author: legacy
            st_tp1 = env_transition_func(prms['st_t'], str(action), prms['representation'], adj_mat=prms['adj_mat'], st_positions=prms['st_positions'],
                                            heading_real=prms['heading_real'], heading_latent=prms['heading_latent'])

        ## sanity check
        if sample_adj_states_uniformly:
            # plot_maze(prms['adj_mat'], prms['st_positions'], title='', rwd_states=None, current_position=prms['st_t'], filename = "unspecified", output_dir= './data_out', close=True, show=True, save=False)
            st_tp1 = np.random.choice(get_adj_states(prms['st_t'], prms['adj_mat'])) # if no valid actions, select among the adjacent nodes uniformly
            break
        elif st_tp1 != 'action_unavailable': 
            break
        else:
            del prob_obs_action_dict[action] # remove the action from the dictionary if it is not available
            prob_obs_action_dict = {k: v/sum(prob_obs_action_dict.values()) for k, v in prob_obs_action_dict.items()} # renormalize the probabilities

            if len(prob_obs_action_dict) < 1: # if no valid actions remain, sample uniformly among the adjacent states
                sample_adj_states_uniformly = True 

    ## convert between the two systems
    # print(f"heading before get_ego_direction: {prms['heading']}")
    action_ego = get_ego_direction(prms['st_positions'][prms['st_t']], prms['st_positions'][st_tp1], prms['heading_latent']) 
    action_allo_latent = displacement_to_compass_heading(prms['st_positions'][st_tp1] - prms['st_positions'][prms['st_t']]) 
    action_allo_real = get_direction_from_radian(np.mod(allo_radian_map_dict[action_allo_latent] + (prms['heading_real'] - prms['heading_latent']), 2*np.pi))
    
    return [int(st_tp1)], [action_allo_latent], [action_allo_real], [action_ego] # length 1 lists to support generalization to the n-length sequence actions


def policy_random(**prms):
    ## note: use "get" approach for default values for **prms
    
    while True: # keep attempting actions until the environment supports it
        #### make a choice
        
        ## compute the actions (select which type to use below)
        action_allo = str(np.random.choice(['N', 'S', 'W', 'E']))
        action_ego = str(np.random.choice(['F', 'B', 'L', 'R'])) # overwritten below if prms['representation] in ['allo_real' or 'allo_latent]

        ## select the action
        if prms['representation'] in ['allo_latent', 'allo_real']:
            action_used = action_allo
        elif prms['representation'] == 'ego':
            action_used = action_ego
                
        ## compute the next state based on the action selected
        st_tp1 = env_transition_func(prms['st_t'], action_used, prms['representation'], adj_mat=prms['adj_mat'], st_positions=prms['st_positions'],
                                        heading_real=prms['heading_real'], heading_latent=prms['heading_latent'])
        
        if st_tp1 != 'action_unavailable':
            
            ## allow reversals
            if 'avoid_reversal' not in prms or not prms['avoid_reversal']: 
                break # accept the action and st_tp1 and escape the loop
            
            ## avoid reversals (except for at dead ends or before reversal is properly defined (eg before the number of states is 2 before the candidate action))
            else: 
                ## 1st condition is for the first state, so reversal is not defined; 2nd condition allows the agent to reverse if that is the only valid action
                if st_tp1 != prms['st_tm1'] or prms['st_tm1'] == 'does not yet exist' or len(get_adj_states(prms['st_t'], prms['adj_mat'])) == 1: 
                    break # 250624 Author: correcting logic; need to compare t+1 to t-1, not t to t-1
                else: 
                    pass # force a new action that's not a reversal by retrying the action selection loop

    """ convert between the two reference frames (latent vs real):
            this is involved but we're just computing the missing types not inherited from the calling side or computable from st_tp1 and st_t (next and current state) """
    
    ## allo latent
    assert type(st_tp1) == int and isinstance(prms["st_t"], numbers.Integral), f'Author: st_tp1 and st_t should both be integers but got these respectively: {type(st_tp1)} vs {type(prms["st_t"])}'
    action_allo_latent = displacement_to_compass_heading(prms['st_positions'][st_tp1] - prms['st_positions'][prms['st_t']])
    
    ## ego    
    inferred_action_ego = get_ego_direction(prms['st_positions'][prms['st_t']], prms['st_positions'][st_tp1], prms['heading_latent'])  
    if prms['representation'] != 'ego':
        action_ego = inferred_action_ego
    else:
        assert action_ego == inferred_action_ego, f'Author: action_ego should agree w/ computed inferred action_ego: {action_ego} vs {inferred_action_ego}'
        pass # use the value selected above

    ## allo real
    if prms['representation'] == 'allo_real': 
        action_allo_real = action_allo 
    else: 
        action_allo_real = get_direction_from_radian(np.mod(allo_radian_map_dict[action_allo_latent] + (prms['heading_real'] - prms['heading_latent']), 2*np.pi))
        # observations_adj_nodes_allo_real = deepcopy([get_direction_from_radian(np.mod(allo_radian_map_dict[obs] + (heading_real - heading_latent), 2*np.pi)) for obs in observations_adj_nodes_allo_latent])
    return [int(st_tp1)], [action_allo_latent], [action_allo_real], [action_ego]    
