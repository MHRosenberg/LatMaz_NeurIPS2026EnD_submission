"""
Shelved: POMCP Oracle Agent

Full-information POMCP planning agent that knows the maze structure
perfectly from the start. Shelved because its point-mass belief state
makes it equivalent to fully-observable MCTS — not a realistic model
of animal navigation.

Still runnable via the standard runner interface (MODEL_CLASSES['POMCP']).

Moved from advanced_agents.py on 2025-02-25.
"""

import random
import numpy as np

from advanced_agents import (
    POMCP_AVAILABLE,
    MazeState, MazeAction,
    MazeTransitionModel, MazeObservationModel, MazeRewardModel,
    MazePolicyModel, MazeRolloutPolicy, NoveltySeekingRolloutPolicy,
    MazeBlackboxModel,
)

if POMCP_AVAILABLE:
    import pomdp_py
    from pomdp_py import POMCP
    from advanced_agents import get_adj_states


class POMCPAgent:
    """POMCP planning agent for graph maze navigation.

    Uses Monte Carlo tree search with particle belief representation.
    Requires a perfect model of the maze (transition dynamics, reward
    positions, rebait mechanics).

    This is NOT a learning agent — it re-plans from scratch at each step.
    Use it as a reference upper-bound for RL agent performance.

    Args:
        max_depth: Maximum MCTS tree depth per planning step.
        num_sims: Number of MCTS simulations per action selection.
        exploration_const: UCB exploration constant.
        discount: Discount factor for future rewards.
    """

    def __init__(
        self,
        max_depth: int = 50,
        num_sims: int = 500,
        exploration_const: float = 20.0,
        discount: float = 0.99,
        novelty_rollout: bool = False,
    ):
        if not POMCP_AVAILABLE:
            raise ImportError(
                "pomdp-py required for POMCP. Install: pip install pomdp-py")
        self.max_depth = max_depth
        self.num_sims = num_sims
        self.exploration_const = exploration_const
        self.discount = discount
        self.novelty_rollout = novelty_rollout

    def run_episode(
        self,
        adj_mat: np.ndarray,
        node_positions: np.ndarray,
        start_node: int,
        rewarded_nodes: np.ndarray,
        n_actions: int,
        min_rewarded_states: int = 2,
        seed: int = 42,
        verbose: bool = False,
        prevent_reverse: bool = False,
    ) -> float:
        """Run a single episode using POMCP planning.

        Args:
            adj_mat: Adjacency matrix of the maze graph.
            node_positions: (n_nodes, 2) array of node XY coordinates.
            start_node: Starting node index.
            rewarded_nodes: Initial reward distribution (1.0 for rewarded).
            n_actions: Action budget (number of valid actions allowed).
            min_rewarded_states: Rebait threshold.
            seed: Random seed.
            verbose: Print progress every 100 steps.

        Returns:
            Total reward collected during the episode.
        """
        np.random.seed(seed)
        random.seed(seed)

        rewarded_tuple = tuple(rewarded_nodes.tolist())

        # Build POMDP models
        transition = MazeTransitionModel(
            adj_mat, node_positions, rewarded_tuple, min_rewarded_states)
        observation = MazeObservationModel(adj_mat, node_positions)
        reward = MazeRewardModel()
        agent_policy = MazePolicyModel()
        if self.novelty_rollout:
            rollout_policy = NoveltySeekingRolloutPolicy(transition._transitions)
        else:
            rollout_policy = MazeRolloutPolicy()
        blackbox = MazeBlackboxModel(transition, observation, reward)

        # Initial state and particle belief
        init_state = MazeState(start_node, rewarded_tuple)
        init_belief = pomdp_py.Particles([init_state] * 100)

        # Create pomdp-py agent
        agent = pomdp_py.Agent(
            init_belief, agent_policy, transition, observation, reward,
            blackbox_model=blackbox,
        )

        # Create POMCP planner
        planner = POMCP(
            max_depth=self.max_depth,
            discount_factor=self.discount,
            num_sims=self.num_sims,
            exploration_const=self.exploration_const,
            rollout_policy=rollout_policy,
        )

        # Run episode
        total_reward = 0.0
        current_state = init_state
        prev_node = None

        for step in range(n_actions):
            action = planner.plan(agent)
            next_state = transition.sample(current_state, action)

            # Prevent reverse: if planned action goes back to prev_node,
            # pick a random valid non-reverse action instead
            if prevent_reverse and prev_node is not None and next_state.node == prev_node:
                adj_nodes = list(get_adj_states(current_state.node, adj_mat))
                forward_nodes = [n for n in adj_nodes if n != prev_node]
                if forward_nodes:
                    chosen = random.choice(forward_nodes)
                    # Find the direction that leads to chosen node
                    for dir_idx, adj in transition._transitions[current_state.node].items():
                        if adj == chosen:
                            action = MazeAction(dir_idx)
                            break
                    next_state = transition.sample(current_state, action)

            obs = observation.sample(next_state, action)
            step_reward = reward.sample(current_state, action, next_state)
            total_reward += step_reward
            planner.update(agent, action, obs)
            prev_node = current_state.node
            current_state = next_state

            if verbose and step % 100 == 0 and step > 0:
                print(f"  POMCP step {step}/{n_actions}: "
                      f"reward so far = {total_reward:.0f}")

        return total_reward
