"""
POMCP Agent for Graph Maze Navigation

Partially Observable Monte Carlo Planning (POMCP) agent using pomdp-py.
This is a model-based planning agent (NOT a learning agent) that serves as
a reference upper-bound: it knows the maze structure perfectly and computes
an optimal plan via Monte Carlo tree search with particle belief tracking.

All RL agents (DQN, PPO, A2C, TRPO, QRDQN, RecurrentPPO, DRQN_seq, DRQN_rand)
are in yoked_rl_runner.py. This file contains only the POMCP planning agent.

Requires: pip install pomdp-py

References:
- Silver & Veness (2010). Monte-Carlo Planning in Large POMDPs.
"""

import os
import sys
import random
from collections import deque
from typing import Tuple

import numpy as np

# Add code to path
_this_dir = os.path.dirname(os.path.abspath(__file__))
_code_in_mr = os.path.join(_this_dir, '..', '..', 'code')
if _code_in_mr not in sys.path:
    sys.path.insert(0, _code_in_mr)

from utils_latMaz import (
    get_adj_states,
    displacement_to_compass_heading,
    allo_actions_one_hot_dict,
)

# Import pomdp-py (graceful degradation)
try:
    import pomdp_py
    from pomdp_py import POMCP
    from pomdp_py.framework.basics import (
        State, Action, Observation,
        TransitionModel, ObservationModel, RewardModel,
        PolicyModel, BlackboxModel,
    )
    from pomdp_py.algorithms.po_uct import RolloutPolicy
    POMCP_AVAILABLE = True
except ImportError:
    POMCP_AVAILABLE = False


# =============================================================================
# POMDP Model Classes
# =============================================================================

if POMCP_AVAILABLE:

    class MazeState(State):
        """State: (node, rewards_tuple)."""
        def __init__(self, node: int, rewards: Tuple[float, ...]):
            self.node = node
            self.rewards = rewards

        def __hash__(self):
            return hash((self.node, self.rewards))

        def __eq__(self, other):
            return (isinstance(other, MazeState)
                    and self.node == other.node
                    and self.rewards == other.rewards)

        def __repr__(self):
            return f"MazeState(node={self.node}, n_rewards={sum(self.rewards)})"

    class MazeAction(Action):
        """Action: direction index (0-3)."""
        NAMES = ['N', 'S', 'W', 'E']

        def __init__(self, direction: int):
            self.direction = direction

        def __hash__(self):
            return hash(self.direction)

        def __eq__(self, other):
            return (isinstance(other, MazeAction)
                    and self.direction == other.direction)

        def __repr__(self):
            return f"MazeAction({self.NAMES[self.direction]})"

    class MazeObservation(Observation):
        """Observation: available directions as tuple of booleans."""
        def __init__(self, available: Tuple[bool, ...]):
            self.available = available

        def __hash__(self):
            return hash(self.available)

        def __eq__(self, other):
            return (isinstance(other, MazeObservation)
                    and self.available == other.available)

        def __repr__(self):
            dirs = [MazeAction.NAMES[i]
                    for i, a in enumerate(self.available) if a]
            return f"MazeObs({dirs})"

    class MazeTransitionModel(TransitionModel):
        """Deterministic transition model with PacMan-style rebait."""

        def __init__(self, adj_mat, node_coords, rewarded_initial,
                     min_rewarded):
            self.adj_mat = adj_mat
            self.node_coords = node_coords
            self.rewarded_initial = rewarded_initial
            self.min_rewarded = min_rewarded
            self.n_nodes = len(adj_mat)

            # Precompute direction-to-neighbor mapping per node
            self._transitions = {}
            for node in range(self.n_nodes):
                adj_nodes = list(get_adj_states(node, adj_mat))
                self._transitions[node] = {}
                for adj in adj_nodes:
                    disp = node_coords[adj] - node_coords[node]
                    direction = displacement_to_compass_heading(disp)
                    dir_idx = allo_actions_one_hot_dict[direction]
                    self._transitions[node][dir_idx] = adj

        def probability(self, next_state, state, action):
            expected = self.sample(state, action)
            return 1.0 if next_state == expected else 0.0

        def sample(self, state, action):
            node = state.node
            rewards = list(state.rewards)

            if action.direction in self._transitions.get(node, {}):
                next_node = self._transitions[node][action.direction]
            else:
                next_node = node  # Invalid action: stay in place

            if next_node != node:
                if rewards[next_node] > 0:
                    if sum(rewards) == self.min_rewarded:
                        rewards = list(self.rewarded_initial)
                    rewards[next_node] = 0.0

            return MazeState(next_node, tuple(rewards))

    class MazeObservationModel(ObservationModel):
        """Observation model: see which compass directions are available."""

        def __init__(self, adj_mat, node_coords):
            self.adj_mat = adj_mat
            self.node_coords = node_coords
            self._obs_cache = {}

        def _compute_obs(self, node):
            if node not in self._obs_cache:
                adj_nodes = list(get_adj_states(node, self.adj_mat))
                available = [False, False, False, False]
                for adj in adj_nodes:
                    disp = self.node_coords[adj] - self.node_coords[node]
                    direction = displacement_to_compass_heading(disp)
                    dir_idx = allo_actions_one_hot_dict[direction]
                    available[dir_idx] = True
                self._obs_cache[node] = tuple(available)
            return self._obs_cache[node]

        def probability(self, observation, next_state, action):
            expected = self._compute_obs(next_state.node)
            return 1.0 if observation.available == expected else 0.0

        def sample(self, next_state, action):
            return MazeObservation(self._compute_obs(next_state.node))

    class MazeRewardModel(RewardModel):
        """Reward: +1 for collecting a reward pellet."""

        def probability(self, reward, state, action, next_state):
            expected = self.sample(state, action, next_state)
            return 1.0 if reward == expected else 0.0

        def sample(self, state, action, next_state):
            if state.node != next_state.node:
                return state.rewards[next_state.node]
            return 0.0

    class MazePolicyModel(PolicyModel):
        """Uniform random policy for the pomdp-py Agent."""

        def __init__(self):
            self.actions = [MazeAction(i) for i in range(4)]

        def sample(self, state):
            return random.choice(self.actions)

        def get_all_actions(self, state=None, history=None):
            return self.actions

    class MazeRolloutPolicy(RolloutPolicy):
        """Uniform random rollout policy for POMCP/MCTS planning."""

        def __init__(self):
            self.actions = [MazeAction(i) for i in range(4)]

        def sample(self, state):
            return random.choice(self.actions)

        def get_all_actions(self, state=None, history=None):
            return self.actions

        def rollout(self, state, history=None):
            return random.choice(self.actions)

    class NoveltySeekingRolloutPolicy(RolloutPolicy):
        """Rollout policy that prefers actions leading to nodes with uncollected rewards."""

        def __init__(self, transitions):
            """
            Args:
                transitions: {node: {dir_idx: next_node}} — from either
                             MazeTransitionModel._transitions or
                             _BioTransitionModel._transitions (known_edges).
            """
            self.transitions = transitions
            self.actions = [MazeAction(i) for i in range(4)]

        def rollout(self, state, history=None):
            node = state.node
            rewards = state.rewards
            edges = self.transitions.get(node, {})

            # Tier 1: prefer actions leading to nodes with uncollected rewards
            rewarded = [MazeAction(d) for d, nxt in edges.items()
                        if rewards[nxt] > 0]
            if rewarded:
                return random.choice(rewarded)

            # Tier 2: prefer any valid move (avoid staying in place)
            valid = [MazeAction(d) for d in edges]
            if valid:
                return random.choice(valid)

            # Tier 3: fallback to random (shouldn't happen in connected maze)
            return random.choice(self.actions)

        def sample(self, state):
            return self.rollout(state)

        def get_all_actions(self, state=None, history=None):
            return self.actions

    class _BioTransitionModel(TransitionModel):
        """Transition model built from explored edges only.

        Unknown transitions result in staying in place. This limits MCTS
        planning to the explored subgraph. Supports optional rebait once
        the agent has discovered the rebait mechanic.
        """

        def __init__(self, known_edges, min_rewarded, knows_rebait):
            self.known_edges = known_edges  # {node: {dir_idx: next_node}}
            self.min_rewarded = min_rewarded
            self.knows_rebait = knows_rebait
            # Alias for compatibility with NoveltySeekingRolloutPolicy
            self._transitions = known_edges

        def probability(self, next_state, state, action):
            expected = self.sample(state, action)
            return 1.0 if next_state == expected else 0.0

        def sample(self, state, action):
            node = state.node
            rewards = list(state.rewards)

            edges = self.known_edges.get(node, {})
            if action.direction in edges:
                next_node = edges[action.direction]
            else:
                next_node = node  # Unknown edge: stay in place

            if next_node != node:
                if rewards[next_node] > 0:
                    if self.knows_rebait and sum(rewards) == self.min_rewarded:
                        # Rebait: reset only known nodes to 1.0
                        for n in self.known_edges:
                            if n < len(rewards):
                                rewards[n] = 1.0
                    rewards[next_node] = 0.0

            return MazeState(next_node, tuple(rewards))

    class _BioObservationModel(ObservationModel):
        """Observation model restricted to corridors the agent has seen.

        When the agent visits a node, it sees which compass directions
        have neighbours. This model only returns those observed corridors
        — it has NO access to the full adjacency matrix. For nodes the
        planner might reach that the agent hasn't visited (shouldn't
        happen with _BioTransitionModel, but defensively handled), it
        returns a generic "all-True" observation so the planner doesn't
        crash.
        """

        def __init__(self, remembered_corridors):
            """
            Args:
                remembered_corridors: {node: tuple(bool, bool, bool, bool)}
                    Corridors the agent saw when it visited each node.
            """
            self.remembered = remembered_corridors  # updated in-place

        def _get_obs(self, node):
            if node in self.remembered:
                return self.remembered[node]
            # Unknown node — shouldn't happen in practice
            return (True, True, True, True)

        def probability(self, observation, next_state, action):
            expected = self._get_obs(next_state.node)
            return 1.0 if observation.available == expected else 0.0

        def sample(self, next_state, action):
            return MazeObservation(self._get_obs(next_state.node))

    class MazeBlackboxModel(BlackboxModel):
        """Combined blackbox model for efficient MCTS rollouts."""

        def __init__(self, transition_model, observation_model,
                     reward_model):
            self.transition = transition_model
            self.observation = observation_model
            self.reward = reward_model

        def sample(self, state, action):
            next_state = self.transition.sample(state, action)
            observation = self.observation.sample(next_state, action)
            reward = self.reward.sample(state, action, next_state)
            return (next_state, observation, reward)


# POMCPAgent (omniscient oracle) shelved to shelved_agents/pomcp_oracle.py


# =============================================================================
# POMCP_bio: Biologically-Constrained POMCP
# =============================================================================

class POMCPbioAlloAgent:
    """POMCP agent that discovers the maze through exploration.

    Unlike standard POMCP which has perfect maze knowledge from the start,
    POMCP_bio relaxes two major unrealistic assumptions:

    A) UNKNOWN TRANSITIONS: The agent starts with no knowledge of the maze
       layout. Edges are learned by traversing them (+ reverse edge inferred).
       During MCTS planning, unknown transitions result in staying in place,
       so the planner can only reason about the explored subgraph.

    B) UNKNOWN REWARDS: The agent does not know which nodes have rewards.
       It starts with an optimistic prior (all unvisited nodes might have
       reward). After visiting a node, it learns whether that node had
       reward. The agent also discovers the rebait mechanic after observing
       the first rebait event.

    Decision policy (three modes):
      - EXPLORE: If the current node has unexplored outgoing edges,
        pick a random unexplored direction.
      - NAVIGATE: If the current node is fully explored but other nodes
        still have unexplored edges, BFS to the nearest frontier node.
      - EXPLOIT: If the entire maze is explored, use POMCP planning
        over the known subgraph with believed reward state. The MCTS
        tree is reused across consecutive exploit steps via
        planner.update() for efficiency.

    This creates a natural exploration-exploitation tradeoff: early in the
    episode the agent mostly explores (building its map), later it exploits
    its map to collect rewards efficiently.

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
                "pomdp-py required for POMCP_bio. Install: pip install pomdp-py")
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
        return_trajectory: bool = False,
    ) -> float:
        """Run a single episode with incremental maze learning.

        The agent explores unknown edges and plans over discovered structure.
        The real maze (adj_mat, rewarded_nodes) is hidden from the agent;
        it only learns through its own traversals and reward observations.

        Returns:
            Total reward collected during the episode (float), or
            (total_reward, trajectory) if return_trajectory=True.
            trajectory is a list of (node, step_reward, mode) tuples where
            mode is 'explore', 'navigate', or 'exploit'.
        """
        np.random.seed(seed)
        random.seed(seed)

        n_nodes = len(adj_mat)

        # ============================================================
        # REAL WORLD (hidden from agent — used to compute outcomes)
        # ============================================================
        real_rewards = list(rewarded_nodes.tolist())
        rewarded_initial = tuple(rewarded_nodes.tolist())

        # Precompute direction↔neighbor mapping (real world)
        dir_to_neighbor = {}
        for node in range(n_nodes):
            dir_to_neighbor[node] = {}
            for adj in get_adj_states(node, adj_mat):
                disp = node_positions[adj] - node_positions[node]
                d = displacement_to_compass_heading(disp)
                dir_to_neighbor[node][allo_actions_one_hot_dict[d]] = adj

        # ============================================================
        # AGENT'S LEARNED MODEL (starts empty)
        # ============================================================
        # A: Known transitions — {node: {dir_idx: next_node}}
        known_edges = {}

        # B: Believed rewards — 0 for unknown, 1 for discovered-but-unvisited
        #    (optimistic prior applied only to nodes the agent knows about)
        believed_rewards = {}  # {node_id: reward_value}
        visited_nodes = set()
        knows_rebait = False

        # Observation model: agent remembers corridors at visited nodes.
        # When visiting a node, it sees which directions have neighbours.
        # This dict is populated incrementally — NO access to full adj_mat.
        remembered_corridors = {}  # {node: (bool, bool, bool, bool)}
        obs_model = _BioObservationModel(remembered_corridors)

        # Helper: record corridors the agent sees at a node
        def _observe_corridors(node):
            if node not in remembered_corridors:
                avail = [False, False, False, False]
                for d_idx in dir_to_neighbor.get(node, {}):
                    avail[d_idx] = True
                remembered_corridors[node] = tuple(avail)

        # ============================================================
        # Episode loop
        # ============================================================
        current_node = start_node
        _observe_corridors(current_node)  # look around at start
        prev_node = None
        total_reward = 0.0
        explore_count = 0
        exploit_count = 0
        nav_to_frontier_count = 0
        if return_trajectory:
            trajectory = [(start_node, 0.0, 'start')]

        # POMCP planner state (persists across exploit steps for tree reuse)
        _pomcp_planner = None
        _pomcp_agent = None
        _last_exploit_action = None

        for step in range(n_actions):
            # --- Available directions at current node ---
            avail_dirs = set(dir_to_neighbor[current_node].keys())
            known_from_here = set(known_edges.get(current_node, {}).keys())
            unexplored = list(avail_dirs - known_from_here)

            # Apply prevent_reverse to unexplored list
            if prevent_reverse and prev_node is not None:
                unexplored = [d for d in unexplored
                              if dir_to_neighbor[current_node].get(d) != prev_node]

            # --- DECIDE: explore, navigate-to-frontier, or exploit ---
            step_mode = 'exploit'
            if unexplored:
                # EXPLORE: at a frontier node, pick random unexplored direction
                action = MazeAction(random.choice(unexplored))
                explore_count += 1
                step_mode = 'explore'
            elif self._has_frontier(known_edges, remembered_corridors):
                # NAVIGATE: frontier nodes exist but not here — BFS to nearest
                nav_to_frontier_count += 1
                step_mode = 'navigate'
                next_dir = self._bfs_to_frontier(
                    current_node, known_edges, remembered_corridors,
                    prev_node if prevent_reverse else None)
                if next_dir is not None:
                    action = MazeAction(next_dir)
                else:
                    # Fallback: random known direction
                    action = MazeAction(random.choice(list(known_from_here)))
            else:
                # EXPLOIT: entire maze explored, use POMCP with learned model
                exploit_count += 1

                # Can we reuse the tree from the previous exploit step?
                can_reuse = (_pomcp_planner is not None
                             and _last_exploit_action is not None)

                # Build rewards tuple from dict of discovered nodes
                max_id = max(believed_rewards.keys()) + 1 if believed_rewards else 0
                bio_rewards_tuple = tuple(
                    believed_rewards.get(n, 0.0) for n in range(max_id))

                if can_reuse:
                    # Carry the MCTS tree forward via planner.update()
                    last_obs = obs_model.sample(
                        MazeState(current_node, bio_rewards_tuple),
                        _last_exploit_action)
                    try:
                        _pomcp_planner.update(
                            _pomcp_agent, _last_exploit_action, last_obs)
                    except ValueError:
                        # Particle deprivation — fall back to full rebuild
                        can_reuse = False

                if not can_reuse:
                    # Full rebuild: first exploit step or returning from
                    # explore/navigate mode
                    bio_transition = _BioTransitionModel(
                        known_edges, min_rewarded_states, knows_rebait)

                    bio_state = MazeState(current_node, bio_rewards_tuple)
                    bio_belief = pomdp_py.Particles([bio_state] * 100)

                    reward_model = MazeRewardModel()
                    policy = MazePolicyModel()
                    if self.novelty_rollout:
                        rollout_policy = NoveltySeekingRolloutPolicy(
                            bio_transition._transitions)
                    else:
                        rollout_policy = MazeRolloutPolicy()
                    blackbox = MazeBlackboxModel(
                        bio_transition, obs_model, reward_model)

                    _pomcp_agent = pomdp_py.Agent(
                        bio_belief, policy, bio_transition, obs_model,
                        reward_model, blackbox_model=blackbox)

                    _pomcp_planner = POMCP(
                        max_depth=self.max_depth,
                        discount_factor=self.discount,
                        num_sims=self.num_sims,
                        exploration_const=self.exploration_const,
                        rollout_policy=rollout_policy)

                action = _pomcp_planner.plan(_pomcp_agent)
                _last_exploit_action = action

                # Prevent reverse on POMCP output
                if prevent_reverse and prev_node is not None:
                    planned_next = dir_to_neighbor[current_node].get(
                        action.direction, current_node)
                    if planned_next == prev_node:
                        forward_dirs = [
                            d for d in avail_dirs
                            if dir_to_neighbor[current_node][d] != prev_node]
                        if forward_dirs:
                            action = MazeAction(random.choice(forward_dirs))
                            _last_exploit_action = action

            # --- EXECUTE in real world ---
            next_node = dir_to_neighbor[current_node].get(
                action.direction, current_node)

            if next_node != current_node:
                step_reward = float(real_rewards[next_node])
                if step_reward > 0:
                    if sum(real_rewards) == min_rewarded_states:
                        if not knows_rebait:
                            knows_rebait = True
                        real_rewards = list(rewarded_initial)
                    real_rewards[next_node] = 0.0
            else:
                step_reward = 0.0

            total_reward += step_reward
            if return_trajectory:
                trajectory.append((next_node, step_reward, step_mode))

            # --- LEARN from experience ---
            if next_node != current_node:
                # Observe corridors at new position (look around)
                _observe_corridors(next_node)

                # A: Learn transition (both directions)
                newly_discovered = []
                if current_node not in known_edges:
                    known_edges[current_node] = {}
                    newly_discovered.append(current_node)
                known_edges[current_node][action.direction] = next_node

                # Infer reverse edge: N(0)↔S(1), W(2)↔E(3)
                if next_node not in known_edges:
                    known_edges[next_node] = {}
                    newly_discovered.append(next_node)
                rev_idx = action.direction ^ 1
                known_edges[next_node][rev_idx] = current_node

                # Set optimistic prior for newly discovered nodes
                for nd in newly_discovered:
                    if nd not in visited_nodes:
                        believed_rewards[nd] = 1.0  # might have reward

                # B: Learn reward at destination (by visiting)
                visited_nodes.add(next_node)
                old_believed = believed_rewards.get(next_node, 0.0)

                # Handle believed rebait: CHECK → REBAIT → REMOVE
                if old_believed > 0:
                    believed_sum = sum(believed_rewards.values())
                    if knows_rebait and believed_sum == min_rewarded_states:
                        # Agent expects rewards to reset at all known nodes
                        for nd in known_edges:
                            believed_rewards[nd] = 1.0
                believed_rewards[next_node] = 0.0  # observed: consumed or empty

                prev_node = current_node
            current_node = next_node

            # Invalidate POMCP tree when leaving exploit mode
            if step_mode != 'exploit':
                _pomcp_planner = None
                _pomcp_agent = None
                _last_exploit_action = None

            if verbose and step > 0 and step % 100 == 0:
                n_known = sum(len(v) for v in known_edges.values())
                pct = 100.0 * len(known_edges) / n_nodes
                print(f"  POMCP_bio step {step}/{n_actions}: "
                      f"reward={total_reward:.0f}, "
                      f"nodes={len(known_edges)}/{n_nodes} ({pct:.0f}%), "
                      f"edges={n_known}, explore={explore_count}, "
                      f"exploit={exploit_count}")

        if verbose:
            n_known = sum(len(v) for v in known_edges.values())
            print(f"  POMCP_bio final: reward={total_reward:.0f}, "
                  f"nodes_discovered={len(known_edges)}/{n_nodes}, "
                  f"edges={n_known}, "
                  f"explore={explore_count}, nav={nav_to_frontier_count}, "
                  f"exploit={exploit_count}")

        if return_trajectory:
            return total_reward, trajectory
        return total_reward

    @staticmethod
    def _seen_dirs(node, remembered_corridors):
        """Directions the agent saw as available when visiting a node."""
        obs = remembered_corridors.get(node)
        if obs is None:
            return set()
        return {i for i, avail in enumerate(obs) if avail}

    @staticmethod
    def _has_frontier(known_edges, remembered_corridors):
        """Check if any discovered node has unexplored outgoing edges.

        Uses only the agent's remembered corridor observations — no access
        to the real adjacency matrix.
        """
        for node in known_edges:
            seen = POMCPbioAlloAgent._seen_dirs(node, remembered_corridors)
            known_dirs = set(known_edges[node].keys())
            if seen - known_dirs:
                return True
        return False

    @staticmethod
    def _bfs_to_frontier(current_node, known_edges, remembered_corridors,
                         blocked_node=None):
        """BFS over known edges to find shortest path to nearest frontier node.

        A frontier node is a known node that has at least one unexplored
        outgoing edge (based on remembered corridors, not real topology).
        Returns the direction (int) for the first step from current_node,
        or None if unreachable.

        Args:
            current_node: Where the agent is now.
            known_edges: {node: {dir_idx: next_node}} learned so far.
            remembered_corridors: {node: (bool,bool,bool,bool)} corridors
                the agent saw at visited nodes.
            blocked_node: If not None, the BFS won't step to this node
                          on the very first hop (prevent_reverse).
        """
        from collections import deque

        queue = deque()
        visited = {current_node}

        # Seed BFS with neighbors reachable via known edges
        for d, nxt in known_edges.get(current_node, {}).items():
            if blocked_node is not None and nxt == blocked_node:
                continue
            if nxt not in visited:
                visited.add(nxt)
                queue.append((nxt, d))

        while queue:
            node, first_dir = queue.popleft()

            # Is this node a frontier? (has unexplored outgoing edges)
            seen = POMCPbioAlloAgent._seen_dirs(node, remembered_corridors)
            known_dirs = set(known_edges.get(node, {}).keys())
            if seen - known_dirs:
                return first_dir  # direction for the first step

            # Expand via known edges
            for d, nxt in known_edges.get(node, {}).items():
                if nxt not in visited:
                    visited.add(nxt)
                    queue.append((nxt, first_dir))

        return None  # No frontier reachable


# =============================================================================
# POMCP_bio_egoOnly: Egocentric-Only POMCP (observation tree)
# =============================================================================

# Ego→Allo conversion tables (used by simulator AND agent's dead reckoning)
# heading_idx: 0=N, 1=S, 2=W, 3=E  (agent uses these as relative frame)
# ego_action:  0=F, 1=B, 2=L, 3=R
# result:      relative dir index (0=N, 1=S, 2=W, 3=E)
EGO_TO_ALLO = {
    0: {0: 0, 1: 1, 2: 2, 3: 3},  # facing N: F=N, B=S, L=W, R=E
    1: {0: 1, 1: 0, 2: 3, 3: 2},  # facing S: F=S, B=N, L=E, R=W
    2: {0: 2, 1: 3, 2: 1, 3: 0},  # facing W: F=W, B=E, L=S, R=N
    3: {0: 3, 1: 2, 2: 0, 3: 1},  # facing E: F=E, B=W, L=N, R=S
}
# Inverse: Allo→Ego conversion
ALLO_TO_EGO = {
    h: {allo: ego for ego, allo in mapping.items()}
    for h, mapping in EGO_TO_ALLO.items()
}
# Dead-reckoning: relative direction → (dx, dy) displacement
_DR_DELTA = {0: (0, 1), 1: (0, -1), 2: (-1, 0), 3: (1, 0)}


class POMCPbioEgoAgent(POMCPbioAlloAgent):
    """POMCP agent that sees ONLY egocentric observations (F/B/L/R corridors + reward).

    The agent receives only 4-bit ego corridors (F/B/L/R) and scalar reward.
    No true node IDs, no compass, no allocentric observations.

    Cycle detection uses (ego_obs, last_N_actions) as a fingerprint to merge
    tree nodes.  The action history partially encodes heading (approach
    direction), allowing the tree to close cycles without cumulative
    position tracking.  Longer history reduces false merges at the cost
    of more tree nodes.

    Args:
        history_depth: Number of past actions to include in fingerprint.
            depth=1: 16 × 5 = 80 possible fingerprints (original behavior).
            depth=2: 16 × 25 = 400 fingerprints (far fewer collisions).
            depth=3: 16 × 125 = 2000 fingerprints (very few collisions).

    The three-phase decision policy (explore -> navigate -> exploit) is
    identical to POMCP_bio, but operates on tree node IDs with ego actions
    (F/B/L/R).
    """

    def __init__(self, history_depth: int = 1, **kwargs):
        super().__init__(**kwargs)
        self.history_depth = history_depth

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
        return_trajectory: bool = False,
    ) -> float:
        """Run a single episode using egocentric observation tree.

        The agent only accesses the hidden world through _sim_step(), which
        returns (ego_obs, reward, moved). It never sees true node IDs,
        the adjacency matrix, or node positions directly.

        Tree nodes are identified by (ego_obs, last_action) fingerprint.
        When the agent arrives at a state matching an existing fingerprint,
        the existing tree node is reused (cycle closure).

        Returns:
            Total reward (float), or (total_reward, trajectory) if
            return_trajectory=True. Trajectory entries are
            (tree_node, step_reward, mode).
        """
        np.random.seed(seed)
        random.seed(seed)

        n_nodes = len(adj_mat)

        # ============================================================
        # HIDDEN WORLD (agent never accesses these directly)
        # ============================================================
        real_rewards = list(rewarded_nodes.tolist())
        rewarded_initial = tuple(rewarded_nodes.tolist())

        # Precompute direction↔neighbor mapping (real world)
        dir_to_neighbor = {}
        for node in range(n_nodes):
            dir_to_neighbor[node] = {}
            for adj in get_adj_states(node, adj_mat):
                disp = node_positions[adj] - node_positions[node]
                d = displacement_to_compass_heading(disp)
                dir_to_neighbor[node][allo_actions_one_hot_dict[d]] = adj

        # Hidden state
        real_node = start_node
        real_heading_idx = 0  # N (π/2) — matches heading_latent_initial

        def _compute_ego_obs(node, heading_idx):
            """Compute 4-bit ego corridor observation at (node, heading)."""
            ego_obs = [False, False, False, False]
            for allo_dir in dir_to_neighbor[node]:
                ego_dir = ALLO_TO_EGO[heading_idx][allo_dir]
                ego_obs[ego_dir] = True
            return tuple(ego_obs)

        def _sim_step(ego_action):
            """Execute ego action in hidden world. Returns (ego_obs, reward, moved)."""
            nonlocal real_node, real_heading_idx, real_rewards

            allo_dir = EGO_TO_ALLO[real_heading_idx][ego_action]
            next_node = dir_to_neighbor[real_node].get(allo_dir, real_node)

            if next_node != real_node:
                real_heading_idx = allo_dir  # heading = direction of movement
                real_node = next_node

                reward = float(real_rewards[real_node])
                if reward > 0:
                    if sum(real_rewards) == min_rewarded_states:
                        real_rewards = list(rewarded_initial)
                    real_rewards[real_node] = 0.0

                ego_obs = _compute_ego_obs(real_node, real_heading_idx)
                return ego_obs, reward, True
            else:
                return _compute_ego_obs(real_node, real_heading_idx), 0.0, False

        # ============================================================
        # AGENT STATE — observation tree with (obs, last_action) merging
        # ============================================================
        initial_ego_obs = _compute_ego_obs(real_node, real_heading_idx)

        tree_obs = {}                # tree_id → (F,B,L,R) bools
        tree_edges = {}              # tree_id → {ego_action → next_tree_id}
        believed_rewards = {}        # tree_id → float
        next_tree_id = 0
        knows_rebait = False

        # Fingerprint → tree_id for cycle detection
        # Key = (ego_obs_tuple, tuple(action_history))
        fingerprint_to_tree = {}

        def _get_or_create_tree_node(ego_obs, action_hist_tuple):
            """Get existing tree node or create new one for (obs, action_history)."""
            nonlocal next_tree_id
            key = (ego_obs, action_hist_tuple)
            if key in fingerprint_to_tree:
                return fingerprint_to_tree[key], False  # existing
            tid = next_tree_id
            next_tree_id += 1
            fingerprint_to_tree[key] = tid
            tree_obs[tid] = ego_obs
            believed_rewards[tid] = 1.0  # optimistic prior
            return tid, True  # newly created

        # Action history deque (maxlen = history_depth)
        action_history = deque([-1], maxlen=self.history_depth)

        # Create start node
        current_tree_node, _ = _get_or_create_tree_node(
            initial_ego_obs, tuple(action_history))
        believed_rewards[current_tree_node] = 0.0  # at start, no reward

        # Track the live ego obs (from simulator, not cached)
        current_ego_obs = initial_ego_obs

        # ============================================================
        # Episode loop
        # ============================================================
        total_reward = 0.0
        explore_count = 0
        exploit_count = 0
        nav_count = 0
        if return_trajectory:
            trajectory = [(current_tree_node, 0.0, 'start')]

        _pomcp_planner = None
        _pomcp_agent = None
        _last_exploit_action = None

        for step in range(n_actions):
            # --- Available ego directions from live observation ---
            avail_ego_dirs = {i for i, passable in enumerate(current_ego_obs)
                              if passable}
            known_from_here = set(tree_edges.get(current_tree_node, {}).keys())
            unexplored = list(avail_ego_dirs - known_from_here)

            # --- DECIDE: explore, navigate, or exploit ---
            step_mode = 'exploit'
            if unexplored:
                # EXPLORE: pick a random unexplored ego direction
                ego_action = random.choice(unexplored)
                explore_count += 1
                step_mode = 'explore'
            elif self._has_frontier(tree_edges, tree_obs):
                # NAVIGATE: BFS on tree_edges to nearest frontier
                nav_count += 1
                step_mode = 'navigate'
                next_dir = self._bfs_to_frontier(
                    current_tree_node, tree_edges, tree_obs, None)
                if next_dir is not None:
                    ego_action = next_dir
                else:
                    ego_action = random.choice(list(known_from_here))
            else:
                # EXPLOIT: POMCP over observation tree
                exploit_count += 1

                can_reuse = (_pomcp_planner is not None
                             and _last_exploit_action is not None)

                # Build rewards tuple from tree IDs
                max_id = max(believed_rewards.keys()) + 1 if believed_rewards else 0
                bio_rewards_tuple = tuple(
                    believed_rewards.get(n, 0.0) for n in range(max_id))

                obs_model = _BioObservationModel(tree_obs)

                if can_reuse:
                    last_obs = obs_model.sample(
                        MazeState(current_tree_node, bio_rewards_tuple),
                        _last_exploit_action)
                    try:
                        _pomcp_planner.update(
                            _pomcp_agent, _last_exploit_action, last_obs)
                    except ValueError:
                        can_reuse = False

                if not can_reuse:
                    bio_transition = _BioTransitionModel(
                        tree_edges, min_rewarded_states, knows_rebait)
                    bio_state = MazeState(current_tree_node, bio_rewards_tuple)
                    bio_belief = pomdp_py.Particles([bio_state] * 100)

                    reward_model = MazeRewardModel()
                    policy = MazePolicyModel()
                    if self.novelty_rollout:
                        rollout_policy = NoveltySeekingRolloutPolicy(
                            bio_transition._transitions)
                    else:
                        rollout_policy = MazeRolloutPolicy()
                    blackbox = MazeBlackboxModel(
                        bio_transition, obs_model, reward_model)

                    _pomcp_agent = pomdp_py.Agent(
                        bio_belief, policy, bio_transition, obs_model,
                        reward_model, blackbox_model=blackbox)

                    _pomcp_planner = POMCP(
                        max_depth=self.max_depth,
                        discount_factor=self.discount,
                        num_sims=self.num_sims,
                        exploration_const=self.exploration_const,
                        rollout_policy=rollout_policy)

                action_obj = _pomcp_planner.plan(_pomcp_agent)
                ego_action = action_obj.direction
                _last_exploit_action = action_obj

            # --- EXECUTE via simulator ---
            new_ego_obs, step_reward, moved = _sim_step(ego_action)
            total_reward += step_reward

            # --- UPDATE observation tree ---
            if moved:
                prev_tree_node = current_tree_node
                action_history.append(ego_action)

                # Get or create tree node at new (obs, action_history) fingerprint
                dst_node, is_new = _get_or_create_tree_node(
                    new_ego_obs, tuple(action_history))

                # Record forward edge
                if prev_tree_node not in tree_edges:
                    tree_edges[prev_tree_node] = {}
                tree_edges[prev_tree_node][ego_action] = dst_node

                current_tree_node = dst_node
                current_ego_obs = new_ego_obs

                # Learn reward at destination: CHECK → REBAIT → REMOVE
                old_believed = believed_rewards.get(current_tree_node, 0.0)
                if old_believed > 0:
                    believed_sum = sum(believed_rewards.values())
                    if believed_sum == min_rewarded_states:
                        if not knows_rebait:
                            knows_rebait = True
                        for tid in believed_rewards:
                            believed_rewards[tid] = 1.0
                believed_rewards[current_tree_node] = 0.0  # visited

            if return_trajectory:
                trajectory.append((current_tree_node, step_reward, step_mode))

            # Invalidate POMCP tree when leaving exploit mode
            if step_mode != 'exploit':
                _pomcp_planner = None
                _pomcp_agent = None
                _last_exploit_action = None

            if verbose and step > 0 and step % 100 == 0:
                n_tree_nodes = len(tree_obs)
                n_tree_edges = sum(len(v) for v in tree_edges.values())
                print(f"  POMCP_bio_egoOnly step {step}/{n_actions}: "
                      f"reward={total_reward:.0f}, "
                      f"tree_nodes={n_tree_nodes} (phys={n_nodes}), "
                      f"tree_edges={n_tree_edges}, "
                      f"explore={explore_count}, nav={nav_count}, "
                      f"exploit={exploit_count}")

        if verbose:
            n_tree_nodes = len(tree_obs)
            n_tree_edges = sum(len(v) for v in tree_edges.values())
            print(f"  POMCP_bio_egoOnly final: reward={total_reward:.0f}, "
                  f"tree_nodes={n_tree_nodes}/{n_nodes} (×{n_tree_nodes/n_nodes:.1f}), "
                  f"tree_edges={n_tree_edges}, "
                  f"explore={explore_count}, nav={nav_count}, "
                  f"exploit={exploit_count}")

        if return_trajectory:
            return total_reward, trajectory
        return total_reward


# =============================================================================
# CSCG_bio_BFS: Clone Structured Cognitive Graph + BFS exploit
# =============================================================================

class CSCGBioBFSAgent(POMCPbioAlloAgent):
    """Bio-plausible agent using Clone Structured Cognitive Graph + BFS exploit.

    Uses a cloned HMM (George et al., Nature Comms 2021) to resolve
    perceptual aliasing: each 4-bit ego observation type has M clones.
    Transitions connect clones, not raw observations.  Learning happens
    online via hard-EM on the transition tensor.

    After sufficient exploration, a deterministic graph is extracted from
    the clone transition tensor and BFS is used in exploit mode to navigate
    to the nearest believed-rewarded clone.  Note: this agent does NOT use
    POMCP for planning — see CSCGBioPOMCPAgent for the POMCP variant.

    Args:
        n_clones: Number of clones per observation type (default 10).
        cscg_pseudocount: Dirichlet smoothing for transition tensor.
        cscg_edge_threshold: Min T[a,h,h'] to count as confident edge.
        explore_budget_frac: Fraction of action budget reserved for
            explore+navigate before forcing exploit mode (default 0.5).
    """

    def __init__(
        self,
        n_clones: int = 10,
        cscg_pseudocount: float = 1e-3,
        cscg_edge_threshold: float = 0.3,
        explore_budget_frac: float = 0.5,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.n_clones = n_clones
        self.cscg_pseudocount = cscg_pseudocount
        self.cscg_edge_threshold = cscg_edge_threshold
        self.explore_budget_frac = explore_budget_frac

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
        return_trajectory: bool = False,
    ) -> float:
        """Run episode using CSCG representation + POMCP planning.

        The agent only accesses the hidden world through _sim_step(), which
        returns (ego_obs, reward, moved). CSCG maintains a belief distribution
        over clone states and learns transitions online.

        Returns:
            Total reward (float), or (total_reward, trajectory) if
            return_trajectory=True.
        """
        np.random.seed(seed)
        random.seed(seed)

        n_nodes = len(adj_mat)
        E = 16  # possible 4-bit ego observations (2^4)
        A = 4   # ego actions (F/B/L/R)
        M = self.n_clones
        H = M * E  # total hidden states

        # ============================================================
        # HIDDEN WORLD (agent never accesses these directly)
        # ============================================================
        real_rewards = list(rewarded_nodes.tolist())
        rewarded_initial = tuple(rewarded_nodes.tolist())

        # Precompute direction→neighbor mapping (real world)
        dir_to_neighbor = {}
        for node in range(n_nodes):
            dir_to_neighbor[node] = {}
            for adj in get_adj_states(node, adj_mat):
                disp = node_positions[adj] - node_positions[node]
                d = displacement_to_compass_heading(disp)
                dir_to_neighbor[node][allo_actions_one_hot_dict[d]] = adj

        # Hidden state
        real_node = start_node
        real_heading_idx = 0  # N

        def _compute_ego_obs(node, heading_idx):
            """Compute 4-bit ego corridor observation at (node, heading)."""
            ego_obs = [False, False, False, False]
            for allo_dir in dir_to_neighbor[node]:
                ego_dir = ALLO_TO_EGO[heading_idx][allo_dir]
                ego_obs[ego_dir] = True
            return tuple(ego_obs)

        def _sim_step(ego_action):
            """Execute ego action in hidden world."""
            nonlocal real_node, real_heading_idx, real_rewards

            allo_dir = EGO_TO_ALLO[real_heading_idx][ego_action]
            next_node = dir_to_neighbor[real_node].get(allo_dir, real_node)

            if next_node != real_node:
                real_heading_idx = allo_dir
                real_node = next_node

                reward = float(real_rewards[real_node])
                if reward > 0:
                    if sum(real_rewards) == min_rewarded_states:
                        real_rewards = list(rewarded_initial)
                    real_rewards[real_node] = 0.0

                ego_obs = _compute_ego_obs(real_node, real_heading_idx)
                return ego_obs, reward, True
            else:
                return _compute_ego_obs(real_node, real_heading_idx), 0.0, False

        def _obs_to_idx(ego_obs):
            """Convert 4-bit ego obs tuple to integer index [0..15]."""
            return sum(v << i for i, v in enumerate(ego_obs))

        def _idx_to_obs(idx):
            """Convert integer index [0..15] back to 4-bit ego obs tuple."""
            return tuple(bool((idx >> i) & 1) for i in range(4))

        # ============================================================
        # CSCG STATE
        # ============================================================
        # Transition counts and normalised tensor
        C = np.zeros((A, H, H))
        T = np.full((A, H, H), self.cscg_pseudocount)
        # Add small random noise to break symmetry
        rng = np.random.RandomState(seed)
        T += rng.uniform(0, self.cscg_pseudocount * 0.1, T.shape)
        # Normalise each row
        for a in range(A):
            row_sums = T[a].sum(axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1.0
            T[a] /= row_sums

        # Forward belief
        alpha = np.zeros(H)

        # Track clone usage counts (for first-visit heuristic)
        clone_usage = np.zeros(H)

        # Current clone assignment
        current_clone = None

        # Clone graph for planning: clone_id → {ego_action → next_clone_id}
        clone_edges = {}
        # Observation and reward per clone
        clone_obs = {}   # clone_id → (F,B,L,R) bools
        believed_rewards = {}  # clone_id → float
        knows_rebait = False

        # Initialise at start observation
        initial_ego_obs = _compute_ego_obs(real_node, real_heading_idx)
        obs_idx = _obs_to_idx(initial_ego_obs)

        # Assign least-used clone for starting observation
        clone_start = obs_idx * M
        clone_end = clone_start + M
        least_used = clone_start + int(np.argmin(clone_usage[clone_start:clone_end]))
        current_clone = least_used
        clone_usage[current_clone] += 1
        alpha[current_clone] = 1.0

        clone_obs[current_clone] = initial_ego_obs
        believed_rewards[current_clone] = 0.0  # at start, no reward

        current_ego_obs = initial_ego_obs

        # ============================================================
        # Episode loop
        # ============================================================
        total_reward = 0.0
        explore_count = 0
        exploit_count = 0
        nav_count = 0
        if return_trajectory:
            trajectory = [(current_clone, 0.0, 'start')]

        explore_budget = int(n_actions * self.explore_budget_frac)
        visited_clones = {current_clone}  # clones actually visited

        for step in range(n_actions):
            # --- Available ego directions from live observation ---
            avail_ego_dirs = {i for i, passable in enumerate(current_ego_obs)
                              if passable}
            known_from_here = set(clone_edges.get(current_clone, {}).keys())
            unexplored = list(avail_ego_dirs - known_from_here)

            # Check if explore budget exhausted
            budget_exhausted = (explore_count + nav_count) >= explore_budget

            # --- DECIDE: explore, navigate, or exploit ---
            step_mode = 'exploit'
            if not budget_exhausted and unexplored:
                ego_action = random.choice(unexplored)
                explore_count += 1
                step_mode = 'explore'
            elif not budget_exhausted and self._has_frontier(clone_edges, clone_obs):
                nav_count += 1
                step_mode = 'navigate'
                next_dir = self._bfs_to_frontier(
                    current_clone, clone_edges, clone_obs, None)
                if next_dir is not None:
                    ego_action = next_dir
                else:
                    ego_action = random.choice(list(known_from_here)) if known_from_here else random.choice(list(avail_ego_dirs))
            else:
                # EXPLOIT: BFS to nearest believed-rewarded clone
                exploit_count += 1

                target_dir = self._bfs_to_reward(
                    current_clone, clone_edges, believed_rewards)
                if target_dir is not None:
                    ego_action = target_dir
                elif known_from_here:
                    ego_action = random.choice(list(known_from_here))
                else:
                    # Current clone not in graph at all — use live obs
                    ego_action = random.choice(list(avail_ego_dirs))

            # --- EXECUTE via simulator ---
            new_ego_obs, step_reward, moved = _sim_step(ego_action)
            total_reward += step_reward

            # --- UPDATE CSCG ---
            if moved:
                prev_clone = current_clone
                new_obs_idx = _obs_to_idx(new_ego_obs)

                # CSCG forward pass: predict → filter → MAP assign
                alpha_pred = T[ego_action].T @ alpha

                # Filter: zero out clones not matching new observation
                mask = np.zeros(H)
                mask[new_obs_idx * M:(new_obs_idx + 1) * M] = 1.0
                alpha_filtered = alpha_pred * mask
                alpha_sum = alpha_filtered.sum()
                if alpha_sum > 0:
                    alpha_filtered /= alpha_sum
                else:
                    # No predicted mass — use first-visit heuristic
                    clone_start = new_obs_idx * M
                    clone_end = clone_start + M
                    least_used_idx = int(np.argmin(
                        clone_usage[clone_start:clone_end]))
                    alpha_filtered = np.zeros(H)
                    alpha_filtered[clone_start + least_used_idx] = 1.0

                # MAP assignment
                clone_start = new_obs_idx * M
                clone_end = clone_start + M
                current_clone = clone_start + int(
                    np.argmax(alpha_filtered[clone_start:clone_end]))
                clone_usage[current_clone] += 1

                # Update transition counts and re-normalise
                C[ego_action, prev_clone, current_clone] += 1
                row = C[ego_action, prev_clone] + self.cscg_pseudocount
                T[ego_action, prev_clone] = row / row.sum()

                # Update belief
                alpha = alpha_filtered

                # Register clone in graph
                clone_obs[current_clone] = new_ego_obs
                visited_clones.add(current_clone)

                # Record edge in clone graph
                if prev_clone not in clone_edges:
                    clone_edges[prev_clone] = {}
                clone_edges[prev_clone][ego_action] = current_clone

                # Full graph extraction from T for all visited clones
                if step_mode != 'exploit':
                    for h in visited_clones:
                        for a in range(A):
                            if C[a, h].sum() > 0:
                                best_next = int(np.argmax(T[a, h]))
                                if T[a, h, best_next] > self.cscg_edge_threshold:
                                    if h not in clone_edges:
                                        clone_edges[h] = {}
                                    if a not in clone_edges[h]:
                                        clone_edges[h][a] = best_next
                                        dest_obs_idx = best_next // M
                                        if best_next not in clone_obs:
                                            clone_obs[best_next] = _idx_to_obs(dest_obs_idx)

                # Learn reward at destination
                if current_clone not in believed_rewards:
                    believed_rewards[current_clone] = 1.0  # optimistic prior

                # Rebait detection (CHECK → REBAIT → REMOVE)
                if believed_rewards.get(current_clone, 0.0) > 0:
                    believed_sum = sum(believed_rewards.values())
                    if believed_sum == min_rewarded_states:
                        if not knows_rebait:
                            knows_rebait = True
                        for cid in believed_rewards:
                            believed_rewards[cid] = 1.0
                believed_rewards[current_clone] = 0.0  # visited

                current_ego_obs = new_ego_obs

            if return_trajectory:
                trajectory.append((current_clone, step_reward, step_mode))

            if verbose and step > 0 and step % 100 == 0:
                n_clones_used = len(clone_obs)
                n_clone_edges = sum(len(v) for v in clone_edges.values())
                print(f"  CSCG_bio_BFS step {step}/{n_actions}: "
                      f"reward={total_reward:.0f}, "
                      f"clones={n_clones_used} (phys={n_nodes}), "
                      f"clone_edges={n_clone_edges}, "
                      f"explore={explore_count}, nav={nav_count}, "
                      f"exploit={exploit_count}")

        if verbose:
            n_clones_used = len(clone_obs)
            n_clone_edges = sum(len(v) for v in clone_edges.values())
            print(f"  CSCG_bio_BFS final: reward={total_reward:.0f}, "
                  f"clones={n_clones_used}/{n_nodes} (×{n_clones_used/n_nodes:.1f}), "
                  f"clone_edges={n_clone_edges}, "
                  f"explore={explore_count}, nav={nav_count}, "
                  f"exploit={exploit_count}")

        if return_trajectory:
            return total_reward, trajectory
        return total_reward

    @staticmethod
    def _bfs_to_reward(current_clone, clone_edges, believed_rewards):
        """BFS over clone_edges to find direction to nearest rewarded clone.

        Returns the ego action (int) for the first step, or None if no
        reachable rewarded clone exists.
        """
        from collections import deque

        if believed_rewards.get(current_clone, 0.0) > 0:
            # Already at a rewarded clone — shouldn't happen in practice
            # but return None to let caller pick any direction
            return None

        queue = deque()
        visited = {current_clone}

        for d, nxt in clone_edges.get(current_clone, {}).items():
            if nxt not in visited:
                visited.add(nxt)
                queue.append((nxt, d))

        while queue:
            node, first_dir = queue.popleft()
            if believed_rewards.get(node, 0.0) > 0:
                return first_dir
            for d, nxt in clone_edges.get(node, {}).items():
                if nxt not in visited:
                    visited.add(nxt)
                    queue.append((nxt, first_dir))

        return None


# Backward compatibility alias
POMCPBioAgent = POMCPbioAlloAgent  # backward compat alias
POMCPBioCSCGAgent = CSCGBioBFSAgent


# =============================================================================
# CSCG_bio_POMCP: Clone Structured Cognitive Graph + POMCP exploit
# =============================================================================

class CSCGBioPOMCPAgent(CSCGBioBFSAgent):
    """CSCG agent that uses POMCP (not BFS) for exploit-mode planning.

    Identical to CSCGBioBFSAgent for state representation (clone HMM) and
    explore/navigate phases.  In exploit mode, builds a POMDP model over
    clone IDs and runs POMCP tree search instead of greedy BFS.

    This mirrors POMCPbioEgoAgent's exploit phase but with CSCG clone IDs
    as the state space (vs observation-tree node IDs).

    Args:
        Same as CSCGBioBFSAgent, plus inherited POMCP params (max_depth,
        num_sims, exploration_const, discount).
    """

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
        return_trajectory: bool = False,
    ) -> float:
        """Run episode using CSCG representation + POMCP exploit planning.

        Identical to CSCGBioBFSAgent.run_episode except the exploit block
        uses POMCP tree search over clone IDs instead of BFS.
        """
        np.random.seed(seed)
        random.seed(seed)

        n_nodes = len(adj_mat)
        E = 16  # possible 4-bit ego observations
        A = 4   # ego actions (F/B/L/R)
        M = self.n_clones
        H = M * E

        # ============================================================
        # HIDDEN WORLD (agent never accesses these directly)
        # ============================================================
        real_rewards = list(rewarded_nodes.tolist())
        rewarded_initial = tuple(rewarded_nodes.tolist())

        dir_to_neighbor = {}
        for node in range(n_nodes):
            dir_to_neighbor[node] = {}
            for adj in get_adj_states(node, adj_mat):
                disp = node_positions[adj] - node_positions[node]
                d = displacement_to_compass_heading(disp)
                dir_to_neighbor[node][allo_actions_one_hot_dict[d]] = adj

        real_node = start_node
        real_heading_idx = 0

        def _compute_ego_obs(node, heading_idx):
            ego_obs = [False, False, False, False]
            for allo_dir in dir_to_neighbor[node]:
                ego_dir = ALLO_TO_EGO[heading_idx][allo_dir]
                ego_obs[ego_dir] = True
            return tuple(ego_obs)

        def _sim_step(ego_action):
            nonlocal real_node, real_heading_idx, real_rewards
            allo_dir = EGO_TO_ALLO[real_heading_idx][ego_action]
            next_node = dir_to_neighbor[real_node].get(allo_dir, real_node)
            if next_node != real_node:
                real_heading_idx = allo_dir
                real_node = next_node
                reward = float(real_rewards[real_node])
                # Rebait (CHECK → REBAIT → REMOVE)
                if reward > 0:
                    if sum(real_rewards) == min_rewarded_states:
                        real_rewards = list(rewarded_initial)
                    real_rewards[real_node] = 0.0
                return _compute_ego_obs(real_node, real_heading_idx), reward, True
            else:
                return _compute_ego_obs(real_node, real_heading_idx), 0.0, False

        def _obs_to_idx(ego_obs):
            return sum(v << i for i, v in enumerate(ego_obs))

        def _idx_to_obs(idx):
            return tuple(bool((idx >> i) & 1) for i in range(4))

        # ============================================================
        # CSCG STATE (identical to CSCGBioBFSAgent)
        # ============================================================
        C = np.zeros((A, H, H))
        T = np.full((A, H, H), self.cscg_pseudocount)
        rng = np.random.RandomState(seed)
        T += rng.uniform(0, self.cscg_pseudocount * 0.1, T.shape)
        for a in range(A):
            row_sums = T[a].sum(axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1.0
            T[a] /= row_sums

        alpha = np.zeros(H)
        clone_usage = np.zeros(H)
        current_clone = None

        clone_edges = {}
        clone_obs = {}
        believed_rewards = {}
        knows_rebait = False

        initial_ego_obs = _compute_ego_obs(real_node, real_heading_idx)
        obs_idx = _obs_to_idx(initial_ego_obs)

        clone_start = obs_idx * M
        least_used = clone_start + int(np.argmin(clone_usage[clone_start:clone_start + M]))
        current_clone = least_used
        clone_usage[current_clone] += 1
        alpha[current_clone] = 1.0

        clone_obs[current_clone] = initial_ego_obs
        believed_rewards[current_clone] = 0.0
        current_ego_obs = initial_ego_obs

        # ============================================================
        # Episode loop
        # ============================================================
        total_reward = 0.0
        explore_count = 0
        exploit_count = 0
        nav_count = 0
        if return_trajectory:
            trajectory = [(current_clone, 0.0, 'start')]

        explore_budget = int(n_actions * self.explore_budget_frac)
        visited_clones = {current_clone}

        # POMCP planner state (persists across exploit steps for tree reuse)
        _pomcp_planner = None
        _pomcp_agent = None
        _last_exploit_action = None

        for step in range(n_actions):
            avail_ego_dirs = {i for i, passable in enumerate(current_ego_obs)
                              if passable}
            known_from_here = set(clone_edges.get(current_clone, {}).keys())
            unexplored = list(avail_ego_dirs - known_from_here)

            budget_exhausted = (explore_count + nav_count) >= explore_budget

            # --- DECIDE: explore, navigate, or exploit ---
            step_mode = 'exploit'
            if not budget_exhausted and unexplored:
                ego_action = random.choice(unexplored)
                explore_count += 1
                step_mode = 'explore'
            elif not budget_exhausted and self._has_frontier(clone_edges, clone_obs):
                nav_count += 1
                step_mode = 'navigate'
                next_dir = self._bfs_to_frontier(
                    current_clone, clone_edges, clone_obs, None)
                if next_dir is not None:
                    ego_action = next_dir
                else:
                    ego_action = random.choice(list(known_from_here)) if known_from_here else random.choice(list(avail_ego_dirs))
            else:
                # EXPLOIT: POMCP over clone graph (NOT BFS)
                exploit_count += 1

                can_reuse = (_pomcp_planner is not None
                             and _last_exploit_action is not None)

                max_id = max(believed_rewards.keys()) + 1 if believed_rewards else 0
                bio_rewards_tuple = tuple(
                    believed_rewards.get(n, 0.0) for n in range(max_id))

                obs_model = _BioObservationModel(clone_obs)

                if can_reuse:
                    last_obs = obs_model.sample(
                        MazeState(current_clone, bio_rewards_tuple),
                        _last_exploit_action)
                    try:
                        _pomcp_planner.update(
                            _pomcp_agent, _last_exploit_action, last_obs)
                    except ValueError:
                        can_reuse = False

                if not can_reuse:
                    bio_transition = _BioTransitionModel(
                        clone_edges, min_rewarded_states, knows_rebait)
                    bio_state = MazeState(current_clone, bio_rewards_tuple)
                    bio_belief = pomdp_py.Particles([bio_state] * 100)

                    reward_model = MazeRewardModel()
                    policy = MazePolicyModel()
                    if self.novelty_rollout:
                        rollout_policy = NoveltySeekingRolloutPolicy(
                            bio_transition._transitions)
                    else:
                        rollout_policy = MazeRolloutPolicy()
                    blackbox = MazeBlackboxModel(
                        bio_transition, obs_model, reward_model)

                    _pomcp_agent = pomdp_py.Agent(
                        bio_belief, policy, bio_transition, obs_model,
                        reward_model, blackbox_model=blackbox)

                    _pomcp_planner = POMCP(
                        max_depth=self.max_depth,
                        discount_factor=self.discount,
                        num_sims=self.num_sims,
                        exploration_const=self.exploration_const,
                        rollout_policy=rollout_policy)

                action_obj = _pomcp_planner.plan(_pomcp_agent)
                ego_action = action_obj.direction
                _last_exploit_action = action_obj

            # --- EXECUTE via simulator ---
            new_ego_obs, step_reward, moved = _sim_step(ego_action)
            total_reward += step_reward

            # --- UPDATE CSCG (identical to CSCGBioBFSAgent) ---
            if moved:
                prev_clone = current_clone
                new_obs_idx = _obs_to_idx(new_ego_obs)

                alpha_pred = T[ego_action].T @ alpha

                mask = np.zeros(H)
                mask[new_obs_idx * M:(new_obs_idx + 1) * M] = 1.0
                alpha_filtered = alpha_pred * mask
                alpha_sum = alpha_filtered.sum()
                if alpha_sum > 0:
                    alpha_filtered /= alpha_sum
                else:
                    cs = new_obs_idx * M
                    ce = cs + M
                    least_used_idx = int(np.argmin(clone_usage[cs:ce]))
                    alpha_filtered = np.zeros(H)
                    alpha_filtered[cs + least_used_idx] = 1.0

                cs = new_obs_idx * M
                ce = cs + M
                current_clone = cs + int(np.argmax(alpha_filtered[cs:ce]))
                clone_usage[current_clone] += 1

                C[ego_action, prev_clone, current_clone] += 1
                row = C[ego_action, prev_clone] + self.cscg_pseudocount
                T[ego_action, prev_clone] = row / row.sum()

                alpha = alpha_filtered

                clone_obs[current_clone] = new_ego_obs
                visited_clones.add(current_clone)

                if prev_clone not in clone_edges:
                    clone_edges[prev_clone] = {}
                clone_edges[prev_clone][ego_action] = current_clone

                # Full graph extraction from T for all visited clones
                if step_mode != 'exploit':
                    for h in visited_clones:
                        for a in range(A):
                            if C[a, h].sum() > 0:
                                best_next = int(np.argmax(T[a, h]))
                                if T[a, h, best_next] > self.cscg_edge_threshold:
                                    if h not in clone_edges:
                                        clone_edges[h] = {}
                                    if a not in clone_edges[h]:
                                        clone_edges[h][a] = best_next
                                        dest_obs_idx = best_next // M
                                        if best_next not in clone_obs:
                                            clone_obs[best_next] = _idx_to_obs(dest_obs_idx)

                if current_clone not in believed_rewards:
                    believed_rewards[current_clone] = 1.0

                # Rebait detection (CHECK → REBAIT → REMOVE)
                if believed_rewards.get(current_clone, 0.0) > 0:
                    believed_sum = sum(believed_rewards.values())
                    if believed_sum == min_rewarded_states:
                        if not knows_rebait:
                            knows_rebait = True
                        for cid in believed_rewards:
                            believed_rewards[cid] = 1.0
                believed_rewards[current_clone] = 0.0

                current_ego_obs = new_ego_obs

            if return_trajectory:
                trajectory.append((current_clone, step_reward, step_mode))

            # Invalidate POMCP tree when leaving exploit mode
            if step_mode != 'exploit':
                _pomcp_planner = None
                _pomcp_agent = None
                _last_exploit_action = None

            if verbose and step > 0 and step % 100 == 0:
                n_clones_used = len(clone_obs)
                n_clone_edges = sum(len(v) for v in clone_edges.values())
                print(f"  CSCG_bio_POMCP step {step}/{n_actions}: "
                      f"reward={total_reward:.0f}, "
                      f"clones={n_clones_used} (phys={n_nodes}), "
                      f"clone_edges={n_clone_edges}, "
                      f"explore={explore_count}, nav={nav_count}, "
                      f"exploit={exploit_count}")

        if verbose:
            n_clones_used = len(clone_obs)
            n_clone_edges = sum(len(v) for v in clone_edges.values())
            print(f"  CSCG_bio_POMCP final: reward={total_reward:.0f}, "
                  f"clones={n_clones_used}/{n_nodes} (×{n_clones_used/n_nodes:.1f}), "
                  f"clone_edges={n_clone_edges}, "
                  f"explore={explore_count}, nav={nav_count}, "
                  f"exploit={exploit_count}")

        if return_trajectory:
            return total_reward, trajectory
        return total_reward


# =============================================================================
# BeliefOracle: Belief-State Oracle Agent (no pomdp_py dependency)
# =============================================================================

class BeliefOracleAgent:
    """Belief-state oracle agent for graph maze navigation.

    Uses explicit belief tracking over a lattice prior instead of
    particle-based MCTS.  Given the approximate graph size (lattice
    dimensions inferred from node_positions), it maintains beliefs about
    which lattice positions are nodes and where rewards are.

    Key insight: corridor observations (NSWE booleans) at each visited
    node allow inferring connectivity of unvisited lattice positions —
    something POMCP_bio cannot do.

    Planning: BFS to nearest high-value target each step (O(V+E) per
    step vs POMCP's 25,000 simulated transitions).

    No external dependencies (no pomdp_py required).

    Args:
        exploration_bonus: Extra value assigned to unvisited nodes during
            BFS planning (controls explore/exploit tradeoff).
        optimistic_reward_prior: Whether unvisited nodes are assumed to
            have reward.
    """

    # Direction index to lattice displacement: N, S, W, E
    DIR_DELTA = {0: (0, 1), 1: (0, -1), 2: (-1, 0), 3: (1, 0)}
    # Reverse direction: N↔S (0↔1), W↔E (2↔3)
    REV_DIR = {0: 1, 1: 0, 2: 3, 3: 2}

    def __init__(self, exploration_bonus=0.5, optimistic_reward_prior=True):
        self.exploration_bonus = exploration_bonus
        self.optimistic_reward_prior = optimistic_reward_prior

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
        """Run a single episode with belief-state tracking.

        The agent discovers the maze through corridor observations and
        plans via BFS over its believed graph.  The real maze (adj_mat,
        rewarded_nodes) is hidden from the agent; it only learns through
        its own traversals and corridor observations.

        Returns:
            Total reward collected during the episode.
        """
        np.random.seed(seed)
        random.seed(seed)

        n_nodes = len(adj_mat)

        # ============================================================
        # REAL WORLD (hidden from agent — used to compute outcomes)
        # ============================================================
        real_rewards = list(rewarded_nodes.tolist())
        rewarded_initial = tuple(rewarded_nodes.tolist())

        # Precompute direction↔neighbor mapping (real world)
        dir_to_neighbor = {}
        for node in range(n_nodes):
            dir_to_neighbor[node] = {}
            for adj in get_adj_states(node, adj_mat):
                disp = node_positions[adj] - node_positions[node]
                d = displacement_to_compass_heading(disp)
                dir_to_neighbor[node][allo_actions_one_hot_dict[d]] = adj

        # Node↔lattice coordinate mappings
        node_to_xy = {}
        xy_to_node = {}
        for node in range(n_nodes):
            xy = (int(round(node_positions[node][0])),
                  int(round(node_positions[node][1])))
            node_to_xy[node] = xy
            xy_to_node[xy] = node

        # Lattice dimensions (the "approximate graph size" prior)
        lattice_w = int(np.max(node_positions[:, 0])) + 1
        lattice_h = int(np.max(node_positions[:, 1])) + 1

        # ============================================================
        # AGENT'S BELIEF STATE
        # ============================================================
        is_node = {}           # (x,y) -> True/False (None = unknown)
        believed_edges = {}    # (x,y) -> {dir_idx: (nx,ny)}
        believed_reward = {}   # (x,y) -> float (0.0 or 1.0)
        visited = set()        # set of (x,y)
        knows_rebait = False

        exploration_bonus = self.exploration_bonus
        optimistic_prior = self.optimistic_reward_prior

        def _in_bounds(x, y):
            return 0 <= x < lattice_w and 0 <= y < lattice_h

        def _observe_node(xy, obs_dirs):
            """Update beliefs when visiting node at lattice position xy.

            Args:
                xy: (x, y) lattice coordinates of the visited node.
                obs_dirs: set of direction indices that have neighbors.
            """
            x, y = xy
            is_node[xy] = True
            visited.add(xy)

            for d_idx in range(4):
                dx, dy = BeliefOracleAgent.DIR_DELTA[d_idx]
                nx, ny = x + dx, y + dy

                if d_idx in obs_dirs:
                    # Corridor exists → neighbor is a node
                    nxy = (nx, ny)
                    is_node[nxy] = True

                    # Record edge in both directions
                    if xy not in believed_edges:
                        believed_edges[xy] = {}
                    believed_edges[xy][d_idx] = nxy

                    if nxy not in believed_edges:
                        believed_edges[nxy] = {}
                    believed_edges[nxy][BeliefOracleAgent.REV_DIR[d_idx]] = xy

                    # Optimistic reward prior for unvisited neighbors
                    if optimistic_prior and nxy not in visited:
                        believed_reward.setdefault(nxy, 1.0)

                elif _in_bounds(nx, ny):
                    # No corridor → that lattice position is not a node
                    nxy = (nx, ny)
                    if nxy not in is_node:
                        is_node[nxy] = False

        def _choose_action(current_xy, prev_xy):
            """BFS over believed edges to find best next step.

            Scores each reachable node as:
                believed_reward + exploration_bonus (if unvisited)
            Returns direction index toward the nearest highest-scoring node.
            """
            from collections import deque

            edges = believed_edges
            if current_xy not in edges or not edges[current_xy]:
                return None

            queue = deque()
            bfs_visited = {current_xy}
            best_dir = None
            best_score = -1.0
            best_dist = float('inf')

            # Seed BFS with immediate neighbors
            for d_idx, nxy in edges[current_xy].items():
                if prevent_reverse and nxy == prev_xy:
                    continue
                if nxy not in bfs_visited:
                    bfs_visited.add(nxy)
                    queue.append((nxy, d_idx, 1))
                    score = believed_reward.get(nxy, 0.0)
                    if nxy not in visited:
                        score += exploration_bonus
                    if (score > best_score
                            or (score == best_score and 1 < best_dist)):
                        best_score = score
                        best_dir = d_idx
                        best_dist = 1

            # Expand BFS
            while queue:
                pos, first_dir, dist = queue.popleft()
                for d_idx, nxy in edges.get(pos, {}).items():
                    if nxy not in bfs_visited:
                        bfs_visited.add(nxy)
                        queue.append((nxy, first_dir, dist + 1))
                        score = believed_reward.get(nxy, 0.0)
                        if nxy not in visited:
                            score += exploration_bonus
                        if (score > best_score
                                or (score == best_score
                                    and dist + 1 < best_dist)):
                            best_score = score
                            best_dir = first_dir
                            best_dist = dist + 1

            if best_dir is not None and best_score > 0:
                return best_dir

            # Fallback: any valid direction (avoid reverse if possible)
            valid_dirs = list(edges.get(current_xy, {}).keys())
            if prevent_reverse and prev_xy is not None:
                forward = [d for d in valid_dirs
                           if edges[current_xy][d] != prev_xy]
                if forward:
                    return random.choice(forward)
            if valid_dirs:
                return random.choice(valid_dirs)
            return None

        # ============================================================
        # Episode loop
        # ============================================================
        current_node = start_node
        current_xy = node_to_xy[current_node]
        prev_xy = None
        total_reward = 0.0

        # Initial observation at start node
        obs_dirs = set(dir_to_neighbor[current_node].keys())
        _observe_node(current_xy, obs_dirs)

        for step in range(n_actions):
            # Choose action via BFS over believed graph
            action_dir = _choose_action(current_xy, prev_xy)

            if action_dir is None:
                # No valid action — stay in place (shouldn't happen)
                continue

            # Execute in real world
            next_node = dir_to_neighbor[current_node].get(
                action_dir, current_node)

            if next_node != current_node:
                step_reward = float(real_rewards[next_node])
                if step_reward > 0:
                    if sum(real_rewards) == min_rewarded_states:
                        if not knows_rebait:
                            knows_rebait = True
                        real_rewards = list(rewarded_initial)
                    real_rewards[next_node] = 0.0
            else:
                step_reward = 0.0

            total_reward += step_reward

            # Update beliefs from observation
            if next_node != current_node:
                next_xy = node_to_xy[next_node]
                obs_dirs = set(dir_to_neighbor[next_node].keys())

                _observe_node(next_xy, obs_dirs)

                # Handle believed rebait (CHECK → REBAIT → REMOVE)
                # Gate on step_reward > 0 (real reward collected), not on
                # belief value — _observe_node no longer clears belief, so
                # the sum is valid at this point.
                if step_reward > 0:
                    if knows_rebait:
                        believed_sum = sum(believed_reward.values())
                        if believed_sum == min_rewarded_states:
                            for xy in believed_reward:
                                believed_reward[xy] = 1.0
                    believed_reward[next_xy] = 0.0

                prev_xy = current_xy
                current_xy = next_xy
                current_node = next_node

            if verbose and step > 0 and step % 100 == 0:
                print(f"  BeliefOracle step {step}/{n_actions}: "
                      f"reward={total_reward:.0f}, "
                      f"visited={len(visited)}")

        if verbose:
            print(f"  BeliefOracle final: reward={total_reward:.0f}, "
                  f"visited={len(visited)}")

        return total_reward
