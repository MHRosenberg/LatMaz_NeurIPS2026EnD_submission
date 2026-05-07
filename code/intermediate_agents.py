"""
Intermediate-complexity agents for the latent maze benchmark.

Four agent classes that sit between a forward-biased random walk and a
full RNN / belief-state planner:

1. VarMarkovAgent  — variable-depth Markov chain policy (online learning)
2. FSCAgent        — finite-state controller (CEM-trained, oracle)
3. OOIAgent        — options with option-observation initiation sets
4. FSCBioAgent     — biologically plausible FSC (online reward-modulated)

All agents use the planning-agent interface: run_episode(adj_mat, ...).

References:
- Rosenberg, Zhang, Perona, Meister (2021). eLife 10:e66175.
- Meuleau, Peshkin, Kim, Kaelbling (1999). Learning FSCs for PO envs.
- Sutton, Precup & Singh (1999). Between MDPs and Semi-MDPs. Artificial Intelligence.
  [Options framework — basis for OOIAgent]
- Steckelmacher, Roijers & Nowé (2018). RL in POMDPs with Memoryless Options and
  Option-Observation Initiation Sets. AAAI.
  [Source of the OOI initiation constraint concept; OOIAgent is a custom hand-designed
  implementation using this concept, not a faithful replication of the published algorithm]
"""

import os
import sys
import random
from collections import defaultdict, deque

import numpy as np

# Path setup (same as advanced_agents.py)
_this_dir = os.path.dirname(os.path.abspath(__file__))
_code_dir = _this_dir
if _code_dir not in sys.path:
    sys.path.insert(0, _code_dir)

from utils_latMaz import (
    get_adj_states,
    displacement_to_compass_heading,
    allo_actions_one_hot_dict,
)
from advanced_agents import EGO_TO_ALLO, ALLO_TO_EGO


# =============================================================================
# Internal Egocentric Simulator
# =============================================================================

class _EgoSimulator:
    """Egocentric maze simulator with PacMan-style rewards.

    Encapsulates the hidden world — agents interact only via step().
    Shared across all three agent classes.
    """

    A = 4   # ego actions: F=0, B=1, L=2, R=3
    O = 16  # possible 4-bit ego observations

    def __init__(self, adj_mat, node_positions, start_node, rewarded_nodes,
                 min_rewarded_states=2, prevent_reverse=False):
        self.n_nodes = len(adj_mat)
        self.start_node = start_node
        self.rewarded_initial = tuple(float(r) for r in rewarded_nodes)
        self.min_rewarded_states = min_rewarded_states
        self.prevent_reverse = prevent_reverse

        # Precompute direction→neighbor mapping
        self.dir_to_neighbor = {}
        for node in range(self.n_nodes):
            self.dir_to_neighbor[node] = {}
            for adj in get_adj_states(node, adj_mat):
                disp = node_positions[adj] - node_positions[node]
                d = displacement_to_compass_heading(disp)
                self.dir_to_neighbor[node][allo_actions_one_hot_dict[d]] = adj

    def reset(self):
        """Reset to initial state. Returns initial ego observation."""
        self.real_node = self.start_node
        self.real_heading_idx = 0  # facing N
        self.real_rewards = list(self.rewarded_initial)
        self.total_reward = 0.0
        self.step_count = 0
        self._prev_node = None
        return self._compute_ego_obs()

    def _compute_ego_obs(self):
        """4-bit egocentric corridor observation at current (node, heading).

        When prevent_reverse is on and more than one exit exists, the
        backward direction is masked (reported as impassable), mirroring
        GraphMazeEnv.prevent_reverse behaviour.
        """
        ego_obs = [False, False, False, False]
        for allo_dir, neighbor in self.dir_to_neighbor[self.real_node].items():
            ego_dir = ALLO_TO_EGO[self.real_heading_idx][allo_dir]
            ego_obs[ego_dir] = True
        # Mask backward direction at env level (unless dead-end)
        if (self.prevent_reverse and self._prev_node is not None
                and sum(ego_obs) > 1):
            ego_obs[1] = False  # B = ego action index 1
        return tuple(ego_obs)

    def step(self, ego_action):
        """Execute ego action. Returns (ego_obs, reward, moved).

        When prevent_reverse is on, backward actions are blocked (treated
        as self-loops) unless the agent is at a dead-end, matching the
        GraphMazeEnv.step prevent_reverse logic.
        """
        allo_dir = EGO_TO_ALLO[self.real_heading_idx][ego_action]
        next_node = self.dir_to_neighbor[self.real_node].get(allo_dir, self.real_node)

        # Block reversal at env level (allow at dead-ends)
        if (self.prevent_reverse and self._prev_node is not None
                and next_node == self._prev_node
                and len(self.dir_to_neighbor[self.real_node]) > 1):
            next_node = self.real_node  # treat as self-loop

        if next_node != self.real_node:
            self._prev_node = self.real_node
            self.real_heading_idx = allo_dir
            self.real_node = next_node
            reward = self.real_rewards[self.real_node]
            # Rebait (CHECK → REBAIT → REMOVE)
            if reward > 0:
                if sum(self.real_rewards) == self.min_rewarded_states:
                    self.real_rewards = list(self.rewarded_initial)
                self.real_rewards[self.real_node] = 0.0
            self.total_reward += reward
            self.step_count += 1
            return self._compute_ego_obs(), reward, True
        else:
            self.step_count += 1
            return self._compute_ego_obs(), 0.0, False

    @staticmethod
    def obs_to_idx(ego_obs):
        """Convert 4-bit ego obs tuple to integer index [0..15]."""
        return sum(int(v) << i for i, v in enumerate(ego_obs))


# =============================================================================
# 1. Variable-Depth Markov Chain Agent
# =============================================================================

class VarMarkovAgent:
    """Variable-depth Markov chain policy with online learning.

    Adapted from Rosenberg/Meister (eLife 2021). Maintains running counts
    of P(action | recent_actions, current_obs) and uses variable-depth
    backoff when counts are sparse.

    Egocentric only — no node IDs, no maze knowledge.

    The history context is (a_{t-k+1}, ..., a_{t-1}, obs_idx) for depth k.
    Depth 1 = reactive policy P(a | obs).
    Depth 2 = P(a | prev_action, obs).
    etc.

    Args:
        max_depth: Maximum history depth (1-6). Higher = more context but sparser.
        min_count: Minimum total count to trust a context (variable-depth threshold).
            Set to 0 to always use max_depth (fixed-depth mode).
        pseudocount: Additive smoothing per action (Laplace prior).
        forward_bias: Extra pseudocount for Forward action (action 0).
        prevent_backward: If True, mask Backward action (ego action 1) when other
            directions are passable. Approximates the forward-random heuristic.
    """

    def __init__(self, max_depth=4, min_count=3, pseudocount=1.0,
                 forward_bias=0.5, prevent_backward=False):
        self.max_depth = max_depth
        self.min_count = min_count
        self.pseudocount = pseudocount
        self.forward_bias = forward_bias
        self.prevent_backward = prevent_backward

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
        """Run episode with online variable-depth Markov policy."""
        np.random.seed(seed)
        random.seed(seed)
        rng = np.random.RandomState(seed)

        sim = _EgoSimulator(adj_mat, node_positions, start_node,
                            rewarded_nodes, min_rewarded_states,
                            prevent_reverse=prevent_reverse)
        ego_obs = sim.reset()

        # Counts: context_tuple → array of 4 action counts
        counts = defaultdict(lambda: np.zeros(4))
        action_history = deque(maxlen=self.max_depth)
        trajectory = []

        for step in range(n_actions):
            obs_idx = sim.obs_to_idx(ego_obs)

            # Get action distribution via variable-depth backoff
            action_probs = self._get_action_probs(
                counts, action_history, obs_idx)

            # Mask impassable directions
            for a in range(4):
                if not ego_obs[a]:
                    action_probs[a] = 0.0

            # Optionally mask backward (ego action 1) unless only option
            if self.prevent_backward and sum(ego_obs) > 1:
                action_probs[1] = 0.0

            total = action_probs.sum()
            if total > 0:
                action_probs /= total
            else:
                action_probs = np.ones(4) / 4.0

            action = rng.choice(4, p=action_probs)

            # Update counts at ALL depths (enables future backoff)
            for d in range(1, self.max_depth + 1):
                ctx = self._build_context(action_history, obs_idx, d)
                if ctx is not None:
                    counts[ctx][action] += 1.0

            # Step
            new_obs, reward, moved = sim.step(action)
            if return_trajectory:
                trajectory.append((sim.real_node, reward, 'markov'))

            action_history.append(action)
            ego_obs = new_obs

        if return_trajectory:
            return sim.total_reward, trajectory
        return sim.total_reward

    def _build_context(self, action_history, obs_idx, depth):
        """Build context tuple: (a_{t-k+1}, ..., a_{t-1}, obs_idx)."""
        n_actions_needed = depth - 1
        if len(action_history) < n_actions_needed:
            return None
        actions = list(action_history)[-n_actions_needed:] if n_actions_needed > 0 else []
        return tuple(actions) + (obs_idx,)

    def _get_action_probs(self, counts, action_history, obs_idx):
        """Variable-depth backoff: deepest sufficient context first."""
        for d in range(self.max_depth, 0, -1):
            ctx = self._build_context(action_history, obs_idx, d)
            if ctx is None:
                continue
            c = counts[ctx]
            if c.sum() >= self.min_count:
                probs = c + self.pseudocount
                probs[0] += self.forward_bias
                return probs / probs.sum()

        # No sufficient context — use prior
        probs = np.full(4, self.pseudocount)
        probs[0] += self.forward_bias
        return probs / probs.sum()


# =============================================================================
# 2. Finite-State Controller Agent (CEM-Trained, Oracle)
# =============================================================================

def _softmax(logits):
    """Numerically stable softmax."""
    e = np.exp(logits - logits.max())
    return e / e.sum()


class FSCAgent:
    """Stochastic finite-state controller trained via CEM.

    Moore-style FSC: action depends on controller node only (psi(a|q)),
    transitions depend on (q, a, obs): eta(q'|q, a, obs).

    Oracle agent: uses full maze knowledge for CEM training (comparable
    to POMCP_bio). For a bio-constrained version, would need online
    REINFORCE instead of CEM.

    Args:
        n_nodes: Number of controller nodes K (2, 4, 8).
        n_cem_iters: CEM iterations.
        n_candidates: Candidates per CEM iteration.
        n_elite: Number of elite samples kept.
        n_eval_episodes: Rollouts per candidate for evaluation.
        train_budget: Max steps per training episode (None = full n_actions).
    """

    def __init__(self, n_nodes=4, n_cem_iters=20, n_candidates=50,
                 n_elite=10, n_eval_episodes=3, train_budget=None):
        self.n_nodes = n_nodes
        self.n_cem_iters = n_cem_iters
        self.n_candidates = n_candidates
        self.n_elite = n_elite
        self.n_eval_episodes = n_eval_episodes
        self.train_budget = train_budget

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
        """Train FSC via CEM, then deploy on the episode."""
        rng = np.random.RandomState(seed)

        sim = _EgoSimulator(adj_mat, node_positions, start_node,
                            rewarded_nodes, min_rewarded_states,
                            prevent_reverse=prevent_reverse)

        K = self.n_nodes
        A = 4   # ego actions
        O = 16  # ego observations

        # Parameter dimensions
        n_psi = K * A
        n_eta = K * A * O * K
        n_params = n_psi + n_eta

        train_steps = self.train_budget or n_actions

        # CEM: Gaussian search over softmax logits
        mean = np.zeros(n_params)
        std = np.ones(n_params) * 2.0
        best_params = mean.copy()
        best_reward = -float('inf')

        for cem_iter in range(self.n_cem_iters):
            candidates = rng.randn(self.n_candidates, n_params) * std + mean
            rewards = np.zeros(self.n_candidates)

            for ci in range(self.n_candidates):
                psi_logits = candidates[ci][:n_psi].reshape(K, A)
                eta_logits = candidates[ci][n_psi:].reshape(K, A, O, K)
                total_r = 0.0
                for ep in range(self.n_eval_episodes):
                    total_r += self._simulate_fsc(
                        sim, psi_logits, eta_logits, train_steps, rng, K, A)
                rewards[ci] = total_r / self.n_eval_episodes

            # Select elite
            elite_idx = np.argsort(rewards)[-self.n_elite:]
            elite = candidates[elite_idx]
            mean = elite.mean(axis=0)
            std = np.clip(elite.std(axis=0), 0.01, 5.0)

            if rewards[elite_idx[-1]] > best_reward:
                best_reward = rewards[elite_idx[-1]]
                best_params = candidates[elite_idx[-1]].copy()

            if verbose:
                print(f"  CEM iter {cem_iter}: best={best_reward:.2f} "
                      f"mean_elite={rewards[elite_idx].mean():.2f}")

        # Deploy best FSC on actual episode
        psi_logits = best_params[:n_psi].reshape(K, A)
        eta_logits = best_params[n_psi:].reshape(K, A, O, K)

        sim.reset()
        ego_obs = sim._compute_ego_obs()
        q = 0  # start in controller node 0
        trajectory = []

        for step in range(n_actions):
            obs_idx = sim.obs_to_idx(ego_obs)

            # Action from psi(a|q), masked by passable directions
            probs = _softmax(psi_logits[q])
            for a in range(A):
                if not ego_obs[a]:
                    probs[a] = 0.0
            total = probs.sum()
            if total > 0:
                probs /= total
            else:
                probs = np.ones(A) / A

            action = rng.choice(A, p=probs)
            ego_obs, reward, moved = sim.step(action)

            if return_trajectory:
                trajectory.append((sim.real_node, reward, f'q{q}'))

            # Controller state transition
            eta_probs = _softmax(eta_logits[q, action, obs_idx])
            q = rng.choice(K, p=eta_probs)

        if return_trajectory:
            return sim.total_reward, trajectory
        return sim.total_reward

    def _simulate_fsc(self, sim, psi_logits, eta_logits, n_steps, rng, K, A):
        """Simulate one episode with given FSC parameters."""
        sim.reset()
        ego_obs = sim._compute_ego_obs()
        q = 0

        for _ in range(n_steps):
            obs_idx = sim.obs_to_idx(ego_obs)
            probs = _softmax(psi_logits[q])
            for a in range(A):
                if not ego_obs[a]:
                    probs[a] = 0.0
            total = probs.sum()
            if total > 0:
                probs /= total
            else:
                probs = np.ones(A) / A

            action = rng.choice(A, p=probs)
            ego_obs, reward, moved = sim.step(action)

            eta_probs = _softmax(eta_logits[q, action, obs_idx])
            q = rng.choice(K, p=eta_probs)

        return sim.total_reward


# =============================================================================
# 3. Options with Option-Observation Initiation Sets (OOI)
# =============================================================================

class OOIAgent:
    """Custom options-based spatial navigation agent.

    Implements the Options framework (Sutton et al. 1999) with hand-designed options
    for egocentric maze navigation. Uses the option-observation initiation set concept
    (Steckelmacher et al. 2018): an option can only initiate if its initial direction
    is currently passable. NOT a faithful replication of either published OOI paper —
    options are fixed/hand-crafted rather than discovered or learned. See internal tracking notes.

    Egocentric only — no node IDs, no maze knowledge.

    Options (multi-step):
        0: follow_forward — go F while in corridor (2 exits), stop at junction/dead-end
        1: explore_left   — go L once, then follow corridor
        2: explore_right  — go R once, then follow corridor
        3: backtrack      — go B once, then follow corridor

    Top-level policy: P(next_option | prev_option, obs) learned online via counts.
    Initiation constraint: an option can only initiate if its initial direction is passable.

    Args:
        pseudocount: Additive smoothing for top-level selection.
        forward_bias: Extra pseudocount for follow_forward option.
        max_corridor_steps: Max steps within a single corridor-follow phase.
        prevent_backward: If True, exclude backtrack option unless only option.
    """

    N_OPTIONS = 4
    # Map option → initial ego action
    OPTION_ACTION = {0: 0, 1: 2, 2: 3, 3: 1}  # F, L, R, B

    def __init__(self, pseudocount=1.0, forward_bias=1.0,
                 max_corridor_steps=20, prevent_backward=False):
        self.pseudocount = pseudocount
        self.forward_bias = forward_bias
        self.prevent_backward = prevent_backward
        self.max_corridor_steps = max_corridor_steps

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
        """Run episode with OOI-based multi-step option selection."""
        np.random.seed(seed)
        random.seed(seed)
        rng = np.random.RandomState(seed)

        sim = _EgoSimulator(adj_mat, node_positions, start_node,
                            rewarded_nodes, min_rewarded_states,
                            prevent_reverse=prevent_reverse)
        ego_obs = sim.reset()

        O = 16
        N = self.N_OPTIONS

        # Top-level counts: P(next_option | prev_option, obs_at_decision)
        counts = np.full((N, O, N), self.pseudocount)
        counts[:, :, 0] += self.forward_bias  # bias toward forward

        prev_option = 0
        steps_used = 0
        trajectory = []

        while steps_used < n_actions:
            obs_idx = sim.obs_to_idx(ego_obs)

            # Determine which options are initiable (direction must be passable)
            allowed = []
            for opt in range(N):
                init_action = self.OPTION_ACTION[opt]
                if ego_obs[init_action]:
                    allowed.append(opt)

            # OOI constraint: exclude backtrack (opt 3) if prevent_backward
            if self.prevent_backward and len(allowed) > 1 and 3 in allowed:
                allowed.remove(3)
            if not allowed:
                # No passable direction (shouldn't happen in connected graph)
                # Take a random action to unstick
                ego_obs, _, _ = sim.step(rng.randint(4))
                steps_used += 1
                continue

            # Top-level selection from allowed options
            option_scores = counts[prev_option, obs_idx].copy()
            mask = np.zeros(N)
            for opt in allowed:
                mask[opt] = 1.0
            option_scores *= mask
            total = option_scores.sum()
            if total > 0:
                option_probs = option_scores / total
            else:
                option_probs = mask / mask.sum()

            option = rng.choice(N, p=option_probs)

            # Execute option
            ego_obs, n_steps = self._execute_option(
                sim, option, n_actions - steps_used, trajectory if return_trajectory else None)
            steps_used += n_steps

            # Update top-level counts
            counts[prev_option, obs_idx, option] += 1.0
            prev_option = option

        if return_trajectory:
            return sim.total_reward, trajectory
        return sim.total_reward

    def _execute_option(self, sim, option, max_steps, trajectory):
        """Execute a multi-step option. Returns (final_obs, steps_used)."""
        init_action = self.OPTION_ACTION[option]
        steps = 0

        # Initial action
        if steps >= max_steps:
            return sim._compute_ego_obs(), steps

        ego_obs, reward, moved = sim.step(init_action)
        steps += 1
        if trajectory is not None:
            trajectory.append((sim.real_node, reward, f'opt{option}'))

        # Follow corridor: continue F while exactly 2 exits and F is passable
        corridor_steps = 0
        while steps < max_steps and corridor_steps < self.max_corridor_steps:
            n_exits = sum(ego_obs)
            if n_exits != 2 or not ego_obs[0]:
                break  # junction, dead end, or forward blocked
            ego_obs, reward, moved = sim.step(0)  # Forward
            steps += 1
            corridor_steps += 1
            if trajectory is not None:
                trajectory.append((sim.real_node, reward, f'opt{option}_corridor'))

        return ego_obs, steps


# =============================================================================
# 3b. OOI Variants — Ablating Hand-Designed Features
# =============================================================================

class OOIVariantAgent(OOIAgent):
    """OOI with configurable termination rule for ablation studies.

    The standard OOI uses hand-designed corridor-following termination:
    continue Forward while exactly 2 exits and Forward is passable.

    This variant supports alternative termination rules:
    - 'corridor': standard corridor detection (same as OOIAgent)
    - 'single_step': no continuation — each option is one primitive action
    - 'fixed_N': continue Forward for exactly N steps (regardless of topology)
    - 'stochastic': continue Forward with probability (1-p_term) per step
    - 'any_forward': continue Forward whenever Forward is passable (not just corridors)

    Args:
        termination: Termination rule name (see above).
        term_param: Parameter for the termination rule:
            - 'fixed_N': number of continuation steps (default 3)
            - 'stochastic': termination probability per step (default 0.3)
            - 'corridor'/'single_step'/'any_forward': ignored
        (All other args inherited from OOIAgent.)
    """

    def __init__(self, termination='corridor', term_param=None,
                 pseudocount=1.0, forward_bias=1.0,
                 max_corridor_steps=20, prevent_backward=False):
        super().__init__(pseudocount=pseudocount, forward_bias=forward_bias,
                         max_corridor_steps=max_corridor_steps,
                         prevent_backward=prevent_backward)
        self.termination = termination
        self.term_param = term_param

    def _execute_option(self, sim, option, max_steps, trajectory):
        """Execute option with configurable termination rule."""
        init_action = self.OPTION_ACTION[option]
        steps = 0

        if steps >= max_steps:
            return sim._compute_ego_obs(), steps

        ego_obs, reward, moved = sim.step(init_action)
        steps += 1
        if trajectory is not None:
            trajectory.append((sim.real_node, reward, f'opt{option}'))

        if self.termination == 'single_step':
            return ego_obs, steps

        if self.termination == 'fixed_N':
            n_continue = self.term_param if self.term_param else 3
            for _ in range(n_continue):
                if steps >= max_steps or not ego_obs[0]:
                    break
                ego_obs, reward, moved = sim.step(0)
                steps += 1
                if trajectory is not None:
                    trajectory.append((sim.real_node, reward, f'opt{option}_fwd'))
            return ego_obs, steps

        if self.termination == 'stochastic':
            p_term = self.term_param if self.term_param else 0.3
            rng = np.random.RandomState()
            cont_steps = 0
            while steps < max_steps and cont_steps < self.max_corridor_steps:
                if not ego_obs[0]:
                    break
                if rng.random() < p_term:
                    break
                ego_obs, reward, moved = sim.step(0)
                steps += 1
                cont_steps += 1
                if trajectory is not None:
                    trajectory.append((sim.real_node, reward, f'opt{option}_stoch'))
            return ego_obs, steps

        if self.termination == 'any_forward':
            cont_steps = 0
            while steps < max_steps and cont_steps < self.max_corridor_steps:
                if not ego_obs[0]:
                    break
                ego_obs, reward, moved = sim.step(0)
                steps += 1
                cont_steps += 1
                if trajectory is not None:
                    trajectory.append((sim.real_node, reward, f'opt{option}_anyfwd'))
            return ego_obs, steps

        # Default: 'corridor' — same as OOIAgent
        corridor_steps = 0
        while steps < max_steps and corridor_steps < self.max_corridor_steps:
            n_exits = sum(ego_obs)
            if n_exits != 2 or not ego_obs[0]:
                break
            ego_obs, reward, moved = sim.step(0)
            steps += 1
            corridor_steps += 1
            if trajectory is not None:
                trajectory.append((sim.real_node, reward, f'opt{option}_corridor'))

        return ego_obs, steps


# =============================================================================
# 4. Biologically Plausible FSC (Online Reward-Modulated)
# =============================================================================

class FSCBioAgent:
    """Biologically plausible finite-state controller with online learning.

    K=2 internal states: explore (0) and exploit (1).
    Mealy-style: psi(a|q,obs) — action depends on controller state AND observation.
    No maze knowledge, no adjacency matrix access — purely egocentric + reward.

    Learning: reward-modulated count updates. When reward is received,
    increment counts for recent (q, obs, action) triples weighted by
    recency (eligibility trace with exponential decay). This is analogous
    to dopamine-modulated synaptic plasticity.

    Transitions (deterministic):
        explore -> exploit: on reward collection
        exploit -> explore: after exploit_timeout steps without reward

    Both states use prevent_backward heuristic by default.

    Args:
        exploit_timeout: Steps without reward before switching exploit->explore.
        trace_decay: Exponential decay for eligibility traces (0-1).
        pseudocount: Base smoothing per (q, obs, action).
        forward_bias: Extra bias for Forward action in explore state.
        prevent_backward: Mask backward action unless only option.
        exploit_greediness: How strongly to bias toward rewarded actions in
            exploit state. Higher = more greedy.
    """

    def __init__(self, exploit_timeout=10, trace_decay=0.8,
                 pseudocount=1.0, forward_bias=1.0,
                 prevent_backward=True, exploit_greediness=3.0):
        self.exploit_timeout = exploit_timeout
        self.trace_decay = trace_decay
        self.pseudocount = pseudocount
        self.forward_bias = forward_bias
        self.prevent_backward = prevent_backward
        self.exploit_greediness = exploit_greediness

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
        """Run episode with online reward-modulated FSC."""
        rng = np.random.RandomState(seed)

        sim = _EgoSimulator(adj_mat, node_positions, start_node,
                            rewarded_nodes, min_rewarded_states,
                            prevent_reverse=prevent_reverse)
        ego_obs = sim.reset()

        K = 2    # controller states: 0=explore, 1=exploit
        A = 4    # ego actions
        O = 16   # ego observations

        # Reward-modulated counts: how often (q, obs, action) led to reward
        reward_counts = np.zeros((K, O, A))

        # Eligibility trace: list of recent (q, obs_idx, action) triples
        trace = []

        q = 0  # start exploring
        steps_since_reward = 0
        trajectory = []

        for step in range(n_actions):
            obs_idx = sim.obs_to_idx(ego_obs)

            # Build action probabilities: psi(a | q, obs)
            action_probs = self._get_action_probs(
                q, obs_idx, ego_obs, reward_counts, rng)

            action = rng.choice(A, p=action_probs)

            # Record in eligibility trace
            trace.append((q, obs_idx, action))

            # Step
            ego_obs, reward, moved = sim.step(action)
            steps_since_reward += 1

            if return_trajectory:
                mode = 'explore' if q == 0 else 'exploit'
                trajectory.append((sim.real_node, reward, mode))

            # Reward-modulated learning
            if reward > 0:
                # Reinforce recent (q, obs, action) triples
                decay = 1.0
                for t_idx in range(len(trace) - 1, -1, -1):
                    tq, to, ta = trace[t_idx]
                    reward_counts[tq, to, ta] += decay
                    decay *= self.trace_decay
                    if decay < 0.01:
                        break

                # Transition: explore -> exploit
                q = 1
                steps_since_reward = 0

            elif q == 1 and steps_since_reward >= self.exploit_timeout:
                # Transition: exploit -> explore
                q = 0
                # Clear trace on mode switch to avoid stale associations
                trace.clear()

        if return_trajectory:
            return sim.total_reward, trajectory
        return sim.total_reward

    def _get_action_probs(self, q, obs_idx, ego_obs, reward_counts, rng):
        """Compute action probs for state q at observation obs_idx."""
        A = 4

        if q == 0:
            # Explore: uniform + forward bias + slight reward-count influence
            probs = np.full(A, self.pseudocount)
            probs[0] += self.forward_bias
            # Still use reward info but weakly
            probs += reward_counts[q, obs_idx] * 0.5
        else:
            # Exploit: strongly biased toward rewarded actions
            probs = np.full(A, self.pseudocount)
            probs += reward_counts[q, obs_idx] * self.exploit_greediness
            # Also use explore-state reward info (transfer)
            probs += reward_counts[0, obs_idx] * self.exploit_greediness * 0.5

        # Mask impassable directions
        for a in range(A):
            if not ego_obs[a]:
                probs[a] = 0.0

        # Mask backward unless only option
        if self.prevent_backward and sum(ego_obs) > 1:
            probs[1] = 0.0

        total = probs.sum()
        if total > 0:
            probs /= total
        else:
            probs = np.ones(A) / A
        return probs


# =============================================================================
# 5. Simple Egocentric Heuristics
# =============================================================================

class FullFwdAgent:
    """Always prefer Forward if passable; otherwise uniform over valid actions.

    Egocentric heuristic. No learning, no memory.
    Ego actions: F=0, B=1, L=2, R=3.
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
        prevent_reverse: bool = False,
        **kwargs,
    ) -> float:
        rng = np.random.RandomState(seed)
        sim = _EgoSimulator(adj_mat, node_positions, start_node,
                            rewarded_nodes, min_rewarded_states,
                            prevent_reverse=prevent_reverse)
        ego_obs = sim.reset()
        for _ in range(n_actions):
            if ego_obs[0]:  # Forward passable
                action = 0
            else:
                valid = [i for i, v in enumerate(ego_obs) if v]
                action = int(rng.choice(valid)) if valid else int(rng.randint(4))
            ego_obs, _, _ = sim.step(action)
        return sim.total_reward


class NoBkFullFwdAgent:
    """Prefer Forward; exclude backward unless it is the only valid action.

    NoBk+FullFwd heuristic: first removes backward (ego index 1) from the
    candidate set (unless it is the only passable direction), then picks
    Forward if available, otherwise uniform over remaining valid actions.
    Ego actions: F=0, B=1, L=2, R=3.
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
        prevent_reverse: bool = False,
        **kwargs,
    ) -> float:
        rng = np.random.RandomState(seed)
        sim = _EgoSimulator(adj_mat, node_positions, start_node,
                            rewarded_nodes, min_rewarded_states,
                            prevent_reverse=prevent_reverse)
        ego_obs = sim.reset()
        for _ in range(n_actions):
            valid = [i for i, v in enumerate(ego_obs) if v]
            no_bk = [i for i in valid if i != 1]   # remove backward (index 1)
            candidates = no_bk if no_bk else valid
            if 0 in candidates:  # Forward passable and allowed
                action = 0
            else:
                action = int(rng.choice(candidates)) if candidates else int(rng.randint(4))
            ego_obs, _, _ = sim.step(action)
        return sim.total_reward


# =============================================================================
# 6. Behavioral Cloning — Marginal Distribution Agent
# =============================================================================

class CloningAgent:
    """Samples from the mouse's empirical marginal action distribution.

    At each step the distribution is restricted to currently valid (passable)
    actions and renormalized. Falls back to uniform over valid actions when no
    valid action has non-zero probability in the marginal.

    Requires a per-session ``prob_dict`` from the yoking dataframe column
    ``prob_dict_ego``, ``prob_dict_allo_real``, or ``prob_dict_allo_latent``.
    Pass the already-parsed dict (use ``ast.literal_eval`` on the CSV string).

    Args:
        prob_dict: Parsed n-gram distribution, e.g.
            ``{(1, 1): {'F': 0.248, 'L': 0.198, 'R': 0.554}, (2, 1): {...}}``
            Ego keys: 'F'/'B'/'L'/'R'.
            Allo keys: 'N'/'S'/'E'/'W'.
        frame: Reference frame — 'ego', 'allo_real', or 'allo_latent'.
    """

    # Ego action indices
    _EGO_LETTER = {'F': 0, 'B': 1, 'L': 2, 'R': 3}

    def __init__(self, prob_dict: dict, frame: str = 'ego'):
        self.prob_dict = prob_dict
        self.frame = frame
        self.marginal = prob_dict.get((1, 1), {})

    def run_episode(
        self,
        adj_mat: np.ndarray,
        node_positions: np.ndarray,
        start_node: int,
        rewarded_nodes: np.ndarray,
        n_actions: int,
        min_rewarded_states: int = 2,
        seed: int = 42,
        prevent_reverse: bool = False,
        **kwargs,
    ) -> float:
        rng = np.random.RandomState(seed)
        sim = _EgoSimulator(adj_mat, node_positions, start_node,
                            rewarded_nodes, min_rewarded_states,
                            prevent_reverse=prevent_reverse)
        ego_obs = sim.reset()
        for _ in range(n_actions):
            if self.frame == 'ego':
                action = self._sample_ego(ego_obs, rng)
            else:
                action = self._sample_allo(ego_obs, sim.real_heading_idx, rng)
            ego_obs, _, _ = sim.step(action)
        return sim.total_reward

    def _sample_ego(self, ego_obs, rng):
        valid = {i for i, v in enumerate(ego_obs) if v}
        weights = {self._EGO_LETTER[k]: v
                   for k, v in self.marginal.items()
                   if k in self._EGO_LETTER and self._EGO_LETTER[k] in valid}
        return self._draw(weights, valid, rng)

    def _sample_allo(self, ego_obs, heading_idx, rng):
        valid_ego = {i for i, v in enumerate(ego_obs) if v}
        weights = {}
        for compass, p in self.marginal.items():
            allo_idx = allo_actions_one_hot_dict.get(compass)
            if not isinstance(allo_idx, int):
                continue
            ego_idx = ALLO_TO_EGO[heading_idx].get(allo_idx)
            if ego_idx is not None and ego_idx in valid_ego:
                weights[ego_idx] = weights.get(ego_idx, 0.0) + p
        return self._draw(weights, valid_ego, rng)

    @staticmethod
    def _draw(weights, valid, rng):
        if not weights:
            opts = list(valid) or list(range(4))
            return int(rng.choice(opts))
        indices = list(weights.keys())
        probs = np.array([weights[i] for i in indices], dtype=float)
        probs /= probs.sum()
        return int(rng.choice(indices, p=probs))


# =============================================================================
# Conditional Behavioral Cloning Agent
# =============================================================================

class ConditionalCloningAgent:
    """Samples from P(action | 4-bit ego observation) learned from the mouse trajectory.

    Unlike CloningAgent (which uses the global marginal P(action)), this agent
    conditions on the current egocentric observation.  With only 16 possible
    observations and 4 actions, the table is a small 16x4 histogram built by
    replaying the mouse's trajectory through _EgoSimulator.

    Falls back to uniform over valid actions for observations not seen in the
    mouse's data.
    """

    def __init__(self, states_visited, adj_mat, node_positions, start_node,
                 rewarded_nodes, min_rewarded_states=2):
        """Build P(action | obs) from the mouse's actual trajectory."""
        self.obs_action_counts = np.zeros((16, 4), dtype=float)

        sim = _EgoSimulator(adj_mat, node_positions, start_node,
                            rewarded_nodes, min_rewarded_states)
        ego_obs = sim.reset()

        for t in range(len(states_visited) - 1):
            current_state = states_visited[t]
            next_state = states_visited[t + 1]

            if current_state == next_state:
                continue  # self-loop, skip

            # Determine which ego action was taken
            allo_dir = None
            for d, neighbor in sim.dir_to_neighbor[current_state].items():
                if neighbor == next_state:
                    allo_dir = d
                    break
            if allo_dir is None:
                continue  # transition not in adjacency, skip

            ego_action = ALLO_TO_EGO[sim.real_heading_idx][allo_dir]
            obs_idx = _EgoSimulator.obs_to_idx(ego_obs)
            self.obs_action_counts[obs_idx, ego_action] += 1

            # Advance simulator to track heading
            ego_obs, _, _ = sim.step(ego_action)

        # Normalize to probabilities per observation
        self.obs_action_probs = np.zeros_like(self.obs_action_counts)
        for o in range(16):
            total = self.obs_action_counts[o].sum()
            if total > 0:
                self.obs_action_probs[o] = self.obs_action_counts[o] / total

    def run_episode(
        self,
        adj_mat: np.ndarray,
        node_positions: np.ndarray,
        start_node: int,
        rewarded_nodes: np.ndarray,
        n_actions: int,
        min_rewarded_states: int = 2,
        seed: int = 42,
        prevent_reverse: bool = False,
        **kwargs,
    ) -> float:
        rng = np.random.RandomState(seed)
        sim = _EgoSimulator(adj_mat, node_positions, start_node,
                            rewarded_nodes, min_rewarded_states,
                            prevent_reverse=prevent_reverse)
        ego_obs = sim.reset()

        for _ in range(n_actions):
            obs_idx = _EgoSimulator.obs_to_idx(ego_obs)
            valid = [i for i, v in enumerate(ego_obs) if v]

            probs = self.obs_action_probs[obs_idx].copy()
            # Mask invalid actions
            mask = np.zeros(4)
            for a in valid:
                mask[a] = 1.0
            probs *= mask

            total = probs.sum()
            if total > 0:
                probs /= total
                action = rng.choice(4, p=probs)
            else:
                action = rng.choice(valid) if valid else rng.randint(4)

            ego_obs, _, _ = sim.step(action)

        return sim.total_reward


# =============================================================================
# Latent-State-Biased Agent (privileged baseline)
# =============================================================================

class LatentStateBiasedAgent:
    """Markov-chain sampler over latent-state transitions observed in the yoked mouse session.

    Builds a per-session count table c(s, s') from the mouse's `states_visited`
    trajectory. At each step, samples the next latent node s' from
        pi(s_{t+1} | s_t) = c(s_t, s_{t+1}) / sum_{s'} c(s_t, s')
    restricted to the agent's currently-passable neighbors via the adjacency
    matrix. Falls back to uniform over valid neighbors when no observed
    mouse-transition from s_t has a valid target.

    REQUIRES privileged latent-state access: at each step the agent must know
    s_t (the current latent node), which is NOT part of the 4-bit egocentric
    observation Omega. This makes it an explicitly-privileged baseline,
    assessing the relative role of latent-state occupancy preference wholly
    independent of reward representation (which is not provided to this agent).

    The agent does NOT receive the reward state. Combined with privileged
    latent-state access, this isolates "where the mouse tends to go" from
    "where rewards currently are".

    Args:
        states_visited: list[int] of latent node IDs the mouse visited in
            the yoked session. Self-loops (state == prev_state) are skipped
            when building counts.

    Returns: total_reward (float) from run_episode, same interface as the
    other intermediate agents.

    Reference: Rosenberg, Zhang, Perona, Meister (2021), where a related
    transition-sampling baseline is used to test whether mouse occupancy
    statistics alone explain navigation efficiency.
    """

    def __init__(self, states_visited):
        self.transition_counts = {}  # s -> {s': count}
        for i in range(len(states_visited) - 1):
            s, s_next = int(states_visited[i]), int(states_visited[i + 1])
            if s == s_next:
                continue  # self-loop, skip (no movement)
            self.transition_counts.setdefault(s, {})
            self.transition_counts[s][s_next] = self.transition_counts[s].get(s_next, 0) + 1

    def run_episode(
        self,
        adj_mat: np.ndarray,
        node_positions: np.ndarray,
        start_node: int,
        rewarded_nodes: np.ndarray,
        n_actions: int,
        min_rewarded_states: int = 2,
        seed: int = 42,
        prevent_reverse: bool = False,
        **kwargs,
    ) -> float:
        rng = np.random.RandomState(seed)
        sim = _EgoSimulator(adj_mat, node_positions, start_node,
                            rewarded_nodes, min_rewarded_states,
                            prevent_reverse=prevent_reverse)
        sim.reset()

        for _ in range(n_actions):
            current = sim.real_node
            # Enumerate valid neighbors via the simulator's dir->neighbor map
            dir_to_n = sim.dir_to_neighbor[current]
            if not dir_to_n:
                # Isolated node (no passable neighbors); step a no-op
                sim.step(0)  # any action; will self-loop
                continue
            valid_neighbors = list(dir_to_n.values())
            valid_dirs = list(dir_to_n.keys())  # parallel to valid_neighbors

            # Apply prevent_reverse mask if active and not at a dead-end
            if prevent_reverse and sim._prev_node is not None and len(valid_neighbors) > 1:
                # Drop the prev-node reverse-target from candidates
                keep = [(d, n) for d, n in zip(valid_dirs, valid_neighbors) if n != sim._prev_node]
                if keep:
                    valid_dirs, valid_neighbors = list(zip(*keep))
                    valid_dirs, valid_neighbors = list(valid_dirs), list(valid_neighbors)

            # Get mouse-observed transition counts from current state, restricted to valid
            counts = self.transition_counts.get(current, {})
            weights = np.array([counts.get(n, 0) for n in valid_neighbors], dtype=float)
            total = weights.sum()

            if total > 0:
                probs = weights / total
                idx = rng.choice(len(valid_neighbors), p=probs)
            else:
                # Mouse never transitioned from this state to any of the agent's
                # currently-valid neighbors (or never visited this state at all).
                # Uniform fallback over valid neighbors.
                idx = rng.choice(len(valid_neighbors))

            # Convert chosen next-node to ego action via the current heading
            chosen_allo_dir = valid_dirs[idx]
            ego_action = ALLO_TO_EGO[sim.real_heading_idx][chosen_allo_dir]
            sim.step(ego_action)

        return sim.total_reward
