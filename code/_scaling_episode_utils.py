"""
_scaling_episode_utils.py

Canonical agent-dispatch + single-episode runner for synthetic-scaling
experiments (e.g., V=36..4000 sweeps). Standardises the helper that
previously lived in two places:
  - 260502_run_scaling_sweep.py:run_one_episode  (active 2026-05-02 sweep)
  - 260503_run_scaling_sweep_v2.py:_run_one_episode  (next sweep)

Both inlined the same Random / Random_forward / Greedy_oracle loops;
this module is the single source of truth going forward. The 260502
script's local copy is kept for archival/reproducibility — it produced
the canonical c260502-124509 sweep CSV, do not edit retroactively.

Usage:
    from _scaling_episode_utils import run_one_episode, DEFAULT_POMCP_CONFIG
    result = run_one_episode(
        agent_name='POMCP', adj_mat=adj_mat, st_pos=st_pos,
        start_node=0, rewarded_nodes=rewarded_nodes,
        n_actions=n_actions, seed=seed,
        min_rewarded_remaining=2, pomcp_num_sims=200)
    # → {'total_reward': float, 'wall_time_s': float}

Conventions:
- `rewarded_nodes` is a 0/1 binary indicator vector of shape (n_nodes,),
  NOT a list of indices. (Matches 260225 focused-scaling and the canonical
  GraphMazeEnv expectation.)
- `min_rewarded_remaining` triggers full re-baiting when the count of
  remaining rewarded nodes drops below the threshold (matches mouse rebait
  semantics; see GraphMazeEnv.step in yoked_rl_runner.py).
- POMCP num_sims=200 corresponds to the canonical 'fewer_sims' Table 1
  configuration (c=20, max_depth=30). Other call sites may override.

Wrapped agent classes are imported from existing project modules:
  POMCPAgent          ← shelved_agents/pomcp_oracle.py
  POMCPbioAlloAgent   ← advanced_agents.py (Frontier-Plan allocentric)
  POMCPbioEgoAgent    ← advanced_agents.py (Frontier-Plan egocentric)
  FSCBioAgent         ← intermediate_agents.py
"""
from __future__ import annotations
import time
from collections import deque

import numpy as np


# Canonical Table 1 POMCP config (c=20, fewer_sims, num_sims=200).
DEFAULT_POMCP_CONFIG = {
    'num_sims': 200,
    'max_depth': 30,
    'exploration_const': 20.0,
}

# FSC_bio canonical config (260331_fsc_bio_correct_params.py defaults).
DEFAULT_FSC_BIO_CONFIG = {
    'exploit_timeout': 1,
    'exploit_greediness': 3,
}


def run_one_episode(agent_name, adj_mat, st_pos, start_node, rewarded_nodes,
                    n_actions, seed, min_rewarded_remaining=2,
                    pomcp_num_sims=DEFAULT_POMCP_CONFIG['num_sims'],
                    pomcp_max_depth=DEFAULT_POMCP_CONFIG['max_depth'],
                    pomcp_exploration_const=DEFAULT_POMCP_CONFIG['exploration_const'],
                    fsc_bio_exploit_timeout=DEFAULT_FSC_BIO_CONFIG['exploit_timeout'],
                    fsc_bio_exploit_greediness=DEFAULT_FSC_BIO_CONFIG['exploit_greediness']):
    """Run one episode of `agent_name` on the given maze instance.

    Returns: {'total_reward': float, 'wall_time_s': float}.

    Supported agent_name values:
      'Random', 'Random_forward', 'Greedy_oracle'         — inline implementations
      'POMCP'                                              — POMCPAgent (oracle)
      'POMCP_bio_allo' / 'Frontier-Plan_allo'              — POMCPbioAlloAgent
      'POMCP_bio_ego'  / 'Frontier-Plan_ego'               — POMCPbioEgoAgent
      'FSC_bio'        / 'FSC_bio_t1_g3'                   — FSCBioAgent

    `rewarded_nodes` MUST be a binary 0/1 indicator vector of shape
    (n_nodes,). Pass `np.ones(n)` for "all nodes initially rewarded"
    (the canonical mouse-experiment setup).
    """
    t0 = time.time()
    initial_rewarded = set(int(i) for i in np.where(rewarded_nodes > 0)[0])

    if agent_name == 'Random':
        rng = np.random.RandomState(seed)
        cur = int(start_node)
        rwd_set = set(initial_rewarded)
        total_reward = 0
        for _ in range(n_actions):
            adj_nodes = np.where(adj_mat[cur] == 1)[0]
            if len(adj_nodes) == 0:
                break
            cur = int(rng.choice(adj_nodes))
            if cur in rwd_set:
                total_reward += 1
                rwd_set.discard(cur)
                if len(rwd_set) < min_rewarded_remaining:
                    rwd_set = set(initial_rewarded)
        return {'total_reward': total_reward, 'wall_time_s': time.time() - t0}

    if agent_name == 'Random_forward':
        rng = np.random.RandomState(seed)
        cur = int(start_node)
        prev = -1
        rwd_set = set(initial_rewarded)
        total_reward = 0
        for _ in range(n_actions):
            adj_nodes = list(np.where(adj_mat[cur] == 1)[0])
            non_reverse = [int(n) for n in adj_nodes if int(n) != prev]
            choices = non_reverse if non_reverse else [int(n) for n in adj_nodes]
            if not choices:
                break
            new_cur = int(rng.choice(choices))
            prev = cur
            cur = new_cur
            if cur in rwd_set:
                total_reward += 1
                rwd_set.discard(cur)
                if len(rwd_set) < min_rewarded_remaining:
                    rwd_set = set(initial_rewarded)
        return {'total_reward': total_reward, 'wall_time_s': time.time() - t0}

    if agent_name == 'Greedy_oracle':
        cur = int(start_node)
        rwd_set = set(initial_rewarded)
        total_reward = 0
        for _ in range(n_actions):
            if not rwd_set:
                rwd_set = set(initial_rewarded)
            visited = {cur}
            queue = deque([(cur, [cur])])
            target_path = None
            while queue:
                node, path = queue.popleft()
                if node in rwd_set and node != cur:
                    target_path = path
                    break
                for nb in np.where(adj_mat[node] == 1)[0]:
                    if int(nb) not in visited:
                        visited.add(int(nb))
                        queue.append((int(nb), path + [int(nb)]))
            if target_path is None or len(target_path) < 2:
                break
            cur = target_path[1]
            if cur in rwd_set:
                total_reward += 1
                rwd_set.discard(cur)
                if len(rwd_set) < min_rewarded_remaining:
                    rwd_set = set(initial_rewarded)
        return {'total_reward': total_reward, 'wall_time_s': time.time() - t0}

    # Wrapped agents — defer imports so this module doesn't pull in heavy deps unless needed.
    if agent_name == 'POMCP':
        from shelved_agents.pomcp_oracle import POMCPAgent
        agent = POMCPAgent(num_sims=pomcp_num_sims, max_depth=pomcp_max_depth,
                            exploration_const=pomcp_exploration_const)
    elif agent_name in ('POMCP_bio_allo', 'Frontier-Plan_allo'):
        from advanced_agents import POMCPbioAlloAgent
        agent = POMCPbioAlloAgent()
    elif agent_name in ('POMCP_bio_ego', 'Frontier-Plan_ego'):
        from advanced_agents import POMCPbioEgoAgent
        agent = POMCPbioEgoAgent()
    elif agent_name in ('FSC_bio', 'FSC_bio_t1_g3'):
        from intermediate_agents import FSCBioAgent
        agent = FSCBioAgent(exploit_timeout=fsc_bio_exploit_timeout,
                             exploit_greediness=fsc_bio_exploit_greediness)
    else:
        raise ValueError(f'Unknown agent_name for scaling sweep: {agent_name}')

    total_reward = agent.run_episode(
        adj_mat=adj_mat, node_positions=st_pos, start_node=int(start_node),
        rewarded_nodes=np.array(rewarded_nodes), n_actions=n_actions,
        min_rewarded_states=min_rewarded_remaining, seed=seed,
        prevent_reverse=False)
    return {'total_reward': total_reward, 'wall_time_s': time.time() - t0}
