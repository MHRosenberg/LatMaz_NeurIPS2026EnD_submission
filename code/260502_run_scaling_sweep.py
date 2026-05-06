"""
260502_run_scaling_sweep.py

Maze-size scaling sweep for the LatMaz paper (App N).
Runs 7 agents across V = 36..4000 nodes with action budget T = 3V.

Design (Author-approved 2026-05-01):
  V values:  [36, 100, 225, 400, 900, 1500, 2500, 4000]
  Agents:    Random, Random_forward, Greedy_oracle, POMCP (oracle, c=20, fewer-sims),
             POMCP_bio (allocentric — Table-1 variant), POMCP_bio (egocentric only),
             FSC_bio
  Reps:      3 instances × 5 seeds = 15 sims per (V, agent) cell
  Budget:    T = 3V actions per session
  Reward:    REWARD_FRACTION = 1.0 (all-nodes-rewarded), MIN_REWARDED_REMAINING = 2
             — matches mouse experimental conditions
  Cell timeout: 30 minutes wall-clock per cell. If exceeded, remaining seeds for
                that cell are skipped and flagged with `wall_time_budget_exceeded=True`.

Output: data_out/rl_sims/c{ts}_scaling_sweep_v36_to_v4000.csv
  Incremental writes after each cell (so an OOM/abort doesn't lose finished work).

Launch:
  tmux new -s sweep
  caffeinate -dis conda run -n latMaz_RL python 260502_run_scaling_sweep.py 2>&1 | tee sweep.log
  # Ctrl+B then D to detach. tmux attach -t sweep to re-attach.

Adapted from 260225_run_focused_scaling.py (260226 focused-scaling experiment).
"""
import os
import sys
import time
import signal
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import networkx as nx

# Locate project root + sys.path setup
PROJECT_ROOT = Path('<REPO_ROOT>')
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / 'code'))
sys.path.insert(0, str(Path(__file__).parent))

# Imports from the existing codebase
from advanced_agents import POMCPbioAlloAgent, POMCPbioEgoAgent
from shelved_agents.pomcp_oracle import POMCPAgent
HAVE_EGO_VARIANT = True

from intermediate_agents import FSCBioAgent
from experiment_config import (
    run_random_agent, run_forward_random_agent, run_greedy_agent,
)


# =============================================================================
# Sweep configuration
# =============================================================================
V_VALUES = [36, 100, 225, 400, 900, 1500, 2500, 4000]
N_INSTANCES = 3
N_SEEDS = 5
BUDGET_MULTIPLIER = 3              # T = 3V
REWARD_FRACTION = 1.0              # all reachable nodes rewarded
MIN_REWARDED_REMAINING = 2

# Timeout semantics (Author-clarified 2026-05-03):
#   PRIMARY — per-seed timeout. If a single seed takes longer than
#             SEED_WALL_TIME_LIMIT_S, mark *that* seed as timed out and
#             continue with the next seed in the same cell. The
#             motivation is "oh this approach is taking forever, we can
#             skip THIS attempt", not "skip the whole cell".
#   SECONDARY — per-cell fail-safe. Only triggers if the entire cell is
#             running 10× longer than expected (10 × N_SEEDS × per-seed).
#             Prevents pathological infinite loops without skipping
#             needlessly.
SEED_WALL_TIME_LIMIT_S = 5 * 60    # 5 min/seed
CELL_WALL_TIME_LIMIT_S = 10 * N_SEEDS * SEED_WALL_TIME_LIMIT_S  # 10× expected

OUT_DIR = PROJECT_ROOT / 'data_out' / 'rl_sims'
TS = 'c' + datetime.now().strftime('%y%m%d-%H%M%S')
OUT_CSV = OUT_DIR / f'{TS}_scaling_sweep_v36_to_v4000.csv'
LOG_PATH = OUT_DIR / f'{TS}_scaling_sweep_v36_to_v4000.log'


# =============================================================================
# Timeout handling — per-cell wall-clock budget
# =============================================================================
class CellTimeout(Exception):
    pass

def _alarm(signum, frame):
    raise CellTimeout()


# =============================================================================
# Synthetic maze generation — square-lattice subgraph with REWARD_FRACTION coverage
# =============================================================================
def make_lattice_subgraph_maze(side, instance_seed):
    """Build a square-lattice subgraph of side*side nodes.

    Drops ~20% of edges at random (instance_seed) so the maze isn't trivially
    fully connected. Returns adj_mat (side^2 × side^2) and node_positions.
    Sanity: takes the largest connected component as the navigable subgraph.
    """
    rng = np.random.RandomState(instance_seed)
    n = side * side

    # Lattice positions
    positions = np.zeros((n, 2))
    for r in range(side):
        for c in range(side):
            positions[r * side + c] = (c, r)

    # All lattice edges, then drop ~20%
    edges = []
    for r in range(side):
        for c in range(side):
            i = r * side + c
            if c < side - 1:
                edges.append((i, i + 1))
            if r < side - 1:
                edges.append((i, i + side))
    keep = rng.rand(len(edges)) > 0.2
    kept_edges = [e for e, k in zip(edges, keep) if k]

    # Build adjacency
    adj = np.zeros((n, n), dtype=np.int64)
    for i, j in kept_edges:
        adj[i, j] = 1
        adj[j, i] = 1

    # Take largest connected component
    G = nx.from_numpy_array(adj)
    components = list(nx.connected_components(G))
    largest = max(components, key=len)
    keep_nodes = sorted(largest)
    keep_idx = {old: new for new, old in enumerate(keep_nodes)}

    n2 = len(keep_nodes)
    adj2 = np.zeros((n2, n2), dtype=np.int64)
    for i, j in kept_edges:
        if i in keep_idx and j in keep_idx:
            ni, nj = keep_idx[i], keep_idx[j]
            adj2[ni, nj] = 1
            adj2[nj, ni] = 1
    pos2 = positions[keep_nodes]

    return adj2, pos2


# =============================================================================
# Per-cell agent runner  ─── returns dict of result fields
# =============================================================================
def run_one_episode(agent_name, adj_mat, st_pos, start_node, rewarded_nodes,
                    n_actions, seed):
    """Run one episode of agent_name on the given maze instance.

    rewarded_nodes : np.ndarray of shape (n_nodes,), float
        Binary indicator vector — 1.0 if node is rewarded, 0.0 otherwise.
        (Matches the convention in 260225_run_focused_scaling.py.)

    Returns dict with keys total_reward, wall_time_s.
    """
    t0 = time.time()
    n_nodes = adj_mat.shape[0]
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
                if len(rwd_set) < MIN_REWARDED_REMAINING:
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
                if len(rwd_set) < MIN_REWARDED_REMAINING:
                    rwd_set = set(initial_rewarded)
        return {'total_reward': total_reward, 'wall_time_s': time.time() - t0}

    if agent_name == 'Greedy_oracle':
        from collections import deque
        cur = int(start_node)
        rwd_set = set(initial_rewarded)
        total_reward = 0
        for _ in range(n_actions):
            if not rwd_set:
                rwd_set = set(initial_rewarded)
            # BFS to nearest in rwd_set
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
                if len(rwd_set) < MIN_REWARDED_REMAINING:
                    rwd_set = set(initial_rewarded)
        return {'total_reward': total_reward, 'wall_time_s': time.time() - t0}

    if agent_name == 'POMCP':
        # POMCPAgent is the wrapper in shelved_agents/pomcp_oracle.py
        # (the canonical c=20 fewer-sims oracle from Table 1).
        agent = POMCPAgent(max_depth=30, num_sims=200, exploration_const=20.0)
        total_reward = agent.run_episode(
            adj_mat=adj_mat, node_positions=st_pos, start_node=int(start_node),
            rewarded_nodes=rewarded_nodes, n_actions=n_actions,
            min_rewarded_states=MIN_REWARDED_REMAINING, seed=seed,
            prevent_reverse=False,
        )
        return {'total_reward': total_reward, 'wall_time_s': time.time() - t0}

    if agent_name == 'POMCP_bio_allo':
        agent = POMCPbioAlloAgent()
        total_reward = agent.run_episode(
            adj_mat=adj_mat, node_positions=st_pos, start_node=int(start_node),
            rewarded_nodes=np.array(rewarded_nodes), n_actions=n_actions,
            min_rewarded_states=MIN_REWARDED_REMAINING, seed=seed,
            prevent_reverse=False,
        )
        return {'total_reward': total_reward, 'wall_time_s': time.time() - t0}

    if agent_name == 'POMCP_bio_ego' and HAVE_EGO_VARIANT:
        agent = POMCPbioEgoAgent()
        total_reward = agent.run_episode(
            adj_mat=adj_mat, node_positions=st_pos, start_node=int(start_node),
            rewarded_nodes=np.array(rewarded_nodes), n_actions=n_actions,
            min_rewarded_states=MIN_REWARDED_REMAINING, seed=seed,
            prevent_reverse=False,
        )
        return {'total_reward': total_reward, 'wall_time_s': time.time() - t0}

    if agent_name == 'FSC_bio':
        agent = FSCBioAgent(exploit_timeout=1, exploit_greediness=3)
        total_reward = agent.run_episode(
            adj_mat=adj_mat, node_positions=st_pos, start_node=int(start_node),
            rewarded_nodes=np.array(rewarded_nodes), n_actions=n_actions,
            min_rewarded_states=MIN_REWARDED_REMAINING, seed=seed,
            prevent_reverse=False,
        )
        return {'total_reward': total_reward, 'wall_time_s': time.time() - t0}

    raise ValueError(f"Unknown agent: {agent_name}")


# =============================================================================
# Main sweep
# =============================================================================
AGENTS = ['Random', 'Random_forward', 'Greedy_oracle', 'POMCP',
          'POMCP_bio_allo', 'FSC_bio']
if HAVE_EGO_VARIANT:
    AGENTS.append('POMCP_bio_ego')


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"=== LatMaz scaling sweep — {TS} ===")
    print(f"V values: {V_VALUES}")
    print(f"Agents:   {AGENTS}")
    print(f"Reps:     {N_INSTANCES} instances × {N_SEEDS} seeds")
    print(f"Output:   {OUT_CSV}")
    print(f"Log:      {LOG_PATH}\n")

    results = []
    log = open(LOG_PATH, 'w')

    for V in V_VALUES:
        side = int(np.ceil(np.sqrt(V)))
        T = BUDGET_MULTIPLIER * V

        # Build N_INSTANCES instances for this V
        for inst in range(N_INSTANCES):
            adj_mat, st_pos = make_lattice_subgraph_maze(side, instance_seed=inst)
            n_nodes_actual = adj_mat.shape[0]
            start_node = 0
            # Binary indicator: rewarded_nodes[i]=1.0 if rewarded, 0.0 else.
            # Per REWARD_FRACTION=1.0, every reachable node is rewarded.
            rewarded_nodes = np.ones(n_nodes_actual, dtype=np.float64)

            for agent in AGENTS:
                cell_id = f"V={V} (n={n_nodes_actual}) inst={inst} agent={agent}"
                print(f"[{datetime.now().strftime('%H:%M:%S')}] {cell_id} ...", flush=True)
                log.write(f"{cell_id}\n"); log.flush()

                cell_t0 = time.time()
                seeds_done = 0
                signal.signal(signal.SIGALRM, _alarm)
                cell_aborted = False
                for seed in range(N_SEEDS):
                    elapsed_cell = time.time() - cell_t0
                    cell_remaining = CELL_WALL_TIME_LIMIT_S - elapsed_cell
                    # Per-cell fail-safe: if the cell is already past its 10×-expected
                    # budget, abort the cell entirely.
                    if cell_remaining <= 0:
                        cell_aborted = True
                        log.write(f"  cell fail-safe ({CELL_WALL_TIME_LIMIT_S}s) exceeded "
                                  f"after {seeds_done} seeds; aborting remaining seeds\n")
                        break
                    # Per-seed timeout: bound to min(SEED_WALL_TIME_LIMIT_S, cell_remaining)
                    seed_budget = min(SEED_WALL_TIME_LIMIT_S, int(cell_remaining))
                    signal.alarm(seed_budget)
                    try:
                        out = run_one_episode(
                            agent, adj_mat, st_pos, start_node,
                            rewarded_nodes, T, seed
                        )
                        signal.alarm(0)
                        rpa = out['total_reward'] / max(T, 1)
                        results.append({
                            'V_target': V, 'n_nodes': n_nodes_actual, 'side': side,
                            'instance': inst, 'agent': agent, 'seed': seed,
                            'n_actions': T, 'total_reward': out['total_reward'],
                            'rpa': rpa, 'wall_time_s': out['wall_time_s'],
                            'seed_timed_out': False,
                            'cell_aborted': False,
                            'timestamp': datetime.now().isoformat(timespec='seconds'),
                        })
                        seeds_done += 1
                    except CellTimeout:
                        # Per-seed timeout — record this seed as timed out and
                        # CONTINUE to the next seed in the same cell. Only break
                        # the cell loop if the per-cell fail-safe triggers above.
                        signal.alarm(0)
                        log.write(f"  seed {seed}: per-seed timeout "
                                  f"({seed_budget}s) — skipped\n")
                        log.flush()
                        results.append({
                            'V_target': V, 'n_nodes': n_nodes_actual, 'side': side,
                            'instance': inst, 'agent': agent, 'seed': seed,
                            'n_actions': T, 'total_reward': np.nan,
                            'rpa': np.nan, 'wall_time_s': np.nan,
                            'seed_timed_out': True,
                            'cell_aborted': False,
                            'timestamp': datetime.now().isoformat(timespec='seconds'),
                        })
                    except Exception as e:
                        signal.alarm(0)
                        log.write(f"  ERROR seed {seed}: {e!r}\n")
                        log.write(traceback.format_exc())
                        log.flush()
                        results.append({
                            'V_target': V, 'n_nodes': n_nodes_actual, 'side': side,
                            'instance': inst, 'agent': agent, 'seed': seed,
                            'n_actions': T, 'total_reward': np.nan,
                            'rpa': np.nan, 'wall_time_s': np.nan,
                            'seed_timed_out': False,
                            'cell_aborted': False,
                            'timestamp': datetime.now().isoformat(timespec='seconds'),
                        })
                signal.alarm(0)
                cell_elapsed = time.time() - cell_t0
                msg = (f"  done: {seeds_done}/{N_SEEDS} seeds, "
                       f"{cell_elapsed:.1f}s wall"
                       f"{' (CELL FAIL-SAFE TRIGGERED)' if cell_aborted else ''}\n")
                print(msg, end='', flush=True)
                log.write(msg); log.flush()

                # Incremental save after each cell so abort-recovery is possible
                pd.DataFrame(results).to_csv(OUT_CSV, index=False)

    log.close()
    print(f"\n=== Sweep complete. Results: {OUT_CSV} ===")


if __name__ == '__main__':
    main()
