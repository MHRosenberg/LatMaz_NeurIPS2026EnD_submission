"""
260503_run_scaling_sweep_v2.py

V2 scaling sweep — adapted from 260502_run_scaling_sweep.py with the
useful pieces from the Codex audit variant (now archived in
z_legacy_outdated_reference/cdx_unvalidated_scaling_variants_260502/).

Adopted from cdx variant:
  1. Subprocess hard-kill timeout (multiprocessing spawn) — robust against
     C-extension hangs, replaces signal.alarm.
  2. Status rows {ok, timeout, error, skipped_cell_budget} — replaces single
     wall_time_budget_exceeded boolean. Adds error string column.
  3. Resume from existing CSV via --resume-csv.
  4. Mouse-tessellation maze family (--maze-family mouse_tessellation) —
     tile canonical mouse mazes and bridge with single edges, for "what
     about real mouse-relevant graph structure at scale?" experiments.
  5. Per-instance verification plot (--plot-mazes).
  6. Provenance/config columns (pomcp_num_sims, seed_timeout_s, etc.).
  7. CLI flags via argparse (replaces module-level constants).

Kept from active 260502 sweep:
  - V grid: [36, 100, 225, 400, 900, 1500, 2500, 4000]
  - 7 agents (Random, Random_forward, Greedy_oracle, POMCP, POMCP_bio_allo,
    POMCP_bio_ego, FSC_bio)
  - 3 instances x 5 seeds per cell
  - Action budget T = 3V
  - REWARD_FRACTION = 1.0
  - 5 min/seed primary timeout (per-seed), 10x per-cell fail-safe (secondary)
  - Inline implementations of Random/Random_forward/Greedy_oracle (so the
    sweep doesn't depend on experiment_config helpers that need yoking_df)

Validation: head-to-head against 260502 sweep on 1-2 cells should match
within seed-noise. Run with --smoke first to verify infrastructure.

Launch:
    python 260503_run_scaling_sweep_v2.py --smoke   # tiny test (~5 sec)
    python 260503_run_scaling_sweep_v2.py           # full sweep (~24 h)
    python 260503_run_scaling_sweep_v2.py --resume-csv data_out/rl_sims/<EXISTING>.csv

DO NOT replace the running 260502 sweep — that one is producing valid data
for the paper. This v2 is for the next sweep (e.g., mouse-tessellation
re-run if Author wants real-mouse-maze topology at scale).
"""
from __future__ import annotations
import argparse
import json
import math
import multiprocessing as mp
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from queue import Empty

import numpy as np
import pandas as pd
import networkx as nx

PROJECT_ROOT = Path('<REPO_ROOT>')
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / 'code0_init-readOnly' / 'code'))
sys.path.insert(0, str(Path(__file__).parent))

# Canonical agent dispatcher — single source of truth for synthetic
# scaling-sweep episode runs (consolidated 2026-05-03 from the duplicated
# helpers that previously lived in the v1 sweep script + this file).
from _scaling_episode_utils import run_one_episode as _run_one_episode_canonical

# Defaults (overridable by CLI)
DEFAULT_V_VALUES   = [36, 100, 225, 400, 900, 1500, 2500, 4000]
DEFAULT_AGENTS     = ['Random', 'Random_forward', 'Greedy_oracle',
                     'POMCP', 'POMCP_bio_allo', 'POMCP_bio_ego', 'FSC_bio']
DEFAULT_INSTANCES  = 3
DEFAULT_SEEDS      = 5
DEFAULT_BUDGET_MULT  = 3
DEFAULT_REWARD_FRAC  = 1.0
DEFAULT_MIN_REW_REM  = 2
DEFAULT_EDGE_DROP    = 0.2
DEFAULT_SEED_TIMEOUT = 5 * 60       # primary
DEFAULT_POMCP_NUM_SIMS = 1000       # match active sweep config

DEFAULT_OUT_DIR = PROJECT_ROOT / 'data_out' / 'rl_sims'


# =============================================================================
# Maze families
# =============================================================================
def make_lattice_subgraph_maze(target_nodes, instance_seed, edge_drop_prob=DEFAULT_EDGE_DROP):
    """Square-lattice subgraph; lcc retained. Returns (adj, pos, meta)."""
    side = int(math.ceil(math.sqrt(target_nodes)))
    rng = np.random.RandomState(instance_seed)
    n = side * side
    positions = np.zeros((n, 2), dtype=float)
    for r in range(side):
        for c in range(side):
            positions[r * side + c] = (c, r)
    edges = []
    for r in range(side):
        for c in range(side):
            i = r * side + c
            if c < side - 1: edges.append((i, i + 1))
            if r < side - 1: edges.append((i, i + side))
    kept_edges = [e for e in edges if rng.rand() > edge_drop_prob]
    adj = np.zeros((n, n), dtype=np.int64)
    for i, j in kept_edges:
        adj[i, j] = 1; adj[j, i] = 1
    g = nx.from_numpy_array(adj)
    largest = sorted(max(nx.connected_components(g), key=len))
    old_to_new = {old: new for new, old in enumerate(largest)}
    out = np.zeros((len(largest), len(largest)), dtype=np.int64)
    for i, j in kept_edges:
        if i in old_to_new and j in old_to_new:
            ni, nj = old_to_new[i], old_to_new[j]
            out[ni, nj] = 1; out[nj, ni] = 1
    meta = {'maze_family': 'lattice_subgraph', 'V_target': int(target_nodes),
            'side': int(side), 'edge_drop_prob': float(edge_drop_prob),
            'instance_seed': int(instance_seed), 'n_nodes': int(out.shape[0]),
            'connectors': []}
    return out, positions[largest], meta


def make_mouse_tessellation_maze(target_nodes, instance_seed, n_tiles=None,
                                  tile_spacing=8.0):
    """Tile canonical mouse mazes; bridge with single edges. From cdx variant."""
    from experiment_config import filter_sessions, load_yoking_df
    from utils_latMaz import load_maze
    yoking = load_yoking_df()
    sessions = set(filter_sessions(yoking))
    yoking = yoking[yoking['exp_moment'].isin(sessions)].copy()
    templates = []
    for _, row in yoking[['adj_file','st_pos_file']].drop_duplicates().iterrows():
        adj, pos = load_maze(
            PROJECT_ROOT / 'data_in' / 'mazes' / row['adj_file'],
            PROJECT_ROOT / 'data_in' / 'mazes' / row['st_pos_file'])
        templates.append({'adj': np.asarray(adj, dtype=np.int64),
                          'pos': np.asarray(pos, dtype=float),
                          'adj_file': row['adj_file'],
                          'st_pos_file': row['st_pos_file']})
    if not templates:
        raise RuntimeError('No mouse maze templates from yoking table')
    mean_nodes = np.mean([t['adj'].shape[0] for t in templates])
    if n_tiles is None:
        n_tiles = max(1, int(math.ceil(target_nodes / mean_nodes)))
    rng = np.random.RandomState(instance_seed)
    order = list(rng.choice(len(templates), size=n_tiles, replace=True))
    grid_cols = int(math.ceil(math.sqrt(n_tiles)))
    blocks, positions, tile_ranges, sources = [], [], [], []
    cursor = 0
    for tile_i, template_idx in enumerate(order):
        tpl = templates[int(template_idx)]
        adj, pos = tpl['adj'], tpl['pos'].copy()
        row, col = divmod(tile_i, grid_cols)
        pos = pos + np.array([col * tile_spacing, row * tile_spacing], dtype=float)
        blocks.append(adj); positions.append(pos)
        tile_ranges.append((cursor, cursor + adj.shape[0]))
        sources.append({'tile': int(tile_i), 'adj_file': tpl['adj_file'],
                        'n_nodes': int(adj.shape[0])})
        cursor += adj.shape[0]
    n_nodes = cursor
    out = np.zeros((n_nodes, n_nodes), dtype=np.int64)
    for adj, (lo, hi) in zip(blocks, tile_ranges):
        out[lo:hi, lo:hi] = adj
    pos_all = np.vstack(positions)
    connectors = []
    for left in range(n_tiles - 1):
        lo_a, hi_a = tile_ranges[left]
        lo_b, hi_b = tile_ranges[left + 1]
        pa, pb = pos_all[lo_a:hi_a], pos_all[lo_b:hi_b]
        d = ((pa[:, None, :] - pb[None, :, :]) ** 2).sum(axis=2)
        ia, ib = np.unravel_index(np.argmin(d), d.shape)
        a, b = lo_a + int(ia), lo_b + int(ib)
        out[a, b] = 1; out[b, a] = 1
        connectors.append((int(a), int(b)))
    meta = {'maze_family': 'mouse_tessellation', 'V_target': int(target_nodes),
            'side': None, 'edge_drop_prob': None, 'instance_seed': int(instance_seed),
            'n_nodes': int(n_nodes), 'n_tiles': int(n_tiles),
            'tile_sources': sources, 'connectors': connectors}
    return out, pos_all, meta


def plot_maze_graph(adj_mat, st_pos, out_path, title=None, connectors=None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    out_path = Path(out_path); out_path.parent.mkdir(parents=True, exist_ok=True)
    g = nx.from_numpy_array(np.asarray(adj_mat))
    pos = {i: tuple(st_pos[i]) for i in range(len(st_pos))}
    fig, ax = plt.subplots(figsize=(8, 8))
    nx.draw_networkx_edges(g, pos, ax=ax, width=0.8, edge_color='#555555')
    nx.draw_networkx_nodes(g, pos, ax=ax, node_size=18, node_color='#2a9d8f')
    if connectors:
        nx.draw_networkx_edges(g, pos, edgelist=[tuple(e) for e in connectors],
                               ax=ax, width=2.5, edge_color='#e76f51')
    ax.set_title(title or f'n={adj_mat.shape[0]}')
    ax.set_aspect('equal'); ax.axis('off'); fig.tight_layout()
    fig.savefig(out_path, dpi=200); plt.close(fig)


# =============================================================================
# Episode runners — run a single seed of a single agent on a single maze
# =============================================================================
def _run_one_episode(agent_name, adj_mat, st_pos, start_node, rewarded_nodes,
                      n_actions, seed, min_rewarded_remaining, pomcp_num_sims):
    """Thin wrapper around _scaling_episode_utils.run_one_episode (consolidated 2026-05-03).

    Kept as a wrapper here so the subprocess worker (`_episode_worker` below)
    has a stable local-name to call. The actual agent dispatch logic lives in
    the canonical helper.
    """
    return _run_one_episode_canonical(
        agent_name=agent_name, adj_mat=adj_mat, st_pos=st_pos,
        start_node=start_node, rewarded_nodes=rewarded_nodes,
        n_actions=n_actions, seed=seed,
        min_rewarded_remaining=min_rewarded_remaining,
        pomcp_num_sims=pomcp_num_sims)


def _episode_worker(queue, kwargs):
    """Subprocess target. Posts result to queue."""
    try:
        queue.put({'status': 'ok', **_run_one_episode(**kwargs)})
    except Exception as exc:
        queue.put({'status': 'error', 'total_reward': float('nan'),
                   'wall_time_s': float('nan'), 'error': repr(exc),
                   'traceback': traceback.format_exc()})


def run_episode_with_timeout(timeout_s, **kwargs):
    """Subprocess hard-kill timeout (replaces signal.alarm). From cdx variant."""
    ctx = mp.get_context('spawn')
    queue = ctx.Queue(maxsize=1)
    proc = ctx.Process(target=_episode_worker, args=(queue, kwargs))
    t0 = time.time(); proc.start(); proc.join(timeout_s)
    elapsed = time.time() - t0
    if proc.is_alive():
        proc.terminate(); proc.join(5)
        if proc.is_alive(): proc.kill(); proc.join()
        return {'status': 'timeout', 'total_reward': float('nan'),
                'wall_time_s': elapsed,
                'error': f'episode > timeout_s={timeout_s}', 'traceback': ''}
    try:
        result = queue.get_nowait()
    except Empty:
        result = {'status': 'error', 'total_reward': float('nan'),
                  'wall_time_s': elapsed,
                  'error': f'child exit {proc.exitcode} no result', 'traceback': ''}
    result.setdefault('wall_time_s', elapsed)
    result.setdefault('error', ''); result.setdefault('traceback', '')
    return result


# =============================================================================
# Main loop
# =============================================================================
def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    p.add_argument('--smoke', action='store_true',
                   help='Tiny run: V=36, 1 instance, 1 seed, Random+Greedy only')
    p.add_argument('--resume-csv', type=Path, default=None,
                   help='Existing CSV; finished cells skipped')
    p.add_argument('--v-values', default=','.join(map(str, DEFAULT_V_VALUES)))
    p.add_argument('--agents', default=','.join(DEFAULT_AGENTS))
    p.add_argument('--n-instances', type=int, default=DEFAULT_INSTANCES)
    p.add_argument('--n-seeds', type=int, default=DEFAULT_SEEDS)
    p.add_argument('--seed-timeout-s', type=float, default=DEFAULT_SEED_TIMEOUT)
    p.add_argument('--cell-fail-safe-multiplier', type=float, default=10.0,
                   help='Per-cell fail-safe = N x n_seeds x seed_timeout')
    p.add_argument('--budget-multiplier', type=int, default=DEFAULT_BUDGET_MULT)
    p.add_argument('--reward-fraction', type=float, default=DEFAULT_REWARD_FRAC)
    p.add_argument('--min-rewarded-remaining', type=int, default=DEFAULT_MIN_REW_REM)
    p.add_argument('--maze-family', choices=['lattice', 'mouse_tessellation'],
                   default='lattice')
    p.add_argument('--edge-drop-prob', type=float, default=DEFAULT_EDGE_DROP)
    p.add_argument('--pomcp-num-sims', type=int, default=DEFAULT_POMCP_NUM_SIMS)
    p.add_argument('--plot-mazes', action='store_true')
    p.add_argument('--plot-dir', type=Path,
                   default=PROJECT_ROOT / 'reports' / 'figures' / 'scaling_mazes')
    return p.parse_args()


def main():
    args = parse_args()
    if args.smoke:
        args.v_values = '36'; args.agents = 'Random,Greedy_oracle'
        args.n_instances = 1; args.n_seeds = 1
        args.seed_timeout_s = min(args.seed_timeout_s, 60)

    v_values = [int(x) for x in args.v_values.split(',') if x]
    agents = [s.strip() for s in args.agents.split(',') if s.strip()]
    out_dir = DEFAULT_OUT_DIR; out_dir.mkdir(parents=True, exist_ok=True)
    if args.resume_csv:
        out_csv = args.resume_csv
        ts = out_csv.name.split('_scaling_sweep')[0]
    else:
        ts = 'c' + datetime.now().strftime('%y%m%d-%H%M%S')
        suffix = '_v2_smoke' if args.smoke else '_v2'
        out_csv = out_dir / f'{ts}_scaling_sweep{suffix}.csv'
    log_path = out_csv.with_suffix('.log')
    meta_path = out_csv.with_suffix('.metadata.json')

    existing = pd.read_csv(out_csv) if out_csv.exists() else pd.DataFrame()
    rows = existing.to_dict('records') if not existing.empty else []
    done_keys = set()
    if not existing.empty and 'status' in existing.columns:
        done = existing[existing['status'].isin(
            ['ok', 'timeout', 'error', 'skipped_cell_budget'])]
        done_keys = set(zip(done['V_target'], done['maze_family'],
                            done['instance'], done['agent'], done['seed']))

    cell_fail_safe = (args.cell_fail_safe_multiplier * args.n_seeds *
                       args.seed_timeout_s)

    run_meta = {'timestamp': ts, 'script': str(Path(__file__).resolve()),
                'args': vars(args), 'v_values': v_values, 'agents': agents,
                'cell_fail_safe_s': cell_fail_safe}
    meta_path.write_text(json.dumps(run_meta, indent=2, default=str) + '\n')
    print(f'Writing: {out_csv}')

    with open(log_path, 'a') as log_f:
        log_f.write(f'=== sweep v2 {datetime.now().isoformat(timespec="seconds")} ===\n')
        log_f.write(json.dumps(run_meta, default=str) + '\n')

        for V_target in v_values:
            n_actions = args.budget_multiplier * V_target
            for instance in range(args.n_instances):
                if args.maze_family == 'lattice':
                    adj_mat, st_pos, meta = make_lattice_subgraph_maze(
                        V_target, instance, args.edge_drop_prob)
                else:
                    adj_mat, st_pos, meta = make_mouse_tessellation_maze(
                        V_target, instance)
                n_nodes = adj_mat.shape[0]
                rng = np.random.RandomState(instance)
                start_node = 0
                n_rewarded = max(1, int(math.ceil(args.reward_fraction * n_nodes)))
                rewarded_idx = rng.choice(np.arange(n_nodes),
                                          size=min(n_rewarded, n_nodes),
                                          replace=False)
                rewarded_nodes = np.zeros(n_nodes, dtype=np.float64)
                rewarded_nodes[rewarded_idx] = 1.0

                if args.plot_mazes:
                    plot_path = args.plot_dir / f'{ts}_{args.maze_family}_V{V_target}_inst{instance}.pdf'
                    plot_maze_graph(adj_mat, st_pos, plot_path,
                                    title=f'{args.maze_family} V={V_target} n={n_nodes} inst={instance}',
                                    connectors=meta.get('connectors'))

                for agent in agents:
                    cell_t0 = time.time()
                    seeds_done = 0
                    for seed in range(args.n_seeds):
                        key = (V_target, meta['maze_family'], instance, agent, seed)
                        if key in done_keys:
                            seeds_done += 1; continue
                        cell_elapsed = time.time() - cell_t0
                        if cell_elapsed >= cell_fail_safe:
                            result = {'status': 'skipped_cell_budget',
                                      'total_reward': float('nan'),
                                      'wall_time_s': float('nan'),
                                      'error': 'cell fail-safe exceeded',
                                      'traceback': ''}
                        else:
                            seed_budget = min(args.seed_timeout_s,
                                              cell_fail_safe - cell_elapsed)
                            result = run_episode_with_timeout(
                                seed_budget,
                                agent_name=agent, adj_mat=adj_mat, st_pos=st_pos,
                                start_node=start_node,
                                rewarded_nodes=rewarded_nodes,
                                n_actions=n_actions, seed=seed,
                                min_rewarded_remaining=args.min_rewarded_remaining,
                                pomcp_num_sims=args.pomcp_num_sims)
                        if result['status'] == 'ok':
                            seeds_done += 1
                        rpa = (result['total_reward'] / max(n_actions, 1)
                               if pd.notna(result.get('total_reward')) else float('nan'))
                        row = {
                            'V_target': V_target, 'n_nodes': n_nodes,
                            'maze_family': meta['maze_family'],
                            'side': meta.get('side'), 'instance': instance,
                            'instance_seed': meta['instance_seed'],
                            'agent': agent, 'seed': seed,
                            'status': result['status'],
                            'n_expected_seeds': args.n_seeds,
                            'seeds_done': seeds_done,
                            'n_actions': n_actions,
                            'budget_multiplier': args.budget_multiplier,
                            'reward_fraction': args.reward_fraction,
                            'min_rewarded_remaining': args.min_rewarded_remaining,
                            'n_initial_rewards': int(rewarded_nodes.sum()),
                            'start_node': start_node,
                            'edge_drop_prob': meta.get('edge_drop_prob'),
                            'n_tiles': meta.get('n_tiles'),
                            'pomcp_num_sims': args.pomcp_num_sims,
                            'seed_timeout_s': args.seed_timeout_s,
                            'cell_fail_safe_s': cell_fail_safe,
                            'cell_wall_time_s': time.time() - cell_t0,
                            'total_reward': result.get('total_reward'),
                            'rpa': rpa,
                            'wall_time_s': result.get('wall_time_s'),
                            'error': result.get('error', ''),
                            'timestamp': datetime.now().isoformat(timespec='seconds'),
                        }
                        rows.append(row)
                        pd.DataFrame(rows).to_csv(out_csv, index=False)
                        log_f.write(json.dumps(row, default=str) + '\n')
                        log_f.flush()
                        print(f"  V={V_target:5d} inst={instance} {agent:18s} "
                              f"seed={seed} status={result['status']:6s} "
                              f"rpa={rpa if rpa==rpa else 'NaN':>6} "
                              f"t={result.get('wall_time_s', 0):6.1f}s")

    print(f'\nWrote {len(rows)} rows to {out_csv}')


if __name__ == '__main__':
    main()
