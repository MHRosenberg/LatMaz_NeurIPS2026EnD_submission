"""
260504_run_a044_no_training_agents.py

Run no-training agents on a044's allocentric sessions (and a033's 5
allocentric sessions post-2026-03-14) to extend Appendix~K coverage to
cohort-176. Mirrors the structure of 260414_run_expanded152_no_
training_agents.py but targets only the new allocentric sessions.

Agents (all deterministic-given-yoking-df + seed):
  - Random, Random_forward (heuristics)
  - FullFwd, NoBk+FullFwd (heuristics)
  - Mouse-bias ego / allo-real / allo-latent (CloningAgent in 3 frames)
  - ConditionalCloning ego (ConditionalCloningAgent)
  - FSC_bio (FSCBioAgent)
  - Greedy_oracle (heuristic with full graph)

NOT run here (heavy / require training): POMCP, POMCP_bio, RL HPO-best.

Output:
  data_out/rl_sims/c{ts}_a044_allo_no_training_agents.csv
"""
from __future__ import annotations
import sys
import os
import ast
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path('<REPO_ROOT>')
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(PROJECT_ROOT / 'code0_init-readOnly' / 'code'))

from intermediate_agents import (
    FullFwdAgent, NoBkFullFwdAgent, CloningAgent,
    ConditionalCloningAgent, FSCBioAgent,
)
from experiment_config import (
    load_yoking_df, load_reward_df, build_runner,
    DEFAULT_N_REPEATS, run_random_agent, run_forward_random_agent,
    run_greedy_agent,
)


# Target sessions: a044 (all valid, n_states>50) + a033 post-2026-03-14
TARGET_ANIMALS = ['044']  # a044 entirely allocentric
INCLUDE_a033_POST_260314 = True  # a033's 5 working-allo sessions

N_REPEATS = DEFAULT_N_REPEATS  # 5 seeds


def get_target_sessions(yoking_df):
    sub = yoking_df[yoking_df['animal_ID'].astype(str).str.zfill(3).isin(TARGET_ANIMALS)]
    sub = sub[pd.to_numeric(sub['n_states_visited'], errors='coerce') > 50]
    sessions = list(sub['exp_moment'].dropna().unique())

    if INCLUDE_a033_POST_260314:
        a033 = yoking_df[yoking_df['animal_ID'].astype(str).str.zfill(3) == '033']
        a033 = a033[pd.to_numeric(a033['n_states_visited'], errors='coerce') > 50]
        # Filter by date >= 260314
        a033_allo = a033[a033['exp_moment'].apply(
            lambda s: s.startswith('260314') or s.startswith('260315') or s.startswith('260316') or
                       s.startswith('260317') or s.startswith('260318') or
                       s >= '260319')]
        # Drop pre-260314 ones
        a033_allo = a033_allo[a033_allo['exp_moment'] >= '260314-000000']
        sessions += list(a033_allo['exp_moment'].dropna().unique())
    return sorted(set(sessions))


def main():
    ts = 'c' + datetime.now().strftime('%y%m%d-%H%M%S')
    out = PROJECT_ROOT / 'data_out' / 'rl_sims' / f'{ts}_a044_allo_no_training_agents.csv'
    out.parent.mkdir(parents=True, exist_ok=True)
    print(f'Output: {out}')

    yoking_df = load_yoking_df()
    rwd_df = load_reward_df()
    runner = build_runner(yoking_df, rwd_df)

    sessions = get_target_sessions(yoking_df)
    print(f'Target sessions: {len(sessions)}')
    if not sessions:
        print('No sessions to run.')
        return

    rows = []
    for sess in sessions:
        yoke_row = yoking_df[yoking_df['exp_moment'] == sess].iloc[0]
        a_id = str(yoke_row['animal_ID']).zfill(3)
        adj_mat, st_pos = runner._load_maze(yoke_row['adj_file'], yoke_row['st_pos_file'])
        rewards, reset_val = runner._get_rewarded_states(sess, len(st_pos))
        n_actions = int(yoke_row['n_states_visited']) - 1
        start_node = int(yoke_row['start_state'])
        common = dict(adj_mat=adj_mat, node_positions=st_pos, start_node=start_node,
                       rewarded_nodes=rewards, n_actions=n_actions,
                       min_rewarded_states=int(reset_val))

        for seed in range(N_REPEATS):
            # Heuristics
            for agent_name, fn in [
                ('Random', lambda: run_random_agent(adj_mat, st_pos, start_node,
                                                       rewards, n_actions, int(reset_val),
                                                       seed=seed, prevent_reverse=False)),
                ('Random_forward', lambda: run_forward_random_agent(adj_mat, st_pos, start_node,
                                                                       rewards, n_actions, int(reset_val),
                                                                       seed=seed)),
                ('Greedy_oracle', lambda: run_greedy_agent(adj_mat, st_pos, start_node,
                                                              rewards, n_actions, int(reset_val),
                                                              seed=seed)),
                ('FullFwd', lambda: FullFwdAgent().run_episode(seed=seed, **common)),
                ('NoBkFullFwd', lambda: NoBkFullFwdAgent().run_episode(seed=seed, **common)),
                ('FSC_bio', lambda: FSCBioAgent().run_episode(seed=seed, **common)),
            ]:
                try:
                    rew = fn()
                    rpa = float(rew) / max(n_actions, 1)
                    rows.append({'exp_moment': sess, 'animal_ID': a_id, 'model': agent_name,
                                 'seed': seed, 'total_reward': float(rew),
                                 'n_actions': n_actions, 'rpa': rpa, 'status': 'ok'})
                except Exception as e:
                    rows.append({'exp_moment': sess, 'animal_ID': a_id, 'model': agent_name,
                                 'seed': seed, 'total_reward': float('nan'),
                                 'n_actions': n_actions, 'rpa': float('nan'),
                                 'status': f'error: {repr(e)[:120]}'})

            # Mouse-bias (3 frames) — needs prob_dict
            for frame_key, agent_name in [
                ('prob_dict_ego',         'Mouse-bias_ego'),
                ('prob_dict_allo_real',   'Mouse-bias_allo_real'),
                ('prob_dict_allo_latent', 'Mouse-bias_allo_latent'),
            ]:
                pd_str = yoke_row.get(frame_key)
                if pd_str is None or (isinstance(pd_str, float) and np.isnan(pd_str)):
                    continue
                try:
                    pd_dict = ast.literal_eval(pd_str) if isinstance(pd_str, str) else pd_str
                    frame = frame_key.replace('prob_dict_', '')
                    rew = CloningAgent(prob_dict=pd_dict, frame=frame).run_episode(seed=seed, **common)
                    rpa = float(rew) / max(n_actions, 1)
                    rows.append({'exp_moment': sess, 'animal_ID': a_id, 'model': agent_name,
                                 'seed': seed, 'total_reward': float(rew),
                                 'n_actions': n_actions, 'rpa': rpa, 'status': 'ok'})
                except Exception as e:
                    rows.append({'exp_moment': sess, 'animal_ID': a_id, 'model': agent_name,
                                 'seed': seed, 'total_reward': float('nan'),
                                 'n_actions': n_actions, 'rpa': float('nan'),
                                 'status': f'error: {repr(e)[:120]}'})

            # Conditional cloning (only ego frame)
            try:
                states_v = ast.literal_eval(yoke_row['states_visited']) if isinstance(yoke_row['states_visited'], str) else yoke_row['states_visited']
                cca = ConditionalCloningAgent(
                    states_visited=states_v, adj_mat=adj_mat, node_positions=st_pos,
                    start_node=start_node, rewarded_nodes=rewards.copy(),
                    min_rewarded_states=int(reset_val))
                rew = cca.run_episode(seed=seed, **common)
                rpa = float(rew) / max(n_actions, 1)
                rows.append({'exp_moment': sess, 'animal_ID': a_id, 'model': 'ConditionalClone_ego',
                             'seed': seed, 'total_reward': float(rew),
                             'n_actions': n_actions, 'rpa': rpa, 'status': 'ok'})
            except Exception as e:
                rows.append({'exp_moment': sess, 'animal_ID': a_id, 'model': 'ConditionalClone_ego',
                             'seed': seed, 'total_reward': float('nan'),
                             'n_actions': n_actions, 'rpa': float('nan'),
                             'status': f'error: {repr(e)[:120]}'})

        if (sessions.index(sess) + 1) % 5 == 0 or sessions.index(sess) == len(sessions) - 1:
            pd.DataFrame(rows).to_csv(out, index=False)
            print(f'  [{sessions.index(sess)+1}/{len(sessions)}] sess={sess} a{a_id} rows-so-far={len(rows)}')

    pd.DataFrame(rows).to_csv(out, index=False)
    print(f'\nDone. {len(rows)} rows written to {out}')
    print('Per-agent mean RPA:')
    df = pd.DataFrame(rows)
    print(df[df['status']=='ok'].groupby('model')['rpa'].agg(['count', 'mean', 'std']).to_string())


if __name__ == '__main__':
    main()
