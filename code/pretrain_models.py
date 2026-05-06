"""
Multi-session pretraining experiments for top RL agents.

Compares baseline (single episode, no pretraining) vs pretrained (train on
prior sessions for same animal + same maze before target session).

Agents (best HPO configs):
  - DQN: default
  - DRQN_seq: large_net
  - DRQN_rand: large_net

Usage:
    source ~/miniconda3/etc/profile.d/conda.sh && conda activate latMaz_RL
    python code/pretrain_models.py
"""
import os
import sys
import time
import pandas as pd
import numpy as np
from datetime import datetime

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.dirname(__file__))

from utils_latMaz import get_now_str
from yoked_rl_runner import YokedRLRunner, MemoryMonitor
from experiment_config import (
    BEST_CONFIGS, OBS_CONFIG, HISTORY_LEN, DEFAULT_N_REPEATS, PRETRAIN_AGENTS,
    load_data, build_runner, obs_kwargs, output_path,
)

# Subset of BEST_CONFIGS for pretraining agents
AGENT_CONFIGS = {
    agent: {'config_name': BEST_CONFIGS[agent][0],
            'overrides': BEST_CONFIGS[agent][1]}
    for agent in PRETRAIN_AGENTS
}

N_REPEATS = DEFAULT_N_REPEATS


def main():
    yoking_df, rwd_df, sessions = load_data()
    print(f"Filtered sessions: {len(sessions)}")

    # Print pretrain session counts for each target
    temp_runner = build_runner(yoking_df, rwd_df)
    print("\nPretrain session counts per target:")
    for exp_mom in sessions:
        yoke_row = yoking_df[yoking_df['exp_moment'] == exp_mom].iloc[0]
        pre = temp_runner.get_pretrain_sessions(yoke_row)
        print(f"  {exp_mom}: {len(pre)} prior sessions "
              f"(animal={yoke_row['animal_ID']}, maze={yoke_row['adj_file']})")

    mem_monitor = MemoryMonitor(limit_gb=8.0, verbose=True)
    all_results = []
    sim_count = 0
    timestamp_str = get_now_str()
    final_output = output_path('260211_pretrain_results.csv')
    intermediate = output_path(f'{timestamp_str}_pretrain_PARTIAL.csv')
    _obs = obs_kwargs()

    for model_name, agent_cfg in AGENT_CONFIGS.items():
        config_name = agent_cfg['config_name']
        overrides = agent_cfg['overrides']

        runner = build_runner(yoking_df, rwd_df, model_name, overrides)

        n_total = len(sessions) * N_REPEATS * 2  # baseline + pretrained
        print(f"\n{'='*60}")
        print(f"  {model_name} / {config_name} ({n_total} sims)")
        print(f"{'='*60}")
        t0 = time.time()

        for exp_mom in sessions:
            yoke_row = yoking_df[yoking_df['exp_moment'] == exp_mom].iloc[0]
            pretrain_rows = runner.get_pretrain_sessions(yoke_row)
            n_pretrain = len(pretrain_rows)

            for rep in range(N_REPEATS):
                seed = 42 + rep + hash(f"{model_name}_{config_name}") % 10000

                # --- Baseline (no pretraining) ---
                sim_count += 1
                adj_mat, st_positions = runner._load_maze(
                    yoke_row['adj_file'], yoke_row['st_pos_file'])
                rewards, reset_val = runner._get_rewarded_states(
                    exp_mom, len(st_positions))
                n_actions = int(yoke_row['n_states_visited']) - 1

                baseline_reward = runner.run_single(
                    model_name=model_name,
                    adj_mat=adj_mat,
                    st_positions=st_positions,
                    start_node=int(yoke_row['start_state']),
                    rewards=rewards,
                    n_actions=n_actions,
                    **_obs,
                    min_allowed_rewarded_states=reset_val,
                    seed=seed,
                )

                all_results.append({
                    'exp_moment': exp_mom,
                    'animal_ID': yoke_row.get('animal_ID', ''),
                    'model': model_name,
                    'hpo_config': config_name,
                    'pretrain_type': 'none',
                    'n_pretrain_sessions': 0,
                    'repeat_idx': rep,
                    'total_reward': baseline_reward,
                    'n_actions': n_actions,
                    'seed': seed,
                    'timestamp': datetime.now().isoformat(),
                })

                # --- Pretrained ---
                sim_count += 1
                pretrain_reward, n_pre = runner.run_single_pretrained(
                    model_name=model_name,
                    target_row=yoke_row,
                    pretrain_rows=pretrain_rows,
                    **_obs,
                    seed=seed,
                )

                all_results.append({
                    'exp_moment': exp_mom,
                    'animal_ID': yoke_row.get('animal_ID', ''),
                    'model': model_name,
                    'hpo_config': config_name,
                    'pretrain_type': 'prior_sessions_same_maze',
                    'n_pretrain_sessions': n_pre,
                    'repeat_idx': rep,
                    'total_reward': pretrain_reward,
                    'n_actions': n_actions,
                    'seed': seed,
                    'timestamp': datetime.now().isoformat(),
                })

                # Memory check
                if sim_count % 20 == 0:
                    mem_monitor.check_threshold(
                        f"sim {sim_count} ({model_name})")

                # Periodic save
                if sim_count % 50 == 0 and all_results:
                    pd.DataFrame(all_results).to_csv(
                        intermediate, index=False)
                    print(f"  Saved intermediate ({len(all_results)} rows)")

        elapsed = time.time() - t0
        model_results = [r for r in all_results if r['model'] == model_name]
        baseline_mean = np.mean([r['total_reward'] for r in model_results
                                 if r['pretrain_type'] == 'none'])
        pretrain_mean = np.mean([r['total_reward'] for r in model_results
                                 if r['pretrain_type'] == 'prior_sessions_same_maze'])
        print(f"  {model_name}: baseline={baseline_mean:.1f}, "
              f"pretrained={pretrain_mean:.1f}, "
              f"delta={pretrain_mean - baseline_mean:+.1f}, "
              f"time={elapsed:.0f}s")

    # Save final results
    df = pd.DataFrame(all_results)
    df.to_csv(final_output, index=False)
    print(f"\nSaved {len(df)} results to {final_output}")

    # Clean up intermediate
    if os.path.exists(intermediate):
        os.remove(intermediate)

    # Summary
    print("\n" + "=" * 70)
    print("PRETRAINING RESULTS SUMMARY")
    print("=" * 70)
    for model_name in AGENT_CONFIGS:
        mdf = df[df['model'] == model_name]
        if len(mdf) == 0:
            continue
        base = mdf[mdf['pretrain_type'] == 'none']['total_reward']
        pre = mdf[mdf['pretrain_type'] == 'prior_sessions_same_maze']['total_reward']
        print(f"  {model_name:>12}: baseline={base.mean():.1f} +/- {base.std():.1f}, "
              f"pretrained={pre.mean():.1f} +/- {pre.std():.1f}, "
              f"delta={pre.mean() - base.mean():+.1f}")

    mem_monitor.snapshot("run_end")
    summary = mem_monitor.get_summary()
    print(f"\n[Memory Summary] Peak RSS: {summary.get('process_peak_mb', 0):.0f} MB")


if __name__ == '__main__':
    main()
