"""
Hyperparameter tuning for all 9 models.

Runs multiple HPO configs per model on the filtered session set
(exp_moment > 251001, n_states > 50). Uses ego observation, h_len=0.

Usage:
    source ~/miniconda3/etc/profile.d/conda.sh && conda activate latMaz_RL
    python code/tune_models.py
"""
import os
import sys
import time
import pandas as pd
import numpy as np
from copy import deepcopy

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.dirname(__file__))

from yoked_rl_runner import (
    YokedRLRunner, SimConfig, AlgoConfig, MemoryConfig,
)
from experiment_config import load_data, build_runner, output_path

# =============================================================================
# HPO Configs per model: {config_name: {param_overrides}}
# =============================================================================

HPO_CONFIGS = {
    'A2C': {
        'default': {},
        'high_ent': {'ent_coef': 0.05, 'learning_rate': 1e-3},
        'low_lr': {'learning_rate': 1e-4, 'n_steps': 16, 'gae_lambda': 0.95},
        'fast_update': {'n_steps': 1, 'learning_rate': 5e-4, 'gamma': 0.95},
    },
    'DQN': {
        'default': {},
        'aggressive_explore': {
            'learning_rate': 5e-4, 'learning_starts': 10, 'train_freq': 1,
            'target_update_interval': 100, 'batch_size': 32,
        },
        'high_lr': {
            'learning_rate': 5e-3, 'learning_starts': 10, 'train_freq': 2,
            'target_update_interval': 50, 'batch_size': 32,
        },
        'soft_update': {
            'learning_rate': 1e-3, 'tau': 0.1, 'learning_starts': 20,
            'train_freq': 2, 'target_update_interval': 1, 'batch_size': 32,
        },
    },
    'PPO': {
        'default': {},
        'high_ent': {'ent_coef': 0.05, 'learning_rate': 5e-4, 'n_epochs': 5},
        'small_batch': {'batch_size': 32, 'n_steps': 64, 'n_epochs': 20, 'learning_rate': 1e-3},
        'low_gamma': {'gamma': 0.95, 'gae_lambda': 0.9, 'ent_coef': 0.02},
    },
    'TRPO': {
        'default': {},
        'loose_kl': {'target_kl': 0.05, 'n_steps': 32, 'cg_max_steps': 15, 'learning_rate': 5e-4},
        'tight_kl': {'target_kl': 0.003, 'n_steps': 128, 'cg_max_steps': 20, 'learning_rate': 1e-3},
    },
    'QRDQN': {
        'default': {},
        'aggressive': {
            'learning_rate': 1e-3, 'exploration_fraction': 0.5,
            'exploration_final_eps': 0.05, 'batch_size': 32,
        },
        'conservative': {
            'learning_rate': 1e-4, 'exploration_fraction': 0.3,
            'exploration_final_eps': 0.01, 'batch_size': 128,
        },
    },
    'RecurrentPPO': {
        'default': {},
        'high_ent': {
            'ent_coef': 0.05, 'learning_rate': 5e-4, 'n_epochs': 5,
            'lstm_hidden_size': 64, 'n_lstm_layers': 1,
        },
        'large_lstm': {
            'lstm_hidden_size': 128, 'n_lstm_layers': 1,
            'learning_rate': 1e-4, 'n_steps': 256, 'batch_size': 64,
        },
        'small_fast': {
            'lstm_hidden_size': 32, 'n_lstm_layers': 1,
            'learning_rate': 1e-3, 'n_steps': 64, 'n_epochs': 5, 'ent_coef': 0.02,
        },
    },
    'DRQN_seq': {
        'default': {},
        'fast_decay': {
            'epsilon_decay_steps': 50, 'learning_rate': 5e-4,
            'train_every_n_steps': 2, 'target_update_every': 25,
            'replay_mode': 'sequential',
        },
        'large_net': {
            'hidden_dim': 128, 'lstm_hidden': 128,
            'learning_rate': 5e-4, 'seq_len': 20, 'batch_size': 8,
            'replay_mode': 'sequential',
        },
        'aggressive': {
            'epsilon_decay_steps': 100, 'epsilon_end': 0.1,
            'train_every_n_steps': 1, 'target_update_every': 20,
            'learning_rate': 2e-3, 'batch_size': 32, 'seq_len': 5,
            'replay_mode': 'sequential',
        },
    },
    'DRQN_rand': {
        'default': {},
        'fast_decay': {
            'epsilon_decay_steps': 50, 'learning_rate': 5e-4,
            'train_every_n_steps': 2, 'target_update_every': 25,
            'replay_mode': 'random',
        },
        'large_net': {
            'hidden_dim': 128, 'lstm_hidden': 128,
            'learning_rate': 5e-4, 'seq_len': 20, 'batch_size': 8,
            'replay_mode': 'random',
        },
        'aggressive': {
            'epsilon_decay_steps': 100, 'epsilon_end': 0.1,
            'train_every_n_steps': 1, 'target_update_every': 20,
            'learning_rate': 2e-3, 'batch_size': 32, 'seq_len': 5,
            'replay_mode': 'random',
        },
    },
    'POMCP': {
        'default': {},
        'more_sims': {'num_sims': 1000, 'max_depth': 80},
        'fewer_sims': {'num_sims': 200, 'max_depth': 30},
    },
}


def run_hpo_config(runner, model_name, config_name, param_overrides,
                   sessions, n_repeats=5):
    """Run a single HPO config and return results DataFrame."""
    # Build custom AlgoConfig with overrides
    algo = AlgoConfig()
    base = deepcopy(getattr(algo, model_name))
    base.update(param_overrides)
    setattr(algo, model_name, base)

    # Create a runner with the custom algo config
    custom_runner = YokedRLRunner(
        yoking_df=runner.yoking_df,
        rewarded_states_df=None,  # rwd_lookup already built
        maze_dir=runner.maze_dir,
        output_dir=runner.output_dir,
        algo_config=algo,
    )
    # Copy over the pre-built rwd_lookup
    custom_runner.rwd_lookup = runner.rwd_lookup
    custom_runner._maze_cache = runner._maze_cache

    # Use different seed base per HPO config to avoid identical random streams
    seed_offset = hash(config_name) % 10000
    config = SimConfig(
        models=[model_name],
        history_lengths=[0],
        observation_configs=[{
            'name': 'ego', 'obs_type': 'ego', 'action_type': 'ego',
            'include_prev_action': True, 'include_prev_reward': True,
        }],
        seed_base=42 + seed_offset,
        verbose=False,
    )
    mem = MemoryConfig(enabled=True, limit_gb=8.0, verbose=False,
                       snapshot_every_n_sims=200)

    df = custom_runner.run(config, target_repeats=n_repeats,
                           session_filter=sessions, memory_config=mem)
    df['hpo_config'] = config_name
    return df


def main():
    yoking_df, rwd_df, sessions = load_data()
    print(f"Filtered sessions: {len(sessions)}")

    runner = build_runner(yoking_df, rwd_df)

    all_results = []
    n_repeats = 5

    # POMCP: run on subset (slow)
    pomcp_sessions = sessions[:5]

    for model_name, configs in HPO_CONFIGS.items():
        model_sessions = pomcp_sessions if model_name == 'POMCP' else sessions

        for config_name, overrides in configs.items():
            n_sims = len(model_sessions) * n_repeats
            print(f"\n{'='*60}")
            print(f"  {model_name} / {config_name} "
                  f"({n_sims} sims on {len(model_sessions)} sessions)")
            print(f"  Overrides: {overrides}")
            print(f"{'='*60}")

            t0 = time.time()
            try:
                df = run_hpo_config(
                    runner, model_name, config_name, overrides,
                    model_sessions, n_repeats)
                elapsed = time.time() - t0

                mean_r = df['total_reward'].mean()
                std_r = df['total_reward'].std()
                rpa = (df['total_reward'] / df['n_actions']).mean()
                print(f"  -> mean={mean_r:.1f} +/- {std_r:.1f}, "
                      f"rpa={rpa:.3f}, time={elapsed:.0f}s")

                all_results.append(df)
            except Exception as e:
                print(f"  -> FAILED: {e}")
                continue

    # Combine all results
    if all_results:
        combined = pd.concat(all_results, ignore_index=True)
        out = output_path('260210_hpo_tuning_results.csv')
        combined.to_csv(out, index=False)
        print(f"\n\nSaved {len(combined)} results to {out}")

        # Best config per model
        print("\n" + "=" * 70)
        print("BEST CONFIG PER MODEL (by mean reward)")
        print("=" * 70)
        best = (combined.groupby(['model', 'hpo_config'])['total_reward']
                .agg(['mean', 'std', 'count'])
                .sort_values('mean', ascending=False))
        for model in HPO_CONFIGS.keys():
            model_results = best.loc[best.index.get_level_values('model') == model]
            if len(model_results) > 0:
                best_row = model_results.iloc[0]
                best_cfg = model_results.index[0][1]
                print(f"  {model:>15}: {best_cfg:<20} "
                      f"mean={best_row['mean']:.1f} +/- {best_row['std']:.1f}")

        # Overall ranking
        print("\n" + "=" * 70)
        print("OVERALL RANKING (best config per model, by mean reward)")
        print("=" * 70)
        overall = []
        for model in HPO_CONFIGS.keys():
            model_df = combined[combined['model'] == model]
            if len(model_df) == 0:
                continue
            for cfg in model_df['hpo_config'].unique():
                cfg_df = model_df[model_df['hpo_config'] == cfg]
                overall.append({
                    'model': model,
                    'config': cfg,
                    'mean_reward': cfg_df['total_reward'].mean(),
                    'std': cfg_df['total_reward'].std(),
                    'mean_rpa': (cfg_df['total_reward'] / cfg_df['n_actions']).mean(),
                    'n': len(cfg_df),
                })
        ranking_df = pd.DataFrame(overall).sort_values('mean_reward', ascending=False)
        print(ranking_df.to_string(index=False))


if __name__ == '__main__':
    main()
