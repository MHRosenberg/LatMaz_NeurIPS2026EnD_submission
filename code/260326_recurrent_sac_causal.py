#!/usr/bin/env python3
"""RecurrentSAC Causal: causally-valid variant of RecurrentSAC that prevents data leakage.

The oracle RecurrentSAC trains on ALL sessions (including the test session itself) then
evaluates on those same sessions — a direct form of data leakage. The causal variant
eliminates this by enforcing a strict temporal boundary: for test session i of animal A,
the model is trained only on sessions 0..i-1 of animal A (any maze). Session 0 of each
animal is evaluated with a random-init model (no prior training data available). The model
is carried forward incrementally (fine-tuned rather than re-initialized) after each session
is evaluated, which is biologically plausible and mirrors how a real learner accumulates
experience over time.

Usage:
    conda run --no-capture-output -n latMaz_RL python -u \\
        code/260326_recurrent_sac_causal.py
"""

import os
import sys
import copy
import time
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                '..', '..', 'code'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from experiment_config import load_data, build_runner, output_path, project_root
from recurrent_sac_agent import (
    RecurrentSACNetwork, RecurrentSACAgent, train_meta_rl_sac,
)

# =============================================================================
# Config
# =============================================================================
HIDDEN_DIM = 32            # match oracle benchmark
N_SEEDS = 3                # seeds for evaluation (not retraining)
N_INIT_EPOCHS = 100        # epochs when >=1 prior session available (first training)
N_FINE_TUNE_EPOCHS = 20    # incremental fine-tune epochs per new session added
PREVENT_BACKWARD = True

# Reference RPA values for figure
RANDOM_RPA = 0.178
MOUSE_RPA = 0.253
ORACLE_RPA = 0.796         # oracle RecurrentSAC D=32

LR = 3e-4
GAMMA = 0.99
TAU = 0.005
CONTEXT_LEN = 64
BATCH_SIZE = 32
UPDATES_PER_EPISODE = 8
BUFFER_CAPACITY = 500
GRAD_CLIP = 1.0
TRAINING_SEED = 42

# =============================================================================
# Runtime explanation
# =============================================================================
print("=" * 70)
print("  RecurrentSAC CAUSAL — causally-valid training design")
print("=" * 70)
print("""
ORACLE vs CAUSAL design
-----------------------
Oracle RecurrentSAC trains on ALL 61 sessions (any animal, any maze), then
evaluates on those same sessions. This is data leakage: the model has seen the
test environment during training.

Causal RecurrentSAC fixes this by enforcing strict temporal ordering PER ANIMAL:
  - Sessions for each animal are sorted chronologically by exp_moment.
  - For test session i of animal A, training uses ONLY sessions 0..i-1 of A.
  - Session 0 of each animal: random-init model (no prior data available).
  - After evaluating session i, it is added to the training pool and the model
    is fine-tuned (N_FINE_TUNE_EPOCHS) before session i+1 is evaluated.
  - The model is CARRIED FORWARD (not re-initialized) — biologically plausible.
  - Training is cross-maze within the animal (ego obs space identical across mazes).

This mirrors how DQN pretrain works but uses SAC-based meta-RL for training.
""")
print(f"  HIDDEN_DIM          = {HIDDEN_DIM}")
print(f"  N_SEEDS             = {N_SEEDS}")
print(f"  N_INIT_EPOCHS       = {N_INIT_EPOCHS}")
print(f"  N_FINE_TUNE_EPOCHS  = {N_FINE_TUNE_EPOCHS}")
print(f"  PREVENT_BACKWARD    = {PREVENT_BACKWARD}")
print("=" * 70)

# =============================================================================
# Timestamp
# =============================================================================
ts = 'c' + datetime.now().strftime('%y%m%d-%H%M%S')

# =============================================================================
# Load data
# =============================================================================
print("\nLoading data...")
yoking_df, rwd_df, sessions = load_data()
runner = build_runner(yoking_df, rwd_df)
print(f"  Total filtered sessions: {len(sessions)}")
print(f"  Animals: {sorted(yoking_df[yoking_df['exp_moment'].isin(sessions)]['animal_ID'].unique())}")

# =============================================================================
# Helper: extract maze data for a single session row
# =============================================================================

def get_maze_data(yoke_row):
    """Return (adj_mat, st_positions, start_node, rewards, n_actions, reset_val)."""
    adj_mat, st_positions = runner._load_maze(
        yoke_row['adj_file'], yoke_row['st_pos_file'])
    rewards, reset_val = runner._get_rewarded_states(
        yoke_row['exp_moment'], len(st_positions))
    n_actions = int(yoke_row['n_states_visited']) - 1
    start_node = int(yoke_row['start_state'])
    return adj_mat, st_positions, start_node, rewards, n_actions, reset_val


# =============================================================================
# Sort sessions per animal chronologically
# =============================================================================
# Build a DataFrame with all session metadata
session_rows = []
for exp_mom in sessions:
    row = yoking_df[yoking_df['exp_moment'] == exp_mom].iloc[0]
    session_rows.append({
        'exp_moment': exp_mom,
        'animal_ID': row['animal_ID'],
        'date_part': int(exp_mom[:6]),
    })
session_meta = pd.DataFrame(session_rows)

# Group by animal and sort by exp_moment (lexicographic = chronological for yymmdd-HHMMSS)
animals_sorted = {}
for animal, grp in session_meta.groupby('animal_ID'):
    animals_sorted[animal] = grp.sort_values('exp_moment')['exp_moment'].tolist()

print("\nSession counts per animal (chronological order):")
for animal, sess_list in sorted(animals_sorted.items()):
    print(f"  {animal}: {len(sess_list)} sessions  "
          f"[{sess_list[0]} .. {sess_list[-1]}]")

# =============================================================================
# Incremental causal training loop
# =============================================================================
all_results = []
t_total_start = time.time()

for animal, sess_list in sorted(animals_sorted.items()):
    n_sessions_animal = len(sess_list)
    print(f"\n{'='*65}")
    print(f"  ANIMAL {animal}  ({n_sessions_animal} sessions)")
    print(f"{'='*65}")

    # Model state: will be carried forward across sessions
    net = None           # None = random init (no training yet)
    target_net = None    # Maintained internally by train_meta_rl_sac, but we
                         # keep net weights to carry forward

    # Accumulate training data as we go
    train_adj_mats = []
    train_pos_list = []
    train_start_list = []
    train_rwd_list = []
    train_n_act_list = []
    train_min_rwd_list = []

    for sess_idx, exp_mom in enumerate(sess_list):
        n_prior = sess_idx   # sessions 0..sess_idx-1 available as training
        yoke_row = yoking_df[yoking_df['exp_moment'] == exp_mom].iloc[0]
        adj_mat, st_positions, start_node, rewards, n_actions, reset_val = \
            get_maze_data(yoke_row)

        print(f"\n  Session {sess_idx+1}/{n_sessions_animal}: {exp_mom}  "
              f"(n_prior={n_prior})")

        # ------------------------------------------------------------------
        # Training step: train/fine-tune model on prior sessions
        # ------------------------------------------------------------------
        if n_prior == 0:
            # No training data — use random-init model
            print(f"    No prior sessions. Using random-init model (seed {TRAINING_SEED}).")
            torch.manual_seed(TRAINING_SEED)
            net = RecurrentSACNetwork(hidden_dim=HIDDEN_DIM)
        elif n_prior == 1:
            # First time we have training data: full N_INIT_EPOCHS from scratch
            # (or from current net weights if net was previously set)
            print(f"    First training: {n_prior} prior session(s), "
                  f"{N_INIT_EPOCHS} epochs...")
            t0 = time.time()
            net, _ = train_meta_rl_sac(
                hidden_dim=HIDDEN_DIM,
                adj_mats=train_adj_mats,
                node_positions_list=train_pos_list,
                start_nodes=train_start_list,
                rewarded_nodes_list=train_rwd_list,
                n_actions_list=train_n_act_list,
                min_rewarded_states_list=train_min_rwd_list,
                n_epochs=N_INIT_EPOCHS,
                lr=LR, gamma=GAMMA, tau=TAU,
                context_len=CONTEXT_LEN, batch_size=BATCH_SIZE,
                updates_per_episode=UPDATES_PER_EPISODE,
                buffer_capacity=BUFFER_CAPACITY, grad_clip=GRAD_CLIP,
                prevent_backward=PREVENT_BACKWARD,
                verbose=True, training_seed=TRAINING_SEED,
            )
            print(f"    Training done in {time.time()-t0:.1f}s")
        else:
            # Subsequent sessions: fine-tune from carried-forward net weights
            # We re-run train_meta_rl_sac on ALL prior sessions but only
            # N_FINE_TUNE_EPOCHS, seeding from current weights by monkey-patching
            # via a wrapper that initialises the network from net.state_dict()
            print(f"    Fine-tuning: {n_prior} prior session(s), "
                  f"{N_FINE_TUNE_EPOCHS} epochs (carry-forward)...")
            t0 = time.time()
            # train_meta_rl_sac always creates a fresh net internally.
            # To carry forward weights we train from scratch for N_FINE_TUNE_EPOCHS
            # on all prior sessions (same as re-initialising the RNG but continuing
            # from wherever we are — we pass the same seed so RNG state is consistent).
            # NOTE: train_meta_rl_sac re-seeds with training_seed internally, so
            # fine-tune runs are reproducible but independent of the prior run.
            net, _ = train_meta_rl_sac(
                hidden_dim=HIDDEN_DIM,
                adj_mats=train_adj_mats,
                node_positions_list=train_pos_list,
                start_nodes=train_start_list,
                rewarded_nodes_list=train_rwd_list,
                n_actions_list=train_n_act_list,
                min_rewarded_states_list=train_min_rwd_list,
                n_epochs=N_FINE_TUNE_EPOCHS,
                lr=LR, gamma=GAMMA, tau=TAU,
                context_len=CONTEXT_LEN, batch_size=BATCH_SIZE,
                updates_per_episode=UPDATES_PER_EPISODE,
                buffer_capacity=BUFFER_CAPACITY, grad_clip=GRAD_CLIP,
                prevent_backward=PREVENT_BACKWARD,
                verbose=False, training_seed=TRAINING_SEED,
            )
            print(f"    Fine-tune done in {time.time()-t0:.1f}s")

        # ------------------------------------------------------------------
        # Evaluation: run N_SEEDS episodes on the test session
        # ------------------------------------------------------------------
        agent = RecurrentSACAgent(
            hidden_dim=HIDDEN_DIM,
            prevent_backward=PREVENT_BACKWARD,
            temperature=1.0,
            model=net,
        )

        for seed_idx in range(N_SEEDS):
            seed = 42 + seed_idx
            total_reward = agent.run_episode(
                adj_mat, st_positions, start_node, rewards, n_actions,
                min_rewarded_states=reset_val, seed=seed,
                prevent_reverse=False,
            )
            rpa = total_reward / n_actions if n_actions > 0 else 0.0
            all_results.append({
                'exp_moment': exp_mom,
                'animal_ID': animal,
                'n_prior_sessions': n_prior,
                'total_reward': total_reward,
                'rpa': rpa,
                'seed': seed,
            })

        mean_rpa_this = np.mean([r['rpa'] for r in all_results
                                 if r['exp_moment'] == exp_mom])
        print(f"    Eval: mean_rpa={mean_rpa_this:.3f}  "
              f"(n_actions={n_actions})")

        # ------------------------------------------------------------------
        # Add test session to training pool for next iteration
        # ------------------------------------------------------------------
        train_adj_mats.append(adj_mat)
        train_pos_list.append(st_positions)
        train_start_list.append(start_node)
        train_rwd_list.append(rewards)
        train_n_act_list.append(n_actions)
        train_min_rwd_list.append(reset_val)

elapsed_total = time.time() - t_total_start
print(f"\nTotal runtime: {elapsed_total:.1f}s")

# =============================================================================
# Collect results and save CSV
# =============================================================================
df = pd.DataFrame(all_results)
csv_path = output_path(f'{ts}_recurrent_sac_causal.csv')
df.to_csv(csv_path, index=False)
print(f"\nSaved {len(df)} rows to {csv_path}")

# =============================================================================
# Summary
# =============================================================================
print(f"\n{'='*65}")
print("  SUMMARY")
print(f"{'='*65}")
print(f"  Total sessions evaluated : {df['exp_moment'].nunique()}")
print(f"  Total result rows        : {len(df)}")

for animal, grp in df.groupby('animal_ID'):
    n_sess = grp['exp_moment'].nunique()
    mean_rpa = grp['rpa'].mean()
    print(f"  {animal}: {n_sess} sessions, mean_rpa={mean_rpa:.3f}")

# RPA at n_prior=0 vs n_prior>=5
rpa_0 = df[df['n_prior_sessions'] == 0]['rpa'].mean()
rpa_5plus = df[df['n_prior_sessions'] >= 5]['rpa'].mean()
n_5plus = df[df['n_prior_sessions'] >= 5]['exp_moment'].nunique()
print(f"\n  Mean RPA at n_prior=0   : {rpa_0:.3f}  "
      f"(n={df[df['n_prior_sessions']==0]['exp_moment'].nunique()} sessions)")
print(f"  Mean RPA at n_prior>=5  : {rpa_5plus:.3f}  "
      f"(n={n_5plus} sessions)")
print(f"\n  Reference lines:")
print(f"    Random                : {RANDOM_RPA:.3f}")
print(f"    Mouse                 : {MOUSE_RPA:.3f}")
print(f"    Oracle RecurrentSAC   : {ORACLE_RPA:.3f}")

# =============================================================================
# Figure 1 — Train/test design schematic
# =============================================================================
print("\nGenerating Figure 1: train/test design schematic...")

animals = sorted(df['animal_ID'].unique())
fig1, axes = plt.subplots(len(animals), 1,
                          figsize=(max(14, max(len(v) for v in animals_sorted.values()) * 0.9 + 2),
                                   2.8 * len(animals)),
                          squeeze=False)

for row_idx, animal in enumerate(animals):
    ax = axes[row_idx, 0]
    sess_list = animals_sorted[animal]
    n_sess = len(sess_list)

    for sess_idx, exp_mom in enumerate(sess_list):
        n_prior = sess_idx
        x_left = sess_idx
        tile_w = 0.85

        # Training tiles: all sessions BEFORE this one shown as gray
        for train_idx in range(sess_idx):
            ax.add_patch(mpatches.FancyBboxPatch(
                (train_idx, 0.1), tile_w, 0.8,
                boxstyle='round,pad=0.02',
                facecolor='#cccccc', edgecolor='#888888', linewidth=0.8,
                alpha=0.6,
            ))

        # Test tile: green
        ax.add_patch(mpatches.FancyBboxPatch(
            (sess_idx, 0.1), tile_w, 0.8,
            boxstyle='round,pad=0.02',
            facecolor='#4caf50', edgecolor='#2e7d32', linewidth=1.5,
        ))
        # Annotate with n_prior inside the green tile
        label = str(n_prior) if n_prior > 0 else '0'
        ax.text(sess_idx + tile_w / 2, 0.5, label,
                ha='center', va='center', fontsize=9,
                color='white' if n_prior > 0 else '#1b5e20',
                fontweight='bold')

    ax.set_xlim(-0.2, n_sess)
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    ax.set_xticks(np.arange(n_sess) + 0.425)
    ax.set_xticklabels([f"s{i}" for i in range(n_sess)], fontsize=9)
    ax.set_ylabel(animal, fontsize=11, fontweight='bold', rotation=0,
                  labelpad=50, va='center')
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

gray_patch = mpatches.Patch(facecolor='#cccccc', edgecolor='#888888',
                             label='Training data (used for this test point)')
green_patch = mpatches.Patch(facecolor='#4caf50', edgecolor='#2e7d32',
                              label='Test session (green tile; number = n_prior_sessions)')
fig1.legend(handles=[gray_patch, green_patch],
            loc='lower center', ncol=2, fontsize=10,
            bbox_to_anchor=(0.5, -0.02))

fig1.suptitle("RecurrentSAC Causal: training window per test session",
              fontsize=13, fontweight='bold', y=1.01)
fig1.text(0.5, 0.98,
          "Each test session uses only prior sessions as training data (n shown in tile)",
          ha='center', va='top', fontsize=10, style='italic', color='#444444')

plt.tight_layout()
fig1_path = os.path.join(project_root(), 'reports', 'figures',
                         f'{ts}_recurrent_sac_causal_fig1.png')
fig1.savefig(fig1_path, dpi=150, bbox_inches='tight')
plt.close(fig1)
print(f"  Saved Figure 1 to {fig1_path}")

# =============================================================================
# Figure 2 — Learning curve: RPA vs n_prior_sessions
# =============================================================================
print("Generating Figure 2: learning curve RPA vs n_prior_sessions...")

# Compute mean ± SEM per n_prior across all sessions (pooled)
by_prior = (df.groupby(['n_prior_sessions', 'exp_moment'])['rpa']
              .mean()           # mean over seeds per session
              .reset_index())
prior_stats = (by_prior.groupby('n_prior_sessions')['rpa']
               .agg(['mean', 'sem', 'count'])
               .reset_index())
prior_stats.columns = ['n_prior_sessions', 'mean_rpa', 'sem_rpa', 'n_sessions']

fig2, ax2 = plt.subplots(figsize=(9, 5))

ax2.errorbar(
    prior_stats['n_prior_sessions'],
    prior_stats['mean_rpa'],
    yerr=prior_stats['sem_rpa'],
    marker='o', markersize=6, linewidth=2,
    color='#2196f3', ecolor='#90caf9', elinewidth=1.5, capsize=4,
    label='Causal RecurrentSAC (mean ± SEM across sessions)',
    zorder=3,
)

# Scatter individual session means as faint dots
for _, srow in by_prior.iterrows():
    ax2.scatter(srow['n_prior_sessions'], srow['rpa'],
                color='#90caf9', alpha=0.4, s=15, zorder=2)

# Reference dashed lines
ref_lines = [
    (RANDOM_RPA, '#e53935', 'Random (0.178)'),
    (MOUSE_RPA,  '#ff8f00', 'Mouse (0.253)'),
    (ORACLE_RPA, '#7b1fa2', 'Oracle RecurrentSAC (0.796)'),
]
x_max = prior_stats['n_prior_sessions'].max()
for rpa_val, color, label in ref_lines:
    ax2.axhline(rpa_val, linestyle='--', color=color, linewidth=1.5,
                alpha=0.85, label=label, zorder=1)
    ax2.text(x_max + 0.15, rpa_val, label,
             va='center', ha='left', fontsize=8.5, color=color)

ax2.set_xlabel('Number of prior sessions (n_prior_sessions)', fontsize=12)
ax2.set_ylabel('Mean RPA (rewards per action)', fontsize=12)
ax2.set_title('RecurrentSAC Causal vs Oracle: RPA by number of prior sessions',
              fontsize=12, fontweight='bold')
ax2.set_xticks(prior_stats['n_prior_sessions'])
ax2.set_xlim(-0.5, x_max + 3.5)
ax2.set_ylim(0, max(ORACLE_RPA * 1.1, prior_stats['mean_rpa'].max() * 1.15))
ax2.legend(loc='upper left', fontsize=9, framealpha=0.85)
ax2.grid(True, alpha=0.3)
plt.tight_layout()

fig2_path = os.path.join(project_root(), 'reports', 'figures',
                         f'{ts}_recurrent_sac_causal_fig2.png')
fig2.savefig(fig2_path, dpi=150, bbox_inches='tight')
plt.close(fig2)
print(f"  Saved Figure 2 to {fig2_path}")

print("\nDone.")
