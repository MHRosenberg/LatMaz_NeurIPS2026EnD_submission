#!/usr/bin/env python3
"""
Generate Mouse vs POMCP vs POMCP_bio trajectory comparison GIFs.

Uses the user's plot_maze base layout with a fading trail overlay.
Output: a_visuals/trajectory_gifs/<yymmdd-hhmmss>_*_mouse_pomcp_bio.gif

Usage:
    source ~/miniconda3/etc/profile.d/conda.sh && conda activate latMaz_RL
    python code/generate_pomcp_bio_gifs.py
"""
import os
import sys
import re
import ast
import random
from datetime import datetime

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import imageio.v2 as imageio

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.dirname(__file__))
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from utils_latMaz import (
    load_maze, get_most_recent_file, get_adj_states,
    displacement_to_compass_heading, allo_actions_one_hot_dict,
)
from yoked_rl_runner import YokedRLRunner, AlgoConfig
from shelved_agents.pomcp_oracle import POMCPAgent
from advanced_agents import (
    POMCPbioAlloAgent, MazeState, MazeTransitionModel,
    MazeObservationModel, MazeRewardModel, MazePolicyModel,
    MazeRolloutPolicy, MazeBlackboxModel,
)
import pomdp_py
from pomdp_py import POMCP as POMCPPlanner

OUT_DIR = os.path.join('..', 'a_visuals', 'trajectory_gifs')
os.makedirs(OUT_DIR, exist_ok=True)

AGENT_COLORS = {
    'Mouse':     '#ff7f0e',  # orange
    'POMCP':     '#2ca02c',  # green
    'POMCP_bio': '#9467bd',  # purple
}

MODE_COLORS = {
    'explore':  '#e377c2',   # pink
    'navigate': '#bcbd22',   # olive
    'exploit':  '#9467bd',   # purple
    'start':    '#9467bd',
}


# ---------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------
def load_data():
    yoking_df = pd.read_csv(
        get_most_recent_file('../yoked_dfs/*animal_to_agent_yoking_info*.csv'),
        dtype={'animal_ID': str})
    yoking_df['exp_moment'] = yoking_df['csv_data_path'].apply(
        lambda p: re.search(r'(\d{6}-\d{6})', str(p)).group(1)
        if re.search(r'(\d{6}-\d{6})', str(p)) else None)
    yoking_df['date_part'] = yoking_df['exp_moment'].str[:6].astype(int)

    rwd_path = get_most_recent_file('../data_out/*rewarded_states*.csv')
    rwd_df = pd.read_csv(rwd_path) if rwd_path else None

    filtered = yoking_df[
        (yoking_df['date_part'] > 251001) &
        (yoking_df['n_states_visited'] > 50)
    ]
    return yoking_df, rwd_df, filtered


# ---------------------------------------------------------------
# Get mouse trajectory from yoking data
# ---------------------------------------------------------------
def get_mouse_trajectory(yoke_row, rewards):
    """Parse the mouse's trajectory from yoking data."""
    states_str = yoke_row['states_visited']
    states = ast.literal_eval(states_str)

    # Reconstruct rewards along trajectory
    rwd_state = rewards.copy()
    min_rwd = 2
    traj = [(states[0], 0.0)]
    for i in range(1, len(states)):
        node = states[i]
        r = float(rwd_state[node])
        rwd_state[node] = 0.0
        if sum(rwd_state) < min_rwd:
            rwd_state = rewards.copy()
        traj.append((node, r))
    return traj


# ---------------------------------------------------------------
# Run POMCP and capture trajectory
# ---------------------------------------------------------------
def run_pomcp_trajectory(adj_mat, st_positions, start_node, rewards,
                         n_actions, reset_val, seed=42):
    """Run POMCP and return trajectory as [(node, reward), ...]."""
    np.random.seed(seed)
    random.seed(seed)
    cfg = AlgoConfig()

    rwd_tuple = tuple(rewards.tolist())
    transition = MazeTransitionModel(adj_mat, st_positions, rwd_tuple, reset_val)
    observation = MazeObservationModel(adj_mat, st_positions)
    reward_model = MazeRewardModel()
    agent_policy = MazePolicyModel()
    rollout_policy = MazeRolloutPolicy()
    blackbox = MazeBlackboxModel(transition, observation, reward_model)

    init_state = MazeState(start_node, rwd_tuple)
    init_belief = pomdp_py.Particles([init_state] * 100)
    agent = pomdp_py.Agent(
        init_belief, agent_policy, transition, observation,
        reward_model, blackbox_model=blackbox)

    pomcp_cfg = cfg.POMCP
    planner = POMCPPlanner(
        max_depth=pomcp_cfg['max_depth'],
        discount_factor=pomcp_cfg['discount'],
        num_sims=pomcp_cfg['num_sims'],
        exploration_const=pomcp_cfg['exploration_const'],
        rollout_policy=rollout_policy)

    trajectory = [(start_node, 0.0)]
    current_state = init_state
    for step in range(n_actions):
        action = planner.plan(agent)
        next_state = transition.sample(current_state, action)
        obs = observation.sample(next_state, action)
        step_reward = reward_model.sample(current_state, action, next_state)
        planner.update(agent, action, obs)
        trajectory.append((next_state.node, step_reward))
        current_state = next_state

    return trajectory


# ---------------------------------------------------------------
# Drawing: use plot_maze style with fading trail overlay
# ---------------------------------------------------------------
def draw_maze_base(ax, adj_mat, st_positions):
    """Draw maze edges and node labels using the user's plot_maze style."""
    # Edges: thick black lines with dots at nodes (plot_maze style)
    for i, row in enumerate(adj_mat):
        connected = np.argwhere(adj_mat[i] == 1).flatten().tolist()
        pt1 = st_positions[i]
        for j in connected:
            if j > i:  # only draw once
                pt2 = st_positions[j]
                ax.plot([pt1[0], pt2[0]], [pt1[1], pt2[1]],
                        '-o', c='k', linewidth=3, markersize=4, zorder=1)

    # Node labels
    for node_idx, (x, y) in enumerate(st_positions):
        ax.annotate(node_idx, (x, y), textcoords="offset points",
                    xytext=(9, 7), ha='center', fontsize=6, color='#555')


def draw_rewards(ax, st_positions, rwd_state):
    """Draw reward pellets."""
    for node_idx in range(len(rwd_state)):
        if rwd_state[node_idx] > 0:
            ax.plot(st_positions[node_idx, 0], st_positions[node_idx, 1],
                    'o', markersize=30, color='deepskyblue', alpha=0.5,
                    zorder=2)


def draw_trail(ax, st_positions, trajectory, step, color, trail_len=15):
    """Draw fading trajectory trail."""
    trail_start = max(0, step - trail_len)
    for t in range(trail_start, min(step, len(trajectory) - 1)):
        n1 = trajectory[t][0]
        n2 = trajectory[t + 1][0]
        alpha = 0.15 + 0.85 * (t - trail_start) / max(1, step - trail_start)
        ax.plot([st_positions[n1, 0], st_positions[n2, 0]],
                [st_positions[n1, 1], st_positions[n2, 1]],
                '-', color=color, linewidth=2.5, alpha=alpha, zorder=4)


def draw_agent(ax, st_positions, node, color):
    """Draw current position marker."""
    ax.plot(st_positions[node, 0], st_positions[node, 1],
            'o', markersize=13, color=color,
            markeredgecolor='white', markeredgewidth=2, zorder=5)


def compute_reward_state(trajectory, step, rwd_initial, min_rwd=2):
    """Track reward state through trajectory up to given step."""
    rwd_state = rwd_initial.copy()
    total_reward = 0
    for t in range(min(step + 1, len(trajectory))):
        node, reward = trajectory[t][0], trajectory[t][1]
        total_reward += reward
        if reward > 0 and node < len(rwd_state):
            rwd_state[node] = 0
            if sum(rwd_state) < min_rwd:
                rwd_state = rwd_initial.copy()
    return rwd_state, total_reward


def create_three_panel_gif(adj_mat, st_positions, mouse_traj, pomcp_traj,
                           bio_traj_with_modes, rwd_initial, session_label,
                           output_path, step_skip=2, fps=10):
    """Create 3-panel comparison: Mouse | POMCP | POMCP_bio."""
    agents = [
        ('Mouse', mouse_traj, AGENT_COLORS['Mouse']),
        ('POMCP', pomcp_traj, AGENT_COLORS['POMCP']),
        ('POMCP_bio', bio_traj_with_modes, AGENT_COLORS['POMCP_bio']),
    ]
    max_steps = max(len(t) for _, t, _ in agents)

    frames = []
    for step in range(0, max_steps, step_skip):
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        for ax, (label, traj, base_color) in zip(axes, agents):
            draw_maze_base(ax, adj_mat, st_positions)

            capped = min(step, len(traj) - 1)
            rwd_state, total_r = compute_reward_state(
                traj, capped, rwd_initial)
            draw_rewards(ax, st_positions, rwd_state)

            # For POMCP_bio, color trail by mode
            if label == 'POMCP_bio' and len(traj[0]) == 3:
                trail_start = max(0, capped - 15)
                for t in range(trail_start, min(capped, len(traj) - 1)):
                    n1 = traj[t][0]
                    n2 = traj[t + 1][0]
                    mode = traj[t + 1][2] if len(traj[t + 1]) == 3 else 'exploit'
                    mc = MODE_COLORS.get(mode, base_color)
                    alpha = 0.15 + 0.85 * (t - trail_start) / max(1, capped - trail_start)
                    ax.plot([st_positions[n1, 0], st_positions[n2, 0]],
                            [st_positions[n1, 1], st_positions[n2, 1]],
                            '-', color=mc, linewidth=2.5, alpha=alpha, zorder=4)
                # Current mode label
                if capped < len(traj) and len(traj[capped]) == 3:
                    cur_mode = traj[capped][2]
                else:
                    cur_mode = ''
            else:
                draw_trail(ax, st_positions, traj, capped, base_color)
                cur_mode = ''

            if capped < len(traj):
                draw_agent(ax, st_positions, traj[capped][0], base_color)

            finished = " (done)" if step >= len(traj) - 1 else ""
            mode_str = f"  [{cur_mode}]" if cur_mode else ""
            ax.set_title(f"{label}{finished}{mode_str}\n"
                         f"Step {capped}/{len(traj)-1}  R={total_r:.0f}",
                         fontsize=10, fontweight='bold')
            ax.set_aspect('equal')
            ax.axis('off')

        fig.suptitle(session_label, fontsize=11, y=0.02, color='grey')
        fig.tight_layout(pad=0.5, rect=[0, 0.04, 1, 1])

        fig.canvas.draw()
        buf = fig.canvas.buffer_rgba()
        data = np.asarray(buf)[:, :, :3].copy()
        frames.append(data)
        plt.close(fig)

    # Hold last frame
    if frames:
        for _ in range(int(fps * 2)):
            frames.append(frames[-1])

    imageio.mimsave(output_path, frames, format='GIF',
                    duration=1.0 / fps, loop=0)
    print(f"  Saved: {output_path} ({len(frames)} frames)")


# ---------------------------------------------------------------
# Main
# ---------------------------------------------------------------
def main():
    ts = datetime.now().strftime('%y%m%d-%H%M%S')
    yoking_df, rwd_df, filtered = load_data()
    runner = YokedRLRunner(yoking_df, rwd_df,
                           maze_dir='../data_in/mazes',
                           output_dir='../data_out/rl_sims')

    # Select 3 exemplar sessions: longest per maze
    exemplars = []
    for maze in filtered['adj_file'].unique():
        maze_sessions = filtered[filtered['adj_file'] == maze].sort_values(
            'n_states_visited', ascending=False)
        if len(maze_sessions) > 0:
            exemplars.append(maze_sessions.iloc[0]['exp_moment'])

    print(f"Exemplar sessions: {exemplars}")
    cfg = AlgoConfig()

    for session in exemplars:
        yoke_row = yoking_df[yoking_df['exp_moment'] == session].iloc[0]
        adj_mat, st_positions = runner._load_maze(
            yoke_row['adj_file'], yoke_row['st_pos_file'])
        rewards, reset_val = runner._get_rewarded_states(
            session, len(st_positions))
        n_actions = int(yoke_row['n_states_visited']) - 1
        start_node = int(yoke_row['start_state'])
        maze_name = yoke_row['adj_file'].split('_')[-1].replace(
            '-adjacency.csv', '')
        animal = yoke_row['animal_ID']

        print(f"\n{'='*60}")
        print(f"  Session {session} (animal {animal}, {maze_name}, "
              f"{n_actions} actions)")
        print(f"{'='*60}")

        step_skip = 1  # 1 sim step per frame

        # 1. Mouse trajectory
        print("  Extracting mouse trajectory...")
        mouse_traj = get_mouse_trajectory(yoke_row, rewards)
        mouse_r = sum(r for _, r in mouse_traj)
        print(f"    Mouse: {len(mouse_traj)} steps, R={mouse_r:.0f}")

        # 2. POMCP trajectory
        print("  Running POMCP...")
        pomcp_traj = run_pomcp_trajectory(
            adj_mat, st_positions, start_node, rewards, n_actions,
            reset_val, seed=42)
        pomcp_r = sum(r for _, r in pomcp_traj)
        print(f"    POMCP: {len(pomcp_traj)} steps, R={pomcp_r:.0f}")

        # 3. POMCP_bio trajectory (with mode annotations)
        print("  Running POMCP_bio...")
        bio_agent = POMCPbioAlloAgent(**cfg.POMCP_bio)
        bio_r, bio_traj = bio_agent.run_episode(
            adj_mat, st_positions, start_node, rewards.copy(),
            n_actions, reset_val, seed=42, return_trajectory=True)
        print(f"    POMCP_bio: {len(bio_traj)} steps, R={bio_r:.0f}")
        from collections import Counter
        modes = Counter(m for _, _, m in bio_traj)
        print(f"    Modes: {dict(modes)}")

        # Create comparison GIF
        output_path = os.path.join(
            OUT_DIR, f"{ts}_{session}_mouse_pomcp_bio.gif")
        session_label = f"{session} | animal {animal} | {maze_name} | {n_actions} actions"
        create_three_panel_gif(
            adj_mat, st_positions, mouse_traj, pomcp_traj, bio_traj,
            rewards, session_label, output_path,
            step_skip=step_skip, fps=2)

    print(f"\nAll GIFs saved to: {OUT_DIR}")


if __name__ == '__main__':
    main()
