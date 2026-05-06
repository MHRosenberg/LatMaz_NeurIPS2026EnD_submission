#!/usr/bin/env python3
"""
Generate animated GIF visualizations of agent trajectories on mazes.

Creates side-by-side trajectory animations for Mouse, DQN baseline,
DQN pretrained, and POMCP on selected exemplar sessions. Output is
formatted for embedding in presentation slides.

Usage:
    source ~/miniconda3/etc/profile.d/conda.sh && conda activate latMaz_RL
    python code/visualize_trajectories.py
"""
import os
import sys
import gc
from datetime import datetime
import pandas as pd
import numpy as np
from copy import deepcopy

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import imageio.v2 as imageio
from PIL import Image as PILImage

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.dirname(__file__))

from utils_latMaz import load_maze, get_adj_states
from yoked_rl_runner import (
    YokedRLRunner, AlgoConfig, GraphMazeEnv, HistoryConcatWrapper,
    _POMCP_AVAILABLE,
)
from utils_latMaz_RL import StopAfterFirstEpisode
from experiment_config import (
    OBS_CONFIG, load_data as _load_data, load_yoking_df, load_reward_df,
    filter_sessions, project_root,
)

os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Output directory
OUT_DIR = os.path.join(project_root(), 'a_visuals', 'trajectory_gifs')
os.makedirs(OUT_DIR, exist_ok=True)

# Agent colors
COLORS = {
    'Mouse': '#ff7f0e',
    'DQN_baseline': '#1f77b4',
    'DQN_pretrained': '#0b5394',
    'POMCP': '#2ca02c',
}


def load_data():
    """Load yoking data and select exemplar sessions."""
    yoking_df = load_yoking_df()
    rwd_df = load_reward_df()
    sessions = filter_sessions(yoking_df)

    filtered = yoking_df[yoking_df['exp_moment'].isin(sessions)]

    # Select exemplar sessions: longest action budgets per maze
    exemplars = []
    for maze in filtered['adj_file'].unique():
        maze_sessions = filtered[filtered['adj_file'] == maze].sort_values(
            'n_states_visited', ascending=False)
        if len(maze_sessions) > 0:
            exemplars.append(maze_sessions.iloc[0]['exp_moment'])

    return yoking_df, rwd_df, exemplars


def run_agent_and_get_trajectory(runner, yoking_df, session, model_name,
                                  pretrained=False, seed=42):
    """Run an agent and return its trajectory as list of (node, reward)."""
    yoke_row = yoking_df[yoking_df['exp_moment'] == session].iloc[0]
    adj_mat, st_positions = runner._load_maze(
        yoke_row['adj_file'], yoke_row['st_pos_file'])
    rewards, reset_val = runner._get_rewarded_states(
        session, len(st_positions))
    n_actions = int(yoke_row['n_states_visited']) - 1

    if pretrained:
        pretrain_rows = runner.get_pretrain_sessions(yoke_row)
        # We need to capture the observation log from the pretrained run
        # Run pretraining on prior sessions, then run target with log capture
        from yoked_rl_runner import DRQNAgent
        is_drqn = model_name.startswith('DRQN')
        model = None
        prev_env = None

        for i, (_, ptr_row) in enumerate(pretrain_rows.iterrows()):
            p_adj, p_pos = runner._load_maze(
                ptr_row['adj_file'], ptr_row['st_pos_file'])
            p_exp = ptr_row['exp_moment']
            p_rwd, p_reset = runner._get_rewarded_states(p_exp, len(p_pos))
            p_n_actions = int(ptr_row['n_states_visited']) - 1

            env = runner._create_env(
                p_adj, p_pos, int(ptr_row['start_state']), p_rwd, p_n_actions,
                OBS_CONFIG['obs_type'], OBS_CONFIG['action_type'], 0,
                OBS_CONFIG['include_prev_action'],
                OBS_CONFIG['include_prev_reward'], p_reset)

            if model is None:
                model, total_ts = runner._make_model(
                    model_name, env, p_n_actions, seed)
            else:
                total_ts = max(p_n_actions, 1000)
                model.set_env(env)

            if prev_env is not None:
                try:
                    prev_env.close()
                except Exception:
                    pass

            if is_drqn:
                model.learn()
            else:
                model.learn(total_timesteps=total_ts,
                            callback=[StopAfterFirstEpisode()],
                            reset_num_timesteps=False)
            prev_env = env

        # Target session
        env = runner._create_env(
            adj_mat, st_positions, int(yoke_row['start_state']), rewards,
            n_actions, OBS_CONFIG['obs_type'], OBS_CONFIG['action_type'], 0,
            OBS_CONFIG['include_prev_action'],
            OBS_CONFIG['include_prev_reward'], reset_val)

        if model is None:
            model, total_ts = runner._make_model(
                model_name, env, n_actions, seed)
        else:
            total_ts = max(n_actions, 1000)
            model.set_env(env)

        if prev_env is not None:
            try:
                prev_env.close()
            except Exception:
                pass

        if is_drqn:
            model.learn()
        else:
            model.learn(total_timesteps=total_ts,
                        callback=[StopAfterFirstEpisode()],
                        reset_num_timesteps=False)

        base_env = env.env if hasattr(env, 'env') else env
        trajectory = [(obs['node_current'], obs.get('reward', 0))
                      for obs in base_env.observation_log]
        del model
        try:
            env.close()
        except Exception:
            pass
        gc.collect()
        return trajectory

    else:
        # Simple baseline run
        env = runner._create_env(
            adj_mat, st_positions, int(yoke_row['start_state']), rewards,
            n_actions, OBS_CONFIG['obs_type'], OBS_CONFIG['action_type'], 0,
            OBS_CONFIG['include_prev_action'],
            OBS_CONFIG['include_prev_reward'], reset_val)

        np.random.seed(seed)

        if model_name == 'POMCP':
            # POMCP doesn't use gym env - use its own trajectory tracking
            if not _POMCP_AVAILABLE:
                return []
            from shelved_agents.pomcp_oracle import POMCPAgent
            from advanced_agents import (
                MazeState, MazeTransitionModel,
                MazeObservationModel, MazeRewardModel,
                MazePolicyModel, MazeRolloutPolicy, MazeBlackboxModel,
            )
            import pomdp_py
            from pomdp_py import POMCP as POMCPPlanner
            import random
            random.seed(seed)

            rwd_tuple = tuple(rewards.tolist())
            transition = MazeTransitionModel(adj_mat, st_positions, rwd_tuple, reset_val)
            observation = MazeObservationModel(adj_mat, st_positions)
            reward_model = MazeRewardModel()
            agent_policy = MazePolicyModel()
            rollout_policy = MazeRolloutPolicy()
            blackbox = MazeBlackboxModel(transition, observation, reward_model)

            init_state = MazeState(int(yoke_row['start_state']), rwd_tuple)
            init_belief = pomdp_py.Particles([init_state] * 100)
            agent = pomdp_py.Agent(
                init_belief, agent_policy, transition, observation,
                reward_model, blackbox_model=blackbox)

            planner = POMCPPlanner(
                max_depth=80, discount_factor=0.99, num_sims=1000,
                exploration_const=20.0, rollout_policy=rollout_policy)

            trajectory = [(int(yoke_row['start_state']), 0)]
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

        # Standard RL agent
        ModelClass = runner.MODEL_CLASSES[model_name]
        kwargs = deepcopy(runner.algo_config.get(model_name))
        policy_str = "MlpPolicy"
        total_ts = max(n_actions, 1000)
        if 'n_steps' in kwargs and kwargs['n_steps'] > n_actions:
            kwargs['n_steps'] = max(n_actions // 2, 8)

        model = ModelClass(policy_str, env, verbose=0, device="cpu",
                           seed=seed, **kwargs)
        model.learn(total_timesteps=total_ts,
                    callback=[StopAfterFirstEpisode()])

        base_env = env.env if hasattr(env, 'env') else env
        trajectory = [(obs['node_current'], obs.get('reward', 0))
                      for obs in base_env.observation_log]
        del model
        try:
            env.close()
        except Exception:
            pass
        gc.collect()
        return trajectory


def get_mouse_trajectory(yoking_df, session):
    """Extract mouse trajectory from yoking data (visit sequence)."""
    yoke_row = yoking_df[yoking_df['exp_moment'] == session].iloc[0]
    csv_path = yoke_row['csv_data_path']
    # The mouse trajectory is not directly stored - we only have n_states_visited
    # Return None to indicate we should simulate a placeholder
    return None


def draw_maze_frame(adj_mat, st_positions, trajectory, step, rwd_initial,
                    color='#1f77b4', title='', figsize=(5, 5)):
    """Draw a single frame of the trajectory on the maze.

    Returns a numpy array (RGB image).
    """
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    # Draw edges
    for i in range(len(adj_mat)):
        for j in range(i + 1, len(adj_mat)):
            if adj_mat[i, j] > 0:
                ax.plot([st_positions[i, 0], st_positions[j, 0]],
                        [st_positions[i, 1], st_positions[j, 1]],
                        '-', color='#cccccc', linewidth=2, zorder=1)

    # Draw reward nodes (current state of rewards)
    # Track which rewards have been collected
    rwd_state = rwd_initial.copy()
    total_reward = 0
    for t in range(min(step + 1, len(trajectory))):
        node, reward = trajectory[t]
        total_reward += reward
        if reward > 0 and node < len(rwd_state):
            rwd_state[node] = 0
            # Check rebait
            if sum(rwd_state) < 2:
                rwd_state = rwd_initial.copy()

    # Draw reward positions
    for node_idx in range(len(rwd_state)):
        if rwd_state[node_idx] > 0:
            ax.plot(st_positions[node_idx, 0], st_positions[node_idx, 1],
                    'o', markersize=30, color='deepskyblue', alpha=0.5,
                    zorder=2)

    # Draw all nodes
    ax.scatter(st_positions[:, 0], st_positions[:, 1], s=30,
               c='#333333', zorder=3)

    # Draw trajectory path (fading trail)
    trail_start = max(0, step - 15)
    if step > 0:
        for t in range(trail_start, min(step, len(trajectory) - 1)):
            n1 = trajectory[t][0]
            n2 = trajectory[t + 1][0]
            alpha = 0.2 + 0.8 * (t - trail_start) / max(1, step - trail_start)
            ax.plot([st_positions[n1, 0], st_positions[n2, 0]],
                    [st_positions[n1, 1], st_positions[n2, 1]],
                    '-', color=color, linewidth=2.5, alpha=alpha, zorder=4)

    # Draw current position
    if step < len(trajectory):
        current_node = trajectory[step][0]
        ax.plot(st_positions[current_node, 0], st_positions[current_node, 1],
                'o', markersize=14, color=color, markeredgecolor='white',
                markeredgewidth=2, zorder=5)

    ax.set_title(f"{title}\nStep {step}/{len(trajectory)-1}  "
                 f"Reward: {total_reward:.0f}", fontsize=10)
    ax.set_aspect('equal')
    ax.axis('off')
    fig.tight_layout(pad=0.5)

    # Convert to numpy array
    fig.canvas.draw()
    buf = fig.canvas.buffer_rgba()
    data = np.asarray(buf)[:, :, :3].copy()  # RGBA -> RGB
    plt.close(fig)
    return data


def create_trajectory_gif(adj_mat, st_positions, trajectory, rwd_initial,
                          color, title, output_path, step_skip=1,
                          fps=10, figsize=(5, 5)):
    """Create an animated GIF of an agent's trajectory."""
    frames = []
    n_steps = len(trajectory)

    for step in range(0, n_steps, step_skip):
        frame = draw_maze_frame(adj_mat, st_positions, trajectory, step,
                                rwd_initial, color=color, title=title,
                                figsize=figsize)
        frames.append(frame)

    # Add final frame (hold)
    if frames:
        for _ in range(fps * 2):  # Hold last frame for 2 seconds
            frames.append(frames[-1])

    imageio.mimsave(output_path, frames, format='GIF',
                    duration=1.0/fps, loop=0)
    print(f"  Saved GIF: {output_path} ({len(frames)} frames)")
    return output_path


def create_comparison_gif(adj_mat, st_positions, trajectories, rwd_initial,
                          colors, labels, output_path, step_skip=2,
                          fps=10):
    """Create side-by-side comparison GIF of multiple agents."""
    n_agents = len(trajectories)
    max_steps = max(len(t) for t in trajectories)

    frames = []
    figsize = (5 * n_agents, 5)

    for step in range(0, max_steps, step_skip):
        fig, axes = plt.subplots(1, n_agents, figsize=figsize)
        if n_agents == 1:
            axes = [axes]

        for ax, traj, color, label in zip(axes, trajectories, colors, labels):
            # Draw edges
            for i in range(len(adj_mat)):
                for j in range(i + 1, len(adj_mat)):
                    if adj_mat[i, j] > 0:
                        ax.plot([st_positions[i, 0], st_positions[j, 0]],
                                [st_positions[i, 1], st_positions[j, 1]],
                                '-', color='#cccccc', linewidth=1.5, zorder=1)

            # Track rewards
            rwd_state = rwd_initial.copy()
            total_reward = 0
            capped_step = min(step, len(traj) - 1)
            for t in range(capped_step + 1):
                node, reward = traj[t]
                total_reward += reward
                if reward > 0 and node < len(rwd_state):
                    rwd_state[node] = 0
                    if sum(rwd_state) < 2:
                        rwd_state = rwd_initial.copy()

            # Draw rewards
            for node_idx in range(len(rwd_state)):
                if rwd_state[node_idx] > 0:
                    ax.plot(st_positions[node_idx, 0], st_positions[node_idx, 1],
                            'o', markersize=24, color='deepskyblue', alpha=0.4,
                            zorder=2)

            # Draw nodes
            ax.scatter(st_positions[:, 0], st_positions[:, 1], s=20,
                       c='#555555', zorder=3)

            # Draw trail
            trail_start = max(0, capped_step - 12)
            for t in range(trail_start, min(capped_step, len(traj) - 1)):
                n1 = traj[t][0]
                n2 = traj[t + 1][0]
                alpha = 0.2 + 0.8 * (t - trail_start) / max(1, capped_step - trail_start)
                ax.plot([st_positions[n1, 0], st_positions[n2, 0]],
                        [st_positions[n1, 1], st_positions[n2, 1]],
                        '-', color=color, linewidth=2, alpha=alpha, zorder=4)

            # Current position
            if capped_step < len(traj):
                cn = traj[capped_step][0]
                ax.plot(st_positions[cn, 0], st_positions[cn, 1],
                        'o', markersize=12, color=color,
                        markeredgecolor='white', markeredgewidth=1.5, zorder=5)

            finished = " (done)" if step >= len(traj) - 1 else ""
            ax.set_title(f"{label}{finished}\n"
                         f"Step {min(step, len(traj)-1)}/{len(traj)-1}  "
                         f"R={total_reward:.0f}", fontsize=9, fontweight='bold')
            ax.set_aspect('equal')
            ax.axis('off')

        fig.tight_layout(pad=0.3)

        # Convert to numpy
        fig.canvas.draw()
        buf = fig.canvas.buffer_rgba()
        data = np.asarray(buf)[:, :, :3].copy()  # RGBA -> RGB
        frames.append(data)
        plt.close(fig)

    # Hold last frame
    if frames:
        for _ in range(fps * 2):
            frames.append(frames[-1])

    imageio.mimsave(output_path, frames, format='GIF',
                    duration=1.0/fps, loop=0)
    print(f"  Saved comparison GIF: {output_path} ({len(frames)} frames)")


def main():
    ts = datetime.now().strftime('%y%m%d-%H%M%S')
    yoking_df, rwd_df, exemplar_sessions = load_data()

    runner = YokedRLRunner(yoking_df, rwd_df, maze_dir='../data_in/mazes',
                           output_dir='../data_out/rl_sims')

    print(f"Exemplar sessions: {exemplar_sessions}")

    for session in exemplar_sessions:
        yoke_row = yoking_df[yoking_df['exp_moment'] == session].iloc[0]
        adj_mat, st_positions = runner._load_maze(
            yoke_row['adj_file'], yoke_row['st_pos_file'])
        rewards, reset_val = runner._get_rewarded_states(
            session, len(st_positions))
        n_actions = int(yoke_row['n_states_visited']) - 1
        maze_name = yoke_row['adj_file'].split('_')[-1].replace('-adjacency.csv', '')
        animal = yoke_row['animal_ID']

        print(f"\n{'='*60}")
        print(f"  Session {session} (animal {animal}, {maze_name}, "
              f"{n_actions} actions)")
        print(f"{'='*60}")

        # Determine step_skip based on action budget
        step_skip = max(1, n_actions // 150)  # Aim for ~150 frames

        # Run agents and collect trajectories
        trajectories = {}
        agent_colors = {}
        agent_labels = {}

        # DQN baseline
        print("  Running DQN baseline...")
        traj = run_agent_and_get_trajectory(
            runner, yoking_df, session, 'DQN', pretrained=False, seed=42)
        if traj:
            trajectories['DQN_baseline'] = traj
            agent_colors['DQN_baseline'] = COLORS['DQN_baseline']
            agent_labels['DQN_baseline'] = 'DQN baseline'
            total_r = sum(r for _, r in traj)
            print(f"    DQN baseline: {len(traj)} steps, reward={total_r:.0f}")

        # DQN pretrained
        pretrain_rows = runner.get_pretrain_sessions(yoke_row)
        if len(pretrain_rows) > 0:
            print(f"  Running DQN pretrained ({len(pretrain_rows)} prior)...")
            traj = run_agent_and_get_trajectory(
                runner, yoking_df, session, 'DQN', pretrained=True, seed=42)
            if traj:
                trajectories['DQN_pretrained'] = traj
                agent_colors['DQN_pretrained'] = COLORS['DQN_pretrained']
                agent_labels['DQN_pretrained'] = 'DQN pretrained'
                total_r = sum(r for _, r in traj)
                print(f"    DQN pretrained: {len(traj)} steps, reward={total_r:.0f}")

        # POMCP
        if _POMCP_AVAILABLE:
            print("  Running POMCP...")
            traj = run_agent_and_get_trajectory(
                runner, yoking_df, session, 'POMCP', pretrained=False, seed=42)
            if traj:
                trajectories['POMCP'] = traj
                agent_colors['POMCP'] = COLORS['POMCP']
                agent_labels['POMCP'] = 'POMCP'
                total_r = sum(r for _, r in traj)
                print(f"    POMCP: {len(traj)} steps, reward={total_r:.0f}")

        # Create individual GIFs
        for agent_key, traj in trajectories.items():
            gif_path = os.path.join(
                OUT_DIR,
                f"{ts}_{session}_{agent_key}.gif")
            create_trajectory_gif(
                adj_mat, st_positions, traj, rewards,
                color=agent_colors[agent_key],
                title=f"{agent_labels[agent_key]} - {maze_name}",
                output_path=gif_path,
                step_skip=step_skip,
                fps=5)

        # Create comparison GIF (all agents side by side)
        if len(trajectories) >= 2:
            keys = list(trajectories.keys())
            comparison_path = os.path.join(
                OUT_DIR,
                f"{ts}_{session}_comparison.gif")
            create_comparison_gif(
                adj_mat, st_positions,
                [trajectories[k] for k in keys],
                rewards,
                [agent_colors[k] for k in keys],
                [agent_labels[k] for k in keys],
                output_path=comparison_path,
                step_skip=step_skip,
                fps=5)

    print(f"\nAll GIFs saved to: {OUT_DIR}")
    print("Embed in slides: drag GIF directly into PowerPoint/Keynote/Google Slides")


if __name__ == '__main__':
    main()
